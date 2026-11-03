"""Image handling, SAM output parsing and overlay rendering."""
from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import fastmask

MAX_SIDE = 1024
_RECORD = re.compile(
    r"(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([\d.]+))?\|>"
    r"<\|mask;x=0;y=0(?:;c=[\d.]+)?;data=(\d+),(\d+),(.*?)\|>",
    re.S,
)
PALETTE = {"verified": (22, 163, 74), "contradicted": (220, 38, 38), "not_visible": (217, 119, 6), "info": (37, 99, 235)}


@dataclass
class Instance:
    box: tuple[int, int, int, int]
    confidence: float
    mask: np.ndarray | None  # bool array sized to the box
    id: int = 0              # stable across frames of one video response

    @property
    def short_side(self) -> int:
        return min(self.box[2] - self.box[0], self.box[3] - self.box[1])


def prepare_image(raw: bytes) -> bytes:
    """Normalize any input image to an RGB JPEG of at most MAX_SIDE."""
    from PIL import ImageOps
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def parse_sam(text: str, min_confidence: float = 0.4) -> list[Instance]:
    out = []
    for m in _RECORD.finditer(text):
        conf = float(m.group(8)) if m.group(8) else 1.0
        if conf < min_confidence:
            continue
        box = tuple(int(m.group(i)) for i in range(2, 6))
        h, w, payload = int(m.group(9)), int(m.group(10)), m.group(11)
        try:
            mask = fastmask.decode(payload, h, w)
        except Exception:
            mask = None  # the box remains usable
        out.append(Instance(box, conf, mask, int(m.group(1))))
    return out


def parse_video(lines: list[str], min_confidence: float = 0.4) -> dict[int, list[Instance]]:
    """SAM video output, one line per frame, to {frame index: instances}."""
    frames: dict[int, list[Instance]] = {}
    for line in lines:
        m = re.match(r"\s*<(\d+)f>", line)
        if m:
            frames[int(m.group(1))] = parse_sam(line, min_confidence)
    return frames


def crop(jpeg: bytes, box: tuple[int, int, int, int], pad: float = 0.12, min_side: int = 512) -> bytes:
    im = Image.open(io.BytesIO(jpeg)).convert("RGB")
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    c = im.crop((max(0, x1 - px), max(0, y1 - py), min(im.width, x2 + px), min(im.height, y2 + py)))
    if min(c.size) < min_side:
        s = min_side / min(c.size)
        c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    c.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def sharpness(jpeg: bytes, box=None) -> float:
    a = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if box:
        a = a[box[1]:box[3], box[0]:box[2]]
    return float(cv2.Laplacian(a, cv2.CV_64F).var()) if a.size else 0.0


def _font(size: int):
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(jpeg: bytes, layers: list[dict]) -> bytes:
    """Draw masks, boxes and labels. Each layer: {instance, label, status}."""
    im = Image.open(io.BytesIO(jpeg)).convert("RGBA")
    tint = Image.new("RGBA", im.size, (0, 0, 0, 0))
    td = ImageDraw.Draw(tint)
    font = _font(max(14, im.width // 55))
    labels = []
    for layer in layers:
        inst, color = layer["instance"], PALETTE.get(layer.get("status", "info"), PALETTE["info"])
        x1, y1, x2, y2 = inst.box
        if inst.mask is not None and x2 > x1 and y2 > y1:
            m = Image.fromarray((inst.mask * 110).astype(np.uint8)).resize((x2 - x1, y2 - y1))
            tint.paste(Image.new("RGBA", m.size, color + (110,)), (x1, y1), m)
        td.rectangle((x1, y1, x2, y2), outline=color + (255,), width=max(3, im.width // 300))
        labels.append(((x1, y1), layer.get("label", ""), color))
    out = Image.alpha_composite(im, tint)
    d = ImageDraw.Draw(out)
    for (x, y), text, color in labels:
        if not text:
            continue
        l, t, r, b = d.textbbox((0, 0), text, font=font)
        tw, th = r - l + 12, b - t + 10
        x = min(x, max(0, out.width - tw))
        y = y - th if y >= th else y
        d.rectangle((x, y, x + tw, y + th), fill=color + (255,))
        d.text((x + 6, y + 3 - t), text, fill="white", font=font)
    buf = io.BytesIO()
    out.convert("RGB").save(buf, "JPEG", quality=86)
    return buf.getvalue()


def image_size(jpeg: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(jpeg)) as im:
        return im.size


def full_mask(inst: Instance, size: tuple[int, int]) -> np.ndarray:
    """The instance as a 0/255 mask the size of the frame."""
    w, h = size
    out = np.zeros((h, w), np.uint8)
    x1, y1, x2, y2 = inst.box
    if inst.mask is None:
        out[y1:y2, x1:x2] = 255
        return out
    mh, mw = inst.mask.shape
    rh, rw = min(mh, h - y1), min(mw, w - x1)
    out[y1:y1 + rh, x1:x1 + rw] = inst.mask[:rh, :rw].astype(np.uint8) * 255
    return out


def _main_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def mask_path(inst: Instance, size: tuple[int, int]) -> str:
    """SVG path of the instance outline in frame pixels, simplified so it can stream to the browser each tick."""
    contours, _ = cv2.findContours(full_mask(inst, size), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parts = []
    for c in contours:
        if cv2.contourArea(c) < 40:
            continue
        points = cv2.approxPolyDP(c, 1.6, True).reshape(-1, 2)
        parts.append("M" + "L".join(f"{x},{y}" for x, y in points) + "Z")
    return "".join(parts)


def rectify(jpeg: bytes, inst: Instance, long_side: int = 1100) -> bytes:
    """Flatten the evidence with its mask: a homography if it is four-cornered, otherwise a rotation.

    The result is shown beside the full frame, so anything the mask clipped can still be seen by the reader.
    """
    im = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = im.shape[:2]
    contour = _main_contour(full_mask(inst, (w, h)))
    if contour is None:
        return crop(jpeg, inst.box)
    hull = cv2.convexHull(contour)
    quad = cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    if angle > 45:
        angle, rw, rh = angle - 90, rh, rw
    pad = 1.10
    if len(quad) == 4 and cv2.contourArea(quad) > 0.85 * cv2.contourArea(hull):
        s = quad.sum(1)
        d = np.diff(quad.astype(float), axis=1).ravel()
        tl, br, tr, bl = quad[s.argmin()], quad[s.argmax()], quad[d.argmin()], quad[d.argmax()]
        centre = np.mean([tl, tr, br, bl], axis=0)
        source = np.float32([centre + (p - centre) * pad for p in (tl, tr, br, bl)])
        out_w = max(8, int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)) * pad))
        out_h = max(8, int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)) * pad))
        target = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
        flat = cv2.warpPerspective(im, cv2.getPerspectiveTransform(source, target), (out_w, out_h),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        turned = cv2.warpAffine(im, cv2.getRotationMatrix2D((cx, cy), angle, 1.0), (w, h), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        flat = cv2.getRectSubPix(turned, (max(8, int(rw * pad)), max(8, int(rh * pad))), (cx, cy))
    scale = long_side / max(flat.shape[:2])
    if scale > 1:
        flat = cv2.resize(flat, None, fx=min(scale, 4), fy=min(scale, 4), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", flat, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else crop(jpeg, inst.box)


def swatch(jpeg: bytes, inst: Instance) -> str:
    """Median colour of the pixels inside the mask, as hex."""
    im = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    mask = full_mask(inst, (im.shape[1], im.shape[0]))
    core = cv2.erode(mask, np.ones((7, 7), np.uint8))
    pixels = im[core > 0]
    if pixels.size == 0:
        return ""
    b, g, r = np.median(pixels, axis=0).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


A4_MM = (297.0, 210.0)


def long_side_px(inst: Instance, size: tuple[int, int]) -> tuple[float, float, list] | None:
    """Long and short side of the instance's tightest rotated rectangle, plus its corners."""
    contour = _main_contour(full_mask(inst, size))
    if contour is None:
        return None
    rect = cv2.minAreaRect(contour)
    a, b = rect[1]
    return max(a, b), min(a, b), cv2.boxPoints(rect).astype(int).tolist()


def measure(item: Instance, reference: Instance, size: tuple[int, int]) -> dict:
    """Length of the item's long side, using a flat sheet of A4 in the same frame as the scale.

    A skewed view changes the sheet's proportions, so a sheet that isn't A4-shaped in the image means no measurement.
    """
    item_rect, ref_rect = long_side_px(item, size), long_side_px(reference, size)
    if not item_rect or not ref_rect or ref_rect[1] <= 0:
        return {"mm": None, "reason": "no_outline"}
    ratio = ref_rect[0] / ref_rect[1]
    if abs(ratio - A4_MM[0] / A4_MM[1]) > 0.12:
        return {"mm": None, "reason": "reference_skewed", "ratio": round(ratio, 2)}
    mm_per_px = (A4_MM[0] / ref_rect[0] + A4_MM[1] / ref_rect[1]) / 2
    return {"mm": item_rect[0] * mm_per_px, "reason": "", "ratio": round(ratio, 2), "corners": item_rect[2],
            "reference_corners": ref_rect[2]}


def hero(jpeg: bytes, instances: list[Instance], backdrop: bytes) -> tuple[bytes, dict]:
    """Place the real product pixels on a generated backdrop. Only the surroundings are new.

    The function checks that every pixel well inside the mask is identical to the source before encoding.
    """
    src = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = src.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for inst in instances:
        mask = cv2.bitwise_or(mask, full_mask(inst, (w, h)))
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("empty mask")
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    margin = int(max(x2 - x1, y2 - y1) * 0.22)
    canvas = int(max(x2 - x1, y2 - y1) + 2 * margin)
    bg = cv2.imdecode(np.frombuffer(backdrop, np.uint8), cv2.IMREAD_COLOR)
    scale = max(canvas / bg.shape[1], canvas / bg.shape[0])
    bg = cv2.resize(bg, (int(bg.shape[1] * scale) + 1, int(bg.shape[0] * scale) + 1), interpolation=cv2.INTER_AREA)
    out = bg[:canvas, :canvas].astype(np.float32)
    ox, oy = (canvas - (x2 - x1)) // 2, (canvas - (y2 - y1)) // 2
    placed = np.zeros((canvas, canvas), np.uint8)
    placed[oy:oy + (y2 - y1), ox:ox + (x2 - x1)] = mask[y1:y2, x1:x2]
    product = np.zeros((canvas, canvas, 3), np.float32)
    product[oy:oy + (y2 - y1), ox:ox + (x2 - x1)] = src[y1:y2, x1:x2]
    # Soft contact shadow, then a slightly feathered edge. Pixels well inside the mask stay at alpha 1.
    drop = cv2.GaussianBlur(np.roll(placed, (int(canvas * 0.018), int(canvas * 0.008)), (0, 1)), (0, 0), canvas * 0.016)
    out *= (1 - (drop.astype(np.float32) / 255) * 0.38)[..., None]
    edge = cv2.GaussianBlur(placed, (0, 0), 1.2).astype(np.float32) / 255
    core = cv2.erode(placed, np.ones((7, 7), np.uint8)) > 0
    edge[core] = 1.0
    composed = out * (1 - edge[..., None]) + product * edge[..., None]
    final = np.clip(np.round(composed), 0, 255).astype(np.uint8)
    exact = bool(np.array_equal(final[core], product[core].astype(np.uint8)))
    ok, buf = cv2.imencode(".jpg", final, [cv2.IMWRITE_JPEG_QUALITY, 93])
    if not ok:
        raise ValueError("encode failed")
    return buf.tobytes(), {"product_pixels": int(core.sum()), "pixel_exact": exact}


def to_data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
