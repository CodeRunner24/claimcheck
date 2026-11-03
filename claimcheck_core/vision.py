"""Image handling, SAM output parsing and overlay rendering."""
from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass

import cv2
import numpy as np
from meta_sam_parser import SegmentationMask, decode_mask_to_raster
from PIL import Image, ImageDraw, ImageFont

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
        encoding = "one_bit" if payload.startswith("!") else "lossless"
        try:
            raster = decode_mask_to_raster(SegmentationMask(encoding, payload, w, h))
            mask = np.frombuffer(raster, dtype=np.uint8).reshape(h, w).astype(bool)
        except Exception:
            mask = None  # the box remains usable
        out.append(Instance(box, conf, mask))
    return out


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


def to_data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
