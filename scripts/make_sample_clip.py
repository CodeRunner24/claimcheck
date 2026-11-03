"""Build assets/samples/sample_take.webm: AI-generated stills moved like a shaky handheld camera. Disclosed in the app.

One black pocket t-shirt with a small hole near the hem, cut out with SAM 3.1 and laid on one oak table, so the far shot and
the close shot are the same object in the same place. The label close-up is a separate generated photo.
"""
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from claimcheck_core import meta_api as api
from claimcheck_core import vision
from claimcheck_core.take import FFMPEG

SAMPLES = ROOT / "assets" / "samples"
W, H, FPS = 1280, 720, 10
TABLE = SAMPLES / "take_far.jpg"  # an oak table with a very small shirt in the middle, always covered below
SHIRT = SAMPLES / "take_wide.jpg"
CUTOUT = SAMPLES / "take_shirt.png"


def cutout() -> np.ndarray:
    """The shirt with its mask as alpha, cached so the clip can be rebuilt without an API call."""
    if not CUTOUT.exists():
        found = vision.parse_sam(api.sam_raw("t-shirt", vision.prepare_image(SHIRT.read_bytes())))
        best = vision.largest(found)
        image = cv2.imdecode(np.frombuffer(vision.prepare_image(SHIRT.read_bytes()), np.uint8), cv2.IMREAD_COLOR)
        mask = vision.full_mask(best, (image.shape[1], image.shape[0]))
        x1, y1, x2, y2 = best.box
        rgba = np.dstack([image, cv2.GaussianBlur(mask, (0, 0), 1.0)])[y1:y2, x1:x2]
        cv2.imwrite(str(CUTOUT), rgba)
    return cv2.imread(str(CUTOUT), cv2.IMREAD_UNCHANGED)


def lay(table: np.ndarray, shirt: np.ndarray, height: int, cx: int, cy: int) -> np.ndarray:
    """Put the shirt on the table with a soft contact shadow."""
    scale = height / shirt.shape[0]
    piece = cv2.resize(shirt, (int(shirt.shape[1] * scale), height), interpolation=cv2.INTER_AREA)
    out = table.astype(np.float32)
    x0, y0 = cx - piece.shape[1] // 2, cy - height // 2
    alpha = np.zeros(table.shape[:2], np.float32)
    alpha[y0:y0 + height, x0:x0 + piece.shape[1]] = piece[..., 3] / 255
    drop = cv2.GaussianBlur(np.roll(alpha, (max(2, height // 90), max(1, height // 180)), (0, 1)), (0, 0), max(1.5, height / 70))
    out *= (1 - 0.4 * drop)[..., None]
    colors = np.zeros_like(out)
    colors[y0:y0 + height, x0:x0 + piece.shape[1]] = piece[..., :3]
    return np.clip(out * (1 - alpha[..., None]) + colors * alpha[..., None], 0, 255).astype(np.uint8)


shirt = cutout()
table = cv2.imread(str(TABLE))
th, tw = table.shape[:2]
# The source table has its own very small shirt in the middle: replace it with a neighbouring piece of the same wood.
middle = np.zeros((th, tw), np.uint8)
middle[int(th * 0.36):int(th * 0.62), int(tw * 0.40):int(tw * 0.60)] = 255
dark = cv2.dilate(((cv2.cvtColor(table, cv2.COLOR_BGR2GRAY) < 90) & (middle > 0)).astype(np.uint8) * 255, np.ones((25, 25), np.uint8))
soft = (cv2.GaussianBlur(dark, (0, 0), 6).astype(np.float32) / 255)[..., None]
table = (table * (1 - soft) + np.roll(table, int(tw * 0.22), axis=1) * soft).astype(np.uint8)
far_image = lay(table, shirt, int(th * 0.24), tw // 2 + 14, int(th * 0.49))
fit = int(H * 1.14)
far_image = cv2.resize(far_image, (int(tw * fit / th), fit), interpolation=cv2.INTER_AREA)
pad = max(0, (int(W * 1.14) - far_image.shape[1]) // 2 + 2)
far_image = cv2.copyMakeBorder(far_image, 0, 0, pad, pad, cv2.BORDER_REFLECT_101)
# Close: a piece of the same table enlarged, and the shirt filling most of the height with table on every side.
piece = table[int(th * 0.05):int(th * 0.05) + 720, int(tw * 0.25):int(tw * 0.25) + 1280]
close_image = cv2.resize(piece, (int(W * 1.1), int(H * 1.1)), interpolation=cv2.INTER_CUBIC)
close_image = lay(close_image, shirt, int(H * 0.86), close_image.shape[1] // 2, close_image.shape[0] // 2)
label = cv2.imread(str(SAMPLES / "demo_label.jpg"))
scale = max(W / label.shape[1], H / label.shape[0])
label = cv2.resize(label, (int(label.shape[1] * scale) + 2, int(label.shape[0] * scale) + 2), interpolation=cv2.INTER_CUBIC)

SEGMENTS = [(far_image, 5, 1.0, 1.1), (close_image, 6, 1.0, 1.05), (label, 7, 1.0, 1.35)]
raw = SAMPLES / "take_raw.avi"
writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
t, previous = 0.0, None
for image, seconds, zoom_from, zoom_to in SEGMENTS:
    h, w = image.shape[:2]
    frames = seconds * FPS
    for i in range(frames):
        zoom = zoom_from + (zoom_to - zoom_from) * i / max(1, frames - 1)
        cw, ch = int(W / zoom), int(H / zoom)
        dx = (w - cw) / 2 + 12 * math.sin(t * 0.9) + 4 * math.sin(t * 3.1)
        dy = (h - ch) / 2 + 8 * math.cos(t * 1.1) + 3 * math.sin(t * 2.7)
        x, y = int(max(0, min(w - cw, dx))), int(max(0, min(h - ch, dy)))
        frame = cv2.resize(image[y:y + ch, x:x + cw], (W, H), interpolation=cv2.INTER_LINEAR)
        if previous is not None and i < 6:
            frame = cv2.addWeighted(frame, i / 6, previous, 1 - i / 6, 0)
        writer.write(frame)
        t += 1 / FPS
    previous = frame
writer.release()
subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libvpx", "-b:v", "1500k", "-quality", "good",
                "-cpu-used", "4", "-an", str(SAMPLES / "sample_take.webm")], check=True)
raw.unlink()
print("wrote sample_take.webm", flush=True)
