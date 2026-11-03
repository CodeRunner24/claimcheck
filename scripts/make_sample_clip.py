"""Build assets/samples/sample_take.webm: AI-generated stills moved like a shaky handheld camera. Disclosed in the app."""
import math
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from claimcheck_core import meta_api as api
from claimcheck_core.take import FFMPEG

SAMPLES = ROOT / "assets" / "samples"
W, H, FPS = 1280, 720, 10
far = SAMPLES / "take_far.jpg"
if not far.exists():
    far.write_bytes(api.generate_image(
        "Phone photo from above of one plain black t-shirt laid flat on a light oak table, far away, the shirt is small and "
        "centered with lots of empty table around it, natural window light, realistic, slightly imperfect amateur photo. No text.",
        purpose="sample"))
SEGMENTS = [(far, 5, 1.0, 1.1), (far, 6, 2.0, 2.15), (SAMPLES / "demo_label.jpg", 7, 1.0, 1.35)]
raw = SAMPLES / "take_raw.avi"
writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
t, previous = 0.0, None
for path, seconds, zoom_from, zoom_to in SEGMENTS:
    image = cv2.imread(str(path))
    h, w = image.shape[:2]
    if path == far:  # square source: fit it whole, then continue the table sideways by reflection
        height = int(H * 1.14)
        image = cv2.resize(image, (int(w * height / h), height), interpolation=cv2.INTER_AREA)
        pad = max(0, (int(W * 1.14) - image.shape[1]) // 2 + 2)
        image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_REFLECT_101)
    else:
        scale = max(W / w, H / h)
        image = cv2.resize(image, (int(w * scale) + 2, int(h * scale) + 2), interpolation=cv2.INTER_CUBIC)
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
