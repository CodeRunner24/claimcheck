"""One continuous take: normalise it, track evidence through it with SAM 3.1, cut a proof reel from it."""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import vision

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"

TAKE_FPS = 6
MAX_SECONDS = 90
STATUS_BGR = {"verified": (0, 221, 255), "contradicted": (28, 33, 227), "not_visible": (255, 255, 255), "open": (0, 221, 255)}


@dataclass
class Take:
    mp4: bytes
    frames: list[bytes]  # JPEG per frame
    size: tuple[int, int]
    fps: int = TAKE_FPS


def _ffmpeg(args: list[str]) -> None:
    done = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True, check=False)
    if done.returncode != 0:
        raise ValueError(f"ffmpeg failed: {done.stderr[-300:].decode(errors='ignore')}")


def load(raw: bytes, max_side: int = 960) -> Take:
    """Any browser or phone clip to H.264 at a fixed low frame rate, the form SAM accepts and bills least for."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in", Path(d) / "out.mp4"
        src.write_bytes(raw)
        _ffmpeg(["-i", str(src), "-t", str(MAX_SECONDS), "-an", "-vf", f"fps={TAKE_FPS},scale='min({max_side},iw)':-2",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)])
        mp4 = dst.read_bytes()
        cap = cv2.VideoCapture(str(dst))
        frames: list[bytes] = []
        size = (0, 0)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            size = (f.shape[1], f.shape[0])
            ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 88])
            frames.append(buf.tobytes())
        cap.release()
    if not frames:
        raise ValueError("No frames in this video.")
    return Take(mp4, frames, size)


def subset(take: Take, indices: list[int], fps: int | None = None) -> Take:
    """A shorter clip from chosen frames, re-encoded so SAM can track it."""
    frames = [take.frames[i] for i in indices]
    fps = fps or take.fps
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / "out.mp4"
        w, h = take.size
        pipe = subprocess.Popen([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                                 "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p",
                                 "-movflags", "+faststart", str(dst)], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for jpeg in frames:
            pipe.stdin.write(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).tobytes())
        pipe.stdin.close()
        pipe.wait()
        return Take(dst.read_bytes(), frames, take.size, fps)


def moment_windows(take: Take, moments: list[dict], before: float = 2.0, after: float = 3.5, limit_seconds: float = 25.0) -> list[int]:
    """Frame indices around the moments that matter. A seller hunting for a label for a minute needn't be tracked or rewatched."""
    keep = set()
    for m in moments:
        keep.update(i for i in range(len(take.frames)) if m["t"] - before <= i / take.fps <= m["t"] + after)
    indices = sorted(keep) or list(range(len(take.frames)))
    return indices[:int(limit_seconds * take.fps)]


def audio_wav(raw: bytes) -> bytes | None:
    """The clip's sound as 16 kHz mono WAV, or None if it has no audio track."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in", Path(d) / "out.wav"
        src.write_bytes(raw)
        done = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-t", str(MAX_SECONDS), "-vn",
                               "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)], capture_output=True, check=False)
        return dst.read_bytes() if done.returncode == 0 and dst.exists() and dst.stat().st_size > 1000 else None


def most_at_once(frames: dict[int, list[vision.Instance]], hold: int = 4) -> tuple[int, int | None]:
    """The most instances visible together for `hold` consecutive frames, and a frame showing them.

    SAM ids do not survive an object leaving and coming back, so counting ids would overcount.
    """
    best, at = 0, None
    run_count, run = 0, 0
    previous = None
    for index in sorted(frames):
        n = len(frames[index])
        run = run + 1 if n == run_count and previous == index - 1 else 1
        run_count, previous = n, index
        if run >= hold and n > best:
            best, at = n, index
    return best, at


def best_read_frame(frames: dict[int, list[vision.Instance]], take: Take) -> tuple[int, vision.Instance] | None:
    """The frame where the evidence is largest and sharpest. Only the largest candidates are scored for sharpness."""
    candidates = []
    for index, instances in frames.items():
        if instances and index < len(take.frames):
            inst = max(instances, key=lambda i: (i.box[2] - i.box[0]) * (i.box[3] - i.box[1]))
            candidates.append((inst.short_side, index, inst))
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[0])
    best, score = None, -1.0
    for side, index, inst in candidates[:24]:
        s = side * min(vision.sharpness(take.frames[index], inst.box), 400.0) ** 0.5
        if s > score:
            best, score = (index, inst), s
    return best


def _caption(width: int, text: str, bgr: tuple[int, int, int]) -> np.ndarray:
    font = vision._font(max(18, width // 44))
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    l, t, r, b = probe.textbbox((0, 0), text, font=font)
    pad = max(10, width // 80)
    im = Image.new("RGBA", (r - l + 2 * pad + 8, b - t + 2 * pad), (10, 12, 18, 225))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 7, im.height), fill=bgr[::-1] + (255,))
    d.text((pad + 8 - l, pad - t), text, font=font, fill="white")
    return np.array(im)


def render_reel(take: Take, layers: list[dict], moments: list[dict], out: Path, on_frame=None) -> None:
    """Burn tracked masks and the agent's verdict moments into a shareable clip.

    layers:  [{"frames": {index: instances}, "status": str, "label": str}]
    moments: [{"t": seconds, "text": str, "status": str}]; each shows for three seconds
    """
    w, h = take.size
    font = vision._font(max(14, w // 62))
    captions = {i: _caption(w, m["text"], STATUS_BGR.get(m["status"], STATUS_BGR["open"])) for i, m in enumerate(moments)}
    pipe = subprocess.Popen([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{w}x{h}", "-r", str(take.fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                             "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        for index, jpeg in enumerate(take.frames):
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            tint = frame.copy()
            outlines, labels = [], []
            for layer in layers:
                color = STATUS_BGR.get(layer["status"], STATUS_BGR["open"])
                for inst in layer["frames"].get(index, []):
                    mask = vision.full_mask(inst, (w, h))
                    tint[mask > 0] = color
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    outlines.append((contours, color))
                    labels.append((inst.box[0], inst.box[1], layer["label"], color))
            frame = cv2.addWeighted(tint, 0.16, frame, 0.84, 0)
            for contours, color in outlines:
                cv2.drawContours(frame, contours, -1, color, max(2, w // 480), cv2.LINE_AA)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            for x, y, text, color in labels:
                l, t, r, b = draw.textbbox((0, 0), text, font=font)
                top = max(0, y - (b - t) - 10)
                draw.rectangle((x, top, x + r - l + 12, top + b - t + 10), fill=color[::-1])
                draw.text((x + 6 - l, top + 5 - t), text, font=font, fill="white")
            seconds = index / take.fps
            stamp = f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"
            draw.text((w - draw.textlength(stamp, font=font) - 14, 12), stamp, font=font, fill="white",
                      stroke_width=2, stroke_fill="black")
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            y = h - 16
            for i, m in reversed(list(enumerate(moments))):
                if m["t"] <= seconds <= m["t"] + 3.0:
                    c = captions[i]
                    ch, cw = c.shape[:2]
                    y -= ch + 6
                    if y < 0 or cw > w - 16:
                        continue
                    alpha = c[..., 3:4] / 255
                    region = frame[y:y + ch, 16:16 + cw]
                    frame[y:y + ch, 16:16 + cw] = (c[..., 2::-1] * alpha + region * (1 - alpha)).astype(np.uint8)
            pipe.stdin.write(frame.tobytes())
            if on_frame:
                on_frame(index)
    finally:
        pipe.stdin.close()
        pipe.wait()
