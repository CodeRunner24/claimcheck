"""Thin Meta Model API client with a hard spend cap and a call trace."""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

BASE_URL = os.getenv("META_API_BASE_URL", "https://api.meta.ai/v1")
SPARK = "muse-spark-1.3"
SAM = "sam-3.1"
IMAGE = "muse-image-1.0"
SPEND_CAP_USD = float(os.getenv("CLAIMCHECK_SPEND_CAP_USD", "40"))
LEDGER = ROOT / ".spend.json"

# USD, from https://dev.meta.ai/docs/pricing-rate-limits
SPARK_IN, SPARK_CACHED, SPARK_OUT = 1.25e-6, 0.15e-6, 4.25e-6
SAM_IMAGE = 2.50 / 1000
SAM_VIDEO_FRAME = 0.20 / 1000  # each video request is billed for at least 50 frames
IMAGE_GEN = 0.01
VOICE_SECOND = 0.18 / 3600
VOICE = "muse-voice-transcribe-1.0"


class SpendCapReached(RuntimeError):
    pass


class ApiError(RuntimeError):
    pass


@dataclass
class CallRecord:
    model: str
    purpose: str
    ms: int
    usd: float


class Ledger:
    """Persistent spend total; every call is refused once the cap is reached."""

    def __init__(self, path: Path = LEDGER, cap: float = SPEND_CAP_USD):
        self.path, self.cap = path, cap
        self._lock = threading.Lock()
        self.total = json.loads(path.read_text())["usd"] if path.exists() else 0.0

    def check(self) -> None:
        if self.total >= self.cap:
            raise SpendCapReached(f"Spend cap ${self.cap:.2f} reached (${self.total:.2f} used).")

    def add(self, usd: float) -> None:
        with self._lock:
            self.total += usd
            self.path.write_text(json.dumps({"usd": self.total}))


LEDGER_INSTANCE = Ledger()


class Trace(list):
    """Finished calls, plus the models running right now so the UI can light them up."""

    def __init__(self):
        super().__init__()
        self.active: dict[str, int] = {}
        self._lock = threading.Lock()

    def begin(self, model: str) -> None:
        with self._lock:
            self.active[model] = self.active.get(model, 0) + 1

    def end(self, model: str) -> None:
        with self._lock:
            self.active[model] = max(0, self.active.get(model, 0) - 1)

    def running(self) -> list[str]:
        with self._lock:
            return [m for m, n in self.active.items() if n > 0]


@contextmanager
def _running(trace, model: str):
    if isinstance(trace, Trace):
        trace.begin(model)
    try:
        yield
    finally:
        if isinstance(trace, Trace):
            trace.end(model)


def data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def _key() -> str:
    key = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")
    if not key:
        raise ApiError("META_API_KEY is not set. Copy .env.example to .env.")
    return key


def _post(path: str, body: dict, timeout: float = 120) -> dict:
    LEDGER_INSTANCE.check()
    key = _key()
    last = None
    for attempt in range(3):
        try:
            r = httpx.post(
                f"{BASE_URL}{path}",
                headers={"Authorization": f"Bearer {key}"},
                json=body,
                timeout=timeout,
            )
        except httpx.HTTPError as e:
            last = str(e)
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code >= 400:
            raise ApiError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()
    raise ApiError(f"Meta API unavailable after retries: {last}")


def _output_text(resp: dict) -> str:
    parts = []
    for item in resp.get("output", []):
        if item.get("type") == "message":
            parts += [c.get("text", "") for c in item.get("content", []) if c.get("type") == "output_text"]
    return "".join(parts)


def _spark_cost(usage: dict) -> float:
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    fresh = usage.get("input_tokens", 0) - cached
    return fresh * SPARK_IN + cached * SPARK_CACHED + usage.get("output_tokens", 0) * SPARK_OUT


def parse_json(text: str):
    """Parse JSON from model text, tolerating code fences or surrounding prose."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def spark_json(prompt: str, images: list[bytes] | None = None, purpose: str = "spark",
               effort: str = "low", trace: list | None = None):
    content = [{"type": "input_text", "text": prompt}]
    content += [{"type": "input_image", "image_url": data_url(i)} for i in images or []]
    body = {
        "model": SPARK,
        "reasoning": {"effort": effort},
        "store": False,
        "input": [{"type": "message", "role": "user", "content": content}],
    }
    error = None
    for _ in range(2):
        t0 = time.time()
        with _running(trace, SPARK):
            resp = _post("/responses", body)
        usd = _spark_cost(resp.get("usage") or {})
        LEDGER_INSTANCE.add(usd)
        if trace is not None:
            trace.append(CallRecord(SPARK, purpose, int((time.time() - t0) * 1000), usd))
        try:
            return parse_json(_output_text(resp))
        except json.JSONDecodeError as e:
            error = e
    raise ApiError(f"Spark did not return JSON for {purpose}: {error}")


def sam_raw(concept: str, jpeg: bytes, trace: list | None = None) -> str:
    body = {
        "model": SAM,
        "store": False,
        "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": concept},
            {"type": "input_image", "image_url": data_url(jpeg)},
        ]}],
    }
    t0 = time.time()
    with _running(trace, SAM):
        resp = _post("/responses", body)
    LEDGER_INSTANCE.add(SAM_IMAGE)
    if trace is not None:
        trace.append(CallRecord(SAM, f'segment "{concept}"', int((time.time() - t0) * 1000), SAM_IMAGE))
    return _output_text(resp)


def generate_image(prompt: str, trace: list | None = None, purpose: str = "shot guide") -> bytes:
    t0 = time.time()
    with _running(trace, IMAGE):
        resp = _post("/images/generations", {"model": IMAGE, "prompt": prompt, "n": 1}, timeout=180)
    LEDGER_INSTANCE.add(IMAGE_GEN)
    if trace is not None:
        trace.append(CallRecord(IMAGE, purpose, int((time.time() - t0) * 1000), IMAGE_GEN))
    return base64.b64decode(resp["data"][0]["b64_json"])


def sam_video(concept: str, mp4: bytes, trace: list | None = None, on_frame=None) -> list[str]:
    """Track a concept through a clip. Streams one line per frame; on_frame(index) reports progress."""
    LEDGER_INSTANCE.check()
    body = {
        "model": SAM,
        "store": False,
        "stream": True,
        "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": concept},
            {"type": "input_video", "video_url": "data:video/mp4;base64," + base64.b64encode(mp4).decode()},
        ]}],
    }
    lines: list[str] = []
    frames = 0
    t0 = time.time()
    with _running(trace, SAM):
        with httpx.stream("POST", f"{BASE_URL}/responses", headers={"Authorization": f"Bearer {_key()}"},
                          json=body, timeout=300) as r:
            if r.status_code >= 400:
                raise ApiError(f"HTTP {r.status_code}: {r.read()[:300]!r}")
            for raw in r.iter_lines():
                if not raw.startswith("data:"):
                    continue
                try:
                    event = json.loads(raw[5:].strip())
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "response.output_text.delta":
                    lines.append(event["delta"])
                    if on_frame:
                        m = re.match(r"\s*<(\d+)f>", event["delta"])
                        on_frame(int(m.group(1)) if m else len(lines))
                elif event.get("type") == "response.completed":
                    frames = (event["response"].get("usage") or {}).get("video_frames_processed", 0)
    usd = max(frames, 50) * SAM_VIDEO_FRAME
    LEDGER_INSTANCE.add(usd)
    if trace is not None:
        trace.append(CallRecord(SAM, f'track "{concept}" through {frames} frames', int((time.time() - t0) * 1000), usd))
    return lines


def transcribe(wav: bytes, trace: list | None = None) -> str:
    """Mono 16-bit PCM WAV to text. Language is detected, so Turkish and English both work."""
    LEDGER_INSTANCE.check()
    t0 = time.time()
    settings = {"mode": "ENDPOINTING", "model": VOICE, "audioEncoding": "WAV"}
    with _running(trace, VOICE):
        r = httpx.post(f"{BASE_URL}/asr/transcribe", headers={"Authorization": f"Bearer {_key()}"}, timeout=90,
                       files={"request": (None, json.dumps(settings), "application/json"),
                              "audio": ("speech.wav", wav, "audio/wav")})
    if r.status_code >= 400:
        raise ApiError(f"HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    usd = data.get("audioDurationMs", 0) / 1000 * VOICE_SECOND
    LEDGER_INSTANCE.add(usd)
    if trace is not None:
        trace.append(CallRecord(VOICE, "transcribe spoken listing", int((time.time() - t0) * 1000), usd))
    return str(data.get("transcript", "")).strip()
