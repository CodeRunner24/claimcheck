"""Live speech: browser audio in, Muse Voice Transcribe in the middle, captions and finished sentences out.

The API key stays on the server, so the browser talks to this relay rather than to Meta.
"""
from __future__ import annotations

import asyncio
import json
import time

import websockets

from . import meta_api as api

URL = "wss://api.meta.ai/v1/asr/realtime"
KEYWORDS = ["cotton", "polyester", "t-shirt", "linen", "wool", "made in"]
MAX_RECONNECTS = 4
STABLE_SECONDS = 20.0  # a stream that lasted this long counts as having worked


async def keep_listening(next_audio, send_caption, on_sentence, trace=None) -> None:
    """The relay, reconnected when the stream drops. A busy machine can fall behind real time, and that shouldn't end the listening."""
    failures = 0
    while True:
        began = time.time()
        try:
            await relay(next_audio, send_caption, on_sentence, trace)
            return
        except (api.ApiError, websockets.WebSocketException, OSError):
            failures = 1 if time.time() - began > STABLE_SECONDS else failures + 1
            if failures > MAX_RECONNECTS:
                raise
            await send_caption("")
            await asyncio.sleep(0.2)


async def relay(next_audio, send_caption, on_sentence, trace=None) -> None:
    """Forward PCM until next_audio() returns None. send_caption gets partials; on_sentence gets each finished turn."""
    handshake = {
        "authorization": {"accessToken": f"Bearer {api._key()}"},
        "audioEncoding": "PCM_24KHZ", "model": api.VOICE, "mode": "ENDPOINTING", "partialMode": "CUMULATIVE",
        "emitAudioProgress": False, "languageBias": ["English"], "keywords": KEYWORDS,
    }
    api.LEDGER_INSTANCE.check()
    started = time.time()
    billed_ms = 0
    async with websockets.connect(URL, open_timeout=20, max_size=None) as upstream:
        await upstream.send(json.dumps(handshake))
        ack = json.loads(await upstream.recv())
        if "sessionId" not in ack:
            raise api.ApiError(f"Voice handshake failed: {str(ack)[:200]}")

        async def pump():
            while True:
                chunk = await next_audio()
                if chunk is None:
                    await upstream.send(json.dumps({"type": "endStream"}))
                    return
                await upstream.send(chunk)

        async def listen():
            nonlocal billed_ms
            async for message in upstream:
                if isinstance(message, bytes):
                    continue
                event = json.loads(message)
                kind = event.get("type")
                if kind == "transcript" and not event.get("final"):
                    await send_caption(str(event.get("transcript", "")))
                elif kind == "speechComplete":
                    text = str(event.get("transcript", "")).strip()
                    done_ms = int(event.get("audioProcessedMs") or 0)
                    usd = max(0, done_ms - billed_ms) / 1000 * api.VOICE_SECOND
                    billed_ms = max(billed_ms, done_ms)
                    api.LEDGER_INSTANCE.add(usd)
                    if trace is not None and text:
                        trace.append(api.CallRecord(api.VOICE, f'heard "{text[:60]}"', 0, usd))
                    await send_caption("")
                    if text:
                        await on_sentence(text)
                elif kind == "error":
                    raise api.ApiError(str(event.get("message", "voice error"))[:200])

        tasks = [asyncio.create_task(pump()), asyncio.create_task(listen())]
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
    remainder = max(0.0, time.time() - started - billed_ms / 1000)
    api.LEDGER_INSTANCE.add(remainder * api.VOICE_SECOND)
