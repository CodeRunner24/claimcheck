"""Two plain endpoints the websocket state can't serve well: the recorded take, and live microphone audio."""
from __future__ import annotations

import asyncio

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from claimcheck_core import agent, speech

from .state import SESSIONS, TAKES_BY_KEY

MAX_TAKE_BYTES = 300 * 1024 * 1024


async def receive_take(request: Request) -> JSONResponse:
    """The recorded take, keyed by the session's random folder name."""
    key = request.path_params["key"]
    if not any(s.out_dir and s.out_dir.name == key for s in SESSIONS.values()):
        return JSONResponse({"error": "unknown session"}, status_code=404)
    body = await request.body()
    if len(body) > MAX_TAKE_BYTES:
        return JSONResponse({"error": "take too large"}, status_code=413)
    TAKES_BY_KEY[key] = body
    return JSONResponse({"ok": True})


async def microphone(socket: WebSocket) -> None:
    """Browser audio to Muse Voice Transcribe, captions back, finished sentences into the agent."""
    key = socket.path_params["key"]
    session = next((s for s in SESSIONS.values() if s.out_dir and s.out_dir.name == key), None)
    await socket.accept()
    if session is None:
        await socket.close(code=1008)
        return

    async def next_audio():
        try:
            message = await socket.receive()
        except WebSocketDisconnect:
            return None
        if message.get("type") == "websocket.disconnect" or message.get("text") == "end":
            return None
        return message.get("bytes") or b""

    async def send_caption(text: str):
        try:
            await socket.send_text(text)
        except Exception:
            pass

    async def on_sentence(text: str):
        asyncio.ensure_future(asyncio.to_thread(agent.hear, session, text))

    try:
        await speech.relay(next_audio, send_caption, on_sentence, session.trace)
    except Exception as e:
        session.progress["voice_error"] = str(e)[:200]
        await send_caption("")


take_api = Starlette(
    routes=[Route("/cc/take/{key}", receive_take, methods=["POST"]), WebSocketRoute("/cc/mic/{key}", microphone)],
    middleware=[Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])],
)
