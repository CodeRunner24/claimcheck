"""The app's state and events. It only moves data between the browser and claimcheck_core; decisions live in the core."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import json
import shutil
import time
from pathlib import Path

import reflex as rx

from claimcheck_core import agent, report, rules


SESSIONS: dict[str, agent.Session] = {}
PENDING_FRAME: dict[str, bytes] = {}
PENDING_TAKE: dict[str, bytes] = {}
TAKES_BY_KEY: dict[str, bytes] = {}
DEFAULT_LISTING = "Set of 2 black t-shirts. 100% cotton. Made in Portugal. Super comfortable, our best seller!"
SAMPLE_SCRIPT = [(2.5, "This is a set of 2 black t-shirts."), (6.5, "It's 100% cotton and made in Portugal.")]
MASK_COLOR = {"open": "#ffdd00", "verified": "#ffdd00", "contradicted": "#e3211c", "not_visible": "#ffffff"}
STICKER_WORD = {"verified": "True", "contradicted": "Not true"}


def _svg(overlay: dict | None) -> str:
    if not overlay or not overlay.get("paths"):
        return ""
    w, h = overlay["size"]
    color = MASK_COLOR.get(overlay["status"], MASK_COLOR["open"])
    width = max(2, w // 260)
    shapes = "".join(
        f'<path d="{html.escape(p)}" fill="{color}" fill-opacity="{0.22 if overlay["status"] == "open" else 0.12}" '
        f'stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>'
        for p in overlay["paths"] if p
    )
    for ruler in overlay.get("rulers", []):
        points = " ".join(f"{x},{y}" for x, y in ruler["corners"])
        shapes += f'<polygon points="{points}" fill="none" stroke="#fff" stroke-width="{width}" stroke-dasharray="{width * 4} {width * 3}"/>'
        if ruler.get("reference"):
            points = " ".join(f"{x},{y}" for x, y in ruler["reference"])
            shapes += f'<polygon points="{points}" fill="none" stroke="#ffdd00" stroke-width="{width}"/>'
        x, y = ruler["corners"][0]
        size = max(20, w // 26)
        shapes += (f'<text x="{x}" y="{max(size, y - 10)}" fill="#fff" font-size="{size}" font-weight="800" font-family="Archivo, Arial" '
                   f'paint-order="stroke" stroke="#000" stroke-width="{width * 2}">{html.escape(str(ruler["label"]))}</text>')
    return f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">{shapes}</svg>'


class State(rx.State):
    phase: str = "start"  # start | live | cutting | reveal
    pasting: bool = False
    listing: str = DEFAULT_LISTING
    source: str = ""
    claims: list[dict[str, str]] = []
    noticed: list[dict[str, str]] = []
    segments: list[dict[str, str]] = []
    seen: list[str] = []
    direction: str = ""
    wait: int = 0
    camera_on: bool = False
    ready: bool = False
    live: bool = False
    busy: bool = False
    error: str = ""
    # Viewfinder
    masks: str = ""
    ratio: str = "1.778"
    frozen: str = ""
    flash_key: int = 0
    stickers: list[dict[str, str]] = []
    print_url: str = ""
    sticker_until: float = 0.0
    ticker: str = ""
    # Finish
    progress: int = 0
    title: str = ""
    corrected: str = ""
    hero: str = ""
    hero_note: str = ""
    reel: str = ""
    cost: str = ""
    calls: list[dict[str, str]] = []
    warning: str = ""

    @rx.var
    def caught(self) -> list[dict[str, str]]:
        return [c for c in self.claims if c["verdict"] == "contradicted"]

    @rx.var
    def established(self) -> list[dict[str, str]]:
        return [c for c in self.claims if c["verdict"] in ("verified", "contradicted") and c["human"] != "disagree" and c["source"] != "noticed"]

    @rx.var
    def unproven(self) -> str:
        return ", ".join(c["text"] for c in self.claims if c["source"] == "said" and c["verdict"] in ("open", "not_visible"))

    @rx.var
    def caught_title(self) -> str:
        n = len(self.caught)
        if not n:
            return ""
        return "One thing you said isn't true." if n == 1 else f"{n} things you said aren't true."

    @rx.var
    def has_hero(self) -> bool:
        return self.hero != ""

    def _token(self) -> str:
        return self.router.session.client_token

    def _session(self) -> agent.Session | None:
        return SESSIONS.get(self._token())

    def _url(self, session: agent.Session, name: str) -> str:
        return f"cc/{session.out_dir.name}/{name}" if name and session.out_dir else ""

    def _lights(self, s: agent.Session) -> None:
        if s.trace:
            t = s.trace[-1]
            self.ticker = f"{t.model} · {t.purpose} · {t.ms / 1000:.1f}s · ${s.cost():.3f} so far"
        p = s.progress
        if p.get("voice_error"):
            self.error = f"I can't hear you ({p['voice_error']}). You can still paste a listing."
        tracks = [v for k, v in p.items() if k.startswith("track:")]
        parts = [sum(tracks) / len(tracks) if tracks else 0.0, p.get("burn", 0.0), p.get("listing", 0.0), p.get("hero", 0.0)]
        self.progress = int(100 * sum(parts) / 4)

    def _sync(self, s: agent.Session) -> None:
        self._lights(s)
        self.claims = [{
            "id": c.id, "text": c.text, "attribute": c.attribute, "source": c.source, "verdict": c.verdict,
            "value": agent.pretty(c.observed) or c.claimed,
            "claimed": c.claimed, "reason": c.reason, "rule": c.rule, "human": c.human,
            "thumb": self._url(s, c.flat or c.evidence), "evidence": self._url(s, c.evidence), "flat": self._url(s, c.flat),
        } for c in s.claims if not (c.source == "plan" and c.verdict != rules.VERIFIED)]
        self.noticed = [c for c in self.claims if c["source"] == "noticed"]
        self.segments = agent.listing_segments(s)
        self.seen = [c.observed or c.text for c in s.claims if c.source == "seen"]

    def _show(self, overlay: dict | None) -> None:
        self.masks = _svg(overlay)
        if overlay:
            w, h = overlay["size"]
            self.ratio = f"{w / h:.4f}"

    async def _track(self, session: agent.Session, work) -> object:
        """Await blocking agent work while keeping the frozen frame and progress current."""
        task = asyncio.ensure_future(work)
        seen = None
        while not task.done():
            p = session.progress
            key = (len(session.trace), int(p.get("burn", 0) * 50), p.get("listing"), p.get("hero"),
                   tuple(sorted(int(v * 50) for k, v in p.items() if k.startswith("track:"))))
            shutter, session.shutter = session.shutter, None
            if key != seen or shutter:
                seen = key
                async with self:
                    self._lights(session)
                    if shutter:
                        self.frozen, self.flash_key = shutter["frame"], self.flash_key + 1
                        self._show(shutter["overlay"])
                        self.direction = shutter["line"]
            await asyncio.sleep(0.12)
        return task.result()

    # ---------- Start ----------
    @rx.event
    def set_listing(self, value: str):
        self.listing = value

    @rx.event
    def set_corrected(self, value: str):
        self.corrected = value

    @rx.event
    def toggle_paste(self):
        self.pasting = not self.pasting

    @rx.event
    def reset_all(self):
        SESSIONS.pop(self._token(), None)
        self.reset()
        return rx.call_script("cc.micStop(); cc.stream && cc.source!=='sample' && cc.stream.getTracks().forEach(t=>t.stop())")

    @staticmethod
    def _prune() -> None:
        """Drop sessions older than two hours, and their files."""
        cutoff = time.time() - 2 * 3600
        for token, session in list(SESSIONS.items()):
            if session.created < cutoff:
                SESSIONS.pop(token, None)
                if session.out_dir:
                    shutil.rmtree(session.out_dir, ignore_errors=True)

    @rx.event(background=True)
    async def begin(self, source: str):
        """source: camera (film and talk) | sample | typed (check a pasted listing with the camera) | upload"""
        async with self:
            if self.busy:
                return
            self._prune()
            token = self._token()
            folder = hashlib.sha256(f"{token}{time.time()}".encode()).hexdigest()[:16]
            typed = source == "typed" or (source == "upload" and self.pasting)
            session = agent.Session(self.listing.strip() if typed else "", spoken=not typed, out_dir=rx.get_upload_dir() / "cc" / folder)
            if source == "sample":
                session.script = list(SAMPLE_SCRIPT)
            SESSIONS[token] = session
            self.phase, self.source, self.busy, self.error = "live", source, True, ""
            self.camera_on = self.ready = self.live = False
            self.claims, self.masks, self.frozen, self.stickers, self.print_url = [], "", "", [], ""
            self.segments, self.seen = [], []
            self.direction = "" if typed else session.line("up")
            key = folder
        if source in ("camera", "sample", "typed"):
            yield rx.call_script(f"cc.start('{'sample' if source == 'sample' else 'camera'}')", callback=State.camera_result)
        if source == "camera":
            mic = rx.config.get_config().api_url.replace("http", "ws", 1) + f"/cc/mic/{key}"
            yield rx.call_script(f"cc.micStart({json.dumps(mic)})", callback=State.mic_result)
        try:
            if typed:
                await self._track(session, asyncio.to_thread(agent.extract_claims, session))
                first = agent.next_direction(session)
                async with self:
                    self._sync(session)
                    if not session.claims:
                        self.error = "Nothing in this listing can be checked with a camera."
                    self.direction = first["ask"] if first else ""
            async with self:
                self.ready, self.busy = True, False
            if source == "upload":
                yield State.process_upload
            else:
                yield State.maybe_roll
        except Exception as e:
            async with self:
                self.error, self.busy = str(e)[:300], False

    @rx.event
    def camera_result(self, result: str):
        if result.startswith("ok"):
            self.camera_on = True
            self.ratio = result.split(":", 1)[1] if ":" in result else self.ratio
            return State.maybe_roll
        self.error = f"Camera unavailable ({result.replace('error:', '')}). Upload a video instead."
        self.phase = "start"
        self.busy = False

    @rx.event
    def mic_result(self, result: str):
        if result != "ok" and self.phase == "live":
            self.error = f"Microphone unavailable ({result.replace('error:', '')}). You can paste a listing instead."

    @rx.event
    def maybe_roll(self):
        if self.live or not (self.camera_on and self.ready):
            return
        s = self._session()
        if not s:
            return
        self.live = True
        s.started = time.time()
        return State.live_tick

    # ---------- The live loop ----------
    @rx.event
    def live_tick(self):
        if self.live:
            return rx.call_script("cc.nextFrame(0)", callback=State.live_frame)

    @rx.event
    def live_frame(self, url: str):
        if not self.live:
            return
        if not url:
            return State.live_tick
        PENDING_FRAME[self._token()] = base64.b64decode(url.split(",", 1)[1])
        return State.live_process

    @rx.event(background=True)
    async def live_process(self):
        async with self:
            token = self._token()
            raw = PENDING_FRAME.pop(token, None)
            session = SESSIONS.get(token)
            if raw is None or session is None or not self.live:
                return
        try:
            step = await self._track(session, asyncio.to_thread(agent.live_step, session, raw))
        except Exception as e:
            async with self:
                self.error = str(e)[:200]
            await asyncio.sleep(1.5)
            yield State.live_tick
            return
        for sentence in step["heard"]:  # sample mode: the scripted seller "speaks"
            asyncio.ensure_future(asyncio.to_thread(agent.hear, session, sentence))
        async with self:
            if not self.live:
                return
            self.error = ""
            self._show(step["overlay"])
            self.wait = int(step["wait"] + 0.99)
            if step["direction"]:
                self.direction = step["direction"]
            now = time.time()
            if step["fired"] and (step["settled"] or step["flat"]):
                shown = [m for m in step["settled"] if m["source"] == "said" or m["status"] == rules.CONTRADICTED] or step["settled"]
                self.stickers = [{"status": m["status"], "word": STICKER_WORD[m["status"]],
                                  "said": m["said"] if m["status"] == rules.CONTRADICTED else "",
                                  "seen": m["seen"] or m["said"]} for m in shown][:3]
                self.print_url = self._url(session, step["flat"])
                self.sticker_until = now + 5.0
            elif self.sticker_until and now > self.sticker_until:
                self.stickers, self.print_url, self.sticker_until = [], "", 0.0
            self._sync(session)
        if step["fired"]:
            kind = "wrong" if any(m["status"] == rules.CONTRADICTED for m in step["settled"]) else "right"
            yield rx.call_script(f"cc.sound('{kind}')")
            await asyncio.sleep(2.2 if not step["done"] else 3.5)  # let the verdict land on the frozen frame
            async with self:
                self.frozen, self.masks = "", ""
        yield rx.call_script(f"cc.cue('{self._cue(session)}')")
        if step["done"]:
            yield State.finish_take
        elif step["idle"]:
            yield rx.call_script("cc.nextFrame(1600)", callback=State.live_frame)
        else:
            yield State.live_tick

    @staticmethod
    def _cue(session: agent.Session) -> str:
        pending = session.open_claims
        if not pending or session.script or session.hearing:  # the scripted seller is still talking
            return "wide" if session.shots else "far"
        claim = pending[0]
        if claim.check in ("composition", "text", "read"):
            return "label"
        return "wide" if claim.attempts > 0 else "far"

    @rx.event
    def dismiss_noticed(self, claim_id: str):
        """The seller knows the product: a noticed flaw they reject leaves the facts, and they can edit the text."""
        s = self._session()
        if not s:
            return
        for c in s.claims:
            if c.id == claim_id:
                c.human = "disagree"
        self._sync(s)
        self.noticed = [c for c in self.noticed if c["id"] != claim_id]

    # ---------- Uploads ----------
    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        if not files:
            return
        data = await files[0].read()
        kind = (files[0].content_type or "") + (files[0].filename or "").lower()
        token = self._token()
        if "video" in kind or kind.endswith((".mp4", ".mov", ".webm")):
            PENDING_TAKE[token] = data
        else:
            PENDING_FRAME[token + ":still"] = data
        return State.begin("upload")

    @rx.event(background=True)
    async def process_upload(self):
        async with self:
            token = self._token()
            session = SESSIONS.get(token)
            video, still = PENDING_TAKE.pop(token, None), PENDING_FRAME.pop(token + ":still", None)
            if session is None or (video is None and still is None) or self.busy:
                return
            self.busy, self.error = True, ""
            self.phase, self.progress = "cutting", 0
        try:
            if video is not None:
                await self._track(session, asyncio.to_thread(agent.analyze_take, session, video))
            else:
                if not session.claims:
                    await self._track(session, asyncio.to_thread(agent.discover, session, still))
                await self._track(session, asyncio.to_thread(agent.analyze_shot, session, still))
            async with self:
                self.busy = False
            yield State.cut
        except Exception as e:
            async with self:
                self.error, self.busy, self.phase = str(e)[:300], False, "start"

    # ---------- Cutting ----------
    @rx.event
    def finish_take(self):
        self.live = False
        s = self._session()
        key = s.out_dir.name if s and s.out_dir else "none"
        target = f"{rx.config.get_config().api_url}/cc/take/{key}"
        self.phase, self.progress = "cutting", 0
        return rx.call_script(f"cc.stopTake({json.dumps(target)})", callback=State.take_ready)

    @rx.event
    def take_ready(self, result: str):
        s = self._session()
        if s and s.out_dir and s.out_dir.name in TAKES_BY_KEY:
            PENDING_TAKE[self._token()] = TAKES_BY_KEY.pop(s.out_dir.name)
        return State.cut

    @rx.event(background=True)
    async def cut(self):
        async with self:
            token = self._token()
            session = SESSIONS.get(token)
            if session is None:
                return
            self.phase, self.live, self.busy, self.error = "cutting", False, True, ""
            raw = PENDING_TAKE.pop(token, None)
        await self._track(session, asyncio.to_thread(agent.finish, session, raw))
        async with self:
            self._sync(session)
            self.title = session.title or (session.product or "").capitalize()
            self.corrected = session.corrected
            self.hero = self._url(session, session.hero)
            self.reel = self._url(session, session.reel)
            info = session.hero_info
            if info:
                self.hero_note = ("Your product is the camera's real pixels"
                                  + (f", {info['product_pixels']:,} of them checked identical. " if info["pixel_exact"] else ", but they could not be confirmed identical. ")
                                  + "Only the background was generated.")
            self.warning = session.leak_warning or "; ".join(session.progress.get("errors", []))
            self.cost = f"${session.cost():.3f}"
            self.calls = [{"model": t.model, "purpose": t.purpose, "ms": f"{t.ms / 1000:.1f}s", "usd": f"${t.usd:.4f}"} for t in session.trace]
            self.phase, self.busy = "reveal", False

    @rx.event
    def copy_listing(self):
        return rx.set_clipboard(f"{self.title}\n\n{self.corrected}")

    @rx.event
    def download_proof(self):
        s = self._session()
        if not s:
            return
        s.title = self.title
        page = report.build(s, self.corrected)
        name = "claimcheck_proof.html"
        (s.out_dir or Path(".")).mkdir(parents=True, exist_ok=True)
        ((s.out_dir or Path(".")) / name).write_text(page, encoding="utf-8")
        return rx.download(url=rx.get_upload_url(self._url(s, name)), filename=name)
