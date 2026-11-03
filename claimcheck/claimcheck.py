"""ClaimCheck Reflex UI. All decisions live in claimcheck_core; this file only displays them."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import json
import time
from pathlib import Path

import reflex as rx
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from claimcheck_core import agent, report, rules, speech

ROOT = Path(__file__).resolve().parent.parent
SESSIONS: dict[str, agent.Session] = {}
PENDING_FRAME: dict[str, bytes] = {}
PENDING_TAKE: dict[str, bytes] = {}
TAKES_BY_KEY: dict[str, bytes] = {}
MAX_TAKE_BYTES = 300 * 1024 * 1024

DEFAULT_LISTING = "Set of 2 black t-shirts. 100% cotton. Made in Portugal. Super comfortable, our best seller!"
SAMPLE_SCRIPT = [(2.5, "This is a set of 2 black t-shirts."), (6.5, "It's 100% cotton and made in Portugal.")]
MASK_COLOR = {"open": "#ffdd00", "verified": "#ffdd00", "contradicted": "#e3211c", "not_visible": "#ffffff"}
STICKER_WORD = {"verified": "True", "contradicted": "Not true"}


def _svg(overlay: dict | None) -> str:
    if not overlay or not overlay.get("paths"):
        return ""
    w, h = overlay["size"]
    color = MASK_COLOR.get(overlay["status"], MASK_COLOR["open"])
    width = max(3, w // 300)
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
        return [c for c in self.claims if c["verdict"] in ("verified", "contradicted") and c["human"] != "disagree"]

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

    @rx.event(background=True)
    async def begin(self, source: str):
        """source: camera (film and talk) | sample | typed (check a pasted listing with the camera) | upload"""
        async with self:
            if self.busy:
                return
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
        if result == "ok":
            self.camera_on = True
            return State.maybe_roll
        self.error = f"Camera unavailable ({result.replace('error:', '')}). Upload a video instead."
        self.phase = "start"
        self.busy = False

    @rx.event
    def mic_result(self, result: str):
        if result != "ok":
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
        if not pending:
            return "wide" if session.shots else "far"
        claim = pending[0]
        if claim.check in ("composition", "text", "read"):
            return "label"
        return "wide" if claim.attempts > 0 else "far"

    @rx.event
    def human_mark(self, claim_id: str, mark: str):
        s = self._session()
        if not s:
            return
        for c in s.claims:
            if c.id == claim_id:
                c.human = mark
        self._sync(s)

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


# ---------- Components ----------
def start_view() -> rx.Component:
    return rx.el.main(
        rx.el.h1("Film it. Say it. Sell what's true."),
        rx.cond(
            State.pasting,
            rx.el.div(
                rx.el.div(
                    rx.el.textarea(value=State.listing, on_change=State.set_listing, aria_label="Your listing"),
                    rx.el.button("Check it on camera", on_click=State.begin("typed"), disabled=State.busy, class_name="btn", id="go-typed"),
                    rx.el.button("Back", on_click=State.toggle_paste, class_name="quiet"),
                    class_name="paste",
                ),
                class_name="poster-foot",
            ),
            rx.el.div(
                rx.el.p("Point your camera at what you're selling and talk about it. Your listing writes itself, "
                        "and anything the camera proves wrong never makes it in."),
                rx.el.div(
                    rx.el.button("Start filming", on_click=State.begin("camera"), disabled=State.busy, class_name="start-btn", id="go-live"),
                    rx.el.div(
                        rx.upload.root(
                            rx.el.button("Upload a video", class_name="quiet", type="button"),
                            id="start-upload", accept={"video/*": [], "image/*": []}, max_files=1, no_drag=True, class_name="upload-plain",
                            on_drop=State.handle_upload(rx.upload_files(upload_id="start-upload")),
                        ),
                        rx.el.button("Paste a listing", on_click=State.toggle_paste, class_name="quiet", id="go-paste"),
                        rx.el.button("Watch a sample", on_click=State.begin("sample"), disabled=State.busy, class_name="quiet", id="go-sample"),
                        class_name="others",
                    ),
                ),
                class_name="poster-foot",
            ),
        ),
        class_name="poster",
    )


def seg(s) -> rx.Component:
    return rx.fragment(
        rx.el.span(s["text"], class_name=f"seg {s['verdict']}"),
        rx.cond(s["fix"] != "", rx.el.span(s["fix"], class_name="fix")),
    )


def live_view() -> rx.Component:
    return rx.el.main(
        rx.el.div(
            rx.el.video(id="cc-video", auto_play=True, muted=True, plays_inline=True),
            rx.cond(State.frozen != "", rx.el.img(src=State.frozen, class_name="frozen", alt="", id="cc-frozen")),
            rx.el.div(rx.html(State.masks), class_name="masks"),
            rx.cond(State.flash_key > 0, rx.el.div(key=State.flash_key, class_name="flash")),
            rx.cond(
                State.direction != "",
                rx.el.div(
                    rx.el.span(State.direction, id="cc-direction"),
                    rx.cond(State.wait > 0, rx.el.span(State.wait, class_name="count")),
                    class_name="say",
                ),
            ),
            rx.cond(State.print_url != "", rx.el.div(rx.el.img(src=rx.get_upload_url(State.print_url), alt="The label, flattened"), class_name="print", id="cc-print")),
            rx.el.div(
                rx.foreach(State.stickers, lambda t: rx.el.div(
                    rx.el.small(t["word"]),
                    rx.cond(t["said"] != "", rx.el.s(t["said"])),
                    t["seen"],
                    class_name=f"sticker {t['status']}",
                )),
                class_name="stickers",
            ),
            rx.el.div(
                rx.foreach(State.segments, seg),
                rx.el.span(" "),
                rx.el.span(id="cc-partial", class_name="partial"),
                rx.cond(
                    State.seen.length() > 0,
                    rx.el.div(rx.foreach(State.seen, lambda x: rx.el.b(x)), class_name="seen"),
                ),
                class_name="tag", id="cc-tag",
            ),
            rx.el.button("Done", on_click=State.finish_take, class_name="done-btn", id="finish"),
            rx.cond(State.source == "sample", rx.el.div("Sample: AI-generated clip, scripted seller", class_name="sample-note")),
            rx.el.div(State.ticker, class_name="ticker", id="cc-ticker"),
            class_name="stage", style={"--ratio": State.ratio},
        ),
        class_name="live",
    )


def cutting_view() -> rx.Component:
    return rx.el.main(
        rx.el.h2("Writing your listing"),
        rx.el.div(rx.el.i(style={"width": f"{State.progress}%"}), class_name="bar-track"),
        rx.el.p(State.ticker),
        class_name="cutting",
    )


def proof_button(c) -> rx.Component:
    return rx.dialog.root(
        rx.dialog.trigger(
            rx.el.button(rx.cond(c["thumb"] != "", rx.el.img(src=rx.get_upload_url(c["thumb"]), alt="")), class_name="proof", aria_label="See the evidence", disabled=c["thumb"] == "")
        ),
        rx.dialog.content(
            rx.dialog.title(c["value"]),
            rx.cond(c["flat"] != "", rx.el.img(src=rx.get_upload_url(c["flat"]), width="100%", alt="Evidence flattened with its mask", style={"marginBottom": "10px"})),
            rx.cond(c["evidence"] != "", rx.el.img(src=rx.get_upload_url(c["evidence"]), width="100%", alt="The frame with its mask")),
            rx.text(c["reason"], size="2", margin_top="10px"),
            rx.text("Decided by: ", c["rule"], size="1", color="gray"),
            rx.dialog.close(rx.button("Close", variant="soft", color_scheme="gray", margin_top="10px")),
            max_width="620px",
        ),
    )


def fact_row(c) -> rx.Component:
    return rx.el.div(
        rx.el.span(c["attribute"]),
        rx.el.b(c["value"], rx.cond(c["verdict"] == "contradicted", rx.el.em("you said " + c["claimed"], class_name="fixed-tag"))),
        proof_button(c),
        class_name="fact claim", data_verdict=c["verdict"],
    )


def reveal_view() -> rx.Component:
    return rx.el.main(
        rx.el.div(
            rx.el.div(rx.el.strong("ClaimCheck"), rx.el.button("New listing", on_click=State.reset_all, class_name="quiet"), class_name="top"),
            rx.cond(State.warning != "", rx.el.div(State.warning, class_name="warn")),
            rx.el.div(
                rx.el.div(
                    rx.cond(
                        State.has_hero,
                        rx.fragment(
                            rx.el.img(src=rx.get_upload_url(State.hero), class_name="photo", id="cc-hero", alt="Product photo"),
                            rx.el.p(State.hero_note, class_name="photo-note"),
                        ),
                        rx.cond(
                            State.reel != "",
                            rx.el.video(src=rx.get_upload_url(State.reel), controls=True, muted=True, auto_play=True, loop=True, plays_inline=True, class_name="reel"),
                        ),
                    ),
                ),
                rx.el.div(
                    rx.cond(
                        State.caught.length() > 0,
                        rx.el.div(
                            rx.el.h2(State.caught_title),
                            rx.el.ul(rx.foreach(State.caught, lambda c: rx.el.li(rx.el.s(c["text"]), " → ", c["value"]))),
                            class_name="caught", id="cc-caught",
                        ),
                    ),
                    rx.el.h1(State.title, id="cc-title"),
                    rx.el.textarea(value=State.corrected, on_change=State.set_corrected, class_name="desc", id="cc-corrected", aria_label="Listing description"),
                    rx.el.div(rx.foreach(State.established, fact_row), class_name="facts"),
                    rx.cond(State.unproven != "", rx.el.p("Left out, because the camera couldn't check it: ", State.unproven, class_name="left-out")),
                    rx.el.div(
                        rx.el.button("Copy listing", on_click=State.copy_listing, class_name="btn", id="copy"),
                        rx.el.button("Download proof", on_click=State.download_proof, class_name="btn alt", id="approve"),
                        class_name="actions",
                    ),
                    rx.el.p("Nothing is published. A match is what the camera saw, not a certification, and printed labels can be wrong too.", class_name="small"),
                ),
                class_name="product",
            ),
            rx.cond(
                State.has_hero & (State.reel != ""),
                rx.el.div(
                    rx.el.h2("The take, with every piece of evidence tracked"),
                    rx.el.video(src=rx.get_upload_url(State.reel), controls=True, muted=True, auto_play=True, loop=True, plays_inline=True, class_name="reel", id="cc-reel"),
                    class_name="reel-wrap",
                ),
            ),
            rx.el.details(
                rx.el.summary("How it knows: ", State.calls.length(), " live Meta model calls, ", State.cost),
                rx.foreach(State.calls, lambda t: rx.el.div(rx.el.span(t["model"]), rx.el.span(t["purpose"]), rx.el.span(t["ms"]), rx.el.span(t["usd"]))),
                class_name="machine",
            ),
            class_name="page-in",
        ),
        class_name="page",
    )


def index() -> rx.Component:
    return rx.el.div(
        rx.script(src="/director.js"),
        rx.cond(State.error != "", rx.el.div(State.error, class_name="alert", role="alert")),
        rx.match(
            State.phase,
            ("start", start_view()),
            ("live", live_view()),
            ("cutting", cutting_view()),
            reveal_view(),
        ),
        class_name="cc",
    )


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

FONTS = "https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..900&display=swap"
app = rx.App(
    theme=rx.theme(appearance="light", accent_color="gray", radius="none"),
    stylesheets=[FONTS, "/claimcheck.css"],
    api_transformer=take_api,
)
app.add_page(index, title="ClaimCheck", description="Film it. Say it. Sell what's true.")
