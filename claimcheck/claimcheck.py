"""ClaimCheck Reflex UI. All decisions live in claimcheck_core; this file only displays them."""
from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path

import reflex as rx

from claimcheck_core import agent, meta_api as api, rules, vision

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "assets" / "samples"
FEEDBACK = ROOT / "eval" / "feedback.jsonl"
SESSIONS: dict[str, agent.Session] = {}
PENDING: dict[str, bytes] = {}
GUIDE_CACHE: dict[str, str] = {}

DEFAULT_LISTING = ("Set of 2 black t-shirts. 100% cotton. Made in Portugal. "
                   "Super comfortable, our best seller!")
SAMPLE_SHOTS = [
    {"name": "demo_far.jpg", "title": "From far away"},
    {"name": "demo_wide.jpg", "title": "Everything in the box"},
    {"name": "demo_label.jpg", "title": "Care label close-up"},
]
VERDICT_LABEL = {"open": "Waiting", "verified": "Verified", "contradicted": "Contradicted", "not_visible": "Need evidence"}

CAMERA_JS = """
window.ccStream = null;
window.ccCameraStart = async () => {
  try {
    window.ccStream = await navigator.mediaDevices.getUserMedia({video: {facingMode: 'environment', width: {ideal: 1280}}});
    const v = document.getElementById('cc-video');
    v.srcObject = window.ccStream; await v.play();
    return 'ok';
  } catch (e) { return 'error:' + e.message; }
};
window.ccCapture = () => {
  const v = document.getElementById('cc-video');
  if (!v || !v.videoWidth) return '';
  const c = document.createElement('canvas');
  const s = Math.min(1, 1024 / Math.max(v.videoWidth, v.videoHeight));
  c.width = v.videoWidth * s; c.height = v.videoHeight * s;
  c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
  return c.toDataURL('image/jpeg', 0.88);
};
"""


def _claim_view(c: agent.Claim) -> dict[str, str]:
    return {
        "id": c.id, "text": c.text, "check": c.check, "claimed": c.claimed, "concept": c.sam_concept,
        "verdict": c.verdict, "label": VERDICT_LABEL[c.verdict], "observed": c.observed, "reason": c.reason,
        "rule": c.rule, "evidence": c.evidence, "human": c.human,
        "model_only": "yes" if c.rule == "model_judgement" else "",
    }


class State(rx.State):
    phase: str = "start"  # start | checking | review | done
    listing: str = DEFAULT_LISTING
    claims: list[dict[str, str]] = []
    direction: str = ""
    direction_why: list[str] = []
    guide: str = ""
    frame: str = ""
    trace: list[dict[str, str]] = []
    busy: bool = False
    status: str = ""
    error: str = ""
    corrected: str = ""
    cost: str = "$0.000"
    shots: int = 0
    camera_on: bool = False
    next_sample: int = 0

    @rx.var
    def settled(self) -> int:
        return sum(c["verdict"] in ("verified", "contradicted") for c in self.claims)

    @rx.var
    def contradictions(self) -> int:
        return sum(c["verdict"] == "contradicted" for c in self.claims)

    @rx.var
    def all_settled(self) -> bool:
        return bool(self.claims) and self.settled == len(self.claims)

    def _session(self) -> agent.Session | None:
        return SESSIONS.get(self.router.session.client_token)

    def _sync(self, s: agent.Session) -> None:
        self.claims = [_claim_view(c) for c in s.claims]
        self.trace = [{"model": t.model, "purpose": t.purpose, "ms": f"{t.ms / 1000:.1f}s", "usd": f"${t.usd:.4f}"}
                      for t in reversed(s.trace)]
        self.cost = f"${s.cost():.3f}"
        self.shots = s.shots

    @rx.event
    def set_listing(self, value: str):
        self.listing = value

    @rx.event
    def set_corrected(self, value: str):
        self.corrected = value

    @rx.event
    def reset_all(self):
        SESSIONS.pop(self.router.session.client_token, None)
        self.phase, self.claims, self.trace, self.frame = "start", [], [], ""
        self.direction, self.direction_why, self.guide, self.error, self.status = "", [], "", "", ""
        self.corrected, self.cost, self.shots, self.next_sample = "", "$0.000", 0, 0
        self.busy = False

    @rx.event(background=True)
    async def start_check(self):
        async with self:
            if self.busy or not self.listing.strip():
                return
            self.busy, self.error = True, ""
            self.status = "Muse Spark is splitting the listing into checkable claims…"
            session = agent.Session(self.listing.strip())
            SESSIONS[self.router.session.client_token] = session
            self.phase = "checking"
        try:
            await self._track(session, asyncio.to_thread(agent.extract_claims, session))
            direction = agent.next_direction(session)
            async with self:
                self._sync(session)
                if not session.claims:
                    self.error = "This listing has no claim a camera could check."
                self.direction = "Show me the product." if direction else ""
                self.direction_why = []
        except Exception as e:
            async with self:
                self.error = str(e)
        finally:
            async with self:
                self.busy, self.status = False, ""

    async def _track(self, session: agent.Session, work) -> object:
        """Await blocking agent work, streaming each finished model call to the UI."""
        task = asyncio.ensure_future(work)
        seen = -1
        while not task.done():
            if len(session.trace) != seen:
                seen = len(session.trace)
                async with self:
                    self._sync(session)
            await asyncio.sleep(0.25)
        return task.result()

    @rx.event(background=True)
    async def process_pending(self):
        async with self:
            token = self.router.session.client_token
            raw = PENDING.pop(token, None)
            session = SESSIONS.get(token)
            if raw is None or session is None or self.busy:
                return
            self.busy, self.error = True, ""
            self.status = "SAM 3.1 is finding the evidence, Muse Spark is reading it…"
            self.guide = ""
            self.frame = vision.to_data_url(vision.prepare_image(raw))
        direction = None
        try:
            result = await self._track(session, asyncio.to_thread(agent.analyze_shot, session, raw))
            direction = agent.next_direction(session)
            async with self:
                self._sync(session)
                self.frame = result["frame"]
                self.direction = direction["ask"] if direction else ""
                self.direction_why = direction["why"] if direction else []
        except Exception as e:
            async with self:
                self.error = str(e)
        finally:
            async with self:
                self.busy, self.status = False, ""
        if direction:
            await self._load_guide(session, direction)

    async def _load_guide(self, session: agent.Session, direction: dict):
        key = direction["concept"]
        if key not in GUIDE_CACHE:
            prompt = (f"Simple black line-art instruction pictogram on white: a hand holding a smartphone "
                      f"photographing a {key} close-up. No text, no logos.")
            try:
                raw = await asyncio.to_thread(api.generate_image, prompt, session.trace)
                GUIDE_CACHE[key] = vision.to_data_url(vision.prepare_image(raw))
            except Exception:
                return  # the guide is optional; the written direction stands
        async with self:
            if self.direction == direction["ask"]:
                self.guide = GUIDE_CACHE[key]
            self._sync(session)

    @rx.event
    def use_sample(self, name: str):
        PENDING[self.router.session.client_token] = (SAMPLES / name).read_bytes()
        names = [s["name"] for s in SAMPLE_SHOTS]
        self.next_sample = names.index(name) + 1
        return State.process_pending

    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        if files:
            PENDING[self.router.session.client_token] = await files[0].read()
            return State.process_pending

    @rx.event
    def camera_toggle(self):
        return rx.call_script("window.ccCameraStart()", callback=State.camera_result)

    @rx.event
    def camera_result(self, result: str):
        if result == "ok":
            self.camera_on = True
        else:
            self.error = f"Camera unavailable: {result}. Use sample shots or upload a photo."

    @rx.event
    def capture(self):
        return rx.call_script("window.ccCapture()", callback=State.captured)

    @rx.event
    def captured(self, url: str):
        if url and url.startswith("data:image"):
            PENDING[self.router.session.client_token] = base64.b64decode(url.split(",", 1)[1])
            return State.process_pending

    @rx.event
    def human_mark(self, claim_id: str, mark: str):
        s = self._session()
        if not s:
            return
        for c in s.claims:
            if c.id == claim_id:
                c.human = mark
                FEEDBACK.parent.mkdir(exist_ok=True)
                with FEEDBACK.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": int(time.time()), "claim": c.text, "check": c.check, "claimed": c.claimed,
                                        "observed": c.observed, "verdict": c.verdict, "rule": c.rule, "human": mark},
                                       ensure_ascii=False) + "\n")
        self._sync(s)

    @rx.event(background=True)
    async def finish(self):
        async with self:
            session = self._session()
            if not session or self.busy:
                return
            self.busy, self.error = True, ""
            self.status = "Muse Spark is drafting the corrected listing…"
        try:
            text = await self._track(session, asyncio.to_thread(agent.correct_listing, session))
            async with self:
                self.corrected = text
                self.phase = "review"
                self._sync(session)
        except Exception as e:
            async with self:
                self.error = str(e)
        finally:
            async with self:
                self.busy, self.status = False, ""

    @rx.event
    def approve(self):
        self.phase = "done"
        s = self._session()
        lines = ["# ClaimCheck report", "", "## Approved listing", "", self.corrected, "", "## Original listing", "",
                 self.listing, "", "## Claims", ""]
        for c in s.claims if s else []:
            lines.append(f"- **{VERDICT_LABEL[c.verdict]}**: {c.text} | observed: {c.observed or 'n/a'} | "
                         f"rule: {c.rule} | {c.reason}" + (f" | reviewer: {c.human}" if c.human else ""))
        lines += ["", f"Shots: {self.shots}. Model cost: {self.cost}. Final approval by a person."]
        return rx.download(data="\n".join(lines), filename="claimcheck_report.md")


# ---------- Components ----------
INK, MUTED, LINE = "#16181d", "#667085", "#e4e7ec"
TONE = {"verified": "green", "contradicted": "red", "not_visible": "amber", "open": "gray"}


def header() -> rx.Component:
    return rx.hstack(
        rx.hstack(
            rx.box(rx.icon("scan-eye", size=20, color="white"), bg="#3538cd", padding="6px", border_radius="9px"),
            rx.heading("ClaimCheck", size="5", weight="bold"),
            rx.text("Listings, checked against the real product", color=MUTED, size="2", display=["none", "none", "block"]),
            align="center", spacing="3",
        ),
        rx.spacer(),
        rx.hstack(
            rx.badge(rx.icon("camera", size=12), State.shots.to_string() + " shots", variant="soft", color_scheme="gray"),
            rx.badge(rx.icon("coins", size=12), State.cost, variant="soft", color_scheme="gray"),
            rx.icon_button(rx.icon("rotate-ccw", size=15), on_click=State.reset_all, variant="ghost", color_scheme="gray", aria_label="Start over"),
            align="center",
        ),
        width="100%", padding_y="14px", align="center",
    )


def start_view() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.heading("What does your listing promise?", size="7", weight="bold"),
            rx.text("Paste it. ClaimCheck finds what a camera can prove, then tells you exactly what to show it.", color=MUTED, size="3"),
            rx.text_area(value=State.listing, on_change=State.set_listing, rows="5", width="100%", size="3"),
            rx.button(rx.icon("play", size=16), "Check this listing", on_click=State.start_check, size="3", width="100%", color_scheme="indigo"),
            spacing="4", width="min(640px, 100%)", padding_top="8vh",
        ),
        width="100%",
    )


def director() -> rx.Component:
    return rx.box(
        rx.cond(
            State.busy,
            rx.hstack(rx.spinner(size="3"), rx.text(State.status, size="4", weight="medium"), align="center", spacing="3"),
            rx.cond(
                State.direction != "",
                rx.vstack(
                    rx.text("NEXT SHOT", size="1", weight="bold", color="#c7d0ff", letter_spacing="0.08em"),
                    rx.text(State.direction, size="6", weight="bold", color="white", line_height="1.25"),
                    rx.cond(State.direction_why.length() > 0,
                            rx.text("To settle: ", rx.foreach(State.direction_why, lambda w: rx.text.span(w + "  ")), size="2", color="#c7d0ff")),
                    spacing="1",
                ),
                rx.vstack(
                    rx.text("ALL CLAIMS SETTLED", size="1", weight="bold", color="#c7d0ff", letter_spacing="0.08em"),
                    rx.text("Review the results, then draft the corrected listing.", size="5", weight="bold", color="white"),
                    spacing="1",
                ),
            ),
        ),
        bg=rx.cond(State.busy, "#f2f4f7", "#2d31a6"),
        padding="18px 20px", border_radius="14px", width="100%", min_height="92px",
    )


def stage() -> rx.Component:
    return rx.vstack(
        director(),
        rx.box(
            rx.el.video(id="cc-video", auto_play=True, muted=True, plays_inline=True,
                        style={"width": "100%", "display": rx.cond(State.camera_on, "block", "none"), "maxHeight": "280px", "objectFit": "contain", "background": "#000"}),
            rx.cond(
                State.frame != "",
                rx.image(src=State.frame, width="100%", max_height="56vh", object_fit="contain", bg="#101114"),
                rx.center(rx.vstack(rx.icon("image", size=30, color=MUTED), rx.text("Your shots appear here, with SAM 3.1 masks", color=MUTED), align="center"), height="300px", bg="#f8f9fb"),
            ),
            position="relative", width="100%", border_radius="14px", overflow="hidden", border=f"1px solid {LINE}",
        ),
        rx.flex(
            rx.cond(State.camera_on,
                    rx.button(rx.icon("aperture", size=16), "Capture", on_click=State.capture, disabled=State.busy, color_scheme="indigo", size="3"),
                    rx.button(rx.icon("video", size=16), "Use camera", on_click=State.camera_toggle, variant="soft", size="3")),
            rx.upload(
                rx.button(rx.icon("upload", size=16), "Upload photo", variant="soft", size="3"),
                id="shot", accept={"image/*": []}, max_files=1, border="none", padding="0",
                on_drop=State.handle_upload(rx.upload_files(upload_id="shot")),
            ),
            rx.spacer(),
            rx.cond(State.claims.length() > 0,
                    rx.button(rx.icon("pen-line", size=16), "Draft corrected listing", on_click=State.finish, disabled=State.busy,
                              variant=rx.cond(State.all_settled, "solid", "outline"), color_scheme="indigo", size="3")),
            gap="8px", wrap="wrap", width="100%", align="center",
        ),
        rx.vstack(
            rx.text("Sample shots (AI-generated test images, for trying the flow)", size="1", color=MUTED),
            rx.hstack(*[
                rx.box(
                    rx.image(src=f"/samples/{s['name']}", height="64px", width="96px", object_fit="cover", border_radius="8px"),
                    rx.text(s["title"], size="1", color=MUTED, width="96px", line_height="1.2", margin_top="3px"),
                    on_click=State.use_sample(s["name"]), cursor="pointer", class_name="sample", id=f"sample-{i}",
                    opacity=rx.cond(State.busy, "0.5", "1"), pointer_events=rx.cond(State.busy, "none", "auto"),
                ) for i, s in enumerate(SAMPLE_SHOTS)
            ], spacing="3"),
            spacing="1",
        ),
        rx.cond(State.guide != "",
                rx.hstack(rx.image(src=State.guide, height="84px", border_radius="8px", border=f"1px solid {LINE}"),
                          rx.text("Shot guide drawn by Muse Image. An illustration, never evidence.", size="1", color=MUTED, max_width="220px"),
                          align="center")),
        spacing="3", width="100%",
    )


def claim_card(c) -> rx.Component:
    tone = rx.match(c["verdict"], ("verified", "green"), ("contradicted", "red"), ("not_visible", "amber"), "gray")
    return rx.box(
        rx.hstack(
            rx.badge(c["label"], color_scheme=tone, variant="solid", size="1"),
            rx.spacer(),
            rx.cond(c["model_only"] == "yes", rx.tooltip(rx.badge("model judgement", variant="outline", color_scheme="gray", size="1"), content="No deterministic rule applies; this rests on Muse Spark's judgement")),
            rx.badge(c["concept"], variant="soft", color_scheme="gray", size="1"),
            width="100%", align="center",
        ),
        rx.text(c["text"], weight="medium", size="3", margin_top="8px"),
        rx.cond(c["observed"] != "", rx.text("Seen: ", rx.text.strong(c["observed"]), size="2", margin_top="2px")),
        rx.cond(c["reason"] != "", rx.text(c["reason"], size="1", color=MUTED, margin_top="2px")),
        rx.cond(
            (c["verdict"] == "verified") | (c["verdict"] == "contradicted"),
            rx.hstack(
                rx.cond(c["evidence"] != "", rx.dialog.root(
                    rx.dialog.trigger(rx.button(rx.icon("image", size=12), "Evidence", size="1", variant="soft", color_scheme="gray")),
                    rx.dialog.content(rx.dialog.title(c["text"]), rx.image(src=c["evidence"], width="100%"), rx.text(c["rule"], " · ", c["reason"], size="1", color=MUTED), rx.dialog.close(rx.button("Close", variant="soft", margin_top="8px"))),
                )),
                rx.spacer(),
                rx.button(rx.icon("thumbs-up", size=12), size="1", variant=rx.cond(c["human"] == "agree", "solid", "ghost"), color_scheme="green", on_click=State.human_mark(c["id"], "agree"), aria_label="Agree"),
                rx.button(rx.icon("thumbs-down", size=12), size="1", variant=rx.cond(c["human"] == "disagree", "solid", "ghost"), color_scheme="red", on_click=State.human_mark(c["id"], "disagree"), aria_label="Disagree"),
                width="100%", margin_top="8px", align="center",
            ),
        ),
        padding="12px 14px", border_radius="12px", border=f"1px solid {LINE}", bg="white", width="100%", class_name="claim", data_verdict=c["verdict"],
    )


def ledger() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.heading("Claims", size="4"),
            rx.spacer(),
            rx.text(State.settled.to_string(), " of ", State.claims.length().to_string(), " settled", size="2", color=MUTED),
            width="100%", align="center",
        ),
        rx.foreach(State.claims, claim_card),
        trace_panel(),
        spacing="2", width="100%",
    )


def trace_panel() -> rx.Component:
    return rx.vstack(
        rx.text("Live Meta model calls", size="2", weight="bold", margin_top="10px"),
        rx.box(
            rx.foreach(State.trace, lambda t: rx.hstack(
                rx.code(t["model"], size="1", variant="ghost"),
                rx.text(t["purpose"], size="1", color=MUTED, overflow="hidden", white_space="nowrap", text_overflow="ellipsis", flex="1"),
                rx.text(t["ms"], size="1", color=MUTED),
                rx.text(t["usd"], size="1", color=MUTED),
                width="100%", align="center", padding_y="3px",
            )),
            width="100%", max_height="170px", overflow_y="auto",
        ),
        spacing="1", width="100%",
    )


def review_view() -> rx.Component:
    return rx.vstack(
        rx.heading(rx.cond(State.phase == "done", "Approved. Report downloaded.", "Your call"), size="6"),
        rx.text(State.contradictions.to_string(), " contradicted. Unsettled claims stay as written and are listed in the report. Nothing is published without you.", color=MUTED),
        rx.grid(
            rx.vstack(rx.text("Original", weight="bold", size="2"), rx.box(rx.text(State.listing), padding="14px", bg="#fef3f2", border_radius="12px", width="100%", min_height="140px"), width="100%"),
            rx.vstack(rx.text("Corrected, editable", weight="bold", size="2"), rx.text_area(value=State.corrected, on_change=State.set_corrected, rows="6", width="100%", read_only=State.phase == "done"), width="100%"),
            columns=rx.breakpoints(initial="1", md="2"), spacing="4", width="100%",
        ),
        rx.cond(State.phase == "review",
                rx.button(rx.icon("check", size=16), "Approve and download report", on_click=State.approve, size="3", color_scheme="green"),
                rx.button("Check another listing", on_click=State.reset_all, size="3", variant="soft")),
        ledger(),
        spacing="4", width="100%", max_width="900px", margin_x="auto", padding_top="20px",
    )


def index() -> rx.Component:
    return rx.box(
        rx.script(CAMERA_JS),
        rx.box(
            header(),
            rx.cond(State.error != "", rx.callout(State.error, icon="triangle-alert", color_scheme="red", width="100%", margin_bottom="10px")),
            rx.match(
                State.phase,
                ("start", start_view()),
                ("checking", rx.grid(stage(), ledger(), columns=rx.breakpoints(initial="1", md="3fr 2fr"), spacing="5", width="100%")),
                review_view(),
            ),
            max_width="1280px", margin_x="auto", padding_x="16px", padding_bottom="32px",
        ),
        bg="#fbfbfd", min_height="100vh", color=INK,
    )


app = rx.App(theme=rx.theme(accent_color="indigo", gray_color="slate", appearance="light", radius="large"))
app.add_page(index, title="ClaimCheck")
