"""What each phase looks like: a poster, the camera, a progress line, a product page."""
from __future__ import annotations

import reflex as rx

from .state import State


def start_view() -> rx.Component:
    """The poster: start filming, or upload a video, paste a listing or watch the sample."""
    return rx.el.main(
        rx.el.h1("Film it. Say it. Sell what's true."),
        rx.el.div(rx.el.s("100% cotton"), rx.el.b("60% cotton"), class_name="promise", aria_hidden="true"),
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
    """One piece of the transcript, marked with its verdict and any correction."""
    return rx.fragment(
        rx.el.span(s["text"], class_name=f"seg {s['verdict']}"),
        rx.cond(s["fix"] != "", rx.el.span(s["fix"], class_name="fix")),
    )


def live_view() -> rx.Component:
    """The viewfinder: the camera, the masks, one line of direction and the words as they are heard."""
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
            rx.cond(State.source == "sample", rx.el.div("Sample clip", class_name="sample-note")),
            rx.el.div(State.ticker, class_name="ticker", id="cc-ticker"),
            class_name="stage", style={"--ratio": State.ratio},
        ),
        class_name="live",
    )


def cutting_view() -> rx.Component:
    """The short wait after filming."""
    return rx.el.main(
        rx.el.h2("Writing your listing"),
        rx.el.div(rx.el.i(style={"width": f"{State.progress}%"}), class_name="bar-track"),
        rx.el.p(State.ticker),
        rx.el.ul(rx.foreach(State.caught, lambda c: rx.el.li(rx.el.s(c["text"]), " ", rx.el.b(c["value"])))),
        class_name="cutting",
    )


def proof_button(c) -> rx.Component:
    """A fact's evidence thumbnail, opening the full frame in a dialog."""
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
    """One row of the fact table."""
    return rx.el.div(
        rx.el.span(c["attribute"]),
        rx.el.b(c["value"], rx.cond(c["verdict"] == "contradicted", rx.el.em("you said " + c["claimed"], class_name="fixed-tag"))),
        proof_button(c),
        class_name="fact claim", data_verdict=c["verdict"],
    )


def answer_card(a) -> rx.Component:
    """One answered question: the moment that answers it, and how the answer was reached."""
    return rx.el.article(
        rx.el.div(
            rx.cond(a["clip"] != "", rx.el.video(src=rx.get_upload_url(a["clip"]), auto_play=True, muted=True, loop=True, plays_inline=True, class_name="a-clip")),
            rx.cond(a["still"] != "", rx.el.img(src=rx.get_upload_url(a["still"]), class_name="a-clip", alt="The frame that answers this")),
            class_name="a-view",
        ),
        rx.el.div(
            rx.el.p(a["question"], class_name="a-q"),
            rx.el.h3(a["headline"], class_name="a-head"),
            rx.el.p(a["sentence"], class_name="a-said"),
            rx.cond(a["at"] != "", rx.el.p(a["at"], class_name="a-at")),
            rx.cond(a["flat"] != "", rx.el.img(src=rx.get_upload_url(a["flat"]), class_name="a-flat", alt="The label, flattened")),
            rx.el.small(a["how"], class_name="a-how"),
            class_name="a-text",
        ),
        class_name=f"answer {a['tone']}",
    )


def ask_band() -> rx.Component:
    """The footage stays useful after the listing is written: a buyer can question it directly."""
    return rx.el.section(
        rx.el.p("What your buyer gets", class_name="ask-eyebrow"),
        rx.el.h2("Don't trust the seller. Ask the video."),
        rx.el.form(
            rx.el.input(value=State.question, on_change=State.set_question, placeholder="Does it have a pocket?", aria_label="Your question",
                        max_length=200, disabled=State.asking, id="cc-question", name="question", auto_complete="off"),
            rx.el.button("Ask", type="submit", disabled=State.asking, class_name="ask-go", id="cc-ask"),
            rx.el.button(rx.cond(State.listening, "Listening. Tap to stop", "Say it"), type="button", on_click=State.toggle_voice, disabled=State.asking,
                         class_name=rx.cond(State.listening, "ask-say on", "ask-say"), id="cc-say"),
            on_submit=State.ask_typed, class_name="ask-form",
        ),
        rx.el.div(rx.foreach(State.suggestions, lambda q: rx.el.button(q, on_click=State.ask(q), disabled=State.asking, type="button", class_name="chip")), class_name="chips"),
        rx.el.div(
            rx.el.img(src=rx.get_upload_url(State.strip_url), alt="", class_name="strip-img"),
            rx.el.div(rx.foreach(State.strip, lambda s: rx.el.i(class_name=f"cell {s}")), class_name="strip-cells"),
            class_name=rx.cond(State.asking, "strip scanning", "strip"), id="cc-strip",
        ),
        rx.el.p(rx.cond(State.asking, State.ask_stage, "Every second of the seller's video, searchable. If it was never filmed, the answer says so."), class_name="ask-stage", id="cc-stage"),
        rx.el.div(rx.foreach(State.answers, answer_card), id="cc-answers"),
        class_name="ask", id="cc-ask-band",
    )


def reveal_view() -> rx.Component:
    """The finished product page: photo, listing, facts and reel, then the buyer's questions."""
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
                    ),
                    rx.cond(
                        State.reel != "",
                        rx.el.div(
                            rx.el.h2("The take, every piece of evidence tracked"),
                            rx.el.video(src=rx.get_upload_url(State.reel), controls=True, muted=True, auto_play=True, loop=True, plays_inline=True, class_name="reel", id="cc-reel"),
                            class_name="reel-wrap",
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
                    rx.cond(
                        State.noticed.length() > 0,
                        rx.el.div(
                            rx.el.h2("The camera also noticed"),
                            rx.foreach(State.noticed, lambda c: rx.el.div(
                                rx.el.span(c["value"]), proof_button(c),
                                rx.el.button("Not damage", on_click=State.dismiss_noticed(c["id"]), class_name="quiet"),
                                class_name="noticed-row",
                            )),
                            class_name="noticed", id="cc-noticed",
                        ),
                    ),
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
            rx.cond(State.can_ask, ask_band()),
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
    """The page: whichever view matches the current phase."""
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
