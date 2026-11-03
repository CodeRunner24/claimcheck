"""What each phase looks like: a poster, the camera, a progress line, a product page."""
from __future__ import annotations

import reflex as rx

from .state import State


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
