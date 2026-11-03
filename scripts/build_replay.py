"""Build docs/, a static page that replays one real run. No server, no API key and no credit is needed to look at it.

    python -u scripts/build_replay.py [video] ["what the seller said" ...]

Every answer on the page was produced by the real pipeline on the run this script performs, against live Meta models. The page
replays those recorded answers, says so, and can't take new questions. Costs about $0.30.
"""
import json
import shutil
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from claimcheck_core import agent, report, rules

DOCS = ROOT / "docs"
WORK = ROOT / "eval" / "ask_out" / "replay"
QUESTIONS = ["Is it really 100% cotton?", "Do I get two shirts?", "Was it made in Portugal?", "Does it have a pocket?", "Any holes or stains?",
             "Does it have a zipper?", "What does the back look like?", "Is it comfortable?"]
SAYING = ["This is a set of 2 black t-shirts.", "It's 100% cotton and made in Portugal."]
SAMPLE = ROOT / "assets" / "samples" / "sample_take.webm"


def run(video: Path, saying: list[str]) -> agent.Session:
    shutil.rmtree(WORK, ignore_errors=True)
    raw = video.read_bytes()
    session = agent.Session("", spoken=True, out_dir=WORK)
    for sentence in saying:
        agent.hear(session, sentence)
    agent.analyze_take(session, raw)
    agent.finish(session, raw)
    print("listing", session.title, "|", session.corrected, flush=True)
    for question in QUESTIONS:
        card = agent.ask(session, question)
        print(question, "->", card["headline"], card.get("seconds"), card.get("usd"), flush=True)
    return session


def publish(session: agent.Session, name: str) -> str:
    if not name:
        return ""
    media = DOCS / "media"
    media.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(WORK / name, media / name)
    return f"media/{name}"


def facts_html(session: agent.Session) -> str:
    rows = []
    for c in session.claims:
        if c.verdict not in (rules.VERIFIED, rules.CONTRADICTED) or c.source == "noticed" or c.source == "plan":
            continue
        value = agent.pretty(c.observed) or c.claimed
        said = f'<em class="fixed-tag">said {escape(c.claimed)}</em>' if c.verdict == rules.CONTRADICTED else ""
        thumb = publish(session, c.flat or c.evidence)
        rows.append(f'<div class="fact claim"><span>{escape(c.attribute)}</span><b>{escape(value)}{said}</b>'
                    f'<div class="proof">{f"""<img src="{thumb}" alt="">""" if thumb else ""}</div></div>')
    return "".join(rows)


def page(session: agent.Session) -> str:
    wrong = [c for c in session.claims if c.verdict == rules.CONTRADICTED]
    noticed = [c for c in session.claims if c.source == "noticed" and c.verdict == rules.VERIFIED]
    answers = [{"question": a["question"], "tone": a["tone"], "headline": a["headline"], "sentence": a["sentence"],
                "at": f"Seen at {a['at']}" if a["at"] else "", "clip": publish(session, a["clip"]),
                "still": "" if a["clip"] else publish(session, a["still"]), "flat": publish(session, a["flat"]),
                "how": " · ".join(x for x in (a["how"], f"took {a.get('seconds')} live", a.get("usd", "")) if x),
                "cells": a.get("cells", "")} for a in session.answers]
    chips = "".join(f'<button type="button" class="chip" data-i="{i}">{escape(a["question"])}</button>' for i, a in enumerate(answers))
    caught = "".join(f'<li><s>{escape(c.text)}</s> → {escape(agent.pretty(c.observed))}</li>' for c in wrong)
    seen = "".join(f'<div class="noticed-row"><span>{escape(agent.pretty(c.observed)[:1].upper() + agent.pretty(c.observed)[1:])}</span></div>' for c in noticed)
    calls = "".join(f"<div><span>{escape(t.model)}</span><span>{escape(t.purpose)}</span><span>{t.ms / 1000:.1f}s</span><span>${t.usd:.4f}</span></div>" for t in session.trace)
    info = session.hero_info or {}
    hero = publish(session, session.hero)
    reel = publish(session, session.reel)
    strip = publish(session, session.strip)
    template = (Path(__file__).parent / "replay_template.html").read_text(encoding="utf-8")
    return (template
            .replace("{{TITLE}}", escape(session.title))
            .replace("{{DESCRIPTION}}", escape(session.corrected))
            .replace("{{HERO}}", f'<img class="photo" src="{hero}" alt="Product photo"><p class="photo-note">The product is the camera\'s real pixels, {info.get("product_pixels", 0):,} of them checked identical. Only the background was generated.</p>' if hero else "")
            .replace("{{REEL}}", f'<div class="reel-wrap"><h2>The take, every piece of evidence tracked</h2><video class="reel" src="{reel}" muted autoplay loop playsinline controls></video></div>' if reel else "")
            .replace("{{CAUGHT_TITLE}}", "One thing the seller said wasn't true." if len(wrong) == 1 else f"{len(wrong)} things the seller said weren't true.")
            .replace("{{CAUGHT}}", caught)
            .replace("{{NOTICED}}", f'<div class="noticed"><h2>The camera also noticed</h2>{seen}</div>' if seen else "")
            .replace("{{FACTS}}", facts_html(session))
            .replace("{{CHIPS}}", chips)
            .replace("{{STRIP}}", strip)
            .replace("{{CELLS}}", '<i class="cell"></i>' * agent.STRIP_CELLS)
            .replace("{{SECONDS}}", f"{len(session.take.frames) / session.take.fps:.0f}")
            .replace("{{CALLS}}", calls)
            .replace("{{CALL_COUNT}}", str(len(session.trace)))
            .replace("{{COST}}", f"${session.cost():.3f}")
            .replace("{{ANSWERS}}", json.dumps(answers, ensure_ascii=False).replace("</", "<\\/")))


def main() -> None:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE
    saying = sys.argv[2:] or SAYING
    session = run(video, saying)
    shutil.rmtree(DOCS, ignore_errors=True)
    (DOCS / "media").mkdir(parents=True)
    html = page(session)
    shutil.copyfile(ROOT / "assets" / "claimcheck.css", DOCS / "claimcheck.css")
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    (ROOT / "demo" / "sample_proof.html").write_text(report.build(session, session.corrected), encoding="utf-8")
    size = sum(p.stat().st_size for p in DOCS.rglob("*") if p.is_file())
    print("wrote docs/", round(size / 1e6, 1), "MB, cost", round(session.cost(), 3), flush=True)


if __name__ == "__main__":
    main()
