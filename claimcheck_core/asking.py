"""A buyer asks; the footage answers, or says it can't.

The seller's video stays searchable after the listing is written. A question nobody planned for becomes a SAM 3.1 search
through the whole take, and the answer is the moment it was found, spotlighted, rather than a sentence to be trusted.

Saying "it isn't there" is the dangerous answer, so it takes two independent failures: SAM finds nothing the model will
confirm, and Muse Spark can't see it in frames spread across the take. Settled facts are answered from their verdicts in
code, so a question can never talk the system out of something the rules already decided.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import meta_api as api
from . import rules, take as takes, vision
from .perception import store
from .prompts import ASK_PROMPT, LOOK_PROMPT, SUGGEST_PROMPT, SWEEP_PROMPT
from .session import Claim, Session, pretty

STRIP_CELLS = 32
SEARCH_FRAMES = 120    # a longer take is searched at a lower frame rate; the answer clip is then re-tracked at the full rate
SWEEP_FRAMES = 5
BEFORE, AFTER = 1.5, 3.0
MAX_QUESTION = 200
ASK_CAP_USD = float(os.getenv("CLAIMCHECK_ASK_CAP_USD", "3"))  # per listing, on top of what filming cost
ABSENCE = re.compile(r"\s*(no|not|none|nothing|cannot|can't)\b", re.I)


def _clock(seconds: float) -> str:
    """Seconds as the 0:07 a viewer would read on a player."""
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def _cell(index: int, total: int) -> int:
    """The filmstrip cell that covers a frame of the take."""
    return min(STRIP_CELLS - 1, int(index * STRIP_CELLS / max(1, total)))


def filmstrip(session: Session) -> None:
    """The whole take as one strip of thumbnails, so a search has something to light up."""
    take = session.take
    if take is None or session.out_dir is None:
        return
    total = len(take.frames)
    w, h = take.size
    tile_h, tile_w = 120, 60  # tall slices from the middle of each frame: a whole frame would be squeezed unrecognisably
    tiles = []
    for cell in range(STRIP_CELLS):
        frame = cv2.imdecode(np.frombuffer(take.frames[min(total - 1, int((cell + 0.5) * total / STRIP_CELLS))], np.uint8), cv2.IMREAD_COLOR)
        slice_w = min(w, h // 2)
        x0 = (w - slice_w) // 2
        tiles.append(cv2.resize(frame[:, x0:x0 + slice_w], (tile_w, tile_h), interpolation=cv2.INTER_AREA))
    ok, buffer = cv2.imencode(".jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        session.strip = store(session, "strip", buffer.tobytes())


def suggest_questions(session: Session) -> None:
    """Questions worth trying, written for this product. They are only suggestions, and any question can be typed instead."""
    picture = session.hero_source[0] if session.hero_source else session.first_look
    if not picture:
        return
    facts = [{"attribute": c.attribute, "value": pretty(c.observed) or c.claimed, "said": c.text if c.source == "said" else ""}
             for c in session.claims if c.verdict in (rules.VERIFIED, rules.CONTRADICTED)]
    data = api.spark_json(SUGGEST_PROMPT.format(product=session.product or session.product_concept, facts=json.dumps(facts, ensure_ascii=False)),
                          [picture], purpose="suggest buyer questions", trace=session.trace, effort="low")
    session.suggestions = [str(q).strip()[:60] for q in (data.get("questions") or []) if str(q).strip()][:4]


def _search(session: Session, concept: str) -> dict[int, list[vision.Instance]]:
    """Every frame of the take where SAM 3.1 finds the concept, keyed by frame of the take. Hits light up as they stream in."""
    take = session.take
    total = len(take.frames)
    step = max(1, math.ceil(total / SEARCH_FRAMES))
    if session.search_clip is None:  # one copy for every question, so asking again doesn't re-encode the take
        session.search_clip = take if step == 1 else takes.subset(take, list(range(0, total, step)), max(1, take.fps // step))
    clip = session.search_clip
    cells = session.asking["cells"]

    def tick(index: int):
        cells[_cell(index * step, total)] = "hit"

    lines = api.sam_video(concept, clip.mp4, session.trace, tick)
    return {index * step: found for index, found in vision.parse_video(lines).items() if found}


def _paint(session: Session, hits: dict[int, list[vision.Instance]]) -> None:
    """The search is over: show exactly where the thing was found, and nothing else."""
    total = len(session.take.frames)
    lit = {_cell(index, total) for index in hits}
    session.asking["cells"] = ["hit" if cell in lit else "miss" for cell in range(STRIP_CELLS)]


def _window(hits: dict[int, list[vision.Instance]], best: int, total: int, fps: int) -> list[int]:
    """Frames around the best one. It stops early where the object leaves the view, unless that would leave almost nothing."""
    before, after = int(BEFORE * fps), int(AFTER * fps)
    step = 1
    known = sorted(hits)
    if len(known) > 1:
        step = max(1, min(b - a for a, b in zip(known, known[1:])))
    start = best
    while start - step in hits and best - (start - step) <= before:
        start -= step
    end = best
    while end + step in hits and (end + step) - best <= after:
        end += step
    if end - start < 2 * fps:  # a glimpse: keep the frames around it so the clip can still be watched
        start, end = max(0, best - fps), min(total - 1, best + fps)
    return list(range(max(0, start), min(total - 1, end) + 1))


def _clip(session: Session, concept: str, hits: dict[int, list[vision.Instance]], best: int, label: str, tone: str) -> str:
    """The moment as a short clip: the rest of the frame drained of color, the camera pushing in on what was found."""
    take = session.take
    if session.out_dir is None:
        return ""
    indices = _window(hits, best, len(take.frames), take.fps)
    clip = takes.subset(take, indices)
    known = sorted(hits)
    full_rate = len(known) < 2 or min(b - a for a, b in zip(known, known[1:])) == 1
    if full_rate:
        tracked = {n: hits[i] for n, i in enumerate(indices) if i in hits}
    else:
        tracked = {index: found for index, found in vision.parse_video(api.sam_video(concept, clip.mp4, session.trace)).items() if found}
    focus = indices.index(best) if best in indices else 0
    if focus not in tracked:
        tracked[focus] = hits[best]
    session.file_seq += 1
    name = f"answer_{session.file_seq}.mp4"
    session.out_dir.mkdir(parents=True, exist_ok=True)
    takes.render_answer(clip, tracked, focus, label, session.out_dir / name, tone)
    return name


def _best(hits: dict[int, list[vision.Instance]], session: Session) -> tuple[int, vision.Instance] | None:
    """The best view, preferring frames where the thing is entirely in shot rather than filling the frame in a close-up of something else."""
    w, h = session.take.size
    whole = {i: found for i, found in hits.items()
             if found and all(x.in_full_view(w, h) for x in found)}
    return takes.best_read_frame(whole or hits, session.take)


def _candidates(hits: dict[int, list[vision.Instance]], session: Session, moments: int = 3) -> list[tuple[int, vision.Instance]]:
    """The best frame from each part of the hits. A false match in one shot then can't hide a true one in another."""
    order = sorted(hits)
    parts = [order[i * len(order) // moments:(i + 1) * len(order) // moments] for i in range(moments)]
    found = [_best({i: hits[i] for i in part}, session) for part in parts if part]
    return [f for f in found if f]


def _card(question: str, tone: str, headline: str, sentence: str, **extra) -> dict:
    """An answer as the interface shows it. Tone is one of right, wrong, open, unseen or nocamera."""
    return {"question": question, "tone": tone, "headline": headline, "sentence": sentence, "at": "", "clip": "", "still": "", "flat": "",
            "how": "", "concept": "", **extra}


def _sentence_case(value: str) -> str:
    return value[:1].upper() + value[1:]


def _from_fact(session: Session, question: str, claim: Claim, concepts: list[str] | None = None) -> dict:
    """Written in code from the verdict. The question can't change what the rules decided."""
    value = _sentence_case(pretty(claim.observed) or claim.claimed)
    name = claim.evidence_name or claim.sam_concept
    if claim.verdict == rules.CONTRADICTED and claim.check == "count":
        card = _card(question, "wrong", f"Only {claim.observed} in the video", f"The seller said {claim.text}. No more than {claim.observed} {name} was ever in view together.")
    elif claim.verdict == rules.CONTRADICTED:
        card = _card(question, "wrong", value, f'The seller said "{claim.text}". The evidence shows {value}.')
    elif claim.verdict == rules.VERIFIED and claim.source == "noticed":
        card = _card(question, "wrong", value, "The camera noticed this while the seller was filming.")
    elif claim.verdict == rules.VERIFIED:
        said = "The seller said so and the camera confirmed it." if claim.source == "said" else "It's clear on camera."
        card = _card(question, "right", value, said)
    else:
        return _card(question, "open", "Not settled",
                     f'The seller said "{claim.text}", but the camera never got a clear enough look, so this stays open.',
                     how="No verdict: unseen is never treated as wrong.")
    rule = "a plain rule" if claim.rule and claim.rule != "model_judgement" else "Muse Spark"
    card.update(still=claim.evidence, flat=claim.flat, concept=claim.sam_concept, how=f"Decided earlier by {rule}, from the seller's own footage.")
    # Printed evidence lives on the thing the claim named. A visible feature is better shown by the part the buyer asked about.
    wanted = list(concepts or []) if claim.check in ("visual", "color") else []
    tone = "wrong" if card["tone"] == "wrong" else "verified"
    for concept in wanted + [claim.sam_concept]:
        session.asking["stage"] = f'SAM 3.1 is finding "{concept}" in the footage'
        hits = _search(session, concept)
        found = _best(hits, session)
        if found:
            _paint(session, hits)
            card["at"] = _clock(found[0] / session.take.fps)
            card["concept"] = concept
            session.asking["stage"] = "Cutting the moment"
            card["clip"] = _clip(session, concept, hits, found[0], concept, tone)
            break
    return card


def _guarded(session: Session, question: str, short: str, answer: str) -> dict | None:
    """A disproved value must not come back through a free-form answer."""
    leaked = rules.leaked_values(session.claims, f"{short} {answer}")
    if leaked:
        return _from_fact(session, question, next(c for c in session.claims if c.verdict == rules.CONTRADICTED and c.claimed in leaked))
    return None


def _found(session: Session, question: str, concept: str, hits: dict, index: int, short: str, answer: str) -> dict:
    """A confirmed find, with the clip that shows it."""
    card = _card(question, "right", short, answer, concept=concept, at=_clock(index / session.take.fps),
                 how="Found by SAM 3.1, read by Muse Spark. A model's reading, not a rule's verdict.")
    session.asking["stage"] = "Cutting the moment"
    card["clip"] = _clip(session, concept, hits, index, concept, "verified")
    return card


def _look(session: Session, question: str, concepts: list[str]) -> dict:
    """Search the take for something nobody planned for. SAM proposes, Spark has to confirm it with its own eyes."""
    take = session.take
    seconds = len(take.frames) / take.fps
    session.asking["stage"] = f'SAM 3.1 is searching {seconds:.0f} seconds of footage for "{concepts[0]}"'
    with ThreadPoolExecutor(max_workers=2) as pool:
        searched = dict(zip(concepts, pool.map(lambda c: _search(session, c), concepts)))
    union: dict[int, list[vision.Instance]] = {}
    for hits in searched.values():
        union.update(hits)
    _paint(session, union)
    concept = max(searched, key=lambda c: len(searched[c]))
    hits = searched[concept]
    moments = _candidates(hits, session)
    if moments:
        session.asking["stage"] = "Muse Spark is checking what SAM found"
        images = [image for index, instance in moments
                  for image in (vision.shrink(take.frames[index], 768), vision.crop(take.frames[index], instance.box))]
        data = api.spark_json(LOOK_PROMPT.format(question=json.dumps(question, ensure_ascii=False), concept=concept, n=len(moments)),
                              images, purpose=f'check "{concept}"', trace=session.trace, effort="low")
        if data.get("shows") is True:
            try:
                index, _ = moments[int(data.get("moment", 1)) - 1]
            except (ValueError, IndexError, TypeError):
                index, _ = moments[0]
            short, answer = str(data.get("short", "Yes"))[:48], str(data.get("answer", ""))[:220]
            return _guarded(session, question, short, answer) or _found(session, question, concept, hits, index, short, answer)
    return _sweep(session, question, concepts, seconds)


def _sweep(session: Session, question: str, concepts: list[str], seconds: float) -> dict:
    """SAM can miss what is plainly there, so absence needs a second opinion from frames spread across the take."""
    take = session.take
    session.asking["stage"] = "Nothing yet. Muse Spark is looking through the footage itself"
    total = len(take.frames)
    picks = sorted({min(total - 1, int((i + 0.5) * total / SWEEP_FRAMES)) for i in range(SWEEP_FRAMES)})
    data = api.spark_json(SWEEP_PROMPT.format(question=json.dumps(question, ensure_ascii=False), n=len(picks)),
                          [vision.shrink(take.frames[i], 640) for i in picks], purpose="look through the take", trace=session.trace, effort="low")
    short, answer = str(data.get("short", "Yes"))[:48], str(data.get("answer", ""))[:220]
    # "I can see there's no zipper" is an absence wearing the clothes of a finding. Nothing was found, so it isn't reported as found.
    if data.get("shows") is True and not ABSENCE.match(short):
        try:
            index = picks[int(data.get("frame", 0))]
        except (ValueError, IndexError, TypeError):
            index = picks[0]
        guarded = _guarded(session, question, short, answer)
        if guarded:
            return guarded
        concept = str(data.get("sam_concept") or concepts[0]).strip().lower()[:40]
        session.asking["stage"] = f'SAM 3.1 is outlining "{concept}"'
        hits = _search(session, concept)
        _paint(session, hits or {index: []})
        found = _best(hits, session)
        if found:
            return _found(session, question, concept, hits, found[0], short, answer)
        return _card(question, "right", short, answer, concept=concept, at=_clock(index / take.fps),
                     still=store(session, "answer", take.frames[index]),
                     how="Seen by Muse Spark. SAM 3.1 couldn't outline it, so there's no mask.")
    _paint(session, {})
    return _card(question, "unseen", "Not in this video",
                 f'Neither model found "{concepts[0]}" in the {seconds:.0f} seconds filmed. That isn\'t a no: the seller may not have filmed it.',
                 concept=concepts[0], how="Searched by SAM 3.1, then double-checked by Muse Spark.")


def ask(session: Session, question: str) -> dict:
    """Answer one buyer question from the footage. Returns the card to show, and keeps it with the session."""
    question = " ".join(question.split())[:MAX_QUESTION]
    started, cost_before = time.time(), session.cost()
    session.asking = {"stage": "Reading the question", "cells": [""] * STRIP_CELLS}
    if session.take is None:
        return _card(question, "nocamera", "No video to search", "This listing was checked from a photo, so there's no footage to ask.")
    if session.cost() > session.filming_cost + ASK_CAP_USD:
        return _card(question, "nocamera", "That's enough questions", "This listing has used its question budget.")
    facts = [{"id": c.id, "attribute": c.attribute, "value": pretty(c.observed) or c.claimed, "said": c.text, "status": c.verdict}
             for c in session.claims if c.human != "disagree" and not c.skipped and (c.source != "plan" or c.verdict == rules.VERIFIED)]
    plan = api.spark_json(ASK_PROMPT.format(question=json.dumps(question, ensure_ascii=False), product=session.product or session.product_concept,
                                            facts=json.dumps(facts, ensure_ascii=False)),
                          purpose="understand the question", trace=session.trace, effort="minimal")
    kind = plan.get("kind")
    claim = next((c for c in session.claims if c.id == plan.get("claim_id")), None)
    concepts = [str(c).strip().lower()[:40] for c in (plan.get("sam_concepts") or []) if str(c).strip()][:2]
    if kind == "fact" and claim is not None:
        card = _from_fact(session, question, claim, concepts)
    elif kind == "look" and concepts:
        card = _look(session, question, concepts)
    else:
        _paint(session, {})
        card = _card(question, "nocamera", "A camera can't tell you that", "No video could show this, so it isn't guessed. Ask the seller.",
                     how="Muse Spark judged that no footage could answer this.")
    card["seconds"] = f"{time.time() - started:.1f}s"
    card["usd"] = f"${session.cost() - cost_before:.3f}"
    card["cells"] = "".join("1" if c == "hit" else "0" for c in session.asking["cells"])
    session.answers.append(card)
    return card
