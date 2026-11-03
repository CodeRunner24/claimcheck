"""The live loop: find the evidence, coach the seller in one line, fire the shutter when the frame is good."""
from __future__ import annotations

import os
import time

from . import meta_api as api
from . import rules, vision
from .listening import discover
from .perception import MIN_SHARPNESS, inspect_concept, record_verdicts
from .session import Claim, Session

MIN_FILL_LIVE = 0.13   # evidence for reading should span this share of the frame's short side
STEADY_TICKS = 2       # consecutive good frames before the shutter fires
LOOK_AGAIN_SECONDS = 9.0
MAX_LOOKS = 4          # shutter fires on one claim before it is left open
MAX_TICKS = 400        # hard stop so a forgotten tab can't keep spending
SESSION_CAP_USD = float(os.getenv("CLAIMCHECK_SESSION_CAP_USD", "2"))  # one listing can't run away with the budget
PLAN_PATIENCE = 8.0    # s: how long to wait for evidence the agent wanted but the seller never promised
DISCOVER_EVERY = 3.0   # s
WRAP_UP_AFTER = 12.0   # s of quiet with nothing open
STRUGGLE_SECONDS = 8.0 # s of "closer" or "hold still" before the shutter fires anyway
FOCUS_AFTER_TICKS = 3  # blurry readings in a row before blaming focus rather than a shaking hand
TOO_DARK = 40.0        # brightness of the frame's brightest parts, 0-255
LET_SELLER_TALK = 6.0  # s after the last word before the agent asks about things nobody mentioned


def next_direction(session: Session) -> dict | None:
    """The next shot the agent needs, or None when no claim is open."""
    pending = session.open_claims
    if not pending:
        return None
    # Group by evidence so one shot can settle several claims.
    claim = pending[0]
    together = [c for c in pending if c.sam_concept == claim.sam_concept]
    return {"ask": _ask(session, claim), "concept": claim.sam_concept, "why": [c.text for c in together]}


def _ask(session: Session, claim: Claim) -> str:
    name = claim.evidence_name or claim.sam_concept
    if claim.rule == "too_small":
        return session.line("closer")
    if claim.rule == "too_blurry":
        return session.line("still")
    if claim.rule == "count_short":
        want = rules.parse_count(claim.claimed)
        return session.line("short", got=session.last_count if session.last_count >= 0 else "?", want=want)
    if claim.rule == "count_disagreement":
        return session.line("overlap")
    if claim.rule in ("need_reference", "no_outline"):
        return session.line("paper")
    if claim.rule == "reference_skewed":
        return session.line("skew")
    if claim.next_shot:
        return claim.next_shot
    if claim.check == "measure":
        return session.line("paper")
    return claim.shot_request or session.line("show", c=name)


def _overlay(jpeg: bytes, instances: list[vision.Instance], concept: str, status: str) -> dict:
    size = vision.image_size(jpeg)
    return {"size": size, "status": status, "concept": concept, "paths": [vision.mask_path(i, size) for i in instances]}


def _idle_line(session: Session) -> str:
    if not session.product:
        return session.line("up")
    if not any(c.source == "said" for c in session.claims):
        return session.line("tell", product=session.product.capitalize())
    return session.line("more")


def _claims_to_pursue(session: Session, just_heard: bool, now: float) -> list[Claim]:
    """Open claims worth asking for right now. What the agent merely wishes to know gives way to the seller."""
    pending = session.open_claims
    # Give up quietly on what the agent only wished to know if it never shows up.
    if pending and pending[0].source == "plan" and session.target == pending[0].sam_concept \
            and now - session.target_since > PLAN_PATIENCE:
        pending[0].skipped = True
        pending = session.open_claims
    # Let the seller talk first: don't start asking for extras the moment a product appears.
    if pending and pending[0].source == "plan" and (session.script or session.hearing or just_heard
                                                    or now - max(session.discovered_at, session.last_heard) < LET_SELLER_TALK):
        return []
    return pending


def _between_claims(session: Session, tick: bytes, out: dict, now: float) -> dict:
    """Nothing to pursue: keep the product outlined, keep the seller talking, and decide whether the session is over."""
    concept = session.product_concept
    instances = vision.parse_sam(api.sam_raw(concept, tick, session.trace)) if concept else []
    out["overlay"] = _overlay(tick, instances, concept, "open") if concept else None
    out["direction"] = _idle_line(session)
    quiet = now - max(session.last_heard, session.started, session.discovered_at)
    settled_something = any(c.verdict != "open" and c.source != "seen" for c in session.claims)
    waiting_on_words = session.hearing > 0 or bool(session.script) or bool(out["heard"])
    out["done"] = not waiting_on_words and not session.open_claims and (
        not session.spoken or (settled_something and quiet > WRAP_UP_AFTER))
    out["idle"] = not concept  # nothing to look for yet
    return out


def _coached_for_reading(session: Session, group: list[Claim], tick: bytes, instances: list[vision.Instance], out: dict, now: float) -> bool:
    """True if print can't be read in this frame and the seller was told what to fix.

    A cheap webcam may never satisfy these checks: too far to fill the frame, or too close to focus. So this coaches for a
    while and then lets the shot through. Reading is cheap, and an unreadable label ends as an open claim, not a stuck session.
    """
    tried_long_enough = session.struggling_since and now - session.struggling_since > STRUGGLE_SECONDS
    if any(c.reads_print for c in group) and not tried_long_enough:
        best = vision.largest(instances)
        small = best.short_side < MIN_FILL_LIVE * min(vision.image_size(tick))
        blurry = vision.sharpness(tick, best.box) < MIN_SHARPNESS
        if small or blurry:
            session.steady = 0
            session.struggling_since = session.struggling_since or now
            session.blurry_ticks = session.blurry_ticks + 1 if blurry else 0
            # Blurry over and over while being held still is focus, not shake, and asking to hold still would never help.
            out["direction"] = session.line("closer") if small else session.line("focus" if session.blurry_ticks >= FOCUS_AFTER_TICKS else "still")
            return True
    if not tried_long_enough:
        session.struggling_since = 0.0
    return False


def _waiting_to_look_again(session: Session, group: list[Claim], instances: list[vision.Instance], out: dict, now: float) -> bool:
    """True if this evidence was just read and the seller has been given time to change the shot."""
    waiting = [c for c in group if c.retry_at > now]
    if not waiting or len(waiting) != len(group):
        return False
    claim = waiting[0]
    want = rules.parse_count(claim.claimed) if claim.check == "count" else None
    if want is not None and len(instances) >= want:  # every claimed item is now in frame, so there is no reason to wait
        return False
    out["wait"] = claim.retry_at - now
    out["direction"] = _ask(session, claim)
    return True


def _fire(session: Session, concept: str, group: list[Claim], jpeg: bytes, out: dict, now: float) -> dict:
    """The shutter: re-segment the full frame, read it, let the rules decide everything it can settle, and pick the next line."""
    session.shots += 1
    session.steady = 0
    session.struggling_since = 0.0
    session.blurry_ticks = 0
    out["fired"] = True
    at = now - session.started if session.started else -1.0
    session.shutter = {"frame": vision.to_data_url(jpeg), "overlay": out["overlay"], "line": session.line("reading")}
    _, instances, results = inspect_concept(session, concept, group, jpeg)
    out["settled"] = record_verdicts(session, concept, group, instances, results, jpeg, at)
    out["flat"] = next((c.flat for c in group if c.flat), "")
    out["overlay"] = _overlay(jpeg, instances, concept, rules.overall([results[c.id]["verdict"] for c in group]))
    out["overlay"]["rulers"] = [{"corners": r["corners"], "reference": r.get("reference"), "label": r["observed"]}
                                for r in results.values() if r.get("corners")]
    for c in group:
        if c.verdict == rules.NOT_VISIBLE:
            c.retry_at = now + (LOOK_AGAIN_SECONDS if c.rule == "count_short" else 3.0)
            if c.source == "plan":  # the agent's own curiosity gets one look at usable evidence
                c.skipped = True
            elif c.attempts >= MAX_LOOKS:  # stay honest and move on rather than nag or keep spending
                c.skipped = True
                c.reason = f"Left open after {c.attempts} looks. {c.reason}"
    remaining = session.open_claims
    if remaining:
        out["direction"] = _ask(session, remaining[0])
    else:
        out["direction"] = _idle_line(session) if session.spoken else ""
        out["done"] = not session.spoken and not session.hearing and not session.script
    return out


def live_step(session: Session, raw_image: bytes) -> dict:
    """One tick of the live director: find the evidence it needs, coach the seller, fire when the frame is good.

    A tick costs one SAM call. Muse Spark is called only when the shutter fires, so the coaching stays fast and cheap.
    """
    now = time.time()
    session.ticks += 1
    out: dict = {"fired": False, "done": False, "overlay": None, "direction": "", "wait": 0.0, "steady": 0,
                 "settled": [], "flat": "", "idle": False, "heard": [], "stopped": session.ticks > MAX_TICKS or session.cost() > SESSION_CAP_USD}
    if out["stopped"] or session.finish_requested:
        out["done"] = True
        return out
    # Sample mode: the pre-written seller "says" the next line when its time comes.
    while session.script and now - session.started >= session.script[0][0]:
        out["heard"].append(session.script.pop(0)[1])

    jpeg = vision.prepare_image(raw_image)
    # Coaching runs on a small copy so SAM answers faster. The shutter re-segments the full frame before anything is read.
    tick = vision.shrink(jpeg)
    if vision.brightness(tick) < TOO_DARK:  # nothing useful can come of this frame, so don't pay for it
        out["direction"] = session.line("dark")
        out["idle"] = True
        return out
    if session.spoken and not session.product and now - session.last_discover > DISCOVER_EVERY:
        discover(session, jpeg)

    pending = _claims_to_pursue(session, bool(out["heard"]), now)
    if not pending:
        return _between_claims(session, tick, out, now)

    # One thing at a time. A director who changes the subject mid-shot is hard to follow.
    claim = pending[0]
    concept = claim.sam_concept
    group = [c for c in pending if c.sam_concept == concept]
    if concept != session.target:
        session.target, session.target_since, session.steady, session.last_count = concept, now, 0, -1
        session.struggling_since, session.blurry_ticks = 0.0, 0
    instances = vision.parse_sam(api.sam_raw(concept, tick, session.trace))
    out["overlay"] = _overlay(tick, instances, concept, "open")

    if not instances:
        session.steady, session.misses = 0, session.misses + 1
        out["direction"] = _ask(session, claim)
        out["idle"] = session.misses > 4  # nothing to see: slow down and stop spending
        return out
    session.misses = 0

    if _coached_for_reading(session, group, tick, instances, out, now):
        return out

    # The count must hold between ticks, or the frame is still changing.
    session.steady = session.steady + 1 if len(instances) == session.last_count or session.last_count < 0 else 1
    session.last_count = len(instances)
    out["steady"] = session.steady

    if _waiting_to_look_again(session, group, instances, out, now):
        return out
    if session.steady < STEADY_TICKS:
        out["direction"] = session.line("hold")
        return out
    return _fire(session, concept, group, jpeg, out, now)
