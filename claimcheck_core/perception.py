"""From a frame to a verdict: SAM finds the evidence, Spark reports what is visible, the rules decide.

Nothing here decides a fact by itself: rules.py does. This module only gathers observations and records outcomes.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

from . import meta_api as api
from . import rules, vision
from .prompts import OBSERVE_PROMPT
from .session import Claim, Session, pretty

MIN_READ_SIDE = 70     # px: below this, a label cannot be read reliably, so ask to move closer
MIN_SHARPNESS = 12.0   # Laplacian variance of the evidence region
MIN_HERO_SIDE = 220    # px: the product's short side in the source frame
REFERENCE_CONCEPT = "sheet of paper"


def store(session: Session, stem: str, jpeg: bytes) -> str:
    if session.out_dir is None:
        return vision.to_data_url(jpeg)
    session.out_dir.mkdir(parents=True, exist_ok=True)
    session.file_seq += 1
    name = f"{stem}_{session.file_seq}.jpg"
    (session.out_dir / name).write_bytes(jpeg)
    return name


def _measure(session: Session, claims: list[Claim], jpeg: bytes, instances: list[vision.Instance], results: dict) -> None:
    """Physical length from masks alone: no model opinion enters this verdict."""
    size = vision.image_size(jpeg)
    references = vision.parse_sam(api.sam_raw(REFERENCE_CONCEPT, jpeg, session.trace))
    for c in claims:
        if not instances or not references:
            results[c.id] = {**rules.decide_measure(c.claimed, None), "observed": ""}
            continue
        item = max(instances, key=lambda i: (i.box[2] - i.box[0]) * (i.box[3] - i.box[1]))
        reference = max(references, key=lambda i: (i.box[2] - i.box[0]) * (i.box[3] - i.box[1]))
        found = vision.measure(item, reference, size)
        if found["mm"] is None:
            results[c.id] = {"verdict": rules.NOT_VISIBLE, "rule": found["reason"], "observed": "",
                             "detail": f"Reference proportions {found.get('ratio', '?')} are not A4; the view is skewed."}
            continue
        decision = rules.decide_measure(c.claimed, found["mm"])
        results[c.id] = {**decision, "observed": f"{found['mm'] / 10:.1f} cm", "corners": found.get("corners"),
                         "reference": found.get("reference_corners")}


def _inspect_concept(session: Session, concept: str, claims: list[Claim], jpeg: bytes,
                     instances: list[vision.Instance] | None = None, first_look: bool | None = None):
    if instances is None:
        instances = vision.parse_sam(api.sam_raw(concept, jpeg, session.trace))
    results: dict[str, dict] = {}
    measured = [c for c in claims if c.check == "measure"]
    if measured:
        _measure(session, measured, jpeg, instances, results)
    claims = [c for c in claims if c.check != "measure"]
    if not instances:
        for c in claims:
            results[c.id] = {"verdict": rules.NOT_VISIBLE, "rule": "sam_no_match", "observed": "",
                             "detail": f'SAM found no "{concept}" in this shot.'}
        return concept, instances, results

    best = max(instances, key=lambda i: (i.box[2] - i.box[0]) * (i.box[3] - i.box[1]))
    to_ask = []
    for c in claims:
        if c.check in ("composition", "text", "read"):
            if best.short_side < MIN_READ_SIDE:
                results[c.id] = {"verdict": rules.NOT_VISIBLE, "rule": "too_small", "observed": "",
                                 "detail": f"The {concept} is too small to read. Move closer."}
                continue
            if vision.sharpness(jpeg, best.box) < MIN_SHARPNESS:
                results[c.id] = {"verdict": rules.NOT_VISIBLE, "rule": "too_blurry", "observed": "",
                                 "detail": f"The {concept} is blurry. Hold the camera steady."}
                continue
        to_ask.append(c)
    if to_ask:
        listing = json.dumps([{"id": c.id, "claim": c.text, "check": c.check, "claimed": c.claimed} for c in to_ask],
                             ensure_ascii=False)
        ordered = sorted(instances, key=lambda i: -(i.box[2] - i.box[0]) * (i.box[3] - i.box[1]))[:3]
        reading = any(c.check in ("composition", "text", "read") for c in to_ask)
        close_ups = [vision.rectify(jpeg, i) if reading else vision.crop(jpeg, i.box) for i in ordered]
        data = api.spark_json(OBSERVE_PROMPT.format(concept=concept, claims=listing),
                              [jpeg] + close_ups, purpose=f'read "{concept}"', trace=session.trace)
        observations = {o.get("id"): o for o in data.get("observations", [])}
        for c in to_ask:
            obs = observations.get(c.id, {"visible": False, "reason": "No observation returned."})
            look = c.attempts == 0 if first_look is None else first_look
            decision = rules.decide(c.check, c.claimed, obs, len(instances) if c.check == "count" else None, first_look=look)
            results[c.id] = {**decision, "observed": str(obs.get("observed") or ""), "next_shot": str(obs.get("next_shot") or "")}
    return concept, instances, results


def _apply(session: Session, concept: str, claims: list[Claim], instances: list[vision.Instance], results: dict,
           jpeg: bytes, at: float = -1.0) -> list[dict]:
    """Record verdicts on the claims. Returns the moments that just settled."""
    statuses = [results[c.id]["verdict"] for c in claims]
    status = rules.CONTRADICTED if rules.CONTRADICTED in statuses else (
        rules.NOT_VISIBLE if rules.NOT_VISIBLE in statuses else rules.VERIFIED)
    layers = [{"instance": i, "label": f"{concept} {i.confidence:.0%}", "status": status} for i in instances]
    evidence = store(session, "evidence", vision.annotate(jpeg, layers)) if layers else ""
    digest = hashlib.sha256(jpeg).hexdigest()
    best = max(instances, key=lambda i: (i.box[2] - i.box[0]) * (i.box[3] - i.box[1])) if instances else None
    flat = ""
    if best is not None and any(c.check in ("composition", "text", "read") for c in claims):
        flat = store(session, "flat", vision.rectify(jpeg, best))
    settled = []
    for c in claims:
        r = results[c.id]
        c.attempts += 1
        c.verdict, c.rule, c.reason = r["verdict"], r["rule"], r["detail"]
        c.observed, c.next_shot = r.get("observed", ""), r.get("next_shot", "")
        if evidence:
            c.evidence, c.frame_sha256 = evidence, digest
        if flat and c.check in ("composition", "text", "read"):
            c.flat = flat
        if best is not None and c.check == "color":
            c.swatch = vision.swatch(jpeg, best)
        if c.verdict in (rules.VERIFIED, rules.CONTRADICTED):
            c.at = at
            moment = {"t": max(0.0, at), "claim": c.id, "status": c.verdict, "source": c.source, "said": c.text,
                      "seen": pretty(c.observed),
                      "text": f"{c.text}  →  {pretty(c.observed)}" if c.verdict == rules.CONTRADICTED and c.observed else (pretty(c.observed) or c.text)}
            session.moments.append(moment)
            settled.append(moment)
    _consider_hero(session, jpeg, instances, concept)
    return settled


def _consider_hero(session: Session, jpeg: bytes, instances: list[vision.Instance], concept: str) -> None:
    """Keep the largest complete view of the product itself, for the listing photo."""
    if not instances or concept != session.product_concept:
        return
    w, h = vision.image_size(jpeg)
    whole = [i for i in instances if i.box[0] > 2 and i.box[1] > 2 and i.box[2] < w - 3 and i.box[3] < h - 3]
    # A cut-out needs the entire product and enough pixels to look like a photo, not a thumbnail.
    if whole and len(whole) == len(instances) and sum(i.short_side for i in whole) >= MIN_HERO_SIDE:
        area = sum((i.box[2] - i.box[0]) * (i.box[3] - i.box[1]) for i in whole)
        if session.hero_source is None or area > session.hero_source[2]:
            session.hero_source = (jpeg, whole, area)


def analyze_shot(session: Session, raw_image: bytes) -> dict:
    """Check every open claim against one still frame. Settled claims cost nothing further."""
    jpeg = vision.prepare_image(raw_image)
    session.shots += 1
    groups: dict[str, list[Claim]] = {}
    for c in session.open_claims:
        groups.setdefault(c.sam_concept, []).append(c)

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(lambda kv: _inspect_concept(session, kv[0], kv[1], jpeg), groups.items()))

    layers = []
    for concept, instances, results in outcomes:
        _apply(session, concept, groups[concept], instances, results, jpeg)
        statuses = [results[c.id]["verdict"] for c in groups[concept]]
        status = rules.CONTRADICTED if rules.CONTRADICTED in statuses else (
            rules.NOT_VISIBLE if rules.NOT_VISIBLE in statuses else rules.VERIFIED)
        layers += [{"instance": i, "label": f"{concept} {i.confidence:.0%}", "status": status} for i in instances]
    return {"frame": vision.to_data_url(vision.annotate(jpeg, layers)), "objects": len(layers)}
