"""After the last verdict: the recorded take, the listing text, the honest photo and the proof reel."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from . import meta_api as api
from . import rules, take as takes, vision
from .asking import filmstrip, suggest_questions
from .listening import discover, hear
from .perception import consider_hero, inspect_concept, record_verdicts, store
from .prompts import BACKDROP_PROMPTS, CONDITION_PROMPT, CORRECT_PROMPT, WRITE_PROMPT
from .session import Claim, Session, pretty

WHOLE_CLIP_SECONDS = 5.0
HERO_SEARCH_FRAMES = 80


def analyze_take(session: Session, raw_video: bytes) -> dict:
    """The phone path: one recorded clip. What the seller says in it is heard, and the best frames get read."""
    take = takes.load(raw_video)
    session.progress["take"] = take
    if session.spoken:
        wav = takes.audio_wav(raw_video)
        if wav:
            said = api.transcribe(wav, session.trace)
            if said:
                hear(session, said)
        discover(session, take.frames[len(take.frames) // 2])
    groups: dict[str, list[Claim]] = {}
    for c in session.open_claims:
        groups.setdefault(c.sam_concept, []).append(c)
    tracks = _track_all(session, take, list(groups))
    second_look = session.shots > 0
    session.shots += 1
    for concept, group in groups.items():
        frames = tracks.get(concept, {})
        counted = [c for c in group if not c.reads_print]
        read = [c for c in group if c.reads_print]
        if counted:
            n, at = takes.most_at_once(frames)
            index = at if at is not None else (min(frames) if frames else None)
            if index is None or index >= len(take.frames):
                record_verdicts(session, concept, counted, [], _not_found(concept, counted), take.frames[0])
            else:
                # A whole clip is many looks: if the rest never came into view in it, that is not a first impression.
                whole_clip = len(take.frames) / take.fps >= WHOLE_CLIP_SECONDS
                _, inst, results = inspect_concept(session, concept, counted, take.frames[index], frames.get(index, []),
                                                    first_look=not (second_look or whole_clip))
                record_verdicts(session, concept, counted, inst, results, take.frames[index], index / take.fps)
        if read:
            found = takes.best_read_frame(frames, take)
            if found is None:
                record_verdicts(session, concept, read, [], _not_found(concept, read), take.frames[0])
            else:
                index, _ = found
                _, inst, results = inspect_concept(session, concept, read, take.frames[index], frames.get(index, []),
                                                    first_look=not second_look)
                record_verdicts(session, concept, read, inst, results, take.frames[index], index / take.fps)
    for c in session.claims:  # no live coaching on this path, so the agent's own wishes don't hold anything up
        if c.source == "plan" and c.verdict != rules.VERIFIED:
            c.skipped = True
    return {"tracks": tracks, "take": take}


def _not_found(concept: str, claims: list[Claim]) -> dict[str, dict]:
    return {c.id: {"verdict": rules.NOT_VISIBLE, "rule": "sam_no_match", "observed": "",
                   "detail": f'SAM did not find "{concept}" in the take.'} for c in claims}


def _track_all(session: Session, take: takes.Take, concepts: list[str]) -> dict[str, dict]:
    """SAM 3.1 video on every concept, in parallel, with frame-level progress for the UI."""
    total = len(take.frames)

    def run(concept: str):
        def tick(index: int):
            session.progress[f"track:{concept}"] = min(1.0, (index + 1) / total)
        return concept, vision.parse_video(api.sam_video(concept, take.mp4, session.trace, tick))

    if not concepts:
        return {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        return dict(pool.map(run, concepts))


def correct_listing(session: Session) -> str:
    """Typed listing: fix only what was disproved. Spoken: write it from what was established."""
    wrong = [c for c in session.claims if c.verdict == rules.CONTRADICTED and c.human != "disagree"]
    if session.spoken:
        facts = [{"attribute": c.attribute, "value": pretty(c.observed) or c.claimed} for c in session.claims
                 if c.verdict == rules.VERIFIED and c.human != "disagree"]
        facts += [{"attribute": c.attribute, "value": pretty(c.observed)} for c in wrong if c.observed]
        facts = [f for f in facts if f["attribute"] != "condition"]
        unproven = [c.text for c in session.claims if c.source == "said" and c.verdict in ("open", "not_visible")]
        noticed = [c.observed for c in session.claims if c.source == "noticed" and c.human != "disagree"]
        prompt = WRITE_PROMPT.format(said=session.listing, facts=json.dumps(facts, ensure_ascii=False),
                                     wrong=json.dumps([c.text for c in wrong], ensure_ascii=False),
                                     unproven=json.dumps(unproven, ensure_ascii=False),
                                     noticed=json.dumps(noticed, ensure_ascii=False))
    elif not wrong:
        session.corrected = session.listing
        return session.corrected
    else:
        results = json.dumps([{"claim": c.text, "verdict": c.verdict, "observed": c.observed} for c in wrong], ensure_ascii=False)
        prompt = CORRECT_PROMPT.format(listing=session.listing, results=results)
    data = api.spark_json(prompt, purpose="write the listing", trace=session.trace, effort="low")
    text = str(data.get("listing", session.listing))
    repeated = rules.leaked_values(session.claims, text)
    if repeated:
        data = api.spark_json(prompt + f"\n\nYour last draft wrongly repeated: {repeated}. Remove those.",
                              purpose="rewrite the listing", trace=session.trace, effort="low")
        text = str(data.get("listing", text))
    leaked = rules.leaked_values(session.claims, text)
    session.leak_warning = f"Check this before publishing: the draft still mentions {', '.join(leaked)}." if leaked else ""
    session.title, session.corrected = str(data.get("title", "")), text
    return session.corrected


BACKDROPS: dict[str, bytes] = {}


def make_hero(session: Session) -> None:
    """A clean listing photo in which only the surroundings are generated. SAM cuts, Python composites and checks."""
    if session.hero_source is None:
        return
    jpeg, instances, _ = session.hero_source
    color = vision.swatch(jpeg, vision.largest(instances))
    dark_product = bool(color) and (int(color[1:3], 16) + int(color[3:5], 16) + int(color[5:7], 16)) / 3 < 110
    tone = "light" if dark_product else "dark"
    if tone not in BACKDROPS:
        BACKDROPS[tone] = api.generate_image(BACKDROP_PROMPTS[tone], session.trace, purpose="generate backdrop only")
    image, info = vision.hero(jpeg, instances, BACKDROPS[tone])
    session.hero = store(session, "hero", image)
    session.hero_info = info


def notice_condition(session: Session) -> None:
    """Honesty runs both ways: flag damage the seller didn't mention, located with SAM, for them to confirm or dismiss."""
    if session.hero_source is not None:
        jpeg, instances, _ = session.hero_source
        best = vision.largest(instances)
        images = [jpeg, vision.crop(jpeg, best.box)]
    elif session.first_look:
        jpeg, images = session.first_look, [session.first_look]
    else:
        return
    data = api.spark_json(CONDITION_PROMPT, images, purpose="look for damage", trace=session.trace, effort="low")
    for issue in (data.get("issues") or [])[:3]:
        what = str(issue.get("what", "")).strip()
        if not what:
            continue
        concept = str(issue.get("sam_concept") or "damage").strip().lower()
        found = vision.parse_sam(api.sam_raw(concept, jpeg, session.trace))
        layers = [{"instance": i, "label": concept, "status": rules.CONTRADICTED} for i in found]
        session.file_seq += 1
        claim = Claim(id=f"c{session.file_seq}", text=what, attribute="condition", check="visual", claimed="", sam_concept=concept,
                      shot_request="", source="noticed", verdict=rules.VERIFIED, observed=what, rule="model_judgement",
                      reason="Noticed by Muse Spark and located with SAM 3.1. You decide whether it's right." if found
                      else "Noticed by Muse Spark. SAM 3.1 could not locate it, so check this yourself.")
        claim.evidence = store(session, "condition", vision.annotate(jpeg, layers))
        session.claims.append(claim)


def finish(session: Session, raw_video: bytes | None) -> None:
    """Everything after the last verdict: the listing, the honest photo, the tracked proof reel."""
    session.progress.update({"listing": 0.0, "hero": 0.0, "reel": 0.0})

    def reel():
        take = session.progress.get("take")
        if raw_video and take is None:
            take = takes.load(raw_video)
        if take is not None:
            session.take = take  # kept so buyers can question the footage later, even if the reel fails
            filmstrip(session)
            _cut_reel(session, take)
        session.progress["reel"] = 1.0

    def draft():
        correct_listing(session)
        session.progress["listing"] = 1.0

    def photo():
        make_hero(session)
        session.progress["hero"] = 1.0

    def run(*steps):
        """Steps in one batch run together. One failed part must not cost the seller the others."""
        with ThreadPoolExecutor(max_workers=len(steps)) as pool:
            return [str(f.exception())[:200] for f in [pool.submit(step) for step in steps] if f.exception()]

    # The draft has to know about any damage, and the photo wants the reel's best view of the product.
    errors = run(reel, lambda: notice_condition(session)) + run(draft, photo, lambda: suggest_questions(session))
    session.progress.update({"listing": 1.0, "hero": 1.0, "reel": 1.0})
    if errors:
        session.progress["errors"] = errors
    session.progress.pop("take", None)
    session.filming_cost = session.cost()


def _cross_check_counts(session: Session, take: takes.Take) -> None:
    """A person may hold a whole set in view only once, so a disproved count is checked against the entire take."""
    disputed = [c for c in session.claims if c.check == "count" and c.verdict == rules.CONTRADICTED]
    if not disputed:
        return
    sparse = takes.subset(take, list(range(0, len(take.frames), 2)), max(1, take.fps // 2))
    tracks = _track_all(session, sparse, list(dict.fromkeys(c.sam_concept for c in disputed)))
    for c in disputed:
        most, _ = takes.most_at_once(tracks.get(c.sam_concept, {}), hold=3)
        session.video_counts[c.sam_concept] = most
        want = rules.parse_count(c.claimed)
        if want is not None and most >= want:
            c.reason += f" Note: the full take shows {most} together at some point; review this."


def _cut_reel(session: Session, take: takes.Take) -> None:
    """Track evidence only around the verdict moments, so finishing takes seconds, not a minute."""
    indices = takes.moment_windows(take, session.moments)
    clip = takes.subset(take, indices)
    position = {index: n for n, index in enumerate(indices)}
    moments = []
    for m in session.moments:
        nearest = min(indices, key=lambda i: abs(i / take.fps - m["t"]))
        moments.append({**m, "t": position[nearest] / take.fps})
    concepts = list(dict.fromkeys(c.sam_concept for c in session.claims
                                  if c.source != "seen" and not (c.source == "plan" and c.verdict != rules.VERIFIED)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        tracked = pool.submit(_track_all, session, clip, concepts)
        checked = pool.submit(_cross_check_counts, session, take)
        tracks = tracked.result()
        checked.result()
    layers = []
    for concept in concepts:
        status = rules.overall([c.verdict for c in session.claims if c.sam_concept == concept])
        name = next(c.evidence_name or concept for c in session.claims if c.sam_concept == concept)
        layers.append({"frames": tracks.get(concept, {}), "status": status, "label": name})
    product = session.product_concept
    if product and product in tracks:  # the take may hold a better complete view of the product than any fired frame
        view = takes.steadiest_view(tracks[product], clip)
        if view:
            consider_hero(session, clip.frames[view[0]], view[1], product)
    if product and session.hero_source is None:  # the verdict moments never showed all of it: look through the rest of the take
        step = max(1, len(take.frames) // HERO_SEARCH_FRAMES)
        wide = takes.subset(take, list(range(0, len(take.frames), step)), max(1, take.fps // step))
        view = takes.steadiest_view(_track_all(session, wide, [product])[product], wide)
        if view:
            consider_hero(session, wide.frames[view[0]], view[1], product)
    if session.out_dir is None:
        return
    session.out_dir.mkdir(parents=True, exist_ok=True)
    session.file_seq += 1
    name = f"reel_{session.file_seq}.mp4"
    total = max(1, len(clip.frames))

    def tick(i: int):
        session.progress["burn"] = (i + 1) / total

    takes.render_reel(clip, layers, moments, session.out_dir / name, tick)
    session.reel = name


def listing_segments(session: Session) -> list[dict[str, str]]:
    """The listing cut into plain text and the claims quoted from it, so verdicts can mark up the text itself."""
    text = session.listing
    spans = []
    lowered = text.casefold()
    for c in session.claims:
        if c.source != "said":
            continue
        needle = c.text.strip().casefold()
        start = lowered.find(needle) if needle else -1
        if start >= 0 and len(lowered) == len(text):
            spans.append((start, start + len(needle), c))
    spans.sort(key=lambda s: s[0])
    out, position = [], 0
    for start, end, c in spans:
        if start < position:
            continue
        if start > position:
            out.append({"text": text[position:start], "verdict": "plain", "fix": "", "id": ""})
        out.append({"text": text[start:end], "verdict": c.verdict, "id": c.id,
                    "fix": pretty(c.observed) if c.verdict == rules.CONTRADICTED else ""})
        position = end
    if position < len(text):
        out.append({"text": text[position:], "verdict": "plain", "fix": "", "id": ""})
    return out
