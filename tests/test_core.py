"""Everything here runs offline, with no API key."""
import asyncio
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meta_sam_parser import SegmentationMask, decode_mask_to_raster
from meta_sam_parser._mask_codec import _encode_raster

from claimcheck_core import agent, asking, director, fastmask, rules, speech, take, vision
from claimcheck_core.vision import Instance


def jpeg(image: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


def rectangle(x1, y1, x2, y2) -> Instance:
    return Instance((x1, y1, x2, y2), 0.9, np.ones((y2 - y1, x2 - x1), dtype=bool))


def test_fast_decoder_matches_the_official_parser():
    rng = random.Random(3)
    for _ in range(12):
        h, w = rng.randint(5, 90), rng.randint(5, 90)
        yy, xx = np.mgrid[0:h, 0:w]
        blob = ((xx - rng.uniform(0, w)) ** 2 + (yy - rng.uniform(0, h)) ** 2) < rng.uniform(8, 1500)
        payload = _encode_raster(bytes(blob.astype(np.uint8).ravel().tolist()), w, h)
        official = np.frombuffer(decode_mask_to_raster(SegmentationMask("one_bit", payload, w, h)), dtype=np.uint8)
        assert np.array_equal(fastmask.decode(payload, h, w), official.reshape(h, w).astype(bool))


def test_hero_keeps_product_pixels_identical():
    source = np.random.default_rng(1).integers(0, 255, (400, 600, 3), dtype=np.uint8)
    backdrop = np.full((800, 800, 3), 200, np.uint8)
    image, info = vision.hero(jpeg(source), [rectangle(100, 80, 400, 300)], jpeg(backdrop))
    assert info["pixel_exact"] is True and info["product_pixels"] > 50000
    assert cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR) is not None


def test_a4_ruler_reads_a_known_scene_and_rejects_a_skewed_one():
    size = (1280, 720)
    scale = 2.0  # px per mm
    sheet = rectangle(100, 100, 100 + int(297 * scale), 100 + int(210 * scale))
    item = rectangle(760, 100, 760 + int(200 * scale), 300)
    assert abs(vision.measure(item, sheet, size)["mm"] - 200) < 3
    skewed = rectangle(100, 100, 700, 300)  # 3:1 cannot be A4 seen straight on
    found = vision.measure(item, skewed, size)
    assert found["mm"] is None and found["reason"] == "reference_skewed"


def test_measure_rule_abstains_in_the_gap():
    assert rules.parse_length_mm("38 cm long") == 380
    assert abs(rules.parse_length_mm(chr(49) + chr(50) + chr(34)) - 304.8) < 0.01
    assert rules.parse_length_mm("a nice long bag") is None
    assert rules.decide_measure("38 cm", 385)["verdict"] == rules.VERIFIED
    assert rules.decide_measure("38 cm", 300)["verdict"] == rules.CONTRADICTED
    assert rules.decide_measure("38 cm", 335)["verdict"] == rules.NOT_VISIBLE  # 12% off: too close to call
    assert rules.decide_measure("38 cm", None)["verdict"] == rules.NOT_VISIBLE


def test_most_at_once_ignores_flicker_and_new_ids():
    one, two = [rectangle(0, 0, 9, 9)], [rectangle(0, 0, 9, 9), rectangle(20, 0, 29, 9)]
    assert take.most_at_once({0: two, 1: one, 2: one, 3: one, 4: one})[0] == 1  # two for one frame is flicker
    assert take.most_at_once({i: two for i in range(6)}) == (2, 3)
    assert take.most_at_once({}) == (0, None)


def test_video_lines_parse_with_stable_ids():
    payload = _encode_raster(bytes([1] * 100), 10, 10)
    box = "<|box;x1=5;y1=5;x2=14;y2=14;w=64;h=64;c=0.9|><|mask;x=0;y=0;c=0.9;data=10,10," + payload + "|>"
    frames = vision.parse_video([f"<3f>7{box}", f"<4f>7{box},9{box}"])
    assert [i.id for i in frames[3]] == [7] and [i.id for i in frames[4]] == [7, 9]
    assert frames[3][0].mask.all()


def test_listing_segments_mark_up_the_quoted_claims():
    s = agent.Session("Set of 2 black shirts. 100% cotton.")
    s.claims = [agent.Claim("c1", "Set of 2", "", "count", "2", "shirt", "", verdict=rules.CONTRADICTED, observed="1"),
                agent.Claim("c2", "100% COTTON", "", "composition", "100% cotton", "care label", "")]
    parts = agent.listing_segments(s)
    assert "".join(p["text"] for p in parts) == s.listing
    assert parts[0] == {"text": "Set of 2", "verdict": "contradicted", "id": "c1", "fix": "1"}
    assert parts[2]["verdict"] == "open" and parts[2]["text"] == "100% cotton"



def test_live_step_coaches_without_calling_the_reader(monkeypatch):
    """Too small to read: the director says so and Spark is never paid for."""
    image = np.full((720, 1280, 3), 120, np.uint8)
    cv2.rectangle(image, (600, 340), (640, 370), (255, 255, 255), -1)
    s = agent.Session("100% cotton")
    s.claims = [agent.Claim("c1", "100% cotton", "material", "composition", "100% cotton", "care label", "Show the label.", "care label")]
    monkeypatch.setattr(agent.api, "sam_raw", lambda *a, **k: "ignored")
    monkeypatch.setattr(agent.vision, "parse_sam", lambda *a, **k: [rectangle(600, 340, 640, 370)])
    def no_spark(*a, **k):
        raise AssertionError("Spark must not be called for a frame that cannot be read")
    monkeypatch.setattr(agent.api, "spark_json", no_spark)
    step = agent.live_step(s, jpeg(image))
    assert step["fired"] is False and "Closer" in step["direction"]


def test_reading_fills_in_what_nobody_promised():
    assert rules.decide("read", "", {"visible": True, "observed": "M"})["verdict"] == rules.VERIFIED
    assert rules.decide("read", "", {"visible": True, "observed": " "})["verdict"] == rules.NOT_VISIBLE
    assert rules.decide("read", "", {"visible": False})["verdict"] == rules.NOT_VISIBLE


def test_said_claims_go_before_what_the_agent_merely_wants():
    s = agent.Session()
    s.claims = [agent.Claim("c1", "size", "size", "read", "", "care label", "", source="plan"),
                agent.Claim("c2", "Set of 2", "quantity", "count", "2", "shirt", "", source="said")]
    assert [c.id for c in s.open_claims] == ["c2", "c1"]


def test_hearing_adds_claims_and_takes_them_back(monkeypatch):
    s = agent.Session(spoken=True)
    replies = [
        {"language": "en", "claims": [{"text": "made in Portugal", "attribute": "origin", "check": "text", "claimed": "Portugal", "sam_concept": "care label"}]},
        {"language": "en", "retract": ["c1"], "claims": [], "finished": True},
    ]
    monkeypatch.setattr(agent.api, "spark_json", lambda *a, **k: replies.pop(0))
    agent.hear(s, "It's made in Portugal.")
    assert s.claims[0].check == "text" and s.open_claims
    agent.hear(s, "Actually I'm not sure about that. That's it.")
    assert s.claims[0].skipped and not s.open_claims and s.finish_requested
    assert "Portugal" in s.listing


def test_spoken_claim_replaces_what_the_camera_merely_noticed(monkeypatch):
    s = agent.Session(spoken=True)
    s.claims = [agent.Claim("c1", "black", "color", "color", "black", "shirt", "", source="seen", verdict=rules.VERIFIED, observed="black")]
    monkeypatch.setattr(agent.api, "spark_json", lambda *a, **k: {"claims": [
        {"text": "red", "attribute": "color", "check": "color", "claimed": "red", "sam_concept": "shirt"}]})
    agent.hear(s, "It's red.")
    assert [c.source for c in s.claims] == ["said"]


def test_claims_are_pursued_in_the_order_they_were_said():
    s = agent.Session("Set of 2 shirts, 100% cotton")
    s.claims = [agent.Claim("c1", "100% cotton", "material", "composition", "100% cotton", "label", ""),
                agent.Claim("c2", "Set of 2", "quantity", "count", "2", "shirt", "")]
    assert [c.id for c in s.open_claims] == ["c2", "c1"]


def test_shouting_labels_are_tidied_for_reading():
    assert agent.pretty("MADE IN BANGLADESH") == "Made in Bangladesh"
    assert agent.pretty("60% cotton / 40% polyester") == "60% cotton / 40% polyester"
    assert agent.pretty("") == ""


def test_damage_is_flagged_for_the_seller_not_decided(monkeypatch):
    image = jpeg(np.full((300, 300, 3), 120, np.uint8))
    s = agent.Session()
    s.hero_source = (image, [rectangle(50, 50, 250, 250)], 1)
    monkeypatch.setattr(agent.api, "spark_json", lambda *a, **k: {"issues": [{"what": "small hole on the sleeve", "sam_concept": "hole"}]})
    monkeypatch.setattr(agent.api, "sam_raw", lambda *a, **k: "")
    agent.notice_condition(s)
    assert s.claims[0].source == "noticed" and s.claims[0].rule == "model_judgement" and "could not locate" in s.claims[0].reason
    assert s.claims[0] not in s.open_claims


def test_a_session_stops_at_its_spending_cap():
    s = agent.Session(spoken=True)
    s.trace.append(agent.api.CallRecord("sam-3.1", "x", 1, 99.0))
    assert agent.live_step(s, jpeg(np.zeros((90, 160, 3), np.uint8)))["done"] is True


def test_only_the_moments_that_matter_are_kept():
    clip = take.Take(b"", [b""] * 600, (10, 10), 6)  # 100 seconds
    kept = take.moment_windows(clip, [{"t": 50.0}])
    assert kept[0] == int(48 * 6) and kept[-1] == int(53.5 * 6) and len(kept) < 40
    assert len(take.moment_windows(clip, [])) == 25 * 6


def test_a_filmstrip_cell_covers_its_share_of_the_take():
    assert asking._cell(0, 100) == 0
    assert asking._cell(99, 100) == asking.STRIP_CELLS - 1
    assert asking._cell(50, 100) == asking.STRIP_CELLS // 2


def test_an_answer_clip_stops_where_the_object_leaves_the_view():
    hits = {i: [rectangle(1, 1, 5, 5)] for i in range(30, 51)}
    assert asking._window(hits, 45, 200, 6) == list(range(36, 51))


def test_a_glimpse_still_gets_a_clip_long_enough_to_watch():
    assert asking._window({50: [rectangle(1, 1, 5, 5)]}, 50, 200, 6) == list(range(44, 57))


def test_saying_something_is_absent_is_not_a_finding():
    for short in ("No zipper visible", "not visible", "None", "Can't tell"):
        assert asking.ABSENCE.match(short)
    for short in ("Yes", "One chest pocket", "Short sleeves", "Notched collar"):
        assert not asking.ABSENCE.match(short)


def test_a_disproved_value_is_caught_in_generated_text():
    wrong = agent.Claim("c1", "100% cotton", "material", "composition", "100% cotton", "care label", "", verdict=rules.CONTRADICTED, observed="60% cotton")
    assert rules.leaked_values([wrong], "Yes, it is 100% cotton.") == ["100% cotton"]
    assert rules.leaked_values([wrong], "It is 60% cotton.") == []


def test_a_question_cant_be_asked_without_footage():
    card = agent.ask(agent.Session("", spoken=True), "Does it have a pocket?")
    assert card["tone"] == "nocamera"


def test_the_answer_clip_is_only_cropped_and_keeps_the_frame_size(tmp_path):
    frames = [jpeg(np.full((90, 160, 3), 40 + i, np.uint8)) for i in range(12)]
    clip = take.Take(b"", frames, (160, 90), 6)
    tracked = {i: [rectangle(60, 30, 100, 60)] for i in range(12)}
    out = tmp_path / "a.mp4"
    take.render_answer(clip, tracked, 6, "pocket", out)
    video = cv2.VideoCapture(str(out))
    assert (video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (160, 90)
    assert video.get(cv2.CAP_PROP_FRAME_COUNT) >= 12


def test_a_frame_blending_two_shots_is_not_chosen_for_the_photo():
    calm = jpeg(np.full((90, 160, 3), 60, np.uint8))
    other = jpeg(np.full((90, 160, 3), 220, np.uint8))
    clip = take.Take(b"", [calm, calm, calm, other, other, other], (160, 90), 6)
    big, small = [rectangle(10, 10, 150, 80)], [rectangle(40, 30, 100, 60)]
    assert take.steadiest_view({1: small, 2: big, 3: big}, clip)[0] == 1  # 2 and 3 sit on the cut, however large the object looks there


def test_a_black_shirt_filling_the_frame_is_not_a_dark_room():
    shirt = np.full((288, 512, 3), 18, np.uint8)
    shirt[:, :60] = 170  # some table at the edge
    room = np.random.default_rng(2).integers(0, 26, (288, 512, 3), dtype=np.uint8)
    assert vision.brightness(jpeg(shirt)) > director.TOO_DARK > vision.brightness(jpeg(room))


def test_listening_survives_a_dropped_stream(monkeypatch):
    calls = []

    async def flaky(*_):
        calls.append(1)
        if len(calls) < 3:
            raise speech.api.ApiError("Ingress audio slower than real-time")

    async def nothing(*_):
        return None

    monkeypatch.setattr(speech, "relay", flaky)
    asyncio.run(speech.keep_listening(nothing, nothing, nothing))
    assert len(calls) == 3


def test_listening_gives_up_when_the_stream_never_holds(monkeypatch):
    async def broken(*_):
        raise speech.api.ApiError("nope")

    async def nothing(*_):
        return None

    monkeypatch.setattr(speech, "relay", broken)
    with pytest.raises(speech.api.ApiError):
        asyncio.run(speech.keep_listening(nothing, nothing, nothing))


def _bright_frame() -> bytes:
    return jpeg(np.full((720, 1280, 3), 150, np.uint8))


def test_the_shutter_waits_for_a_steady_frame_and_then_the_rules_decide(monkeypatch):
    s = agent.Session("It's red.")
    s.claims = [agent.Claim("c1", "red", "color", "color", "red", "shirt", "Show the shirt.")]
    monkeypatch.setattr(agent.api, "sam_raw", lambda *a, **k: "ignored")
    monkeypatch.setattr(agent.vision, "parse_sam", lambda *a, **k: [rectangle(100, 100, 400, 400)])
    monkeypatch.setattr(agent.api, "spark_json", lambda *a, **k: {"observations": [{"id": "c1", "visible": True, "observed": "black"}]})
    first = agent.live_step(s, _bright_frame())
    assert first["fired"] is False and s.claims[0].verdict == "open"
    second = agent.live_step(s, _bright_frame())
    assert second["fired"] is True and second["settled"][0]["status"] == rules.CONTRADICTED
    assert s.claims[0].verdict == rules.CONTRADICTED and s.claims[0].rule == "color_equal"


def test_a_camera_that_never_gets_the_shot_right_doesnt_stick_the_session(monkeypatch):
    s = agent.Session("100% cotton")
    s.claims = [agent.Claim("c1", "100% cotton", "material", "composition", "100% cotton", "care label", "", "care label")]
    monkeypatch.setattr(agent.api, "sam_raw", lambda *a, **k: "ignored")
    monkeypatch.setattr(agent.vision, "parse_sam", lambda *a, **k: [rectangle(600, 340, 640, 370)])
    monkeypatch.setattr(agent.api, "spark_json", lambda *a, **k: {"observations": [{"id": "c1", "visible": False, "reason": "unreadable"}]})
    assert agent.live_step(s, _bright_frame())["direction"] and s.struggling_since
    s.struggling_since -= director.STRUGGLE_SECONDS + 1
    agent.live_step(s, _bright_frame())
    fired = agent.live_step(s, _bright_frame())
    assert fired["fired"] is True and s.claims[0].verdict == rules.NOT_VISIBLE


def test_what_the_agent_merely_wants_to_know_waits_for_the_seller_to_finish():
    s = agent.Session(spoken=True)
    s.claims = [agent.Claim("c1", "size", "size", "read", "", "care label", "", source="plan")]
    s.last_heard = 100.0
    assert director._claims_to_pursue(s, False, 100.0 + director.LET_SELLER_TALK - 1) == []
    assert director._claims_to_pursue(s, False, 100.0 + director.LET_SELLER_TALK + 1) == s.claims


def test_one_verdict_from_many():
    assert rules.overall([rules.VERIFIED, rules.CONTRADICTED, rules.NOT_VISIBLE]) == rules.CONTRADICTED
    assert rules.overall([rules.VERIFIED, rules.NOT_VISIBLE]) == rules.NOT_VISIBLE
    assert rules.overall([rules.VERIFIED, rules.VERIFIED]) == rules.VERIFIED


def _footage() -> take.Take:
    frames = [jpeg(np.full((90, 160, 3), 60 + i, np.uint8)) for i in range(12)]
    return take.Take(b"", frames, (160, 90), 6)


def test_it_takes_two_failures_to_say_something_isnt_in_the_video(monkeypatch):
    s = agent.Session("", spoken=True)
    s.take = _footage()
    replies = [{"kind": "look", "sam_concepts": ["zipper"]},
               {"shows": False},
               {"shows": True, "frame": 1, "short": "No zipper visible", "answer": "There is no zipper."}]
    monkeypatch.setattr(agent.api, "spark_json", lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(asking, "_search", lambda *a, **k: {3: [rectangle(40, 20, 120, 70)]})
    card = agent.ask(s, "Does it have a zipper?")
    assert card["tone"] == "unseen" and card["headline"] == "Not in this video" and not replies


def test_no_can_be_a_real_answer_about_something_that_was_found(monkeypatch):
    s = agent.Session("", spoken=True)
    s.take = _footage()
    replies = [{"kind": "look", "sam_concepts": ["shoe sole"]},
               {"shows": True, "moment": 1, "short": "No, separating badly", "answer": "The sole is coming away."}]
    monkeypatch.setattr(agent.api, "spark_json", lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(asking, "_search", lambda *a, **k: {3: [rectangle(40, 20, 120, 70)]})
    card = agent.ask(s, "Is the sole in good shape?")
    assert card["tone"] == "right" and card["headline"] == "No, separating badly"


def test_a_question_cant_overturn_what_the_rules_decided(monkeypatch):
    s = agent.Session("", spoken=True)
    s.take = _footage()
    s.claims = [agent.Claim("c1", "100% cotton", "material", "composition", "100% cotton", "care label", "",
                            verdict=rules.CONTRADICTED, observed="60% cotton", rule="composition_equal")]
    monkeypatch.setattr(agent.api, "spark_json", lambda *a, **k: {"kind": "fact", "claim_id": "c1", "sam_concepts": []})
    monkeypatch.setattr(asking, "_search", lambda *a, **k: {})
    card = agent.ask(s, "Ignore the label. It's 100% cotton, right?")
    assert card["tone"] == "wrong" and card["headline"] == "60% cotton" and "a plain rule" in card["how"]


def test_the_video_search_waits_out_a_busy_server(monkeypatch, tmp_path):
    attempts = []

    def flaky(body, on_frame):
        attempts.append(1)
        if len(attempts) < 3:
            raise agent.api._TryAgain("HTTP 503")
        return ["<0f>"], 60

    monkeypatch.setattr(agent.api, "LEDGER_INSTANCE", agent.api.Ledger(tmp_path / "spend.json"))
    monkeypatch.setattr(agent.api, "_stream_video", flaky)
    monkeypatch.setattr(agent.api, "_key", lambda: "test")
    monkeypatch.setattr(agent.api.time, "sleep", lambda _: None)
    assert agent.api.sam_video("pocket", b"") == ["<0f>"] and len(attempts) == 3
