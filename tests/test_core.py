"""Everything here runs offline, with no API key."""
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meta_sam_parser import SegmentationMask, decode_mask_to_raster
from meta_sam_parser._mask_codec import _encode_raster

from claimcheck_core import agent, fastmask, rules, take, vision
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
    image = np.zeros((720, 1280, 3), np.uint8)
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


def test_disproved_values_are_caught_in_the_draft():
    s = agent.Session()
    s.claims = [agent.Claim("c1", "100% cotton", "material", "composition", "100% cotton", "care label", "",
                            verdict=rules.CONTRADICTED, observed="60% cotton 40% polyester")]
    assert agent._leaks(s, "A shirt in 100% Cotton, comfy") == ["100% cotton"]
    assert agent._leaks(s, "A shirt in 60% cotton and 40% polyester") == []


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
