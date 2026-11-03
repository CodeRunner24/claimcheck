import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claimcheck_core.rules import CONTRADICTED, NOT_VISIBLE, VERIFIED, decide, parse_composition, parse_count


def seen(observed, **kw):
    return {"visible": True, "observed": observed, **kw}


def test_composition_formats():
    assert parse_composition("%100 pamuk") == {"cotton": 100}
    assert parse_composition("60% COTTON 40% POLYESTER") == {"cotton": 60, "polyester": 40}
    assert parse_composition("Cotton 95%, Elastane 5%") == {"cotton": 95, "elastane": 5}
    assert parse_composition("%60 Pamuk %40 Polyester") == {"cotton": 60, "polyester": 40}


def test_composition_verdicts():
    assert decide("composition", "100% cotton", seen("60% cotton 40% polyester"))["verdict"] == CONTRADICTED
    assert decide("composition", "%100 pamuk", seen("100% COTTON"))["verdict"] == VERIFIED
    assert decide("composition", "100% cotton", seen("illegible"))["verdict"] == NOT_VISIBLE


def test_unseen_is_never_a_contradiction():
    assert decide("composition", "100% cotton", {"visible": False, "matches_claim": False})["verdict"] == NOT_VISIBLE


def test_count():
    assert parse_count("set of 2") == 2
    assert parse_count("iki adet") == 2
    assert decide("count", "2", seen(1), sam_count=1)["verdict"] == CONTRADICTED
    assert decide("count", "2", seen(2), sam_count=2)["verdict"] == VERIFIED
    assert decide("count", "2", seen(2), sam_count=1)["verdict"] == NOT_VISIBLE


def test_color_across_languages():
    assert decide("color", "siyah", seen("black"))["verdict"] == VERIFIED
    assert decide("color", "black", seen("navy blue"), None)["verdict"] in (CONTRADICTED,)


def test_text_turkish():
    assert decide("text", "Türkiye", seen("MADE IN TÜRKİYE"))["verdict"] == VERIFIED
    assert decide("text", "Turkey", seen("TÜRKİYE'DE ÜRETİLMİŞTİR"))["verdict"] == VERIFIED


def test_text_mismatch_needs_the_model_to_agree():
    assert decide("text", "Portugal", seen("Made in Bangladesh", matches_claim=False))["verdict"] == CONTRADICTED
    assert decide("text", "Portugal", seen("Made in Bangladesh"))["verdict"] == NOT_VISIBLE
    assert decide("text", "Turkey", seen("Türk malı", matches_claim=True))["verdict"] == VERIFIED
