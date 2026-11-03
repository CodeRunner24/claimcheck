"""Deterministic decisions. The models observe; this module rules.

Every function is pure, so a verdict can be replayed from its recorded observation.
"""
from __future__ import annotations

import re
import unicodedata

VERIFIED, CONTRADICTED, NOT_VISIBLE = "verified", "contradicted", "not_visible"

FIBERS = {
    "cotton": ["cotton", "pamuk", "coton", "baumwolle", "algodon"],
    "polyester": ["polyester", "poliester"],
    "elastane": ["elastane", "elastan", "spandex", "lycra"],
    "wool": ["wool", "yun", "laine"],
    "viscose": ["viscose", "viskon", "viskoz", "rayon"],
    "linen": ["linen", "keten", "lin"],
    "silk": ["silk", "ipek", "soie"],
    "nylon": ["nylon", "polyamide", "poliamid"],
    "acrylic": ["acrylic", "akrilik"],
    "leather": ["leather", "deri", "cuir"],
}
COLORS = {
    "black": ["black", "siyah"], "white": ["white", "beyaz"], "red": ["red", "kirmizi"],
    "blue": ["blue", "mavi"], "navy": ["navy", "lacivert"], "green": ["green", "yesil"],
    "yellow": ["yellow", "sari"], "grey": ["grey", "gray", "gri"], "brown": ["brown", "kahverengi"],
    "pink": ["pink", "pembe"], "orange": ["orange", "turuncu"], "purple": ["purple", "mor"],
    "beige": ["beige", "bej"],
}
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "ten": 10, "twelve": 12,
    "bir": 1, "iki": 2, "uc": 3, "dort": 4, "bes": 5, "alti": 6, "sekiz": 8, "on": 10,
}


COUNTRY_ALIASES = {"turkiye": "turkey", "almanya": "germany", "italya": "italy", "portekiz": "portugal",
                   "cin": "china", "hindistan": "india", "misir": "egypt", "banglades": "bangladesh"}


def _canonical_text(text) -> str:
    words = normalize(text).split()
    return " ".join(COUNTRY_ALIASES.get(w, w) for w in words)


def normalize(text: str) -> str:
    text = str(text).replace("ı", "i").replace("İ", "i").casefold()
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w%.°/ ]+", " ", text)).strip()


def parse_composition(text: str) -> dict[str, float]:
    """'60% cotton 40% polyester' or '%100 pamuk' -> {'cotton': 60, 'polyester': 40}."""
    n = normalize(text)
    alias = {a: fiber for fiber, names in FIBERS.items() for a in names}
    names = "|".join(sorted(alias, key=len, reverse=True))
    out: dict[str, float] = {}
    number = r"(?:%\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s*%)"
    first_fiber = re.search(rf"\b({names})\b", n)
    first_number = re.search(number, n)
    if first_fiber and first_number and first_fiber.start() < first_number.start():
        # Label style: "cotton 95% elastane 5%"
        for name, pre, post in re.findall(rf"\b({names})\s*{number}", n):
            out[alias[name]] = float((pre or post).replace(",", "."))
    else:
        # "60% cotton 40% polyester" or "%60 pamuk"
        for pre, post, name in re.findall(rf"{number}\s*({names})\b", n):
            out[alias[name]] = float((pre or post).replace(",", "."))
    if not out:  # "pure cotton" / "cotton" with no percentage is a 100% claim only if said so
        m = re.search(rf"\b(?:pure|saf|full)\s+({names})\b", n)
        if m:
            out[alias[m.group(1)]] = 100.0
    return out


def parse_count(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    n = normalize(value)
    m = re.search(r"\d+", n)
    if m:
        return int(m.group(0))
    for word in n.split():
        if word in NUMBER_WORDS:
            return NUMBER_WORDS[word]
    return None


def color_of(text: str) -> str | None:
    words = normalize(text).split()
    for color, names in COLORS.items():
        if any(name in words for name in names):
            return color
    return None


def parse_length_mm(text) -> float | None:
    """'32 cm', '320mm', '1.2 m', '12 inch' or 12" to millimetres."""
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*(mm|cm|metre|meter|m|inch|in|")(?![a-z])', str(text).casefold())
    if not m:
        return None
    value = float(m.group(1).replace(",", "."))
    return value * {"mm": 1, "cm": 10, "m": 1000, "metre": 1000, "meter": 1000, "inch": 25.4, "in": 25.4, '"': 25.4}[m.group(2)]


def decide_measure(claimed, measured_mm: float | None, ok_within: float = 0.08, wrong_beyond: float = 0.20) -> dict:
    """A photo ruler is rough, so there is a gap between 'matches' and 'contradicted' where it abstains."""
    want = parse_length_mm(claimed)
    if want is None:
        return _result(NOT_VISIBLE, "length_unparsed", f"Could not parse claimed length: {claimed}")
    if measured_mm is None:
        return _result(NOT_VISIBLE, "need_reference", "No usable reference in the frame.")
    off = abs(measured_mm - want) / want
    detail = f"Claimed {want / 10:g} cm, measured {measured_mm / 10:.1f} cm ({off:.0%} off)."
    if off <= ok_within:
        return _result(VERIFIED, "measure_within_tolerance", detail)
    if off >= wrong_beyond:
        return _result(CONTRADICTED, "measure_outside_tolerance", detail)
    return _result(NOT_VISIBLE, "measure_uncertain", detail)


def _fmt_comp(comp: dict[str, float]) -> str:
    return ", ".join(f"{v:g}% {k}" for k, v in comp.items())


def _result(verdict: str, rule: str, detail: str) -> dict:
    return {"verdict": verdict, "rule": rule, "detail": detail}


def decide(check: str, claimed, observation: dict, sam_count: int | None = None, first_look: bool = False) -> dict:
    """Turn one observation into a verdict.

    Unreadable or unseen evidence always yields not_visible, never a contradiction.
    Seeing fewer items than claimed on a first look only means the rest may be out of frame.
    """
    if not observation.get("visible"):
        return _result(NOT_VISIBLE, "evidence_not_visible", observation.get("reason") or "Evidence not visible.")
    observed = observation.get("observed")

    if check == "read":  # the seller promised nothing here: whatever is legibly printed simply becomes the fact
        if observed and str(observed).strip():
            return _result(VERIFIED, "read_from_evidence", f"Read directly from the evidence: {observed}.")
        return _result(NOT_VISIBLE, "nothing_to_read", observation.get("reason") or "Nothing legible.")

    if check == "count":
        want, spark = parse_count(claimed), parse_count(observed)
        if want is None:
            return _result(NOT_VISIBLE, "count_unparsed", f"Could not parse claimed count: {claimed}")
        if sam_count is not None and spark is not None and sam_count != spark:
            return _result(NOT_VISIBLE, "count_disagreement",
                           f"SAM found {sam_count}, Spark counted {spark}; separate the items.")
        got = sam_count if sam_count is not None else spark
        if got is None:
            return _result(NOT_VISIBLE, "count_unparsed", "No usable count.")
        if got < want and first_look:
            return _result(NOT_VISIBLE, "count_short", f"Only {got} of {want} in frame so far.")
        verdict = VERIFIED if got == want else CONTRADICTED
        return _result(verdict, "count_equal", f"Claimed {want}, counted {got}.")

    if check == "composition":
        want, got = parse_composition(str(claimed)), parse_composition(str(observed))
        if not got:
            return _result(NOT_VISIBLE, "composition_unreadable", f"No fibre percentages in: {observed!r}")
        if not want:
            named = [f for f, a in FIBERS.items() if any(x in normalize(claimed).split() for x in a)]
            if not named:
                return _result(NOT_VISIBLE, "composition_unparsed", f"Could not parse claim: {claimed!r}")
            if all(f in got for f in named):
                return _result(VERIFIED, "fibre_present", f"{', '.join(named)} on label: {_fmt_comp(got)}.")
            return _result(CONTRADICTED, "fibre_absent", f"{', '.join(named)} not on label: {_fmt_comp(got)}.")
        verdict = VERIFIED if want == got else CONTRADICTED
        return _result(verdict, "composition_equal", f"Claimed {_fmt_comp(want)}; label says {_fmt_comp(got)}.")

    if check == "color":
        want, got = color_of(str(claimed)), color_of(str(observed))
        if want and got:
            verdict = VERIFIED if want == got else CONTRADICTED
            return _result(verdict, "color_equal", f"Claimed {want}, observed {got}.")

    if check == "text":
        a, b = _canonical_text(claimed), _canonical_text(observed)
        if a and b and (a in b or b in a):
            return _result(VERIFIED, "text_match", f'Claimed "{a}"; read "{b}".')
        # No literal match may only mean another language or wording, so the rule does not
        # declare a contradiction alone: it falls through to the model's reading below.

    # Remaining visual facts rest on the model's judgement, and the record says so.
    match = observation.get("matches_claim")
    if match is True:
        return _result(VERIFIED, "model_judgement", observation.get("reason") or "Model judged a match.")
    if match is False:
        return _result(CONTRADICTED, "model_judgement", observation.get("reason") or "Model judged a mismatch.")
    return _result(NOT_VISIBLE, "model_unsure", observation.get("reason") or "Model was unsure.")
