"""What the seller says and what the first look shows become claims to check."""
from __future__ import annotations

import hashlib
import json
import time

from . import meta_api as api
from . import rules, vision
from .perception import store
from .prompts import CLAIMS_PROMPT, DISCOVER_PROMPT, LISTEN_PROMPT
from .session import Claim, Session


def _claim_from(session: Session, c: dict, source: str = "said") -> Claim:
    check = c.get("check") if c.get("check") in ("count", "composition", "color", "text", "visual", "measure", "read") else "visual"
    session.file_seq += 1
    return Claim(id=f"c{session.file_seq}", text=str(c.get("text", "")), attribute=str(c.get("attribute", "")),
                 check=check, claimed=str(c.get("claimed", "")), sam_concept=str(c.get("sam_concept", "product")).strip().lower(),
                 shot_request=str(c.get("shot_request", "")),
                 evidence_name=str(c.get("evidence_name") or c.get("sam_concept") or "").strip().lower(), source=source)


def extract_claims(session: Session) -> None:
    data = api.spark_json(CLAIMS_PROMPT.format(listing=session.listing), purpose="split listing into claims",
                          trace=session.trace, effort="low")
    for c in data.get("claims", [])[:8]:
        session.claims.append(_claim_from(session, c))
    _set_product_from_claims(session)


def _set_product_from_claims(session: Session) -> None:
    if not session.product_concept:
        session.product_concept = next((c.sam_concept for c in session.claims if c.check in ("count", "color", "visual", "measure")), "")


def hear(session: Session, said: str) -> None:
    """One finished sentence from the seller: note what it claims, drop what it takes back."""
    said = said.strip()
    if not said:
        return
    session.hearing += 1
    session.last_heard = time.time()
    session.listing = f"{session.listing} {said}".strip()
    try:
        with session.hearing_lock:
            known = json.dumps([{"id": c.id, "attribute": c.attribute, "value": c.claimed or c.observed}
                                for c in session.claims if c.source == "said"], ensure_ascii=False)
            data = api.spark_json(LISTEN_PROMPT.format(said=said, known=known), purpose="listen to the seller",
                                  trace=session.trace, effort="low")
            for cid in data.get("retract", []) or []:
                for c in session.claims:
                    if c.id == cid and c.verdict in ("open", "not_visible"):
                        c.skipped, c.reason = True, "The seller took this back."
            for c in (data.get("claims") or [])[:5]:
                claim = _claim_from(session, c)
                # What the seller promises replaces what the agent was curious about or merely noticed.
                if claim.attribute != "feature":
                    session.claims = [o for o in session.claims if not (o.source == "seen" and o.attribute == claim.attribute)]
                for other in session.claims:
                    if other.source == "plan" and other.attribute == claim.attribute and other.verdict != rules.VERIFIED:
                        other.skipped = True
                session.claims.append(claim)
            if data.get("finished"):
                session.finish_requested = True
            _set_product_from_claims(session)
    finally:
        session.hearing -= 1
        session.last_heard = time.time()


def discover(session: Session, jpeg: bytes) -> None:
    """First look: what is this, what is obvious, and what would a buyer want to know that isn't yet readable?"""
    session.last_discover = time.time()
    jpeg = vision.prepare_image(jpeg)
    data = api.spark_json(DISCOVER_PROMPT, [jpeg], purpose="first look at the product", trace=session.trace, effort="low")
    product = str(data.get("product") or "").strip()
    if not product:
        return
    session.product = product
    session.discovered_at = time.time()
    if not session.product_concept:
        session.product_concept = str(data.get("sam_concept") or product).strip().lower()
    for fact in (data.get("seen") or [])[:3]:
        attribute, value = str(fact.get("attribute", "")), str(fact.get("value", ""))
        if value and not any(c.attribute == attribute for c in session.claims):
            c = _claim_from(session, {"text": value, "attribute": attribute, "check": fact.get("check", "visual"), "claimed": value,
                                      "sam_concept": session.product_concept}, "seen")
            c.verdict, c.observed, c.rule, c.reason = rules.VERIFIED, value, "model_judgement", "Seen by Muse Spark on first look."
            c.evidence, c.frame_sha256 = store(session, "first_look", jpeg), hashlib.sha256(jpeg).hexdigest()
            session.claims.append(c)
    for item in (data.get("plan") or [])[:2]:
        attribute = str(item.get("attribute", ""))
        if not any(c.attribute == attribute for c in session.claims):
            session.claims.append(_claim_from(session, {
                "text": attribute, "attribute": attribute, "check": "read", "claimed": "", "sam_concept": item.get("sam_concept"),
                "evidence_name": item.get("evidence_name"), "shot_request": item.get("shot_request")}, "plan"))
