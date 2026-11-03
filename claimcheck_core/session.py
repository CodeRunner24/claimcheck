"""The state of one listing being checked: its claims, and what the agent knows so far."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from . import meta_api as api
from .prompts import LINES


@dataclass
class Claim:
    id: str
    text: str
    attribute: str
    check: str
    claimed: str
    sam_concept: str
    shot_request: str
    evidence_name: str = ""
    source: str = "said"           # said (spoken or typed) | seen (camera noticed) | plan (agent wants to know)
    verdict: str = "open"          # open | verified | contradicted | not_visible
    observed: str = ""
    reason: str = ""
    rule: str = ""
    evidence: str = ""             # annotated frame
    flat: str = ""                 # evidence straightened with its mask
    frame_sha256: str = ""         # of the exact frame the verdict rests on
    swatch: str = ""
    at: float = -1.0               # seconds into the take
    attempts: int = 0
    next_shot: str = ""
    retry_at: float = 0.0
    skipped: bool = False
    human: str = ""                # "" | agree | disagree


@dataclass
class Session:
    listing: str = ""              # what the seller typed, or everything they have said so far
    spoken: bool = False           # the listing comes from the seller's voice as they film
    claims: list[Claim] = field(default_factory=list)
    trace: list = field(default_factory=api.Trace)
    shots: int = 0
    title: str = ""
    corrected: str = ""
    leak_warning: str = ""
    out_dir: Path | None = None    # evidence files go here; without it they stay as data URLs
    # What the agent knows about the thing in front of it
    product: str = ""
    product_concept: str = ""
    discovered_at: float = 0.0
    last_discover: float = 0.0
    last_heard: float = 0.0
    finish_requested: bool = False
    hearing: int = 0               # extractions in flight
    hearing_lock: threading.Lock = field(default_factory=threading.Lock)  # one sentence at a time, in order
    script: list = field(default_factory=list)  # sample mode: (seconds, sentence) the sample seller says
    # Live loop
    ticks: int = 0
    steady: int = 0
    target: str = ""
    target_since: float = 0.0
    last_count: int = -1
    misses: int = 0
    started: float = 0.0
    moments: list[dict] = field(default_factory=list)
    shutter: dict | None = None    # the frame being read right now, shown frozen
    # Finishing
    hero_source: tuple | None = None   # (jpeg, instances, area)
    hero: str = ""
    hero_info: dict = field(default_factory=dict)
    reel: str = ""
    video_counts: dict = field(default_factory=dict)
    progress: dict = field(default_factory=dict)
    file_seq: int = 0

    @property
    def open_claims(self) -> list[Claim]:
        """What the seller said comes first; what the agent would merely like to know waits its turn."""
        waiting = [c for c in self.claims if c.verdict in ("open", "not_visible") and not c.skipped]
        said = sorted((c for c in waiting if c.source == "said"), key=self._position)
        return said + [c for c in waiting if c.source != "said"]

    def _position(self, claim: Claim) -> int:
        found = self.listing.casefold().find(claim.text.strip().casefold())
        return found if found >= 0 else len(self.listing)

    def cost(self) -> float:
        return sum(t.usd for t in self.trace)

    def line(self, key: str, **kw) -> str:
        return LINES[key].format(**kw)


def pretty(value: str) -> str:
    """Labels shout. 'MADE IN BANGLADESH' reads better as 'Made in Bangladesh'."""
    value = (value or "").strip()
    if not value or value != value.upper() or not any(ch.isalpha() for ch in value):
        return value
    small = {"in", "of", "and", "the", "made"}
    words = [w.lower() if w.lower() in small else w.capitalize() for w in value.split()]
    words[0] = words[0].capitalize()
    return " ".join(words)
