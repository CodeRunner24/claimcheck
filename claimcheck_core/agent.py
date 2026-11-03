"""The public face of the agent. The seller films and talks; the listing writes itself from what the camera backs up.

    listening.py    speech and first look -> claims
    perception.py   frame -> observation -> verdict (rules.py decides)
    director.py     the live loop that coaches the seller
    finishing.py    listing text, photo and proof reel
    asking.py       a buyer's question -> a search of the footage -> the moment that answers it
"""
from . import meta_api as api
from . import vision
from .asking import STRIP_CELLS, ask
from .director import live_step, next_direction
from .finishing import analyze_take, finish, listing_segments, notice_condition
from .listening import discover, extract_claims, hear
from .perception import analyze_shot, inspect_concept
from .session import Claim, Session, pretty

__all__ = ["api", "vision", "STRIP_CELLS", "ask", "live_step", "next_direction", "analyze_take", "finish", "listing_segments",
           "notice_condition", "discover", "extract_claims", "hear", "analyze_shot", "inspect_concept", "Claim", "Session", "pretty"]
