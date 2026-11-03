"""The public face of the agent. The seller films and talks; the listing writes itself from what the camera backs up.

    listening.py    speech and first look -> claims
    perception.py   frame -> observation -> verdict (rules.py decides)
    director.py     the live loop that coaches the seller
    finishing.py    listing text, photo and proof reel
"""
from . import meta_api as api  # noqa: F401
from . import vision  # noqa: F401
from .director import live_step, next_direction  # noqa: F401
from .finishing import _leaks, analyze_take, correct_listing, finish, listing_segments, make_hero  # noqa: F401
from .listening import discover, extract_claims, hear  # noqa: F401
from .perception import _inspect_concept, analyze_shot  # noqa: F401
from .session import Claim, Session, pretty  # noqa: F401
