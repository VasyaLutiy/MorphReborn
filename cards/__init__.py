"""
Morph card format for the Morph 2.0 batch orchestrator.

The morph card is the unit of specification the orchestrator compiles and the
batch executor consumes; see ``documentation/batch-orchestrator.md`` for the
design and ``documentation/DEVELOPMENT_PLAN.md`` (Phase 0) for this deliverable.

This package is the format frozen in code, ahead of any consumer:

* :mod:`cards.schema` -- the :class:`MorphCard` dataclass and its validation.
* :mod:`cards.deck` -- loading and whole-deck validation.

Both are pure and stdlib-only (no I/O beyond reading a deck file, no network,
no imports from ``processors/`` or ``flows/``).
"""

from cards.deck import DeckError, load_deck, validate_deck
from cards.schema import CardError, MorphCard

__all__ = [
    "MorphCard",
    "CardError",
    "load_deck",
    "validate_deck",
    "DeckError",
]
