"""
The deck: a validated collection of morph cards.

A *deck* is the backlog file the orchestrator submits -- a JSON list of morph
card dicts (see ``documentation/batch-orchestrator.md``, "The generation
cycle"). This module loads that file into :class:`~cards.schema.MorphCard`
objects and validates the deck *as a whole*: the checks here are about
relationships *between* cards, which a single card cannot verify on its own.

Deck-level rules enforced (Phase 0 of ``documentation/DEVELOPMENT_PLAN.md``):

* ``custom_id`` is unique across the deck (it names batch requests and output
  files, so a collision would silently overwrite results);
* every ``depends_on`` entry refers to a card actually present in the deck;
* the dependency graph is acyclic -- a cycle (self-dependency included) can
  never be split into generations, so it is rejected up front.

Ordering the acyclic deck into generations is the job of a later phase
(``cards/generations.py``); this module only guarantees such an ordering
exists. Pure and stdlib-only: no file writing, no network, no imports from
``processors/`` or ``flows/``.
"""

import json
from typing import Dict, List

from cards.schema import CardError, MorphCard


class DeckError(ValueError):
    """A deck is internally inconsistent (duplicate ids, bad or cyclic deps).

    Distinct from :class:`cards.schema.CardError`, which concerns a single
    malformed card. A deck of individually valid cards can still be an invalid
    deck.
    """


def load_deck(path: str) -> List[MorphCard]:
    """Load and validate a deck from a JSON file.

    The file must contain a JSON list of card dicts (nested or flat form, per
    :meth:`MorphCard.from_dict`). Raises :class:`DeckError` if the top-level
    JSON is not a list, :class:`CardError` if any card is malformed, and
    :class:`DeckError` for deck-level problems.
    """
    with open(path, "r", encoding="utf-8") as handle:
        try:
            raw = json.load(handle)
        except json.JSONDecodeError as exc:
            raise DeckError(f"{path}: not valid JSON: {exc}") from exc

    if not isinstance(raw, list):
        raise DeckError(
            f"{path}: deck must be a JSON list of cards, "
            f"got {type(raw).__name__}"
        )

    cards = [MorphCard.from_dict(item) for item in raw]
    validate_deck(cards)
    return cards


def validate_deck(cards: List[MorphCard]) -> None:
    """Validate a list of already-built cards as a deck.

    Exposed directly so callers holding in-memory cards (not loaded from disk)
    can run the same checks. Raises :class:`DeckError` on the first problem.
    """
    by_id = _index_by_id(cards)
    _check_dependencies_exist(cards, by_id)
    _check_acyclic(cards, by_id)


# -- internals ---------------------------------------------------------------


def _index_by_id(cards: List[MorphCard]) -> Dict[str, MorphCard]:
    """Map custom_id -> card, rejecting duplicates."""
    by_id: Dict[str, MorphCard] = {}
    for card in cards:
        if card.custom_id in by_id:
            raise DeckError(f"duplicate custom_id {card.custom_id!r} in deck")
        by_id[card.custom_id] = card
    return by_id


def _check_dependencies_exist(
    cards: List[MorphCard], by_id: Dict[str, MorphCard]
) -> None:
    """Every depends_on entry must name a card present in the deck."""
    for card in cards:
        for dep in card.depends_on:
            if dep not in by_id:
                raise DeckError(
                    f"card {card.custom_id!r} depends on unknown card {dep!r}"
                )


def _check_acyclic(cards: List[MorphCard], by_id: Dict[str, MorphCard]) -> None:
    """Reject dependency cycles (self-dependency included) via DFS.

    On the first cycle found, raises :class:`DeckError` listing the cycle
    members in dependency order (``a -> b -> c -> a``).
    """
    WHITE, GREY, BLACK = 0, 1, 2
    color: Dict[str, int] = {card.custom_id: WHITE for card in cards}

    def visit(cid: str, stack: List[str]) -> None:
        color[cid] = GREY
        stack.append(cid)
        for dep in by_id[cid].depends_on:
            if color[dep] == GREY:
                # Found a back-edge: the cycle is from dep's position in the
                # current stack up to here, closed back to dep.
                cycle = stack[stack.index(dep):] + [dep]
                raise DeckError(
                    "dependency cycle: " + " -> ".join(repr(c) for c in cycle)
                )
            if color[dep] == WHITE:
                visit(dep, stack)
        stack.pop()
        color[cid] = BLACK

    for card in cards:
        if color[card.custom_id] == WHITE:
            visit(card.custom_id, [])
