"""
File-ownership hazards: what two CONCURRENT cards do to one file.

The invariant the whole batch model rests on: a batch request cannot see another
request's output (``documentation/batch-orchestrator.md``, "The generation
cycle"), so within one generation no card may write a file another card writes,
and no card may read a file another card writes. Until this module existed that
invariant was carried by prose -- a rule in the ``/card`` decomposition prompt,
a line in the README -- and by the orchestrator's memory. Nothing checked it.

WHY that was the deepest defect in the machine, not a planning inconvenience.
The dependency graph is layered by ``depends_on`` (:func:`cards.generations.
split_into_generations`), a CARD-level field written by hand, while the real
dependency is FILE-level and is already fully declared in every card:
``targets`` is its write set, ``context_slice`` its read set. The machine held
both and routed execution by neither. Two cards of one generation writing one
file therefore produced a silent lost update -- the second write simply replaced
the first, both cards reported ``written``, both got a provenance commit, and
the run report was green. Silent loss with a green report is exactly the failure
class this architecture exists to abolish, so the check belongs in the machine.

What this module computes, from the cards alone (no disk, no network):

* :data:`KIND_WRITE_WRITE` -- two cards of one generation write the same file.
  An ERROR: there is no correct automatic answer to "which one wins", and
  serializing them silently is not one either -- the later card would rewrite
  the file from a fresh read and could still drop the earlier card's change
  without anyone being told. The deck is wrong and its author must split it.
* :data:`KIND_READ_WRITE` -- a card of one generation names in its
  ``context_slice`` a file another card of that generation writes. An ERROR,
  but a REPAIRABLE one: the fix is unambiguous (the reader must run after the
  writer), so :func:`repair_deck` adds that edge and the two cards serialize
  into consecutive generations.
* :data:`KIND_IMPLICIT_READ` -- the same thing reached through an EMPTY
  ``context_slice``, which means "the whole project" (``cards/compiler.py``)
  and therefore "every sibling's target too". A WARNING, not an error, for one
  reason: erroring would condemn the documented default, and repairing it
  automatically would make such a card depend on every other card in its
  generation -- collapsing the deck into a chain and turning one batch into N,
  at an hour of provider queue each (``Head_Pains.md`` 3.3). The operator is
  told exactly which files it reads stale and chooses.
* :data:`KIND_UNORDERED_READ` -- a card reads a file another card writes, they
  are in DIFFERENT generations, and no dependency path connects them. A
  WARNING: the batch loop compiles generation N+1 only after N's morphs are on
  disk, so the CONTENT the reader sees is fresh and correct. What is missing is
  the failure cascade -- if the writer fails, the reader is not skipped and
  runs against a file that was never written.

Deliberately NOT wired into :func:`cards.deck.validate_deck`. That function is
what LOADS a deck, and a deck that cannot be loaded cannot be inspected,
repaired or cleared -- a hazardous ``.morph/deck.json`` would brick the very CLI
that owns it. Enforcement sits where money is spent instead: the run preflight
(:func:`cards.store.preflight_deck`, called before a branch is opened or a batch
is submitted) refuses an errored deck and repairs a repairable one, while
``/card`` and ``/deck check`` report hazards without ever refusing a card --
there is no command to EDIT a card yet (``Head_Pains.md`` 2.4), so refusing an
add would leave an operator unable to fix the deck from inside the CLI.

Pure and stdlib-only: no I/O, no imports from ``flows`` or ``processors``.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from cards.deck import DeckError, validate_deck
from cards.generations import split_into_generations
from cards.schema import MorphCard


SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

KIND_WRITE_WRITE = "write-write"
KIND_READ_WRITE = "read-write"
KIND_IMPLICIT_READ = "implicit-read"
KIND_UNORDERED_READ = "unordered-read"


class HazardError(DeckError):
    """A deck violates file ownership and must not be submitted as it stands.

    A subclass of :class:`cards.deck.DeckError` on purpose: every caller that
    already reports "this deck is internally inconsistent" (the ``/card`` paths,
    the decomposition handler, the ``/submit`` guard) reports this too, with its
    own message, without a line of new error handling.

    Carries the offending :class:`Hazard` list so a caller can render more than
    the message -- ``str(error)`` is already the full report.
    """

    def __init__(self, message: str, hazards: Optional[List["Hazard"]] = None):
        super().__init__(message)
        self.hazards = list(hazards or [])


@dataclass(frozen=True)
class Hazard:
    """One contested file between two cards.

    ``left`` and ``right`` are custom_ids in deck order. For the read/write
    kinds ``right`` is always the READER and ``left`` the WRITER, which is what
    makes :func:`repair_deck`'s edge direction (reader depends on writer)
    readable at the call site. ``paths`` are the contested files as the cards
    spell them -- normalised only for comparison, never for display, so an
    operator sees the path they typed. ``generation`` is the computed generation
    the pair shares, or ``None`` for :data:`KIND_UNORDERED_READ`, whose whole
    point is that the two are in different ones.
    """

    kind: str
    severity: str
    left: str
    right: str
    paths: Tuple[str, ...]
    generation: Optional[int] = None

    def message(self) -> str:
        """One line naming the pair, the files and what to do about it."""
        files = ", ".join(self.paths)
        where = (f"generation {self.generation}" if self.generation is not None
                 else "different generations")
        if self.kind == KIND_WRITE_WRITE:
            return (
                f"{self.left!r} and {self.right!r} both write {files} in "
                f"{where}: batch requests cannot see each other, so whichever "
                f"is accepted second silently replaces the first. Split them "
                f"into one card (use 'targets' for a set written atomically) "
                f"or order them with depends_on")
        if self.kind == KIND_READ_WRITE:
            return (
                f"{self.right!r} reads {files}, which {self.left!r} rewrites in "
                f"the same {where}: the reader is compiled from the file as it "
                f"was BEFORE the batch. Add depends_on {self.left!r} to "
                f"{self.right!r} (the run preflight does this for you)")
        if self.kind == KIND_IMPLICIT_READ:
            return (
                f"{self.right!r} has an empty context_slice (= the whole "
                f"project), so it reads {files} -- which {self.left!r} rewrites "
                f"in the same {where} -- as it was BEFORE the batch. Name the "
                f"slice explicitly, or add depends_on {self.left!r}")
        return (
            f"{self.right!r} reads {files}, which {self.left!r} writes, and "
            f"nothing connects them: the content is fresh (the reader's "
            f"generation compiles later), but if {self.left!r} fails, "
            f"{self.right!r} is NOT skipped and runs against a file that was "
            f"never written. Add depends_on {self.left!r} to {self.right!r}")

    def __str__(self) -> str:
        return f"[{self.severity}] {self.message()}"


# -- detection ---------------------------------------------------------------


def _key(path: str) -> str:
    """The comparison identity of one declared path.

    ``a.py`` and ``./a.py`` are one file; the cards may spell them either way,
    and a missed match here is a missed hazard. Normalised for comparison only
    -- every message prints the path as its card spells it.
    """
    return os.path.normpath(path)


def _write_sets(cards: Sequence[MorphCard]) -> Dict[str, Dict[str, str]]:
    """``custom_id -> {normalised path: spelled path}`` for what each card writes."""
    return {card.custom_id: {_key(path): path for path in card.targets}
            for card in cards}


def _read_sets(cards: Sequence[MorphCard]) -> Dict[str, Dict[str, str]]:
    """``custom_id -> {normalised path: spelled path}`` for each EXPLICIT slice.

    A card with an empty ``context_slice`` gets an empty map here and is handled
    separately: its read set is "everything", which is not a set of declared
    paths but a property of the card (see :func:`find_hazards`).
    """
    return {card.custom_id: {_key(path): path for path in card.context_slice}
            for card in cards}


def _reachable(cards: Sequence[MorphCard]) -> Dict[str, Set[str]]:
    """``custom_id -> every card it transitively depends on``.

    Memoised depth-first closure over ``depends_on``. Assumes the deck is
    acyclic (:func:`cards.deck.validate_deck`), which every caller has already
    established -- a cycle cannot be layered into generations and so can never
    reach a hazard question.
    """
    by_id = {card.custom_id: card for card in cards}
    closure: Dict[str, Set[str]] = {}

    def walk(custom_id: str) -> Set[str]:
        if custom_id in closure:
            return closure[custom_id]
        found: Set[str] = set()
        closure[custom_id] = found          # guards a malformed self-reference
        for dep in by_id[custom_id].depends_on:
            if dep in by_id:
                found.add(dep)
                found |= walk(dep)
        closure[custom_id] = found
        return found

    for card in cards:
        walk(card.custom_id)
    return closure


def _contested(writes: Dict[str, str], reads: Dict[str, str]) -> Tuple[str, ...]:
    """The paths in both maps, as the WRITER spells them, in sorted order."""
    return tuple(writes[key] for key in sorted(set(writes) & set(reads)))


def find_hazards(cards: Sequence[MorphCard]) -> List[Hazard]:
    """Every file-ownership hazard in a deck, in a stable order.

    The deck is layered exactly the way the run will layer it
    (:func:`cards.generations.split_into_generations`), because "concurrent"
    means "same computed generation": cards in different generations are
    serialized in time by the run loop, which compiles generation N+1 only after
    N's morphs are on disk.

    Pairs are examined in deck order (``i < j``), and each pair yields at most
    one hazard per direction, so the output is deterministic and reads like the
    deck. See the module docstring for what each kind means and why its severity
    is what it is.
    """
    cards = list(cards)
    generation_of: Dict[str, int] = {}
    for number, generation in enumerate(split_into_generations(cards)):
        for card in generation:
            generation_of[card.custom_id] = number

    writes = _write_sets(cards)
    reads = _read_sets(cards)
    closure = _reachable(cards)
    whole_project = {card.custom_id for card in cards if not card.context_slice}

    hazards: List[Hazard] = []
    for i, left in enumerate(cards):
        for right in cards[i + 1:]:
            a, b = left.custom_id, right.custom_id
            same_generation = generation_of[a] == generation_of[b]
            generation = generation_of[a] if same_generation else None

            if same_generation:
                both = _contested(writes[a], writes[b])
                if both:
                    hazards.append(Hazard(KIND_WRITE_WRITE, SEVERITY_ERROR,
                                          a, b, both, generation))

            # Both directions of "one reads what the other writes". The writer
            # is always ``left`` of the hazard, the reader always ``right``.
            # A path the READER also writes is left out of both read kinds: it
            # is already a write/write hazard above, and naming one file twice
            # in two voices is how a report stops being read.
            for writer, reader in ((a, b), (b, a)):
                contested = {key: value for key, value
                             in writes[writer].items() if key not in writes[reader]}
                shared = _contested(contested, reads[reader])
                if shared:
                    if same_generation:
                        hazards.append(Hazard(KIND_READ_WRITE, SEVERITY_ERROR,
                                              writer, reader, shared, generation))
                    elif writer not in closure[reader]:
                        hazards.append(Hazard(KIND_UNORDERED_READ,
                                              SEVERITY_WARNING, writer, reader,
                                              shared, None))
                elif same_generation and reader in whole_project and contested:
                    # An empty slice reads the whole project, siblings' targets
                    # included. Reported only within a generation: across
                    # generations every empty-slice card would name every other
                    # card, which is noise, and the content it reads there is
                    # fresh anyway.
                    implicit = tuple(contested[key] for key in sorted(contested))
                    hazards.append(Hazard(KIND_IMPLICIT_READ, SEVERITY_WARNING,
                                          writer, reader, implicit, generation))
    return hazards


def errors(hazards: Sequence[Hazard]) -> List[Hazard]:
    """The hazards that must stop a run."""
    return [hazard for hazard in hazards if hazard.severity == SEVERITY_ERROR]


def warnings(hazards: Sequence[Hazard]) -> List[Hazard]:
    """The hazards an operator is told about but may proceed through."""
    return [hazard for hazard in hazards if hazard.severity == SEVERITY_WARNING]


def format_hazards(hazards: Sequence[Hazard], indent: str = "    ") -> str:
    """The hazards as one indented block, one line each."""
    return "\n".join(f"{indent}{hazard}" for hazard in hazards)


# -- repair ------------------------------------------------------------------


def repairable(hazards: Sequence[Hazard]) -> List[Hazard]:
    """The hazards :func:`repair_deck` knows how to fix: the read/write ones.

    A write/write hazard is never repairable -- see the module docstring -- and
    the two warning kinds are not repaired because their repair is worse than
    the hazard (the implicit one) or unnecessary for correctness of content
    (the unordered one).
    """
    return [hazard for hazard in hazards if hazard.kind == KIND_READ_WRITE]


def repair_deck(
    cards: Sequence[MorphCard], hazards: Sequence[Hazard]
) -> Tuple[List[MorphCard], List[Hazard]]:
    """Add the missing dependency edges, returning ``(new cards, edges added)``.

    One edge per :func:`repairable` hazard: the reader gains ``depends_on`` the
    writer, which pushes it into a later generation, where the run compiles it
    from the file the writer actually produced. Cards are never mutated -- a
    fresh :class:`~cards.schema.MorphCard` is built for each reader, the rest
    are passed through -- and deck order is preserved, so the repaired deck is
    the authored deck plus edges.

    Raises :class:`HazardError` if the repair would close a dependency cycle:
    two cards that each read what the other writes cannot be ordered at all, and
    the honest answer is that the deck is wrong (merge them into one changeset
    card, or cut the shared file out of one of the slices).
    """
    edges = repairable(hazards)
    if not edges:
        return list(cards), []

    additions: Dict[str, List[str]] = {}
    for hazard in edges:
        already = additions.setdefault(hazard.right, [])
        if hazard.left not in already:
            already.append(hazard.left)

    repaired: List[MorphCard] = []
    for card in cards:
        extra = [dep for dep in additions.get(card.custom_id, [])
                 if dep not in card.depends_on]
        if not extra:
            repaired.append(card)
            continue
        repaired.append(MorphCard(
            custom_id=card.custom_id,
            intent=card.intent,
            targets=list(card.targets),
            instruction=card.instruction,
            context_slice=list(card.context_slice),
            acceptance=card.acceptance,
            model=card.model,
            variants=card.variants,
            generation=card.generation,
            depends_on=list(card.depends_on) + extra,
        ))

    try:
        validate_deck(repaired)
    except DeckError as error:
        raise HazardError(
            "the deck cannot be ordered: cards read files that each other "
            f"write, and serializing them closes a cycle ({error}). Merge them "
            "into one card that writes the whole set ('targets'), or take the "
            "shared file out of one of the slices.",
            list(edges)) from error
    return repaired, edges


# -- the gate ----------------------------------------------------------------


def check_deck(cards: Sequence[MorphCard], strict: bool = False) -> List[Hazard]:
    """Raise on an unrunnable deck; return the hazards worth mentioning.

    Raises :class:`HazardError` when the deck holds any ERROR hazard (or any
    hazard at all under ``strict``), with every offending line in the message --
    an operator fixing a deck wants the whole list, not the first item of it.
    Otherwise returns the warnings, for the caller to log.

    This does NOT repair: :func:`cards.store.preflight_deck` repairs first and
    calls this with what repair could not fix, so a caller of this function
    always learns about the hazards that are really left.
    """
    hazards = list(find_hazards(cards))
    fatal = list(hazards) if strict else errors(hazards)
    if fatal:
        raise HazardError(
            f"the deck has {len(fatal)} file-ownership problem(s) -- two cards "
            f"of one generation cannot share a file, because a batch request "
            f"never sees another request's output:\n"
            f"{format_hazards(fatal)}",
            fatal)
    return warnings(hazards)
