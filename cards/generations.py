"""
Generations: splitting a deck by dependency depth and running it generation
by generation.

Phase 3 of ``documentation/DEVELOPMENT_PLAN.md``. A batch request cannot see
another request's output (``documentation/batch-orchestrator.md``, "The
generation cycle"), so any card that reads a file another card writes must wait
for the next batch. This module turns a validated deck into an ordered list of
*generations* -- each a set of cards with no dependency on one another -- and
then executes them in order through a single :class:`processors.batch.BatchBackend`:
compile generation N (reading the fresh files generation N-1 just wrote),
submit, poll, collect, write the morphs, then move to generation N+1.

Two responsibilities live here:

* :func:`split_into_generations` -- a pure topological layering of the deck by
  longest dependency path. The card's stored ``generation`` field is advisory
  author input and is ignored: computed placement always wins. Cards are never
  mutated.
* :func:`run_deck` -- the generation execution loop, plus its bookkeeping in
  :class:`DeckResult`.

Constraint (see the development plan): ``cards`` must not import from ``flows``
or ``processors``. The backend arrives as a parameter (duck-typed: only
``submit`` / ``status`` / ``collect`` are called), and the small response-to-file
helper is duplicated from ``flows.morph`` rather than imported -- the same
pattern as the source filter duplicated in :mod:`cards.compiler`.

Known simplification for this phase: ``flows.morph``'s ``append_if_plain`` /
``todo`` append semantics are *not* reproduced. A response with no fenced code
block is written verbatim in mode ``'w'`` (never appended). Machine acceptance
and regeneration with error context are Phase 4; per-card backend routing is
Phase 5. Neither is built here.
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from cards.compiler import compile_card
from cards.schema import MorphCard


# -- splitting ---------------------------------------------------------------


def split_into_generations(cards: List[MorphCard]) -> List[List[MorphCard]]:
    """Layer a deck into generations by longest dependency path.

    Generation 0 holds every card with no ``depends_on``; a card otherwise lands
    in ``1 + max(generation of its dependencies)``. Within a generation the deck
    order is preserved. The result is a list of generations, index ``0`` first.

    Assumes ``cards`` already passed :func:`cards.deck.validate_deck` (acyclic,
    every dependency present), so the recursion below always terminates. The
    stored ``generation`` field on each card is ignored -- computed placement
    wins -- and no card is mutated.
    """
    by_id = {card.custom_id: card for card in cards}

    # Memoized longest-path depth. Safe against re-computation and, on an acyclic
    # deck, guaranteed to terminate.
    depth: Dict[str, int] = {}

    def generation_of(custom_id: str) -> int:
        if custom_id in depth:
            return depth[custom_id]
        card = by_id[custom_id]
        if not card.depends_on:
            result = 0
        else:
            result = 1 + max(generation_of(dep) for dep in card.depends_on)
        depth[custom_id] = result
        return result

    highest = max((generation_of(card.custom_id) for card in cards), default=-1)
    generations: List[List[MorphCard]] = [[] for _ in range(highest + 1)]
    # A second pass in deck order fills each generation, so intra-generation
    # ordering follows the deck rather than the recursion order above.
    for card in cards:
        generations[generation_of(card.custom_id)].append(card)
    return generations


# -- run result --------------------------------------------------------------


@dataclass
class CardOutcome:
    """What happened to one card in a deck run.

    ``status`` is ``"written"`` (``paths`` lists the files written -- one for a
    single-variant card, one per surviving variant otherwise), ``"failed"``
    (every variant response was ``None``, or the whole batch failed), or
    ``"skipped"`` (a dependency failed or was itself skipped; ``reason`` names
    the blocking dependency).
    """

    custom_id: str
    status: str
    paths: List[str] = field(default_factory=list)
    reason: Optional[str] = None

    def __str__(self) -> str:
        if self.status == "written":
            return f"{self.custom_id}: written -> {', '.join(self.paths)}"
        if self.status == "skipped":
            return f"{self.custom_id}: skipped (blocked by {self.reason})"
        return f"{self.custom_id}: failed"


@dataclass
class DeckResult:
    """A summary of one :func:`run_deck` execution.

    ``outcomes`` maps ``custom_id -> CardOutcome`` in the order the cards were
    processed (generation order, deck order within a generation).
    ``generations`` records the composition as executed: the ``custom_id``\\ s of
    each generation, index ``0`` first. The Phase 5 ``/deck`` status view builds
    on this.
    """

    outcomes: Dict[str, CardOutcome] = field(default_factory=dict)
    generations: List[List[str]] = field(default_factory=list)

    def __str__(self) -> str:
        total = len(self.generations)
        lines = [f"deck run: {total} generation(s), {len(self.outcomes)} card(s)"]
        for index, generation in enumerate(self.generations, start=1):
            lines.append(f"  [generation {index}/{total}] {', '.join(generation)}")
        for outcome in self.outcomes.values():
            lines.append(f"  {outcome}")
        return "\n".join(lines)


# -- response to file (mirrors flows.morph) ----------------------------------


def _response_to_file_body(response: str) -> str:
    """Extract the file body from a response text.

    Mirrors ``flows.morph.MorphBot.response_to_file_body``: pull the fenced code
    blocks if any are present, else use the response verbatim. Duplicated (not
    imported) to keep ``cards`` free of any dependency on ``flows``; keep the two
    in step when either changes. Simplification for this phase: the
    ``append_if_plain`` / ``todo`` append mode is dropped -- callers here always
    write mode ``'w'``.
    """
    code_blocks = re.findall(r"```(.*?)\n(.*?)\n```", response, re.DOTALL)
    if 0 < len(code_blocks):
        return "".join(f"{code_block}\n" for _, code_block in code_blocks)
    return response


def _variant_ids(card: MorphCard) -> List[str]:
    """The batch custom_ids a card compiles to, matching :func:`compile_card`."""
    if card.variants == 1:
        return [card.custom_id]
    return [f"{card.custom_id}.v{n}" for n in range(1, card.variants + 1)]


def _output_path(card: MorphCard, variant_custom_id: str, root: str) -> str:
    """Where one (variant) response is written, relative to ``root``.

    A single-variant card writes its ``target``; a multi-variant card writes
    ``<stem>.<variant_custom_id><ext>`` (the ``variant_custom_id`` already
    carries the ``.vN`` suffix), mirroring ``flows.morph.output_file_name``.
    """
    if card.variants == 1:
        return os.path.join(root, card.target)
    stem, ext = os.path.splitext(card.target)
    return os.path.join(root, f"{stem}.{variant_custom_id}{ext}")


# -- the generation cycle ----------------------------------------------------


def run_deck(
    cards: List[MorphCard],
    backend,
    root: str = ".",
    poll_interval: float = 1.0,
    log: Callable[[str], None] = print,
) -> DeckResult:
    """Execute a deck generation by generation through one batch backend.

    For each generation, in order: skip any card whose dependency failed or was
    itself skipped (logged, never compiled or submitted); log the composition of
    what is being submitted *before* submitting; compile the remaining cards --
    which reads the fresh files earlier generations wrote, the whole point of
    generations -- submit them as one batch, poll ``backend.status`` until it
    reaches ``"completed"`` or ``"failed"`` (sleeping ``poll_interval`` between
    polls), collect, and write the morphs.

    Failure semantics for this phase: a card fails when every one of its variant
    responses is ``None``; it is still ``"written"`` if at least one variant
    succeeded. A whole batch reported ``"failed"`` fails every card it carried. A
    failed (or skipped) card's transitive dependents are skipped. Regeneration
    with error context is Phase 4 and is not built here.

    ``backend`` is duck-typed: only ``submit(requests) -> batch_id``,
    ``status(batch_id) -> str`` and ``collect(batch_id) -> {custom_id: text|None}``
    are called. Returns a :class:`DeckResult`.
    """
    generations = split_into_generations(cards)
    total = len(generations)
    composition = [[card.custom_id for card in generation] for generation in generations]

    outcomes: Dict[str, CardOutcome] = {}
    # custom_ids that failed or were skipped: their dependents cannot run.
    blocked: set = set()

    for index, generation in enumerate(generations, start=1):
        runnable: List[MorphCard] = []
        for card in generation:
            blocking = next((dep for dep in card.depends_on if dep in blocked), None)
            if blocking is not None:
                status = outcomes[blocking].status
                log(
                    f"mrph> [generation {index}/{total}] skipping {card.custom_id!r}: "
                    f"dependency {blocking!r} {status}"
                )
                outcomes[card.custom_id] = CardOutcome(
                    card.custom_id, "skipped", reason=blocking
                )
                blocked.add(card.custom_id)
            else:
                runnable.append(card)

        if not runnable:
            continue

        ids = [card.custom_id for card in runnable]
        log(
            f"mrph> [generation {index}/{total}] submitting "
            f"{len(runnable)} card(s): {', '.join(ids)}"
        )

        # Compile only now -- generation N-1's morphs are already on disk, so a
        # dependent card's context slice sees the fresh files.
        requests: List[dict] = []
        for card in runnable:
            requests.extend(compile_card(card, root))

        batch_id = backend.submit(requests)

        while True:
            status = backend.status(batch_id)
            if status in ("completed", "failed"):
                break
            time.sleep(poll_interval)

        if status == "failed":
            log(
                f"mrph> [generation {index}/{total}] batch {batch_id!r} failed; "
                f"{len(runnable)} card(s) failed"
            )
            for card in runnable:
                outcomes[card.custom_id] = CardOutcome(card.custom_id, "failed")
                blocked.add(card.custom_id)
            continue

        results = backend.collect(batch_id)

        for card in runnable:
            written: List[str] = []
            for variant_id in _variant_ids(card):
                response = results.get(variant_id)
                if response is None:
                    continue
                body = _response_to_file_body(response)
                path = _output_path(card, variant_id, root)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(body)
                written.append(path)

            if written:
                outcomes[card.custom_id] = CardOutcome(
                    card.custom_id, "written", paths=written
                )
                log(
                    f"mrph> [generation {index}/{total}] {card.custom_id!r} "
                    f"written: {', '.join(written)}"
                )
            else:
                outcomes[card.custom_id] = CardOutcome(card.custom_id, "failed")
                blocked.add(card.custom_id)
                log(
                    f"mrph> [generation {index}/{total}] {card.custom_id!r} "
                    f"failed: all variants empty"
                )

    return DeckResult(outcomes=outcomes, generations=composition)
