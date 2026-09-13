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

Phase 4 adds mechanical acceptance on top of this loop (see
:mod:`cards.acceptance`): a card with an ``acceptance`` command has its variants
verified best-of-N, and a card that fails verification is regenerated -- its
instruction extended with the acceptance error output -- into a retry batch that
runs before its dependents. A card *without* an ``acceptance`` command keeps the
exact Phase 3 semantics below (write every surviving variant, no verification,
no retries), so ``verify=True`` is a no-op for it.

Known simplification, still in force: ``flows.morph``'s ``append_if_plain`` /
``todo`` append semantics are *not* reproduced. A response with no fenced code
block is written verbatim in mode ``'w'`` (never appended). Per-card backend
routing is Phase 5 and is not built here.
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
    (every variant response was ``None``, or the whole batch failed, or -- for a
    card with acceptance -- verification failed through the last retry), or
    ``"skipped"`` (a dependency failed or was itself skipped; ``reason`` names
    the blocking dependency).

    Phase 4 adds three fields, meaningful only for a card with an ``acceptance``
    command: ``attempts`` is the number of generation-level tries (1 = original
    only, 2 = one retry, and so on); ``winning_variant`` is the batch custom_id
    of the variant that passed acceptance (``None`` unless ``"written"`` via
    verification); ``acceptance_output`` is the final failure's captured output
    (``None`` when the card passed or has no acceptance). A card without
    acceptance keeps ``attempts == 1`` and both others ``None``.
    """

    custom_id: str
    status: str
    paths: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    attempts: int = 1
    winning_variant: Optional[str] = None
    acceptance_output: Optional[str] = None

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


def response_to_file_body(response: str) -> str:
    """Extract the file body from a response text.

    Mirrors ``flows.morph.MorphBot.response_to_file_body``: pull the fenced code
    blocks if any are present, else use the response verbatim. Duplicated (not
    imported) to keep ``cards`` free of any dependency on ``flows``; keep the two
    in step when either changes. Public so :mod:`cards.acceptance` reuses this
    one copy rather than adding a third. Simplification for this phase: the
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


def _submit_poll_collect(
    requests: List[dict], backend, poll_interval: float
) -> Optional[Dict[str, Optional[str]]]:
    """Submit one batch, poll to completion, and collect -- or ``None`` on failure.

    Returns the ``{custom_id: text|None}`` map on a completed batch, or ``None``
    when the backend reports the whole batch ``"failed"`` (in which case there is
    nothing to collect). Sleeps ``poll_interval`` between polls. ``backend`` is
    duck-typed: only ``submit`` / ``status`` / ``collect`` are called.
    """
    batch_id = backend.submit(requests)
    while True:
        status = backend.status(batch_id)
        if status in ("completed", "failed"):
            break
        time.sleep(poll_interval)
    if status == "failed":
        return None
    return backend.collect(batch_id)


def _write_variants(
    card: MorphCard, results: Dict[str, Optional[str]], root: str
) -> List[str]:
    """Write every surviving variant of a card (Phase 3, no-acceptance path).

    A ``None`` response is a failed variant and is skipped. Returns the paths
    written; an empty list means every variant response was ``None``.
    """
    written: List[str] = []
    for variant_id in _variant_ids(card):
        response = results.get(variant_id)
        if response is None:
            continue
        body = response_to_file_body(response)
        path = _output_path(card, variant_id, root)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(body)
        written.append(path)
    return written


def _retry_card(card: MorphCard, attempt: int, result) -> MorphCard:
    """A copy of ``card`` for retry ``attempt``, its instruction carrying the error.

    The batch custom_id is suffixed ``.r<attempt>`` (variants then become
    ``.r<attempt>.v<m>`` via :func:`compile_card`). The instruction gains a
    clearly delimited block with the failed acceptance command and the tail of
    its output (empty when the previous attempt produced none, e.g. a timeout or
    all-``None`` responses) so the executor can see what went wrong.
    """
    output = result.output if result is not None else ""
    error_block = (
        "\n\n---\n"
        f"A previous attempt failed its acceptance check (`{card.acceptance}`):\n"
        f"{output}\n"
        "---\n"
        "Please fix the issues and produce the complete corrected file."
    )
    return MorphCard(
        custom_id=f"{card.custom_id}.r{attempt}",
        intent=card.intent,
        target=card.target,
        instruction=card.instruction + error_block,
        context_slice=list(card.context_slice),
        acceptance=card.acceptance,
        model=card.model,
        variants=card.variants,
        generation=card.generation,
        depends_on=list(card.depends_on),
    )


def run_deck(
    cards: List[MorphCard],
    backend,
    root: str = ".",
    poll_interval: float = 1.0,
    log: Callable[[str], None] = print,
    verify: bool = True,
    acceptance_timeout: float = 300.0,
    max_regenerations: int = 2,
) -> DeckResult:
    """Execute a deck generation by generation through one batch backend.

    For each generation, in order: skip any card whose dependency failed or was
    itself skipped (logged, never compiled or submitted); log the composition of
    what is being submitted *before* submitting; compile the remaining cards --
    which reads the fresh files earlier generations wrote, the whole point of
    generations -- submit them as one batch, poll ``backend.status`` until it
    reaches ``"completed"`` or ``"failed"`` (sleeping ``poll_interval`` between
    polls), collect, and process the morphs.

    A card WITHOUT an ``acceptance`` command keeps the Phase 3 contract: it is
    ``"written"`` if at least one variant response is non-``None`` (all surviving
    variants written), ``"failed"`` when every variant response is ``None``, and
    it is never retried.

    A card WITH an ``acceptance`` command (only when ``verify`` is true, the
    default) is verified best-of-N by :func:`cards.acceptance.verify_card`: the
    first variant to pass acceptance wins and the card is ``"written"``. A card
    that fails verification (or whose responses were all ``None``) is
    *regenerated* -- resubmitted with the acceptance error appended to its
    instruction -- into a retry batch that runs BEFORE this generation's
    dependents, up to ``max_regenerations`` times (default 2, so at most 3 total
    attempts). Its ``CardOutcome`` is recorded under the ORIGINAL custom_id with
    the attempt count and, on success, the winning variant. Only once a card
    exhausts its retries and finally ``"failed"`` are its dependents skipped.

    A whole batch reported ``"failed"`` (a transport/provider failure, no
    responses to judge) fails every card it carried terminally -- no retry. A
    failed or skipped card's transitive dependents are skipped.

    ``backend`` is duck-typed: only ``submit(requests) -> batch_id``,
    ``status(batch_id) -> str`` and ``collect(batch_id) -> {custom_id: text|None}``
    are called. Returns a :class:`DeckResult` whose ``generations`` records the
    static generation composition (retry batches are extra submits, not extra
    generations).
    """
    # Imported lazily to break the cycle: cards.acceptance imports the shared
    # file helpers from this module at import time, so this module can only reach
    # into it once its own definitions exist -- i.e. here, at call time.
    from cards.acceptance import verify_card

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
                    card.custom_id, "skipped", reason=blocking, attempts=0
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

        results = _submit_poll_collect(requests, backend, poll_interval)

        if results is None:
            log(
                f"mrph> [generation {index}/{total}] batch failed; "
                f"{len(runnable)} card(s) failed"
            )
            for card in runnable:
                outcomes[card.custom_id] = CardOutcome(card.custom_id, "failed")
                blocked.add(card.custom_id)
            continue

        # Cards that failed acceptance but have retries left, paired with the
        # verify outcome carrying the error context for their next attempt.
        retry_pending: List[tuple] = []

        for card in runnable:
            if verify and card.acceptance:
                outcome = verify_card(card, results, root, acceptance_timeout, log)
                if outcome.passed:
                    outcomes[card.custom_id] = CardOutcome(
                        card.custom_id,
                        "written",
                        paths=outcome.paths,
                        attempts=1,
                        winning_variant=outcome.winning_custom_id,
                    )
                    log(
                        f"mrph> [generation {index}/{total}] {card.custom_id!r} "
                        f"written (variant {outcome.winning_custom_id!r} passed "
                        f"acceptance): {', '.join(outcome.paths)}"
                    )
                elif max_regenerations > 0:
                    retry_pending.append((card, outcome))
                else:
                    outcomes[card.custom_id] = CardOutcome(
                        card.custom_id,
                        "failed",
                        attempts=1,
                        acceptance_output=_acceptance_output(outcome.result),
                    )
                    blocked.add(card.custom_id)
                    log(
                        f"mrph> [generation {index}/{total}] {card.custom_id!r} "
                        f"failed acceptance (no retries)"
                    )
            else:
                written = _write_variants(card, results, root)
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

        # Retry generations for the failed acceptance cards -- these run BEFORE
        # this generation's dependents (which live in later generations).
        _run_retries(
            retry_pending,
            index,
            total,
            root,
            backend,
            poll_interval,
            log,
            acceptance_timeout,
            max_regenerations,
            outcomes,
            blocked,
        )

    return DeckResult(outcomes=outcomes, generations=composition)


def _acceptance_output(result) -> Optional[str]:
    """The captured output of a failing acceptance result, or ``None`` if none ran."""
    return result.output if result is not None else None


def _run_retries(
    pending: List[tuple],
    index: int,
    total: int,
    root: str,
    backend,
    poll_interval: float,
    log: Callable[[str], None],
    acceptance_timeout: float,
    max_regenerations: int,
    outcomes: Dict[str, CardOutcome],
    blocked: set,
) -> None:
    """Regenerate cards that failed acceptance, up to ``max_regenerations`` times.

    ``pending`` is a list of ``(card, verify_outcome)``. Each retry attempt
    builds a retry card per still-pending card (instruction carrying the previous
    attempt's error), submits them as one retry batch, and re-verifies. A card
    that passes is recorded ``"written"`` under its ORIGINAL custom_id with the
    attempt count; a card that exhausts its retries is recorded ``"failed"`` and
    blocks its dependents. Mutates ``outcomes`` and ``blocked`` in place.
    """
    from cards.acceptance import verify_card

    attempt = 0
    while pending and attempt < max_regenerations:
        attempt += 1

        retry_cards: List[MorphCard] = []
        for card, prev_outcome in pending:
            log(
                f"mrph> [generation {index}/{total}] retry "
                f"{attempt}/{max_regenerations} for card {card.custom_id!r} "
                f"(acceptance failed)"
            )
            retry_cards.append(_retry_card(card, attempt, prev_outcome.result))

        requests: List[dict] = []
        for retry_card in retry_cards:
            requests.extend(compile_card(retry_card, root))

        results = _submit_poll_collect(requests, backend, poll_interval)

        if results is None:
            # The retry batch itself failed wholesale -- fail every pending card
            # terminally, carrying its last real acceptance output.
            log(
                f"mrph> [generation {index}/{total}] retry {attempt} batch "
                f"failed; {len(pending)} card(s) failed"
            )
            for card, prev_outcome in pending:
                outcomes[card.custom_id] = CardOutcome(
                    card.custom_id,
                    "failed",
                    attempts=1 + attempt,
                    acceptance_output=_acceptance_output(prev_outcome.result),
                )
                blocked.add(card.custom_id)
            return

        next_pending: List[tuple] = []
        for retry_card, (card, _prev) in zip(retry_cards, pending):
            outcome = verify_card(retry_card, results, root, acceptance_timeout, log)
            if outcome.passed:
                outcomes[card.custom_id] = CardOutcome(
                    card.custom_id,
                    "written",
                    paths=outcome.paths,
                    attempts=1 + attempt,
                    winning_variant=outcome.winning_custom_id,
                )
                log(
                    f"mrph> [generation {index}/{total}] {card.custom_id!r} "
                    f"written after retry {attempt} (variant "
                    f"{outcome.winning_custom_id!r} passed): "
                    f"{', '.join(outcome.paths)}"
                )
            elif attempt < max_regenerations:
                next_pending.append((card, outcome))
            else:
                outcomes[card.custom_id] = CardOutcome(
                    card.custom_id,
                    "failed",
                    attempts=1 + attempt,
                    acceptance_output=_acceptance_output(outcome.result),
                )
                blocked.add(card.custom_id)
                log(
                    f"mrph> [generation {index}/{total}] {card.custom_id!r} "
                    f"failed acceptance after {attempt} retr"
                    f"{'y' if attempt == 1 else 'ies'}"
                )
        pending = next_pending
