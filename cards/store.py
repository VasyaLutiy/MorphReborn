"""
The store: backlog persistence and split-step (submit / collect) execution.

Phase 5 of ``documentation/DEVELOPMENT_PLAN.md``. Where :mod:`cards.generations`
runs a whole deck synchronously (``run_deck``: submit -> poll -> collect ->
advance, blocking), the interactive ``mrph`` CLI needs the cycle *stepwise*: the
engineer appends cards during the day (``/card``), fires the current generation
in the evening (``/submit``), and picks the morphs up the next morning
(``/collect``) -- possibly after quitting and restarting the CLI in between. That
demands the backlog and the run state survive on disk.

Two things live here:

* :class:`DeckStore` -- the ``.morph/`` persistence layer. ``deck.json`` is the
  backlog (a JSON list of card dicts, exactly what :func:`cards.deck.load_deck`
  reads); ``state.json`` is the run state (which generation, the in-flight batch
  id, the per-card outcomes, the phase). Pure persistence + validation.
* :func:`submit_generation` / :func:`collect_generation` -- the split of
  ``run_deck``'s per-generation cycle into two calls that talk through
  ``state.json``. They are built ON TOP of the existing primitives
  (:func:`cards.generations.resolve_runnable` / ``process_generation`` /
  ``compile_card`` / ``verify_card``), so the best-of-N, rollback, retry and
  skip-cascade semantics are byte-for-byte those of ``run_deck``.
* :func:`record_run` / :func:`recover_orphaned_local_batch` -- the two state
  transitions the split-step pair cannot express: persisting a whole run that
  ``run_deck`` executed in memory (``/nightly``), and letting a run out of a
  phase ``"submitted"`` whose batch died with the CLI process.

DESIGN NOTE (flagged for review): the split-step functions live in this module
rather than a separate one -- they are the store's reason to exist, and they need
nothing from ``processors`` or ``flows`` (the batch backend arrives as a
duck-typed parameter). ``cards`` therefore still imports nothing from
``flows``/``processors``, honouring the layering constraint.

Pure and stdlib-only. No network, no provider SDK, no imports from ``flows`` or
``processors``.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from cards.compiler import compile_card
from cards.deck import load_deck, validate_deck
from cards.generations import (
    CardOutcome,
    process_generation,
    resolve_runnable,
    split_into_generations,
)
from cards.schema import MorphCard


MORPH_DIR = ".morph"
DECK_FILE = "deck.json"
STATE_FILE = "state.json"

# Card display statuses derived from state + backlog (Phase 5).
STATUS_PENDING = "pending"
STATUS_IN_FLIGHT = "in_flight"
STATUS_WRITTEN = "written"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

# Run phases persisted in state.json.
PHASE_IDLE = "idle"            # ready to /submit the current generation
PHASE_SUBMITTED = "submitted"  # a batch is in flight for the current generation
PHASE_DONE = "done"            # every generation has been processed

# ``LocalBatchBackend`` mints its batch ids as "local-<hex>" (see
# processors/batch.py). Duplicated here as a prefix rather than imported: this
# module must not depend on ``processors``, and the shape is part of the state
# file's contract anyway -- it is what tells a batch that died with the CLI
# process apart from a cloud batch that is still sitting on a provider's server.
LOCAL_BATCH_PREFIX = "local-"


class StoreError(RuntimeError):
    """A store operation was requested in a state that does not allow it.

    Distinct from :class:`cards.schema.CardError` / :class:`cards.deck.DeckError`
    (which concern malformed cards/decks): this is about the *run* -- submitting
    twice, collecting with nothing in flight, removing an absent card.
    """


# -- card <-> dict serialization --------------------------------------------


def card_to_dict(card: MorphCard) -> dict:
    """Serialize a card back to the nested backlog JSON shape.

    The inverse of :meth:`MorphCard.from_dict`'s nested form: ``custom_id`` and
    ``instruction`` at the top level, the meta fields under ``"meta"``. Only
    non-default meta values are emitted, so a round-tripped backlog stays as
    clean as a hand-written one (defaults are reapplied on load).
    """
    meta: Dict[str, object] = {"intent": card.intent, "target": card.target}
    if card.context_slice:
        meta["context_slice"] = list(card.context_slice)
    if card.acceptance is not None:
        meta["acceptance"] = card.acceptance
    if card.model is not None:
        meta["model"] = card.model
    if card.variants != 1:
        meta["variants"] = card.variants
    if card.generation:
        meta["generation"] = card.generation
    if card.depends_on:
        meta["depends_on"] = list(card.depends_on)
    return {"custom_id": card.custom_id, "meta": meta, "instruction": card.instruction}


# -- outcome <-> dict serialization -----------------------------------------


def _outcome_to_dict(outcome: CardOutcome) -> dict:
    return {
        "custom_id": outcome.custom_id,
        "status": outcome.status,
        "paths": list(outcome.paths),
        "reason": outcome.reason,
        "attempts": outcome.attempts,
        "winning_variant": outcome.winning_variant,
        "acceptance_output": outcome.acceptance_output,
    }


def _outcome_from_dict(data: dict) -> CardOutcome:
    return CardOutcome(
        custom_id=data["custom_id"],
        status=data["status"],
        paths=list(data.get("paths", [])),
        reason=data.get("reason"),
        attempts=data.get("attempts", 1),
        winning_variant=data.get("winning_variant"),
        acceptance_output=data.get("acceptance_output"),
    )


def _outcomes_from_state(state: dict) -> Dict[str, CardOutcome]:
    return {
        cid: _outcome_from_dict(data)
        for cid, data in state.get("outcomes", {}).items()
    }


def _store_outcomes(state: dict, outcomes: Dict[str, CardOutcome]) -> None:
    state["outcomes"] = {
        cid: _outcome_to_dict(outcome) for cid, outcome in outcomes.items()
    }


def _blocked_from_outcomes(outcomes: Dict[str, CardOutcome]) -> set:
    """The set of custom_ids whose dependents cannot run (failed or skipped).

    Derived rather than stored, so the two can never drift: exactly the set
    ``run_deck`` maintains in memory as ``blocked``.
    """
    return {
        cid
        for cid, outcome in outcomes.items()
        if outcome.status in (STATUS_FAILED, STATUS_SKIPPED)
    }


# -- persistence -------------------------------------------------------------


class DeckStore:
    """Backlog + run-state persistence under ``<project_root>/.morph/``.

    ``deck.json`` holds the backlog (the format :func:`cards.deck.load_deck`
    reads); ``state.json`` holds the run state that lets ``/submit`` and
    ``/collect`` span CLI restarts. Every write goes through validation: a card
    that would make the backlog invalid (duplicate id, dangling/cyclic
    dependency) is rejected before anything is saved.
    """

    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.morph_dir = os.path.join(project_root, MORPH_DIR)
        self.deck_path = os.path.join(self.morph_dir, DECK_FILE)
        self.state_path = os.path.join(self.morph_dir, STATE_FILE)

    def _ensure_dir(self) -> None:
        os.makedirs(self.morph_dir, exist_ok=True)

    # -- backlog ------------------------------------------------------------

    def load_cards(self) -> List[MorphCard]:
        """The backlog as validated cards; an empty list when no file exists."""
        if not os.path.exists(self.deck_path):
            return []
        return load_deck(self.deck_path)

    def _save_cards(self, cards: List[MorphCard]) -> None:
        self._ensure_dir()
        payload = [card_to_dict(card) for card in cards]
        with open(self.deck_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def add_card(self, card_dict: dict) -> MorphCard:
        """Validate a single card, then the resulting deck, then append and save.

        Raises :class:`cards.schema.CardError` if the card itself is malformed
        and :class:`cards.deck.DeckError` if it breaks the deck (a duplicate
        ``custom_id``, a dangling or cyclic dependency). Nothing is written when
        validation fails.
        """
        card = MorphCard.from_dict(card_dict)
        cards = self.load_cards()
        combined = cards + [card]
        validate_deck(combined)
        self._save_cards(combined)
        return card

    def add_cards(self, card_dicts: List[dict]) -> List[MorphCard]:
        """Validate and append several cards as one fragment (decomposition mode).

        The whole fragment is validated together *against the existing backlog*
        (so cross-references within the fragment resolve, and a duplicate against
        an existing card is caught) and either all are added or none are.
        """
        new_cards = [MorphCard.from_dict(item) for item in card_dicts]
        combined = self.load_cards() + new_cards
        validate_deck(combined)
        self._save_cards(combined)
        return new_cards

    def remove_card(self, custom_id: str) -> None:
        """Drop a card from the backlog by id.

        Raises :class:`StoreError` if no such card exists, and
        :class:`cards.deck.DeckError` if removing it would leave a dangling
        dependency (another card depends on it) -- the removal is refused rather
        than silently corrupting the deck.
        """
        cards = self.load_cards()
        remaining = [card for card in cards if card.custom_id != custom_id]
        if len(remaining) == len(cards):
            raise StoreError(f"no card {custom_id!r} in the backlog")
        validate_deck(remaining)
        self._save_cards(remaining)

    def clear(self) -> None:
        """Empty the backlog (remove ``deck.json``). Run state is left untouched."""
        if os.path.exists(self.deck_path):
            os.remove(self.deck_path)

    # -- run state ----------------------------------------------------------

    @staticmethod
    def _default_state() -> dict:
        return {
            "phase": PHASE_IDLE,
            "generation_index": 0,     # 0-based index of the current generation
            "generations": [],         # static composition: list of custom_id lists
            "batch_id": None,          # in-flight batch id (phase "submitted")
            "backend_label": None,     # which processor(s) the in-flight batch runs on
            "submitted_ids": [],       # runnable card ids in the in-flight batch
            "outcomes": {},            # custom_id -> outcome dict
        }

    def load_state(self) -> dict:
        """The run state, merged over defaults so missing keys are always present."""
        state = self._default_state()
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
            state.update(stored)
        return state

    def save_state(self, state: dict) -> None:
        self._ensure_dir()
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def reset_state(self) -> None:
        """Discard the run state (delete ``state.json``); the backlog is kept."""
        if os.path.exists(self.state_path):
            os.remove(self.state_path)

    def load_outcomes(self) -> Dict[str, CardOutcome]:
        """The recorded per-card outcomes, as :class:`CardOutcome` objects.

        A convenience over :meth:`load_state` for callers (the ``/deck`` view)
        that want the paths/attempts detail behind each card's display status.
        """
        return _outcomes_from_state(self.load_state())


# -- display status ----------------------------------------------------------


@dataclass
class DeckStatusView:
    """A testable snapshot of the backlog for the ``/deck`` view.

    ``card_status`` pairs each backlog card (in deck order) with one of
    ``pending`` | ``in_flight`` | ``written`` | ``failed`` | ``skipped``.
    ``generations`` is the run's static composition when a run has started, else
    a fresh preview computed from the current backlog. ``current_generation`` is
    the 0-based index of the generation that is in flight or next to submit.
    """

    empty: bool
    phase: str
    current_generation: int
    generations: List[List[str]]
    card_status: List[Tuple[str, str]]


def build_deck_status(store: DeckStore) -> DeckStatusView:
    """Derive per-card display statuses from the backlog and run state.

    A card with a recorded outcome shows that outcome's status; a card in the
    in-flight batch (phase ``"submitted"``) shows ``in_flight``; everything else
    is ``pending``. This is the function the ``/deck`` transition renders and the
    end-to-end test asserts against.
    """
    cards = store.load_cards()
    state = store.load_state()
    outcomes = _outcomes_from_state(state)
    submitted = set(state.get("submitted_ids", []))
    phase = state.get("phase", PHASE_IDLE)

    card_status: List[Tuple[str, str]] = []
    for card in cards:
        if card.custom_id in outcomes:
            status = outcomes[card.custom_id].status
        elif phase == PHASE_SUBMITTED and card.custom_id in submitted:
            status = STATUS_IN_FLIGHT
        else:
            status = STATUS_PENDING
        card_status.append((card.custom_id, status))

    generations = state.get("generations")
    if not generations:
        generations = [
            [card.custom_id for card in generation]
            for generation in split_into_generations(cards)
        ]

    return DeckStatusView(
        empty=not cards,
        phase=phase,
        current_generation=state.get("generation_index", 0),
        generations=generations,
        card_status=card_status,
    )


# -- whole-run persistence and recovery --------------------------------------


def record_run(store: DeckStore, result, backend_label: Optional[str] = None) -> None:
    """Persist a deck run that :func:`cards.generations.run_deck` held in memory.

    ``/nightly`` runs the whole deck in one blocking pass, writing every morph to
    disk -- but the run itself lived only in the returned ``DeckResult``, so the
    next ``/deck`` reported ``idle`` and every card ``pending`` for work that was
    finished. This records that run in ``state.json`` in exactly the shape the
    split-step path leaves behind, so the ``/deck`` view cannot tell the two
    routes apart: phase ``"done"``, the composition as executed
    (``DeckResult.generations`` is already a list of custom_id lists, the shape
    ``state["generations"]`` wants), and every :class:`CardOutcome`.

    The run state is REPLACED, not merged: a nightly pass computes its own
    generations from the whole backlog, so whatever an earlier split-step run
    left is history. ``backend_label`` is recorded for symmetry with
    :func:`submit_generation`; nothing is in flight, so it is informational.
    """
    state = DeckStore._default_state()
    state["phase"] = PHASE_DONE
    state["generations"] = [list(generation) for generation in result.generations]
    state["generation_index"] = len(result.generations)
    state["backend_label"] = backend_label
    _store_outcomes(state, result.outcomes)
    store.save_state(state)


def recover_orphaned_local_batch(store: DeckStore) -> bool:
    """Put a local batch that died with the CLI process back to ``pending``.

    A :class:`processors.batch.LocalBatchBackend` batch is worker threads and an
    in-memory result dict; its id (``"local-<hex>"``) means nothing to a new
    process. So a CLI restarted between ``/submit`` and ``/collect`` used to be
    wedged forever: ``/submit`` refused (a generation is already submitted) and
    ``/collect`` had nothing to collect from. The honest repair is to admit the
    batch is gone and return its cards to the queue.

    Only a LOCAL batch is recoverable this way. A cloud batch id (OpenAI /
    Anthropic) names work that is genuinely still running on a provider's server
    and collectable later, so it is never touched here.

    Returns ``True`` when something was recovered. The generation composition and
    every recorded outcome survive -- only the in-flight bookkeeping (phase,
    batch id, backend label, submitted ids) is cleared, so the run resumes at the
    same generation on the next ``/submit``. Callers must only invoke this when
    the session holds no live backend for the batch.
    """
    state = store.load_state()
    if state.get("phase") != PHASE_SUBMITTED:
        return False
    batch_id = state.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id.startswith(LOCAL_BATCH_PREFIX):
        return False

    state["phase"] = PHASE_IDLE
    state["batch_id"] = None
    state["backend_label"] = None
    state["submitted_ids"] = []
    store.save_state(state)
    return True


# -- split-step execution ----------------------------------------------------


@dataclass
class SubmitResult:
    """What one :func:`submit_generation` call did.

    ``submitted`` is true when a batch was actually sent; ``done`` is true when
    there was nothing left to run (every remaining generation was skipped or the
    deck was already complete). ``skipped`` lists ``(custom_id, blocking_dep)``
    recorded while advancing to a runnable generation.
    """

    submitted: bool
    done: bool
    generation_number: int          # 1-based; 0 when nothing was submitted
    total_generations: int
    batch_id: Optional[str]
    card_ids: List[str] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class CollectResult:
    """What one :func:`collect_generation` call did.

    ``in_progress`` true means the batch had not finished and nothing changed --
    the caller reports it and the user re-runs ``/collect`` later. Otherwise
    ``outcomes`` carries the :class:`CardOutcome` for every card in the collected
    generation (runnable + any skipped while advancing), and ``phase`` is the new
    run phase (``"idle"`` when a further generation remains, ``"done"`` when the
    deck is finished).
    """

    in_progress: bool
    generation_number: int
    total_generations: int
    phase: str
    outcomes: Dict[str, CardOutcome] = field(default_factory=dict)


def _ensure_run_started(store: DeckStore, state: dict, cards: List[MorphCard]) -> None:
    """Lock in the generation composition on the first submit of a fresh run.

    The composition is computed once, from the backlog as it stands at the first
    ``/submit``, and persisted -- later generations are compiled against the
    files earlier ones write, but which cards live in which generation is fixed
    for the run.
    """
    if not state.get("generations"):
        state["generations"] = [
            [card.custom_id for card in generation]
            for generation in split_into_generations(cards)
        ]
        state["generation_index"] = 0
        state["outcomes"] = {}


def submit_generation(
    store: DeckStore,
    backend,
    root: str = ".",
    backend_label: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> SubmitResult:
    """Compile and submit the current generation's runnable cards.

    On the first call of a run the generation composition is locked in from the
    backlog. Cards whose dependency already failed/was skipped are recorded as
    ``skipped`` and the loop advances to the next generation that has runnable
    cards; that generation is compiled (reading the fresh files earlier
    generations wrote) and submitted as one batch. The batch id and the runnable
    ids are saved and the phase moves to ``"submitted"``.

    Raises :class:`StoreError` if a batch is already in flight (``phase ==
    "submitted"``) or the run is already ``"done"``. When every remaining
    generation is skippable (or the deck is empty), no batch is sent, the phase
    becomes ``"done"`` and ``SubmitResult.done`` is true. ``backend`` is
    duck-typed: only ``submit`` is called here.
    """
    state = store.load_state()
    phase = state.get("phase", PHASE_IDLE)
    if phase == PHASE_SUBMITTED:
        raise StoreError(
            "a generation is already submitted; run /collect before /submit")
    if phase == PHASE_DONE:
        raise StoreError(
            "the deck run is complete; run /deck reset before submitting again")

    cards = store.load_cards()
    _ensure_run_started(store, state, cards)

    by_id = {card.custom_id: card for card in cards}
    composition = state["generations"]
    total = len(composition)
    outcomes = _outcomes_from_state(state)
    blocked = _blocked_from_outcomes(outcomes)

    index = state.get("generation_index", 0)
    skipped_here: List[Tuple[str, str]] = []
    runnable: List[MorphCard] = []

    # Advance past any generation with nothing runnable (all skipped), recording
    # the skips, until a generation has runnable cards or the deck ends.
    while index < total:
        generation = [by_id[cid] for cid in composition[index] if cid in by_id]
        before = set(outcomes)
        runnable = resolve_runnable(generation, index + 1, total, outcomes, blocked, log)
        for cid in outcomes:
            if cid not in before and outcomes[cid].status == STATUS_SKIPPED:
                skipped_here.append((cid, outcomes[cid].reason))
        if runnable:
            break
        index += 1

    if index >= total or not runnable:
        # Nothing left to run: record any trailing skips and finish the run.
        state["phase"] = PHASE_DONE
        state["generation_index"] = total
        state["batch_id"] = None
        state["submitted_ids"] = []
        _store_outcomes(state, outcomes)
        store.save_state(state)
        return SubmitResult(
            submitted=False, done=True, generation_number=0,
            total_generations=total, batch_id=None, skipped=skipped_here)

    ids = [card.custom_id for card in runnable]
    log(
        f"mrph> [generation {index + 1}/{total}] submitting "
        f"{len(runnable)} card(s): {', '.join(ids)}"
    )
    requests: List[dict] = []
    for card in runnable:
        requests.extend(compile_card(card, root))
    batch_id = backend.submit(requests)

    state["phase"] = PHASE_SUBMITTED
    state["generation_index"] = index
    state["batch_id"] = batch_id
    state["backend_label"] = backend_label
    state["submitted_ids"] = ids
    _store_outcomes(state, outcomes)
    store.save_state(state)

    return SubmitResult(
        submitted=True, done=False, generation_number=index + 1,
        total_generations=total, batch_id=batch_id, card_ids=ids,
        skipped=skipped_here)


def collect_generation(
    store: DeckStore,
    backend,
    root: str = ".",
    verify: bool = True,
    acceptance_timeout: float = 300.0,
    max_regenerations: int = 2,
    poll_interval: float = 1.0,
    log: Callable[[str], None] = print,
) -> CollectResult:
    """Poll the in-flight batch once; if finished, process it and advance.

    ``backend.status`` is polled a SINGLE time. If the batch is still running the
    call returns immediately with ``in_progress`` true and changes nothing -- the
    CLI reports it and the user re-runs ``/collect`` later (``LocalBatchBackend``
    briefly blocks inside ``collect`` while its worker threads drain; that is
    fine). Once the batch has ended, its results are handed to
    :func:`cards.generations.process_generation` -- identical best-of-N,
    rollback, and inline-retry semantics to ``run_deck`` -- the outcomes are
    persisted, and the run advances to the next generation (phase ``"idle"``) or
    finishes (phase ``"done"``).

    Raises :class:`StoreError` if nothing is in flight (``phase !=
    "submitted"``). ``backend`` is duck-typed: ``status``, ``collect`` and (for
    retries) ``submit`` are called.
    """
    state = store.load_state()
    if state.get("phase") != PHASE_SUBMITTED:
        raise StoreError(
            "nothing is in flight; run /submit before /collect "
            "(or /deck reset to discard the run state)")

    batch_id = state["batch_id"]
    total = len(state["generations"])
    index = state.get("generation_index", 0)

    status = backend.status(batch_id)
    if status not in ("completed", "failed"):
        return CollectResult(
            in_progress=True, generation_number=index + 1,
            total_generations=total, phase=PHASE_SUBMITTED)

    cards = store.load_cards()
    by_id = {card.custom_id: card for card in cards}
    runnable = [by_id[cid] for cid in state.get("submitted_ids", []) if cid in by_id]

    outcomes = _outcomes_from_state(state)
    blocked = _blocked_from_outcomes(outcomes)

    results = None if status == "failed" else backend.collect(batch_id)

    process_generation(
        runnable, results, index + 1, total, root, backend, poll_interval, log,
        verify, acceptance_timeout, max_regenerations, outcomes, blocked)

    next_index = index + 1
    state["phase"] = PHASE_DONE if next_index >= total else PHASE_IDLE
    state["generation_index"] = next_index
    state["batch_id"] = None
    state["submitted_ids"] = []
    _store_outcomes(state, outcomes)
    store.save_state(state)

    reported = {card.custom_id: outcomes[card.custom_id] for card in runnable}
    return CollectResult(
        in_progress=False, generation_number=index + 1, total_generations=total,
        phase=state["phase"], outcomes=reported)
