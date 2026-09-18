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

REGENERATIONS ARE PERSISTED, NOT POLLED INLINE. A card that fails acceptance is
resubmitted; ``run_deck`` polls that retry batch inside the same call, which is
right for ``/nightly`` (one blocking pass is its whole contract) and was wrong
for ``/collect``, measurably so: one ``/collect`` sat silent for an hour across
three sequential 20-minute retry batches; a second CLI session polling the same
deck submitted its OWN retry for the same card, so four paid batches existed at
once; and a process that died mid-retry orphaned a paid batch nobody could
collect. So ``collect_generation`` submits ONE retry batch, records it in
``state["retries"]`` (keyed by the card's ORIGINAL custom_id) and RETURNS. A
later call finds that record and polls THAT batch -- never submits a
replacement -- so ``/collect`` is idempotent: running it ten times while a
regeneration is in flight costs nothing and changes nothing, and a fresh session
picks the batch up from ``state.json`` after a restart.

PHASE 7 PUTS THE RUN ON A BRANCH AND KEEPS IT. Two additions, both built on
:mod:`cards.repo` (which is inside ``cards``, so the layering constraint below
still holds): a run that starts in a git working copy opens ``morph/<deck-id>``
and turns each accepted card into one commit carrying the card's provenance in
its trailers (:func:`begin_run` / :func:`_open_branch` /
:func:`make_card_committer`); and a run that FINISHES is archived under
``.morph/runs/<deck-id>/`` as the deck it executed plus a :class:`RunReport`
(:func:`archive_run`), because ``deck.json`` holds exactly one deck and the next
run overwrites it -- a deck that lives one run is not the reviewable artifact the
manifest promises. Both degrade to nothing outside a repository, and both can be
declined outright (``use_git=False``, the CLI's ``nogit``).

DESIGN NOTE (flagged for review): the split-step functions live in this module
rather than a separate one -- they are the store's reason to exist, and they need
nothing from ``processors`` or ``flows`` (the batch backend arrives as a
duck-typed parameter). ``cards`` therefore still imports nothing from
``flows``/``processors``, honouring the layering constraint.

Pure and stdlib-only. No network, no provider SDK, no imports from ``flows`` or
``processors``.
"""

import datetime
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from cards import repo
from cards.acceptance import AcceptanceResult
from cards.compiler import CompiledInputs
from cards.deck import load_deck, validate_deck
from cards.generations import (
    CardOutcome,
    REASON_COMPILE,
    build_retry_cards,
    compile_for_batch,
    process_generation,
    process_retry_batch,
    resolve_runnable,
    split_into_generations,
)
from cards.hazards import (
    check_deck,
    errors as hazard_errors,
    find_hazards,
    repair_deck,
    repairable,
    warnings as hazard_warnings,
)
from cards.schema import MorphCard


MORPH_DIR = ".morph"
DECK_FILE = "deck.json"
STATE_FILE = "state.json"
RUNS_DIR = "runs"
REPORT_FILE = "report.json"

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

# The branch a run opens, and how long an acceptance command may be in a commit
# trailer. WHY a cap: a trailer is a single line an operator reads in ``git log``
# alongside four others, and an acceptance command can be a 400-character shell
# pipeline. Cut at a readable width and say so with an ellipsis -- the full
# command is a field of the card, which the run archive keeps verbatim.
BRANCH_PREFIX = "morph/"
ACCEPTANCE_TRAILER_CAP = 120

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
    # One key or the other, never both: the schema accepts exactly one form,
    # and a changeset card must round-trip through the backlog as a changeset.
    if len(card.targets) > 1:
        meta: Dict[str, object] = {"intent": card.intent, "targets": list(card.targets)}
    else:
        meta = {"intent": card.intent, "target": card.target}
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
        # The two git-mode fields (cards.generations.CardOutcome): both None
        # outside a run that opened a branch. _outcome_from_dict reads them
        # with .get, so a state.json or report.json written before they
        # existed keeps loading.
        "commit": outcome.commit,
        "diffstat": (list(outcome.diffstat)
                     if outcome.diffstat is not None else None),
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
        # .get(..., None), never [...]: a state.json or report.json written
        # before these two fields existed carries neither key, and an old
        # record must keep loading as the not-a-commit it was.
        commit=data.get("commit", None),
        diffstat=data.get("diffstat", None),
    )


def _inputs_to_state(state: dict, inputs: Dict[str, CompiledInputs]) -> None:
    """Persist what the batch now in flight was compiled from."""
    state["inputs"] = {custom_id: captured.to_dict()
                       for custom_id, captured in inputs.items()}


def _inputs_from_state(state: dict) -> Dict[str, CompiledInputs]:
    """Read back :func:`_inputs_to_state`.

    A state file written before this record existed simply has none, and the
    staleness check is then skipped for that generation rather than failing it:
    an in-flight batch from an older version must still be collectable.
    """
    stored = state.get("inputs") or {}
    return {custom_id: CompiledInputs.from_dict(data)
            for custom_id, data in stored.items()}


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
        # One directory per finished run. ``deck.json``/``state.json`` describe
        # the CURRENT run and are overwritten by the next one; this is where a
        # run goes to survive that.
        self.runs_dir = os.path.join(self.morph_dir, RUNS_DIR)

    def run_dir(self, deck_id: str) -> str:
        """The archive directory of one run, whether or not it exists yet."""
        return os.path.join(self.runs_dir, deck_id)

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
            "retries": {},             # original custom_id -> in-flight retry dict
            # What each request of the IN-FLIGHT batch was compiled from:
            # (variant-less) custom_id -> CompiledInputs.to_dict(). Collection
            # checks it before believing an answer; see _inputs_from_state.
            "inputs": {},
            "outcomes": {},            # custom_id -> outcome dict
            "deck_id": None,           # this run's id; names its branch and archive
            "branch": None,            # the git branch the run owns, or None
            "batch_ids": [],           # every batch this run submitted, in order
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


# -- the run's identity, its branch and its commits ---------------------------


def make_deck_id(cards: List[MorphCard]) -> str:
    """A run id: a local timestamp plus a short digest of the backlog.

    Two jobs, one string. It has to be READABLE inside a branch name
    (``morph/20260917-114233-9f1c4b02``), because the branch is what an engineer
    types at ``git checkout`` a week later and reads in ``git branch`` next to
    twenty others -- so the date leads, and the list sorts chronologically on its
    own. And it has to be STABLE for the run and distinct BETWEEN runs, because
    it also names the run's archive directory, which is append-only: the digest
    separates two DIFFERENT decks run in the same second, and makes a re-run of
    an identical deck visibly a re-run of the same deck. The one case it cannot
    separate -- the same deck run twice within a second -- is what
    :func:`_unique_deck_id` exists for.

    The digest is over the serialized cards, so it changes when any card's
    instruction, target or dependency does -- the deck's identity, not the file's
    formatting.
    """
    payload = json.dumps([card_to_dict(card) for card in cards],
                         ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{digest}"


def _unique_deck_id(store: DeckStore, cards: List[MorphCard]) -> str:
    """A deck id no archived run already owns.

    The timestamp+digest pair collides only when the SAME backlog is run twice
    within one second -- a scripted re-run, which is exactly the case where
    silently reusing the id would make the archive overwrite its own history. A
    numeric suffix is cheaper than a clock we cannot trust.
    """
    base = make_deck_id(cards)
    candidate = base
    counter = 1
    while os.path.exists(store.run_dir(candidate)):
        counter += 1
        candidate = f"{base}-{counter}"
    return candidate


def _open_branch(root: str, deck_id: str, use_git: bool,
                 log: Callable[[str], None]) -> Optional[str]:
    """Put the run on ``morph/<deck-id>``, or explain why it is not on a branch.

    Returns the branch name, or ``None`` when this run does not touch git at all
    -- and ``None`` is a complete answer, not a failure: it is what makes the
    rest of the run identical to its pre-Phase-7 self. Three ways to get it, each
    announced once at run start so nobody discovers it afterwards by finding no
    commits:

    * ``use_git`` false -- the operator said so (``/submit nogit``);
    * the project is not a git working copy, or the machine has no ``git`` --
      :func:`cards.repo.is_git_repo` folds both into one ``False``;
    * nothing else. A DIRTY tree is the one case that refuses instead: starting
      there would mix the run's commits with the engineer's uncommitted work, and
      the phase's promise -- ``git checkout`` undoes the run -- would quietly
      stop holding. The message names both ways out, because "commit or stash"
      is not the only honest answer when the engineer means to keep the mess.
    """
    if not use_git:
        log("mrph> git: off for this run -- no branch, no commits "
            "(the morphs still land in the working tree).")
        return None
    if not repo.is_git_repo(root):
        log("mrph> git: not a git working copy (or no git on PATH) -- the run "
            "writes straight into the tree, with no branch and no per-card "
            "commits.")
        return None

    try:
        # ``.morph/`` is the orchestrator's own bookkeeping, and ``/card`` wrote
        # to it moments ago: counting it as the engineer's unfinished work would
        # refuse every run that had just been planned.
        if repo.is_dirty(root, exclude=(MORPH_DIR,)):
            raise StoreError(
                "the working tree has uncommitted changes, and a run started "
                "here could not be undone with git checkout. Commit or stash "
                "them (git stash -u), or start the run with \"nogit\" "
                "(/submit nogit, /nightly nogit) to leave git alone entirely.")
        name = BRANCH_PREFIX + deck_id
        previous = repo.current_branch(root)
        repo.create_branch(root, name)
    except repo.GitError as error:
        # git refused something we asked for (an existing branch, a wedged
        # index). That is a run that cannot start, not a run that starts without
        # provenance: silently degrading here would produce commits nobody asked
        # for on a branch nobody expects.
        raise StoreError(f"git: {error}")

    log(f"mrph> git: the run is on branch {name!r}"
        f"{' (branched off ' + previous + ')' if previous else ''}; "
        f"each accepted card becomes one commit.")
    return name


def _cap(text: str, limit: int) -> str:
    """One line, at most ``limit`` characters, with the cut made visible."""
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[:limit - 3].rstrip() + "..."


def _commit_subject(card: MorphCard) -> str:
    """The commit's first line: which card, and what it wrote.

    The card's own ``targets`` rather than ``outcome.paths``: the targets are
    written the way the card spells them (repository-relative), while the paths
    are joined onto the run's root and would put a tempdir prefix in the history
    of every test run.
    """
    return f"morph {card.custom_id}: {', '.join(card.targets)}"


def _commit_trailers(card: MorphCard, outcome: CardOutcome,
                     backend_label: Optional[str]) -> Dict[str, str]:
    """The provenance footer of one card's commit.

    This is the phase's whole point: after this, ``git log`` answers "where did
    this line come from" -- which card asked for it, which model wrote it, which
    of N variants won, and what command was run to accept it -- without anyone
    having kept ``.morph/state.json``.

    ``Morph-Model`` falls back to the run's backend label when the card pins no
    model of its own, because "the model that produced it" is a fact about the
    run, and a card without a ``model`` field was produced by whatever processor
    the run was fired on. ``Morph-Acceptance-Exit`` is ``0`` whenever an
    acceptance command ran: a card is only ever accepted through
    :class:`cards.acceptance.AcceptanceResult` with ``passed`` true, and that is
    exit 0 by construction -- the trailer records that the command RAN and
    returned cleanly, which is the part ``git log`` cannot otherwise show.

    ``Morph-Variant`` appears only for a card that actually HELD a contest
    (``variants > 1``). A single-variant card's "winning variant" is its own
    custom_id, which ``Morph-Card`` already says: a trailer that is always there
    and never carries information is one an operator stops reading, taking the
    informative ones with it. Empty values drop out in :mod:`cards.repo`.
    """
    trailers: Dict[str, str] = {
        "Morph-Card": card.custom_id,
        "Morph-Model": card.model or backend_label or "",
    }
    if card.variants > 1:
        trailers["Morph-Variant"] = outcome.winning_variant or ""
    if card.acceptance:
        trailers["Morph-Acceptance"] = _cap(card.acceptance, ACCEPTANCE_TRAILER_CAP)
        trailers["Morph-Acceptance-Exit"] = "0"
    return trailers


def make_card_committer(
    root: str,
    state: dict,
    log: Callable[[str], None] = print,
) -> Optional[Callable[[MorphCard, CardOutcome], None]]:
    """The :data:`cards.generations.AcceptedHook` that turns a card into a commit.

    Returns ``None`` when the run holds no branch -- not a git repository, or the
    operator opted out -- and a ``None`` hook is what keeps those runs
    byte-identical to their pre-Phase-7 behaviour.

    The returned hook NEVER RAISES. A card whose files are already on disk and
    accepted must not be undone by git having an opinion (a ``pre-commit`` hook
    that rejects the change, an unconfigured identity, a path the deck pointed
    outside the repository): the failure is reported on the spot, naming the card
    and carrying git's own message, and the run carries on. Each commit stages
    only its own card's paths, so one refusal does not contaminate the next
    card's commit either.

    A commit that turns out empty is the ordinary case, not an error: a morph
    that rewrote a file byte-identically changed nothing, and
    :func:`cards.repo.commit_paths` answers ``None`` for it rather than putting a
    provenance record in the history of a line that was never generated.

    A commit that SUCCEEDS is also written onto the outcome it was handed:
    ``commit`` takes the sha and ``diffstat`` what that commit changed, read
    back out of git (:func:`cards.repo.diffstat`). No plumbing carries them
    anywhere: both run routes save their outcomes only after this hook has
    fired, so setting the fields here is what puts them into ``state.json`` and
    the run report. Reading the diffstat can fail where committing succeeded,
    and that failure is caught: both fields stay ``None`` (a sha without its
    measurement is not recorded as provenance), one line says the commit stands
    anyway, and the hook still does not raise.
    """
    branch = state.get("branch")
    if not branch:
        return None
    backend_label = state.get("backend_label")

    def commit(card: MorphCard, outcome: CardOutcome) -> None:
        if outcome.status != STATUS_WRITTEN or not outcome.paths:
            return
        try:
            sha = repo.commit_paths(
                root, outcome.paths,
                _commit_subject(card),
                _commit_trailers(card, outcome, backend_label))
        except repo.GitError as error:
            log(f"mrph> git: {card.custom_id!r} is written but NOT committed: "
                f"{error}")
            return
        if sha is None:
            log(f"mrph> git: {card.custom_id!r} changed nothing on disk -- "
                f"no commit.")
        else:
            # The commit's footprint, read back out of git and carried on the
            # outcome into state.json and the run report. Computed BEFORE either
            # field is set: a failure here leaves both None -- a sha whose
            # diffstat could not be read is recorded as no commit at all, which
            # is the honest half-record -- and the one log line still says the
            # commit itself went through.
            try:
                changed = repo.diffstat(root, sha)
            except repo.GitError as error:
                log(f"mrph> git: {card.custom_id!r} committed as {sha[:10]}; "
                    f"its diffstat could not be read: {error}")
                return
            outcome.commit = sha
            outcome.diffstat = changed
            log(f"mrph> git: {card.custom_id!r} committed as {sha[:10]}.")

    return commit


def preflight_deck(
    store: DeckStore,
    cards: List[MorphCard],
    log: Callable[[str], None] = print,
    strict: bool = False,
) -> List[MorphCard]:
    """Check file ownership before a run spends anything; repair what it can.

    The gate every route into a run passes through. It runs
    :func:`cards.hazards.find_hazards` over the deck as it stands and then:

    * **repairs** the read/write hazards (:func:`cards.hazards.repair_deck`) by
      adding the missing ``depends_on`` edges, which serialize a reader after
      the card that writes what it reads. Each edge is logged and the repaired
      deck is SAVED to the backlog -- the deck as executed has to be the deck on
      disk, or the composition locked into the run state would describe cards
      that ``.morph/deck.json`` does not contain, and the run archive would
      record a deck that never ran;
    * **refuses** the deck if anything is left that cannot be repaired -- two
      cards of one generation writing one file, or a pair whose serialization
      would close a cycle -- by raising :class:`cards.hazards.HazardError`
      (a :class:`cards.deck.DeckError`, so every caller already reports it);
    * **warns**, on the way out, about the hazards that are real but not fatal:
      an empty slice reading a sibling's target, a cross-generation read with no
      edge to carry the failure cascade. ``strict`` turns those into refusals
      too, for an operator who wants the machine hard-line.

    Returns the cards to run -- the repaired list when anything was repaired,
    the input list otherwise. Raises before anything is opened, submitted,
    written or repaired, so a refused deck costs nothing and comes back exactly
    as its author left it: a run that will not start must not also have edited
    the backlog on its way to saying so.
    """
    hazards = find_hazards(cards)
    fixable = set(repairable(hazards))
    blocking = [hazard for hazard in hazard_errors(hazards)
                if hazard not in fixable]
    if blocking or (strict and hazard_warnings(hazards)):
        # Refused as authored. check_deck raises with the whole report -- the
        # repairable hazards included, because an operator fixing a deck wants
        # everything that is wrong with it, not the subset that stopped it.
        check_deck(cards, strict=strict)

    repaired, edges = repair_deck(cards, hazards)
    if edges:
        for hazard in edges:
            log(f"mrph> deck repair: {hazard.right!r} now depends on "
                f"{hazard.left!r} -- it reads {', '.join(hazard.paths)}, which "
                f"{hazard.left!r} writes. They run in consecutive generations.")
        store._save_cards(repaired)
        log(f"mrph> deck repair: {len(edges)} dependency edge(s) added to the "
            f"backlog; the deck runs in the repaired order.")
    for hazard in check_deck(repaired, strict=strict):
        log(f"mrph> deck warning: {hazard.message()}")
    return repaired


def begin_run(
    store: DeckStore,
    cards: List[MorphCard],
    root: str = ".",
    use_git: bool = True,
    backend_label: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Start a whole-deck run: lock the composition, mint the id, open the branch.

    The ``/nightly`` counterpart of the first ``/submit``. ``run_deck`` executes
    a deck in one blocking call, so the branch has to exist BEFORE that call --
    otherwise the first card's morph lands on whatever branch happened to be
    checked out. This writes the run state ``record_run`` will later complete,
    and returns it so the caller can build the commit hook from it.

    Raises :class:`StoreError` on a dirty working tree (see :func:`_open_branch`);
    nothing is written when it does. Raises
    :class:`cards.hazards.HazardError` on a deck whose cards contend for a file
    -- checked here as well as in :func:`preflight_deck`, because this function
    is a public way into a run and a gate with a way around it is not a gate.
    Callers that want the repairable hazards REPAIRED (every CLI route does)
    call :func:`preflight_deck` first and pass its cards here -- and the check
    below then passes silently, which is why it does not log the warnings a
    second time: the caller that repaired the deck has already reported them.
    """
    check_deck(cards)
    state = DeckStore._default_state()
    state["generations"] = [
        [card.custom_id for card in generation]
        for generation in split_into_generations(cards)
    ]
    state["backend_label"] = backend_label
    state["deck_id"] = _unique_deck_id(store, cards)
    state["branch"] = _open_branch(root, state["deck_id"], use_git, log)
    store.save_state(state)
    return state


# -- the run archive ----------------------------------------------------------


@dataclass
class RunReport:
    """What one finished run did, as archived under ``.morph/runs/<deck-id>/``.

    WHY this exists at all: ``.morph/deck.json`` held exactly one deck and every
    new run overwrote it, so yesterday's three-card patch deck was simply gone
    and the only surviving deck of a week was one someone had copied out by hand.
    The manifest calls the deck a reviewable artifact; an artifact that lives one
    run is not one.

    It is deliberately a VIEW over the facts the run already produced, not a
    second shape for them: ``outcomes`` holds real :class:`CardOutcome` objects
    (serialized by the same ``_outcome_to_dict`` the run state uses) and the
    archived deck is written by the same ``card_to_dict``. So a report read back
    is the same data the run worked with, and there is only ever one definition
    of "what happened to this card" to keep correct.
    """

    deck_id: str
    completed_at: str
    branch: Optional[str] = None
    backend_label: Optional[str] = None
    generations: List[List[str]] = field(default_factory=list)
    batch_ids: List[str] = field(default_factory=list)
    outcomes: Dict[str, CardOutcome] = field(default_factory=dict)

    @property
    def counts(self) -> Dict[str, int]:
        """How many cards ended written / failed / skipped."""
        tally = {STATUS_WRITTEN: 0, STATUS_FAILED: 0, STATUS_SKIPPED: 0}
        for outcome in self.outcomes.values():
            if outcome.status in tally:
                tally[outcome.status] += 1
        return tally

    def to_dict(self) -> dict:
        return {
            "deck_id": self.deck_id,
            "completed_at": self.completed_at,
            "branch": self.branch,
            "backend_label": self.backend_label,
            "generations": [list(generation) for generation in self.generations],
            "batch_ids": list(self.batch_ids),
            "outcomes": {cid: _outcome_to_dict(outcome)
                         for cid, outcome in self.outcomes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunReport":
        return cls(
            deck_id=data.get("deck_id", ""),
            completed_at=data.get("completed_at", ""),
            branch=data.get("branch"),
            backend_label=data.get("backend_label"),
            generations=[list(generation)
                         for generation in data.get("generations", [])],
            batch_ids=list(data.get("batch_ids", [])),
            outcomes={cid: _outcome_from_dict(entry)
                      for cid, entry in data.get("outcomes", {}).items()},
        )


def _completed_at() -> str:
    """The moment a run finished, to the millisecond.

    Millisecond precision is not decoration: it is what orders two runs of the
    same deck started within one second of each other, which the deck id (a
    timestamp to the second, then a CONTENT digest) cannot do -- their ids sort
    by digest, which is to say arbitrarily. :func:`list_runs` sorts on this.
    """
    return datetime.datetime.now().isoformat(timespec="milliseconds")


def _report_from_state(state: dict, deck_id: str) -> RunReport:
    return RunReport(
        deck_id=deck_id,
        completed_at=_completed_at(),
        branch=state.get("branch"),
        backend_label=state.get("backend_label"),
        generations=[list(generation) for generation in state.get("generations", [])],
        batch_ids=list(state.get("batch_ids") or []),
        outcomes=_outcomes_from_state(state),
    )


def _write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def archive_run(
    store: DeckStore,
    state: dict,
    cards: List[MorphCard],
    root: str = ".",
    log: Callable[[str], None] = print,
) -> Optional[str]:
    """Copy a finished run into ``.morph/runs/<deck-id>/`` and return that path.

    Two files: ``deck.json`` -- the cards AS EXECUTED, in deck order, filtered to
    the composition this run locked in (a card appended to the backlog after the
    run started was never part of it) -- and ``report.json``, a
    :class:`RunReport`. Together they answer the two questions a review asks a
    week later: what was asked, and what happened.

    APPEND-ONLY. An existing directory is left exactly as it is and ``None`` is
    returned: a deck id names one run forever, and the value of the archive is
    entirely in yesterday's run still being yesterday's run.

    When the run owns a branch, the two files are committed as its FINAL commit,
    so "what was asked" and "what happened" live in the same history as the code
    they produced. A repository that ignores ``.morph/`` refuses that commit, and
    that refusal is reported, not overridden: the ignore is the engineer's own
    instruction, the archive is on disk either way, and a project that wants its
    run records in git says so with a ``!.morph/runs/`` line.
    """
    deck_id = state.get("deck_id") or _unique_deck_id(store, cards)
    state["deck_id"] = deck_id
    directory = store.run_dir(deck_id)
    if os.path.exists(directory):
        return None
    report = _report_from_state(state, deck_id)

    composition = {cid for generation in report.generations for cid in generation}
    executed = [card for card in cards if card.custom_id in composition]

    os.makedirs(directory)
    deck_path = os.path.join(directory, DECK_FILE)
    report_path = os.path.join(directory, REPORT_FILE)
    _write_json(deck_path, [card_to_dict(card) for card in executed])
    _write_json(report_path, report.to_dict())

    if report.branch:
        counts = report.counts
        try:
            sha = repo.commit_paths(
                root, [deck_path, report_path],
                f"morph run {deck_id}: deck and report",
                {
                    "Morph-Run": deck_id,
                    "Morph-Cards": str(len(executed)),
                    "Morph-Written": str(counts[STATUS_WRITTEN]),
                    "Morph-Failed": str(counts[STATUS_FAILED]),
                    "Morph-Skipped": str(counts[STATUS_SKIPPED]),
                },
            )
        except repo.GitError as error:
            log(f"mrph> git: the run archive is on disk but not committed: "
                f"{error}")
        else:
            if sha:
                log(f"mrph> git: run archive committed as {sha[:10]}.")
    return directory


def list_runs(store: DeckStore) -> List[RunReport]:
    """Every archived run, newest first.

    Sorted by ``completed_at`` and then ``deck_id``, descending. Not by the
    directory name alone, tempting as that is: a deck id is a timestamp to the
    second followed by a digest of the deck's CONTENT, so two runs of different
    decks started in the same second sort by digest -- which is to say
    arbitrarily. ``completed_at`` carries milliseconds for exactly this.

    A directory without a readable ``report.json`` is skipped rather than raised
    on -- the archive is a record, and a half-written one from a killed process
    must not stop the operator seeing the other twenty.
    """
    if not os.path.isdir(store.runs_dir):
        return []
    reports: List[RunReport] = []
    for deck_id in sorted(os.listdir(store.runs_dir)):
        path = os.path.join(store.runs_dir, deck_id, REPORT_FILE)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        report = RunReport.from_dict(data)
        if not report.deck_id:
            report.deck_id = deck_id
        reports.append(report)
    reports.sort(key=lambda report: (report.completed_at, report.deck_id),
                 reverse=True)
    return reports


# -- whole-run persistence and recovery --------------------------------------


def record_run(
    store: DeckStore,
    result,
    backend_label: Optional[str] = None,
    root: str = ".",
    run_state: Optional[dict] = None,
    log: Callable[[str], None] = print,
) -> None:
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

    ``run_state`` is what :func:`begin_run` returned for THIS run -- its deck id
    and its branch. It is passed explicitly rather than read back from
    ``state.json`` because inheriting an id is exactly the mistake the archive
    cannot survive: a nightly pass that adopted the previous run's deck id would
    find that run's directory already there and archive nothing. Without it the
    run is archived under a freshly minted id and owns no branch, which is what a
    caller that never opened one wants.

    Finally the run is ARCHIVED (:func:`archive_run`): the state file describes
    only the latest run, and a deck the next run overwrites is not the reviewable
    artifact the manifest promises.
    """
    state = DeckStore._default_state()
    state["phase"] = PHASE_DONE
    state["generations"] = [list(generation) for generation in result.generations]
    state["generation_index"] = len(result.generations)
    state["backend_label"] = backend_label
    state["batch_ids"] = list(getattr(result, "batch_ids", []) or [])
    if run_state:
        state["deck_id"] = run_state.get("deck_id")
        state["branch"] = run_state.get("branch")
    _store_outcomes(state, result.outcomes)
    # Archive first: it settles ``deck_id`` (minting one when the caller opened
    # no run), and that id belongs in the state file the next ``/deck`` reads.
    archive_run(store, state, store.load_cards(), root=root, log=log)
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
    and collectable later, so it is never touched here. That includes a local
    REGENERATION batch (``state["retries"]``): its worker threads died with the
    process exactly as a first-attempt batch's do, so the record is dropped too
    and the card goes back to the attempt it was pending -- otherwise the deck
    wedges on a batch that can never report.

    Returns ``True`` when something was recovered. The generation composition and
    every recorded outcome survive -- only the in-flight bookkeeping (phase,
    batch id, backend label, submitted ids, retries) is cleared, so the run
    resumes at the same generation on the next ``/submit``, which re-sends just
    the cards of that generation that have no outcome yet (see
    :func:`submit_generation`). Callers must only invoke this when the session
    holds no live backend for the batch.
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
    state["retries"] = {}
    # Including what that batch was compiled from: nothing is in flight to check
    # it against any more, and the re-send writes its own record.
    state["inputs"] = {}
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

    ``in_progress`` true means the generation is not collected yet and the caller
    should run ``/collect`` again. Two things look like that: the batch had not
    finished (nothing changed at all), or a card failed acceptance and this call
    submitted its regeneration (``retry_in_flight``), which IS a change -- the
    passing cards' outcomes and the new batch id are persisted -- but not a
    finished generation. Otherwise ``outcomes`` carries the
    :class:`CardOutcome` for every card in the collected generation (runnable +
    any skipped while advancing), and ``phase`` is the new run phase (``"idle"``
    when a further generation remains, ``"done"`` when the deck is finished).

    The ``retry_*`` fields describe the regeneration the run is waiting on (set
    whenever ``retry_in_flight``), so the CLI can say WHICH cards are being
    regenerated, on which attempt of how many, and in which batch.
    ``retry_submitted`` separates the call that SENT that batch (this ``/collect``
    just spent money and has news: those cards failed acceptance) from the calls
    that merely found it still running (which spend and change nothing).
    """

    in_progress: bool
    generation_number: int
    total_generations: int
    phase: str
    outcomes: Dict[str, CardOutcome] = field(default_factory=dict)
    retry_in_flight: bool = False
    retry_submitted: bool = False
    retry_attempt: int = 0
    retry_limit: int = 0
    retry_card_ids: List[str] = field(default_factory=list)
    retry_batch_id: Optional[str] = None


def _ensure_run_started(
    store: DeckStore,
    state: dict,
    cards: List[MorphCard],
    root: str = ".",
    use_git: bool = True,
    log: Callable[[str], None] = print,
) -> List[MorphCard]:
    """Lock in the generation composition on the first submit of a fresh run.

    The composition is computed once, from the backlog as it stands at the first
    ``/submit``, and persisted -- later generations are compiled against the
    files earlier ones write, but which cards live in which generation is fixed
    for the run.

    This is also where the RUN begins for git: the same moment that fixes what
    the run is gives it its id and its branch (:func:`_open_branch`). A later
    ``/submit`` in the same run finds a composition already there and touches
    neither -- which is why ``use_git`` is only ever consulted here, and why
    opting out halfway through a run is not a thing that can happen.

    Raises :class:`StoreError` on a dirty working tree, and
    :class:`cards.hazards.HazardError` on a deck whose cards contend for a file
    -- both before anything has been compiled, submitted or saved.

    Returns the cards the run is to use: the first submit passes the backlog
    through :func:`preflight_deck`, which may add dependency edges, and the
    composition is then locked in from THAT deck. A later submit finds a
    composition already there and returns the cards it was given untouched.
    """
    if not state.get("generations"):
        cards = preflight_deck(store, cards, log=log)
        state["generations"] = [
            [card.custom_id for card in generation]
            for generation in split_into_generations(cards)
        ]
        state["generation_index"] = 0
        state["outcomes"] = {}
        state["batch_ids"] = []
        state["deck_id"] = _unique_deck_id(store, cards)
        state["branch"] = _open_branch(root, state["deck_id"], use_git, log)
    return cards


def submit_generation(
    store: DeckStore,
    backend,
    root: str = ".",
    backend_label: Optional[str] = None,
    log: Callable[[str], None] = print,
    use_git: bool = True,
) -> SubmitResult:
    """Compile and submit the current generation's runnable cards.

    On the first call of a run the generation composition is locked in from the
    backlog. Cards whose dependency already failed/was skipped are recorded as
    ``skipped`` (and a card of this generation that already has an outcome is
    left alone -- see the loop below) and the loop advances to the next
    generation that has runnable cards; that generation is compiled (reading the fresh files earlier
    generations wrote) and submitted as one batch. The batch id and the runnable
    ids are saved and the phase moves to ``"submitted"``.

    Raises :class:`StoreError` if a batch is already in flight (``phase ==
    "submitted"``) or the run is already ``"done"``. When every remaining
    generation is skippable (or the deck is empty), no batch is sent, the phase
    becomes ``"done"`` and ``SubmitResult.done`` is true. ``backend`` is
    duck-typed: only ``submit`` is called here.

    ``use_git`` is consulted on the FIRST submit of a run only, where the branch
    is opened (:func:`_ensure_run_started`); false leaves the working copy
    untouched by the run. A dirty tree raises :class:`StoreError` there, before
    anything is compiled or paid for.
    """
    state = store.load_state()
    phase = state.get("phase", PHASE_IDLE)
    if phase == PHASE_SUBMITTED:
        raise StoreError(
            "a generation is already submitted; collect it first "
            "(/collect in the REPL, `mrph collect` headless)")
    if phase == PHASE_DONE:
        raise StoreError(
            "the deck run is complete; discard the run state before submitting "
            "again (/deck reset in the REPL, `mrph deck reset` headless)")

    cards = store.load_cards()
    if not cards:
        # Пустой бэклог — это не прогон. Раньше ветка открывалась ДО того, как
        # выяснялось, что отправлять нечего, и каждый холостой /submit оставлял
        # в истории ветку и коммит пустого архива. ``run`` на том же входе
        # всегда отвечал честным no-op; теперь обе команды согласованы.
        return SubmitResult(
            submitted=False, done=True, generation_number=0,
            total_generations=0, batch_id=None)
    cards = _ensure_run_started(
        store, state, cards, root=root, use_git=use_git, log=log)
    # Persist the run's identity the moment it has one. Waiting until the batch
    # is away would leave a branch that exists on disk and nowhere in the state
    # file if compiling or submitting then failed -- and the next /submit would
    # open a SECOND branch on top of it. The phase is still "idle" here, so a
    # failure below is still a /submit that can simply be run again.
    store.save_state(state)

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
        # A card of THIS generation that already has an outcome is settled --
        # written or failed -- and must not be sent again. It only happens after
        # a recovery (:func:`recover_orphaned_local_batch` returns a lost local
        # regeneration to pending while its generation-mates keep their
        # outcomes); re-sending them would pay for morphs already on disk and
        # overwrite accepted files.
        runnable = [card for card in runnable if card.custom_id not in before]
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
        # A run can end HERE as well as in /collect -- every remaining card
        # skipped behind a failure. It is still a run, and a run that ended
        # badly is the one most worth having archived.
        archive_run(store, state, cards, root=root, log=log)
        store.save_state(state)
        return SubmitResult(
            submitted=False, done=True, generation_number=0,
            total_generations=total, batch_id=None, skipped=skipped_here)

    ids = [card.custom_id for card in runnable]
    log(
        f"mrph> [generation {index + 1}/{total}] submitting "
        f"{len(runnable)} card(s): {', '.join(ids)}"
    )
    # Compiling reads the project and can fail on one card (a slice naming a
    # file that is not there is the ordinary case). That card fails; the rest of
    # the generation still goes out. See ``cards.generations.compile_for_batch``
    # for why this is not allowed to take the run with it.
    requests, inputs, compiled, failures = compile_for_batch(runnable, root)
    for custom_id, error in failures:
        log(f"mrph> [generation {index + 1}/{total}] {custom_id!r} could not be "
            f"compiled and was NOT submitted: {error}")
        outcomes[custom_id] = CardOutcome(
            custom_id, "failed", reason=REASON_COMPILE, acceptance_output=error)
        blocked.add(custom_id)

    if not compiled:
        # Nothing survived compilation: the generation is settled without a
        # batch. Advance so the next /submit moves on (the failures block their
        # dependents, which resolve_runnable will skip) and finish the run if
        # this was the last generation -- an empty batch must never be sent.
        state["generation_index"] = index + 1
        state["phase"] = PHASE_DONE if index + 1 >= total else PHASE_IDLE
        state["batch_id"] = None
        state["submitted_ids"] = []
        state["inputs"] = {}
        _store_outcomes(state, outcomes)
        if state["phase"] == PHASE_DONE:
            archive_run(store, state, cards, root=root, log=log)
        store.save_state(state)
        return SubmitResult(
            submitted=False, done=state["phase"] == PHASE_DONE,
            generation_number=index + 1, total_generations=total, batch_id=None,
            skipped=skipped_here)

    ids = [card.custom_id for card in compiled]
    batch_id = backend.submit(requests)

    state["phase"] = PHASE_SUBMITTED
    state["generation_index"] = index
    state["batch_id"] = batch_id
    state["backend_label"] = backend_label
    state["submitted_ids"] = ids
    state["batch_ids"] = list(state.get("batch_ids") or []) + [batch_id]
    _inputs_to_state(state, inputs)
    _store_outcomes(state, outcomes)
    store.save_state(state)

    return SubmitResult(
        submitted=True, done=False, generation_number=index + 1,
        total_generations=total, batch_id=batch_id, card_ids=ids,
        skipped=skipped_here)


def _pending_from_retries(
    retries: dict, by_id: Dict[str, MorphCard]
) -> Tuple[List[tuple], int]:
    """Rebuild a recorded regeneration into ``(pending, attempt)``.

    ``pending`` is the ``(card, previous AcceptanceResult or None)`` shape
    :func:`cards.generations.process_retry_batch` judges -- the card read back
    from the backlog, the error context read back from the record (as a stand-in
    :class:`AcceptanceResult`, since only its ``output`` is ever used: it is what
    the NEXT attempt's instruction would quote). A recorded card that has since
    left the backlog is dropped, the same way every other id read from state is
    filtered through ``by_id``.

    ``attempt`` is the attempt number the recorded batch IS (1 = first
    regeneration). By construction every entry of one record shares it -- they
    were submitted together -- so the maximum is that number and is also correct
    for a record hand-edited into disagreement.
    """
    pending: List[tuple] = []
    attempt = 0
    for custom_id, entry in retries.items():
        attempt = max(attempt, int(entry.get("attempt", 1)))
        card = by_id.get(custom_id)
        if card is None:
            continue
        output = entry.get("acceptance_output")
        previous = None if output is None else AcceptanceResult(
            passed=False, exit_code=entry.get("exit_code"), output=output,
            timed_out=bool(entry.get("timed_out")))
        pending.append((card, previous))
    return pending, attempt


def _record_retry(
    store: DeckStore,
    state: dict,
    pending: List[tuple],
    attempt: int,
    batch_id: str,
    outcomes: Dict[str, CardOutcome],
    inputs: Optional[Dict[str, CompiledInputs]] = None,
) -> None:
    """Persist a just-submitted regeneration batch as the run's in-flight work.

    The phase stays ``"submitted"`` and ``generation_index`` does not move -- the
    generation is not collected until every card of it is settled -- but
    ``batch_id`` now points at the RETRY batch, so ``/deck`` names what is
    actually in a queue, a restarted session polls the right thing, and
    :func:`recover_orphaned_local_batch` sees a ``local-`` id and can free a
    local retry that died with its process. ``submitted_ids`` is deliberately
    left as the whole generation's runnable list: the cards that already passed
    now have outcomes (so ``/deck`` shows them written), the regenerating ones do
    not (so ``/deck`` shows them in flight), and the final ``CollectResult``
    still reports the generation entire.
    """
    state["batch_id"] = batch_id
    state["batch_ids"] = list(state.get("batch_ids") or []) + [batch_id]
    state["retries"] = {
        card.custom_id: {
            "attempt": attempt,
            "batch_id": batch_id,
            "backend_label": state.get("backend_label"),
            # The previous attempt's acceptance output, so a regeneration that
            # outlives the process still knows what to tell the executor.
            "acceptance_output": previous.output if previous is not None else None,
            # ...and whether a command actually RAN for it. Without this a
            # failure read back after a restart cannot be told from an answer
            # that was rejected unread, and the next attempt's prompt would
            # blame the executor for a test that was never executed
            # (``cards.generations._retry_card``).
            "exit_code": previous.exit_code if previous is not None else None,
            "timed_out": bool(previous.timed_out) if previous is not None else False,
        }
        for card, previous in pending
    }
    # The retry batch is now the in-flight one, so its compiled inputs replace
    # the generation's: state["inputs"] always describes what is in a queue.
    _inputs_to_state(state, inputs or {})
    _store_outcomes(state, outcomes)
    store.save_state(state)


def _log_provider_rejection(backend, batch_id: str,
                            log: Callable[[str], None]) -> None:
    """Say why the provider rejected a batch, when the backend can say.

    A batch that polls as ``"failed"`` reaches the rest of ``collect_generation``
    as ``results = None``: every card of the generation is recorded failed, and
    until now the log said nothing about why -- the reason sat on the batch
    object the provider returned, readable only by hand
    (:meth:`OpenRouterBatchBackend.failure_reason` surfaces it, e.g. ``HTTP 400:
    invalid batch inference job: job-submission-count for account alex-79b5d6,
    in use: 16, quota: 16``). This asks the backend for that message and logs it
    once, naming the batch.

    Deliberately defensive at both ends, because this is a diagnosis and must
    never become a second way for collection to fail: the method is looked up
    with :func:`getattr`, so the local, OpenAI and Anthropic backends -- which
    have no ``failure_reason`` -- are skipped without a word; a backend whose
    ``failure_reason`` raises is treated as one with nothing to say rather than
    being allowed to break the collection that already succeeded in polling; and
    a ``None``/empty answer prints nothing, because "no reason available" is not
    a diagnosis.
    """
    ask = getattr(backend, "failure_reason", None)
    if ask is None:
        return
    try:
        message = ask(batch_id)
    except Exception:
        return
    if message:
        log(f"mrph> the provider rejected batch {batch_id}: {message}")


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

    ``backend.status`` is polled a SINGLE time, for whichever batch the run is
    waiting on -- the generation's own, or a regeneration submitted by an earlier
    ``/collect``. If it is still running the call returns immediately with
    ``in_progress`` true and changes nothing; the CLI reports it and the user
    re-runs ``/collect`` later (``LocalBatchBackend`` briefly blocks inside
    ``collect`` while its worker threads drain; that is fine).

    Once the batch has ended its results are judged -- a first batch by
    :func:`cards.generations.process_generation` (identical best-of-N and
    rollback semantics to ``run_deck``), a regeneration by
    :func:`cards.generations.process_retry_batch` -- and then:

    * every card settled: the outcomes are persisted and the run advances to the
      next generation (phase ``"idle"``) or finishes (phase ``"done"``);
    * a card failed acceptance with regenerations left: ONE retry batch is
      submitted and RECORDED (:func:`_record_retry`), and the call returns with
      ``in_progress`` and ``retry_in_flight`` true. It is NOT polled here -- that
      is the difference from ``run_deck``, and the whole point: a second
      ``/collect`` (in this session or a later one) finds the record and polls
      that same batch instead of paying for a second one.

    Attempt counting is unchanged: ``max_regenerations`` (default 2) allows at
    most 3 attempts in total, and the final :class:`CardOutcome` is the one
    ``run_deck`` would have recorded.

    Every card accepted along the way becomes one commit on the run's branch
    (:func:`make_card_committer`), and the ``/collect`` that finishes the deck
    also archives it (:func:`archive_run`) -- both are no-ops for a run that
    holds no branch, which is what keeps a non-git project's behaviour exactly
    what it was.

    Raises :class:`StoreError` if nothing is in flight (``phase !=
    "submitted"``). ``backend`` is duck-typed: ``status``, ``collect`` and (for
    retries) ``submit`` are called.
    """
    state = store.load_state()
    if state.get("phase") != PHASE_SUBMITTED:
        raise StoreError(
            "nothing is in flight; submit a generation first, or discard the "
            "run state (/submit and /deck reset in the REPL, `mrph submit` and "
            "`mrph deck reset` headless)")

    batch_id = state["batch_id"]
    total = len(state["generations"])
    index = state.get("generation_index", 0)
    retries = state.get("retries") or {}

    status = backend.status(batch_id)
    if status not in ("completed", "failed"):
        # Nothing is written and nothing is submitted -- this is the call that
        # must stay free, however many times a polling loop makes it.
        return CollectResult(
            in_progress=True, generation_number=index + 1,
            total_generations=total, phase=PHASE_SUBMITTED,
            retry_in_flight=bool(retries),
            retry_attempt=max((int(entry.get("attempt", 1))
                               for entry in retries.values()), default=0),
            retry_limit=max_regenerations,
            retry_card_ids=sorted(retries),
            retry_batch_id=batch_id if retries else None)

    if status == "failed":
        # The batch failed as a WHOLE: results will be None below and every card
        # of the generation recorded failed. The provider often said why on the
        # batch object itself; say it here, once, before the outcomes bury the
        # question. The helper is defensive -- a backend with no
        # failure_reason, or one whose failure_reason raises, must not turn a
        # diagnosis into a second way for collection to fail.
        _log_provider_rejection(backend, batch_id, log)

    cards = store.load_cards()
    by_id = {card.custom_id: card for card in cards}
    submitted_ids = [cid for cid in state.get("submitted_ids", []) if cid in by_id]

    outcomes = _outcomes_from_state(state)
    blocked = _blocked_from_outcomes(outcomes)

    results = None if status == "failed" else backend.collect(batch_id)

    # The hook that turns each card accepted below into one commit. Built from
    # the run state, so it exists only for a run that opened a branch.
    on_accepted = make_card_committer(root, state, log)

    # What the batch being collected was compiled from, so an answer written
    # against files that have since changed is discarded unread instead of
    # overwriting whatever changed them (``cards.generations`` stale guard).
    inputs = _inputs_from_state(state)

    if retries:
        pending, attempt = _pending_from_retries(retries, by_id)
        retry_cards = build_retry_cards(
            pending, attempt, index + 1, total, max_regenerations,
            log=lambda _line: None)  # already logged when it was submitted
        pending = process_retry_batch(
            retry_cards, pending, results, attempt, index + 1, total, root, log,
            acceptance_timeout, max_regenerations, outcomes, blocked, on_accepted,
            inputs=inputs)
    else:
        attempt = 0
        pending = process_generation(
            runnable=[by_id[cid] for cid in submitted_ids],
            results=results, index=index + 1, total=total, root=root,
            backend=backend, poll_interval=poll_interval, log=log, verify=verify,
            acceptance_timeout=acceptance_timeout,
            max_regenerations=max_regenerations, outcomes=outcomes,
            blocked=blocked, inline_retries=False, on_accepted=on_accepted,
            inputs=inputs)

    if pending:
        # Submit the next regeneration, persist it, and stop. The generation
        # stays in flight; /collect run again polls exactly this batch.
        attempt += 1
        next_cards = build_retry_cards(
            pending, attempt, index + 1, total, max_regenerations, log)
        requests, retry_inputs, _compiled, failures = compile_for_batch(
            next_cards, root)
        if failures:
            # A retry that will not compile never reaches the provider, so there
            # is nothing left to judge for that card: it fails here, carrying
            # the compiler's message as its diagnosis.
            errors = dict(failures)
            kept: List[tuple] = []
            for retry_card, pair in zip(next_cards, pending):
                error = errors.get(retry_card.custom_id)
                if error is None:
                    kept.append(pair)
                    continue
                card, _previous = pair
                log(f"mrph> [generation {index + 1}/{total}] retry for "
                    f"{card.custom_id!r} could not be compiled: {error}")
                outcomes[card.custom_id] = CardOutcome(
                    card.custom_id, "failed", attempts=attempt,
                    reason=REASON_COMPILE, acceptance_output=error)
                blocked.add(card.custom_id)
            pending = kept

    if pending:
        retry_batch_id = backend.submit(requests)
        _record_retry(store, state, pending, attempt, retry_batch_id, outcomes,
                      retry_inputs)
        return CollectResult(
            in_progress=True, generation_number=index + 1,
            total_generations=total, phase=PHASE_SUBMITTED,
            retry_in_flight=True, retry_submitted=True, retry_attempt=attempt,
            retry_limit=max_regenerations,
            retry_card_ids=[card.custom_id for card, _previous in pending],
            retry_batch_id=retry_batch_id)

    next_index = index + 1
    state["phase"] = PHASE_DONE if next_index >= total else PHASE_IDLE
    state["generation_index"] = next_index
    state["batch_id"] = None
    state["submitted_ids"] = []
    state["retries"] = {}
    # Nothing is in flight any more, so nothing is waiting to be checked
    # against a compile; the next submit writes its own record.
    state["inputs"] = {}
    _store_outcomes(state, outcomes)
    if state["phase"] == PHASE_DONE:
        archive_run(store, state, cards, root=root, log=log)
    store.save_state(state)

    reported = {cid: outcomes[cid] for cid in submitted_ids if cid in outcomes}
    return CollectResult(
        in_progress=False, generation_number=index + 1, total_generations=total,
        phase=state["phase"], outcomes=reported)
