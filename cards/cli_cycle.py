"""
``mrph run``: the whole nightly cycle in one blocking call, for a caller that
is a script and not a person.

WHY this exists: an agent driving this system today spends a third of its
budget typing REPL commands and polling a 20-40 minute queue by hand. A call
an agent makes must correspond to a DECISION, not to a step -- and "run the
deck" is one decision. Preflight, the branch, generation after generation,
the regenerations, the per-card commits and the archive are mechanics that
need no model in the loop, so none of them should cost the agent a round
trip. This module writes NO new run logic: every step it takes is a call into
something that already exists, composed in the one order those pieces compose
correctly -- and the order is load-bearing twice:

* ``begin_run`` BEFORE ``run_deck``. The branch has to exist before the first
  morph lands, or the first card's commit goes on whatever branch happened to
  be checked out; and ``begin_run``'s dirty-tree refusal
  (:class:`cards.store.StoreError`) must happen before anything is spent. See
  ``cards.store.begin_run``.
* ``record_run`` receives the state ``begin_run`` returned as its
  ``run_state``. Inheriting a deck id any other way is exactly the mistake
  the archive cannot survive: a nightly pass that adopted the previous run's
  id would find that run's directory already there and archive nothing. See
  ``cards.store.record_run``.

The pieces, in the order they run:

1. ``cards.store.DeckStore.load_cards`` -- the backlog. An EMPTY backlog is
   not an error: "there was nothing to run" is a completed decision, answered
   with ``EXIT_OK`` and a payload that says so.
2. ``cards.store.preflight_deck`` -- file ownership FIRST, so a refused deck
   costs nothing. Its :class:`cards.hazards.HazardError` propagates (the
   error table maps it to exit 2), and the cards it RETURNS are the cards
   that run: its repair may have added dependency edges and saved them to the
   backlog, and the deck that runs must be the deck on disk.
3. ``cards.cli_backend.resolve_backend`` (skipped when a backend is injected)
   and ``cards.cli_wait.ResilientBackend`` -- the whole of the network
   resilience, in one wrapper. ``run_deck`` polls through the proxy and never
   sees a blinked GET; the wrapper holds ONE deadline for the entire wait, so
   a deck that failed once per poll cannot stretch the budget. A
   :class:`cards.cli_wait.WaitTimeout` propagates to exit 3.
4. ``cards.store.begin_run`` -- the deck id and the branch, before anything
   is spent.
5. ``cards.generations.run_deck`` -- the generation cycle itself, with the
   commit hook ``cards.store.make_card_committer`` builds from the state step
   4 returned, so every accepted card becomes one commit carrying its
   provenance trailers. A run that holds no branch passes no committer: the
   builder already answers ``None`` for exactly that case.
6. ``cards.store.record_run`` -- persistence and archive.
7. The report: the ARCHIVED one, read back with ``cards.store.list_runs`` for
   this run's deck id and rendered through
   ``cards.cli_views.report_to_dict`` -- the one report shape, the archived
   one. Should the read-back be unreachable, a ``cards.store.RunReport`` is
   built over the result and rendered by the same serialiser, so the shape is
   the same either way.

Errors beyond the paths it completes are not this function's vocabulary: a
:class:`cards.hazards.HazardError`, a :class:`cards.store.StoreError`, a
:class:`cards.cli_wait.WaitTimeout` and a
:class:`cards.cli_backend.BackendError` all PROPAGATE, for
``cards.cli_json``'s table to map onto refusals (2), transport (3) and usage
(4). What this function owns is the SUCCESS split: ``EXIT_OK`` only when
EVERY card ended ``written``; ``EXIT_INCOMPLETE`` when any ended ``failed``
or ``skipped`` -- derived from the report's own ``counts``.

It prints nothing and passes ``print`` nowhere: progress goes to the
one-argument ``log`` callable the caller hands in (``None`` means silence),
because stdout belongs to the one JSON document the headless contract
promises. Nothing here imports from ``flows``.
"""

from typing import Callable, Dict, Optional, Tuple

from cards import cli_views
from cards.cli_backend import resolve_backend
from cards.cli_json import EXIT_INCOMPLETE, EXIT_OK
from cards.cli_wait import ResilientBackend
from cards.generations import run_deck
from cards.store import (
    DeckStore,
    RunReport,
    # The one definition of a run's finish timestamp; imported rather than
    # re-spelled so the fallback report below stamps time the same way the
    # archived one does.
    _completed_at,
    begin_run,
    list_runs,
    make_card_committer,
    preflight_deck,
    record_run,
)

__all__ = ["run"]


def _quiet(_line: str) -> None:
    """The sink a ``log=None`` caller gets: the run's progress goes nowhere.

    Every store call and ``run_deck`` defaults its ``log`` to ``print``, which
    would put a progress line on stdout the moment a run had something to say
    -- and stdout belongs to the one JSON document a headless caller parses.
    A caller that passes no ``log`` gets this instead: the default is
    silence, never ``print``.
    """


def run(
    root: str = ".",
    processor: Optional[str] = None,
    nogit: bool = False,
    timeout: float = 6 * 3600.0,
    log: Optional[Callable[[str], None]] = None,
    backend=None,
    sleep: Optional[Callable[[float], None]] = None,
    now: Optional[Callable[[], float]] = None,
) -> Tuple[dict, int]:
    """Run the whole deck once, blocking, and return ``(payload, exit_code)``.

    The pair is exactly what a ``cards.cli_json.run_cli`` handler returns, so
    the headless wiring around this function stays a pass-through. The order,
    which is the contract:

    1. Load the backlog from ``<root>/.morph/deck.json``. An EMPTY backlog is
       not an error: the call returns ``{"empty": True, "counts": {all zero}}``
       with ``EXIT_OK`` -- "there was nothing to run" is a completed decision,
       and it is answered before the processor registry is even consulted.
    2. :func:`cards.store.preflight_deck` checks file ownership first, so a
       refused deck costs nothing. Its :class:`cards.hazards.HazardError`
       propagates to the error table (exit code 2). The cards it RETURNS are
       used from here on: its repair may have added dependency edges and
       saved them to the backlog, and the deck that runs must be the deck on
       disk.
    3. The backend: :func:`cards.cli_backend.resolve_backend` unless one is
       injected (an injected backend is the test seam that keeps a caller off
       the network and off the processor registry entirely; it carries no
       label), then wrapped in :class:`cards.cli_wait.ResilientBackend` under
       the ``timeout`` budget -- six hours of wall clock for the whole wait,
       not per call. Injected ``sleep``/``now`` pass through to the wrapper,
       so tests run a fake clock. A :class:`cards.cli_wait.WaitTimeout`
       propagates to exit code 3.
    4. :func:`cards.store.begin_run` mints the deck id and opens the branch
       ``morph/<deck-id>`` -- BEFORE anything is spent. It raises
       :class:`cards.store.StoreError` on a dirty working tree (exit code 2).
       ``nogit`` skips the branch, not the run.
    5. :func:`cards.generations.run_deck` executes the deck generation by
       generation through the proxy, with the commit hook
       :func:`cards.store.make_card_committer` builds from the state step 4
       returned -- every accepted card becomes one commit with its provenance
       trailers, exactly as on the existing blocking route. A run that holds
       no branch passes no committer: the builder answers ``None`` for
       exactly that case.
    6. :func:`cards.store.record_run` persists the run and archives it under
       ``.morph/runs/<deck-id>/``. ``run_state`` is the state step 4
       returned, so this run keeps its own deck id instead of adopting the
       previous run's.
    7. The payload: the ARCHIVED report for this run's deck id, read back
       with :func:`cards.store.list_runs` and rendered by
       :func:`cards.cli_views.report_to_dict` -- the one report shape, the
       archived one. If the read-back is not reachable, a
       :class:`cards.store.RunReport` is built over the result and rendered
       by the same serialiser, so the shape does not change.

    The exit code comes from the report's ``counts``: ``EXIT_OK`` only when
    every card of the deck ended ``written``; ``EXIT_INCOMPLETE`` when any
    card ended ``failed`` or ``skipped`` -- or failed to end at all, which
    counts as not-written.

    ``log`` is a one-argument callable passed into every store call and into
    ``run_deck``; ``None`` means silence. ``print`` is never passed and
    nothing is printed here: stdout belongs to the caller's one JSON
    document. ``backend`` is a duck-typed batch backend
    (``submit``/``status``/``collect``), the test seam.
    """
    if log is None:
        log = _quiet

    store = DeckStore(root)

    # 1. The backlog. Empty is a completed decision, not an error -- answered
    #    before the registry or the network is anywhere in sight. ``counts``
    #    is here so a script reads one counts shape on every outcome.
    cards = store.load_cards()
    if not cards:
        return ({"empty": True,
                 "counts": {"written": 0, "failed": 0, "skipped": 0}},
                EXIT_OK)

    # 2. Ownership first: a refused deck costs nothing, and the cards that run
    #    are the cards preflight saved.
    cards = preflight_deck(store, cards, log=log)

    # 3. The backend, and around it the whole of the network resilience. An
    #    injected backend (the test seam) goes through the same proxy, so the
    #    resilience is a property of the run, not of how the backend was
    #    obtained; run_deck never sees a blinked GET either way.
    if backend is not None:
        raw = backend
        label = None
    else:
        raw, label = resolve_backend(processor)
    injected: Dict[str, Callable] = {}
    if sleep is not None:
        injected["sleep"] = sleep
    if now is not None:
        injected["now"] = now
    proxy = ResilientBackend(raw, timeout=timeout, log=log, **injected)

    # 4. The run's identity and its branch -- before anything is spent. A
    #    dirty tree raises StoreError here, still at zero cost.
    state = begin_run(store, cards, root=root, use_git=not nogit,
                      backend_label=label, log=log)

    # 5. The generation cycle itself. The committer is built from the state
    #    begin_run returned, exactly as the blocking route builds it; a run
    #    with no branch (nogit, or no repository) gets None and no commits.
    committer = make_card_committer(root, state, log=log)
    result = run_deck(cards, proxy, root=root, log=log, on_accepted=committer)

    # 6. Persist and archive. run_state is what keeps this run's own deck id
    #    instead of adopting the previous run's.
    record_run(store, result, backend_label=label, root=root,
               run_state=state, log=log)

    # 7. The report the run left behind -- the archived one, in the one shape.
    deck_id = state.get("deck_id") or ""
    archived = None
    for report in list_runs(store):
        if report.deck_id == deck_id:
            archived = report
            break
    if archived is None:
        # The read-back is not reachable; the shape may not change. Built over
        # the result, rendered by the same serialiser below.
        archived = RunReport(
            deck_id=deck_id,
            completed_at=_completed_at(),
            branch=state.get("branch"),
            backend_label=label,
            generations=[list(generation) for generation in result.generations],
            batch_ids=list(result.batch_ids),
            outcomes=dict(result.outcomes),
        )
    payload = cli_views.report_to_dict(archived)

    # The exit-code rule, derived from the report's own counts: every card
    # ended written, or the run is incomplete. A card with no outcome at all
    # is not a written card, hence the literal count against the deck.
    counts = payload["counts"]
    if counts["failed"] or counts["skipped"] or counts["written"] != len(cards):
        return payload, EXIT_INCOMPLETE
    return payload, EXIT_OK
