"""
The two headless step subcommands that talk to a provider: ``submit`` and
``collect``.

The split-step cycle a headless ``mrph`` runs -- preflight the deck's file
ownership, lock the composition, open the run's branch, compile the current
generation, submit one batch, wait, judge the morphs, regenerate the failures,
commit each accepted card, archive the run -- is already written, all of it, in
:mod:`cards.store`. A headless caller must not have to re-implement any of it,
and must not have to know that "send tonight's generation" means building a
store, resolving a backend and threading a logger through several store calls.
This module is the seam between the two: two plain functions that take a
project root and a processor label and return the JSON-ready dict the command
prints plus one exit code. No argument parsing, no printing, no ``sys`` -- and
nothing imported from ``flows``, the same wall every other ``cards`` module
sits behind.

Two contracts shape everything here:

* **One document on stdout, always** (:mod:`cards.cli_json`). These functions
  are handler code: they compute and return, they never write. Every line the
  store would have printed goes to the caller's ``log`` callable -- the
  dispatcher points it at stderr -- so a ``log=None`` call is given a logger
  that DROPS the lines, never ``print``: the store's own default would put
  progress on stdout and tear the contract for any caller that forgot the
  logger, and bare ``None`` would stop the store dead at its first
  ``log(...)``.
* **The coarse exit codes** (:mod:`cards.cli_json`'s table). A submit that
  sent a batch is ``EXIT_OK``; a settled generation or a submit that only
  recorded skips carries ``EXIT_INCOMPLETE``; a refused deck rides
  :class:`cards.hazards.HazardError` to ``EXIT_REFUSED``; an expired wait
  rides :class:`cards.cli_wait.WaitTimeout` to ``EXIT_TRANSPORT``. Nothing
  here invents a code or a shape: the payloads come from
  :mod:`cards.cli_views`, the codes from :mod:`cards.cli_json`.

The backend comes from :func:`cards.cli_backend.resolve_backend` unless one is
injected -- the seam the tests drive -- and the waiting is
:mod:`cards.cli_wait`'s: the backend a ``collect`` polls through is wrapped in
a :class:`cards.cli_wait.ResilientBackend` under the ``timeout`` budget, so a
blinking network costs a backoff pause instead of an hour-old run. No new run
logic lives here: the store already does the work, including the branch, the
per-card commits, the regeneration bookkeeping and the run archive.
"""

from typing import Callable, Optional, Tuple

from cards import cli_views
from cards.cli_backend import resolve_backend
from cards.cli_json import EXIT_INCOMPLETE, EXIT_OK
from cards.cli_wait import ResilientBackend, wait_until
from cards.store import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    CollectResult,
    DeckStore,
    collect_generation,
    preflight_deck,
    submit_generation,
)

__all__ = ["submit", "collect"]


def _quiet(line: str) -> None:
    """Drop a log line: the logger a ``log=None`` call passes to the store.

    The store functions' own default is ``print``, which would put progress
    lines on stdout and break the one-document contract of ``cards.cli_json``;
    ``None`` passed through unmodified would stop the store at its first
    ``log(...)``. Dropping the lines is the one behaviour that is both silent
    and safe, so ``None`` never reaches the store and ``print`` is never
    substituted for it.
    """


def _resolve(backend, processor: Optional[str]) -> Tuple[object, Optional[str]]:
    """The backend to use, and the label to record for it.

    An injected ``backend`` -- the seam the tests drive -- is used as-is and
    recorded under ``processor`` (``None`` when none is named). Otherwise
    :func:`cards.cli_backend.resolve_backend` turns the ``--processor`` label
    into a batch backend, raising :class:`cards.cli_backend.BackendError` (a
    ``ValueError``, so exit code 4 in the error table) for a label that names
    nothing configured.
    """
    if backend is not None:
        return backend, processor
    return resolve_backend(processor)


def _clocks(
    sleep: Optional[Callable[[float], None]],
    now: Optional[Callable[[], float]],
) -> dict:
    """Only the clock callables that were actually injected.

    :mod:`cards.cli_wait`'s defaults are the real ``time.sleep`` and
    ``time.monotonic``; handing either of them ``None`` would crash, so an
    absent one is left at its default and only a given one is passed on --
    which is what lets the tests run a fake clock instead of a real six
    hours.
    """
    kwargs: dict = {}
    if sleep is not None:
        kwargs["sleep"] = sleep
    if now is not None:
        kwargs["now"] = now
    return kwargs


def submit(
    root: str,
    processor: Optional[str] = None,
    nogit: bool = False,
    log: Optional[Callable[[str], None]] = None,
    backend=None,
) -> Tuple[dict, int]:
    """Compile and submit the current generation of the deck at ``root``.

    The work is :func:`cards.store.submit_generation`'s -- resolving the
    runnable cards, locking the composition and the run identity on the first
    submit, opening the branch (unless ``nogit``), compiling against the files
    earlier generations wrote, submitting one batch and persisting the run
    state. This function adds only what a headless caller must not have to
    re-derive:

    * **the refusal gate, first**: :func:`cards.store.preflight_deck` checks
      file ownership over the backlog as it stands, and an unrepairable
      conflict -- two cards of one generation writing one file -- raises
      :class:`cards.hazards.HazardError` HERE, before the branch is opened and
      before any money is spent; the error table maps it to exit code 2. The
      cards preflight returns are passed nowhere: ``submit_generation``
      reloads the backlog itself, and preflight has already SAVED any repair
      it made, so the deck the run executes is the deck on disk.
    * **the backend**: :func:`cards.cli_backend.resolve_backend` turns
      ``processor`` into one -- unless ``backend`` is injected, which is how
      the tests drive this function.
    * **the log**: ``log`` -- the caller's logger; ``None`` drops the lines --
      is passed to every store call as its ``log=`` argument, so the store's
      progress lines go wherever the dispatcher points the logger (stderr)
      and never to stdout.

    Returns ``(payload, exit_code)`` with the payload of
    :func:`cards.cli_views.submit_to_dict`. The code is ``EXIT_OK`` when a
    batch went out or the deck was already done, and ``EXIT_INCOMPLETE`` when
    nothing was submitted and cards were skipped on the way -- a submit that
    only records skips has news, and the news is that part of the deck will
    never run.
    """
    logger = log if log is not None else _quiet
    resolved, label = _resolve(backend, processor)

    store = DeckStore(root)
    # Refuse before anything store-side exists: the branch, the run identity
    # and the batch all belong to submit_generation, and none of them should
    # be created for a deck that cannot run as authored. (The cards preflight
    # returns are dropped on purpose: submit_generation reloads the backlog
    # that preflight has just saved, repairs included.)
    preflight_deck(store, store.load_cards(), log=logger)

    result = submit_generation(
        store, resolved, root=root, backend_label=label, log=logger,
        use_git=not nogit)
    payload = cli_views.submit_to_dict(result)
    if not result.submitted and result.skipped:
        return payload, EXIT_INCOMPLETE
    return payload, EXIT_OK


def collect(
    root: str,
    processor: Optional[str] = None,
    wait: bool = False,
    timeout: float = 6 * 3600.0,
    log: Optional[Callable[[str], None]] = None,
    backend=None,
    sleep: Optional[Callable[[float], None]] = None,
    now: Optional[Callable[[], float]] = None,
) -> Tuple[dict, int]:
    """Poll the in-flight batch once -- or, with ``wait``, until it settles.

    The work is :func:`cards.store.collect_generation`'s: one status poll of
    whichever batch the run is waiting on (the generation's own, or a
    regeneration an earlier collect submitted), and -- once the batch has
    finished -- the judging, the best-of-N acceptance, the rollbacks, the
    per-card commits, the regeneration bookkeeping and, at the end of the
    deck, the archive. This function adds the two things a headless caller
    must not have to re-derive:

    * **the retry budget.** The backend is wrapped in
      :class:`cards.cli_wait.ResilientBackend` with ``timeout`` as the WHOLE
      wait's budget, so one connection reset in the small hours costs a
      backoff pause instead of a run that has been queued for an hour. An
      injected ``backend`` is wrapped like any other; ``sleep`` and ``now``,
      when given, are injected into the wrapper and into the wait loop, so
      the tests run a fake clock instead of a real six hours.
    * **the wait.** Without ``wait``, exactly one
      :func:`cards.store.collect_generation` call: a batch still running
      returns ``in_progress`` true and nothing else happens. With ``wait``,
      :func:`cards.cli_wait.wait_until` drives the loop, and its ``step`` is
      one ``collect_generation`` reporting ``(not result.in_progress,
      result)`` -- so the loop ends the moment a poll settles the generation,
      and a batch that is already completed is collected without waiting even
      one backoff delay.

    Returns ``(payload, exit_code)`` with the payload of
    :func:`cards.cli_views.collect_to_dict`. The code is ``EXIT_OK`` for a
    still-in-progress poll -- an unfinished poll is not a failure, it is news
    -- and for a settled generation whose every outcome was ``written``;
    ``EXIT_INCOMPLETE`` when any outcome of the collected generation is
    ``failed`` or ``skipped``.

    A :class:`cards.cli_wait.WaitTimeout` is deliberately NOT caught: it
    propagates, and the error table maps it to exit code 3 (transport --
    retrying is meaningful). That is safe because of one property of the step
    the wait drives: the deadline is only ever checked BETWEEN polls, and a
    poll that finds the batch still running has persisted nothing --
    ``collect_generation`` writes state only once the batch has finished and
    its results are being judged. An expired wait therefore leaves
    ``state.json`` exactly as the submit that sent the batch left it --
    phase ``submitted``, batch id recorded, no card half-judged -- so the
    generation stays collectable by a later ``mrph collect``, in this session
    or a fresh one. A timeout must never be a mid-acceptance abort, and under
    this design it cannot be one.
    """
    logger = log if log is not None else _quiet
    # (collect records no backend label: the submit that sent the batch did.)
    resolved, _label = _resolve(backend, processor)
    clocks = _clocks(sleep, now)

    store = DeckStore(root)
    resilient = ResilientBackend(resolved, timeout=timeout, log=logger,
                                 **clocks)

    def step() -> Tuple[bool, CollectResult]:
        result = collect_generation(store, resilient, root=root, log=logger)
        return (not result.in_progress, result)

    if wait:
        result = wait_until(step, timeout=timeout, log=logger,
                            what="generation", **clocks)
    else:
        result = collect_generation(store, resilient, root=root, log=logger)

    payload = cli_views.collect_to_dict(result)
    if result.in_progress:
        return payload, EXIT_OK
    incomplete = any(outcome.status in (STATUS_FAILED, STATUS_SKIPPED)
                     for outcome in result.outcomes.values())
    if incomplete:
        return payload, EXIT_INCOMPLETE
    return payload, EXIT_OK
