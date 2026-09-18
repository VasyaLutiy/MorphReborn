"""
One line, out to whatever channel the operator already has, when a run ends.

A run that finishes at 03:40 tells nobody. The report lands in
``.morph/runs/`` and the operator learns the outcome by coming back to the
terminal -- which at night means the morning. The whole feature is one line,
pushed through a channel the operator has configured, and the line is
worthless without NUMBERS in it: an operator who has to go and open the
report anyway has been told nothing.

Exactly two entry points, nothing else.

``build_run_message(report)``
    ONE line, no newline inside it, built from an archived run -- a
    :class:`~cards.store.RunReport`, the same object whose ``to_dict()``
    writes every ``report.json`` in the run archive. The line names, all of
    them: the deck id; the branch (``no branch`` when the run owned none);
    the written / failed / skipped counts as ``<N> written``, ``<N> failed``,
    ``<N> skipped`` -- number before the word, in that order, because that
    is what a reader scans for; the number of generations; the whole minutes
    the run took, named with ``min``.

``send_notification(message, command=None)``
    Hands ``message`` to a shell command on its STDIN, as UTF-8 bytes, and
    answers whether the command took it. The command defaults to
    ``$MORPH_NOTIFY_CMD``, read at call time. No command configured (the
    variable unset or empty, or an empty ``command`` passed) is NOT an
    error: nothing is sent, ``False`` comes back, nothing is raised and
    nothing is printed -- an operator who never set the variable must not
    have their run report a failure. A command that exits non-zero, cannot
    be started or hangs past a short timeout is also a ``False`` and never
    a raise: a notification is the last thing a run does, so it must never
    be able to fail the run -- which is also why the channel's own output
    is discarded instead of leaking into the run's own.

The minutes come from the report alone, because nothing else survives the
run: the deck id begins ``YYYYMMDD-HHMMSS-`` -- the moment the run started --
and ``completed_at`` is an ISO-8601 local timestamp of the moment it
finished; the whole minutes between the two are the number. A deck id or a
``completed_at`` that will not parse costs the minutes, not the message: the
line still goes out, with a readable ``?`` where the number would have been.

Imports stop at ``cards.store``, for the ``RunReport`` type. This module
sits below the CLI and touches neither ``flows`` nor ``processors``.
Stdlib-only.
"""

import os
import subprocess
from datetime import datetime
from typing import Optional, Union

from cards.store import RunReport

#: The variable :func:`send_notification` consults when no command is
#: passed. Read at CALL time, never at import, so it can change between
#: calls under a test's feet.
_NOTIFY_ENV_VAR = "MORPH_NOTIFY_CMD"

#: How long the notification command may run before it is judged to have
#: hung. A push to a channel takes well under a second; a hang must cost the
#: notification, never the run.
_NOTIFY_TIMEOUT_SECONDS = 10.0


# -- the line ----------------------------------------------------------------


def _parse_deck_start(deck_id: str) -> Optional[datetime]:
    """The moment the run started, read off the deck id's timestamp head.

    A deck id begins ``YYYYMMDD-HHMMSS-``; the tail after the second dash is
    the run's random suffix and is ignored here. ``None`` when the id does
    not carry that head -- malformed or hand-written -- in which case the
    minutes are simply not knowable.
    """
    parts = (deck_id or "").split("-")
    if len(parts) < 2:
        return None
    try:
        return datetime.strptime(
            "{0}-{1}".format(parts[0], parts[1]), "%Y%m%d-%H%M%S"
        )
    except ValueError:
        return None


def _parse_completed_at(completed_at: object) -> Optional[datetime]:
    """The moment the run finished, from the report's ``completed_at``.

    The archive holds an ISO-8601 local timestamp string, so that is the
    case parsed here; a :class:`~datetime.datetime` a caller already holds
    passes straight through. ``None`` when it is neither, or will not parse.
    """
    if isinstance(completed_at, datetime):
        return completed_at
    if not isinstance(completed_at, str):
        return None
    try:
        return datetime.fromisoformat(completed_at.strip())
    except ValueError:
        return None


def _run_minutes(report: RunReport) -> Union[int, str]:
    """Whole minutes the run took, or the readable ``'?'`` when unknowable.

    Both moments come from the report alone -- the start from the deck id's
    ``YYYYMMDD-HHMMSS-`` head, the finish from ``completed_at`` -- because
    nothing else survives the run. Whole minutes: the line is scanned, not
    billed by the second. Two moments that will not parse, or that cannot be
    subtracted from each other (one timezone-aware, one naive), cost the
    line a ``?`` and nothing else.
    """
    start = _parse_deck_start(report.deck_id)
    end = _parse_completed_at(report.completed_at)
    if start is None or end is None:
        return "?"
    try:
        elapsed = end - start
    except TypeError:
        return "?"
    return int(elapsed.total_seconds() // 60)


def build_run_message(report: RunReport) -> str:
    """ONE line summarising a finished run, its numbers included.

    The shape, exactly::

        mrph run <deck_id> (<branch>|no branch): <W> written, <F> failed, <S> skipped; <G> generations, <M> min

    for instance::

        mrph run 20260918-150640-34c1caaf (no branch): 3 written, 0 failed, 0 skipped; 3 generations, 130 min

    ``<W> <F> <S>`` are ``report.counts`` (keyed ``written`` / ``failed`` /
    ``skipped``) rendered number-before-word in that order, ``<G>`` is
    ``len(report.generations)`` and ``<M>`` the whole minutes from
    :func:`_run_minutes` -- ``?`` when the report's two moments will not
    give them. No newline is ever inside: this line travels through channels
    that deliver one line as one message.
    """
    counts = report.counts
    generations = len(report.generations)
    return (
        "mrph run {deck_id} ({branch}): "
        "{written} written, {failed} failed, {skipped} skipped; "
        "{generations} {generation_word}, {minutes} min"
    ).format(
        deck_id=report.deck_id,
        branch=report.branch if report.branch else "no branch",
        written=counts["written"],
        failed=counts["failed"],
        skipped=counts["skipped"],
        generations=generations,
        generation_word="generation" if generations == 1 else "generations",
        minutes=_run_minutes(report),
    )


# -- the transport -----------------------------------------------------------


def send_notification(message: str, command: Optional[str] = None) -> bool:
    """Hand ``message`` to a shell command's STDIN; ``True`` iff it exits 0.

    ``command`` is a shell command line, run with ``shell=True``; the
    message goes down its stdin as UTF-8 bytes, so the channel is anything
    that reads a pipe and answers with an exit code -- an append to a log,
    a ``curl --data-binary @-`` to a webhook, whatever the operator already
    has. When ``command`` is ``None``, ``$MORPH_NOTIFY_CMD`` is read
    instead, at call time.

    Returns ``True`` only when a command was configured, ran and exited 0.
    Every other outcome is a ``False`` and never a raise:

    * nothing configured -- the variable unset or empty, or an empty or
      blank ``command`` passed -- sends nothing, raises nothing and prints
      nothing: an operator who never set the variable must not have their
      run report a failure;
    * a command that exits non-zero, cannot be started, or outlives
      ``_NOTIFY_TIMEOUT_SECONDS`` costs the notification, not the run, and
      its output is discarded rather than leaked into the run's own -- a
      notification is the last thing a run does, so it must never be able
      to fail the run.
    """
    if command is None:
        command = os.environ.get(_NOTIFY_ENV_VAR)
    if not command or not command.strip():
        return False
    try:
        completed = subprocess.run(
            command,
            input=message.encode("utf-8"),
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_NOTIFY_TIMEOUT_SECONDS,
        )
    except Exception:
        # Deliberately broad. A missing shell, a fork failure, a channel
        # that hung past the timeout -- none of it may surface, because
        # nothing in the run comes after this to catch it. BaseException is
        # left alone so a Ctrl-C still interrupts.
        return False
    return completed.returncode == 0
