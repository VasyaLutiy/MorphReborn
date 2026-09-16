"""
Acceptance: the mechanical verifier and best-of-N variant selector.

Phase 4 of ``documentation/DEVELOPMENT_PLAN.md``. A morph card carries a
machine-checkable ``acceptance`` criterion (``documentation/batch-orchestrator.md``,
"The morph card"): a shell command that returns exit 0 when the morph is good.
This module runs that command and, for a card with several variants, sandboxes
the trial so acceptance judges each candidate *at the file's real location* and
the losers leave no trace.

Two responsibilities live here:

* :func:`run_acceptance` -- run one acceptance command and report the outcome
  (:class:`AcceptanceResult`): exit 0 passes, a non-zero exit or a timeout
  fails, and the combined stdout+stderr is preserved -- clipped to the budget
  by :func:`clip_output`, head and tail both kept -- as the error context a
  retry generation feeds back to the executor.
* :func:`verify_card` -- the best-of-N + rollback protocol
  (:class:`VerifyOutcome`): capture the target's original state, try each
  variant in order at the *real* target path, keep the first that passes (plus
  its suffixed variant file for a multi-variant card), roll every rejected
  write back, and hand :mod:`cards.generations` the winner or the last failure.

Constraint (see the development plan): ``cards`` must not import from ``flows``
or ``processors``. The response-to-file helper and the variant/output-path
helpers are imported from :mod:`cards.generations` (their single home) rather
than duplicated a third time; :mod:`cards.generations` in turn imports
:func:`verify_card` lazily, inside ``run_deck``, to keep this an acyclic import.
"""

import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from cards.generations import (
    _output_path,
    _variant_ids,
    ensure_parent_dir,
    response_to_file_body,
)
from cards.schema import MorphCard


# The combined stdout+stderr of an acceptance run is kept only as failure
# context for a retry; a runaway test log would otherwise bloat the next
# prompt. The budget is spent head-and-tail (:func:`clip_output`): for the
# most typical acceptance command of all, ``pytest``, the cause of the
# failure -- the first failing traceback -- sits near the TOP of the log
# while the bottom holds only the summary line, so the tail-only cut this
# constant once paid for threw away exactly the part a regenerating executor
# needed. The middle is noise.
_OUTPUT_TAIL_CAP = 4000


# The marker :func:`clip_output` puts where the elided middle was; its one
# placeholder is the number of characters dropped.
_ELISION_TEMPLATE = "\n... [{} characters elided] ...\n"


def _elision_line(dropped: int) -> str:
    """The elision marker for ``dropped`` characters, as clip_output prints it."""
    return _ELISION_TEMPLATE.format(dropped)


def clip_output(text: str) -> str:
    """Clip ``text`` to the :data:`_OUTPUT_TAIL_CAP` budget, head AND tail kept.

    Text at or under the budget is returned unchanged. Longer text keeps its
    first quarter -- where a failing suite puts its first, and most causal,
    traceback -- and its last three quarters (the summary, the final errors),
    with the noisy middle replaced by a line naming how many characters were
    dropped. The marker is paid for out of the tail's three-quarter share, so
    the result never exceeds the budget.

    The dropped count in that marker is exact: the number of characters of
    ``text`` absent from the result. The marker's length depends on that
    count's width, and the count on the marker's length, so the two are
    settled together -- each pass moves the count only by the marker's own
    length, which converges in two or three passes for any output a command
    can produce, and the loop below is bounded regardless.
    """
    if len(text) <= _OUTPUT_TAIL_CAP:
        return text
    head_len = _OUTPUT_TAIL_CAP // 4           # the first-quarter share
    tail_budget = _OUTPUT_TAIL_CAP - head_len  # the last-three-quarters share
    # Start as if the marker were free, then settle marker length against tail
    # length: settled means head_len + len(line) + tail_len == the cap exactly,
    # with the count the marker prints equal to the characters it replaces.
    tail_len = tail_budget
    line = _elision_line(len(text) - head_len - tail_len)
    for _ in range(8):
        settled = tail_budget - len(line)
        if settled == tail_len:
            break
        tail_len = settled
        line = _elision_line(len(text) - head_len - tail_len)
    # Pure insurance (the loop above always settles): whatever the pair agreed
    # on, the marker may never push the result past the budget.
    tail_len = min(tail_len, _OUTPUT_TAIL_CAP - head_len - len(line))
    tail_text = text[-tail_len:] if tail_len > 0 else ""
    return text[:head_len] + line + tail_text


# -- the stale-bytecode guard ------------------------------------------------
#
# WHY this exists at all. CPython validates a cached ``.pyc`` against a pair
# taken from the source file: its mtime truncated to WHOLE SECONDS, and its
# size in bytes. :func:`verify_card` writes every variant of a card to the SAME
# real target path, milliseconds apart -- so two variants that happen to be the
# same byte size are, to the import system, the same file, and the second
# variant's acceptance run silently executes the FIRST variant's bytecode.
# Best-of-N then rejects a correct morph, the card burns its retries and fails.
# It bites the most typical acceptance command there is, ``pytest``, and it is
# invisible on macOS system python (``sys.pycache_prefix`` puts the cache in
# ``~/Library/Caches``, so no ``__pycache__`` appears next to the file to hint
# at it). Two cheap independent guards below: acceptance never writes bytecode,
# and every write gets an mtime no other write has used.

# The last whole-second stamp handed out by :func:`_stamp_distinct_mtime`, so
# stamps are unique across the WHOLE process, not merely within one card: a
# single-variant card is rewritten at the same variant index on every retry, and
# the clock does not necessarily move between the two.
_last_mtime_stamp = None


def _stamp_distinct_mtime(path: str) -> int:
    """Give ``path`` a whole-second mtime no earlier write has used.

    Now, or one second before the previous stamp when the clock has not moved --
    strictly decreasing, so a stamp is never in the future (a future mtime upsets
    ordinary build tooling) and never repeats. A cached ``.pyc`` keyed on an
    earlier variant therefore cannot validate against this file, whatever its
    size. Not thread-safe, and need not be: variants of a card are verified in
    sequence (concurrent writes to one target would be the larger problem).
    """
    global _last_mtime_stamp
    now = int(time.time())
    stamp = now if _last_mtime_stamp is None else min(now, _last_mtime_stamp - 1)
    _last_mtime_stamp = stamp
    os.utime(path, (stamp, stamp))
    return stamp


# -- the mechanical check ----------------------------------------------------


@dataclass
class AcceptanceResult:
    """The outcome of running one acceptance command.

    ``passed`` is true only on a clean exit 0. ``exit_code`` is the process
    return code, or ``None`` when the command timed out (``timed_out`` true).
    ``output`` is the combined stdout+stderr, clipped by :func:`clip_output`
    to at most :data:`_OUTPUT_TAIL_CAP` characters (head and tail kept, the
    elided middle named) -- it is the error context a failed card carries into
    its retry generation.
    """

    passed: bool
    exit_code: Optional[int]
    output: str
    timed_out: bool


def run_acceptance(command: str, root: str, timeout: float = 300.0) -> AcceptanceResult:
    """Run one acceptance ``command`` in ``root`` and report the outcome.

    The command runs through the shell (``shell=True``) with ``cwd=root`` and
    combined stdout+stderr captured. Exit 0 passes; any non-zero exit fails; a
    command that exceeds ``timeout`` fails with ``timed_out`` set and whatever
    output was produced before the kill preserved. The output is always
    clipped to the :data:`_OUTPUT_TAIL_CAP` budget by :func:`clip_output`
    (head and tail kept, the elided middle named).

    The command inherits a COPY of the environment with
    ``PYTHONDONTWRITEBYTECODE=1`` added (the parent's own environment is never
    mutated), so an acceptance run leaves no ``.pyc`` behind for the next variant
    to be judged by -- see the stale-bytecode note above.
    """
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        # exc.output holds the bytes/str captured before the timeout kill; it
        # may be empty, and is bytes when the process wrote non-decodable data.
        output = exc.output or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return AcceptanceResult(
            passed=False,
            exit_code=None,
            output=clip_output(output),
            timed_out=True,
        )

    return AcceptanceResult(
        passed=(completed.returncode == 0),
        exit_code=completed.returncode,
        output=clip_output(completed.stdout or ""),
        timed_out=False,
    )


# -- best-of-N + rollback ----------------------------------------------------


@dataclass
class VerifyOutcome:
    """The result of verifying one card's variants against its acceptance.

    ``passed`` marks that some variant cleared acceptance. ``winning_custom_id``
    is that variant's batch custom_id (``None`` on failure); ``paths`` lists the
    files kept for it (the real target, plus its suffixed variant file for a
    multi-variant card). ``attempts`` is how many variants were actually run
    (``None`` responses are skipped). ``result`` is the winning
    :class:`AcceptanceResult`, or -- on failure -- the last failing one, whose
    ``output`` becomes the retry's error context (``None`` only when every
    response was ``None`` and nothing ran).
    """

    passed: bool
    winning_custom_id: Optional[str] = None
    paths: List[str] = field(default_factory=list)
    attempts: int = 0
    result: Optional[AcceptanceResult] = None


def _capture_original(path: str) -> Optional[bytes]:
    """Snapshot the target's bytes, or ``None`` if it does not exist yet."""
    if os.path.exists(path):
        with open(path, "rb") as handle:
            return handle.read()
    return None


def _restore_original(path: str, original: Optional[bytes]) -> None:
    """Put the target back the way :func:`_capture_original` found it.

    Bytes are rewritten; an originally absent file (``original is None``) is
    deleted if a rejected variant left one behind. A DIRECTORY created for that
    write (see :func:`cards.generations.ensure_parent_dir`) is left in place on
    purpose -- an empty directory is harmless, and removing it is not ours to
    do; only the file is rolled back.
    """
    if original is None:
        if os.path.exists(path):
            os.remove(path)
    else:
        with open(path, "wb") as handle:
            handle.write(original)


def verify_card(
    card: MorphCard,
    responses: Dict[str, Optional[str]],
    root: str,
    timeout: float = 300.0,
    log: Callable[[str], None] = print,
) -> VerifyOutcome:
    """Best-of-N acceptance with a sandboxed rollback, for a card WITH acceptance.

    The target's original state is captured up front. Each variant custom_id is
    tried IN ORDER (``None`` responses skipped): its body is written to the
    *real* ``card.target`` -- acceptance must test the file where it will live --
    and ``card.acceptance`` is run. The first variant to pass WINS: the target
    keeps its winning bytes, a multi-variant card also gets the winner's suffixed
    file (naming as in :mod:`cards.generations`) while every losing variant's
    suffixed file is removed, and a passed :class:`VerifyOutcome` is returned.

    Each write is also stamped with an mtime no other write has used, so an
    acceptance command that imports the target cannot be answered by bytecode
    cached for an earlier variant of the same byte size -- the stale-``.pyc``
    trap documented above :func:`_stamp_distinct_mtime`.

    A failing variant is rolled back to the captured original before the next is
    tried. If every variant fails (or every response was ``None``), the original
    state is restored and a failed :class:`VerifyOutcome` is returned carrying
    the last failure's :class:`AcceptanceResult` as the retry's error context.

    Callers must only invoke this for a card that has an ``acceptance`` command;
    a card without one keeps Phase 3 behaviour and never reaches the verifier.
    """
    target_path = os.path.join(root, card.target)
    original = _capture_original(target_path)

    variant_ids = _variant_ids(card)
    attempts = 0
    last_result: Optional[AcceptanceResult] = None

    for variant_id in variant_ids:
        response = responses.get(variant_id)
        if response is None:
            continue

        attempts += 1
        body = response_to_file_body(response)
        # The target may name a package that does not exist yet; create it
        # before the first variant lands (idempotent for the ones after it).
        ensure_parent_dir(target_path)
        with open(target_path, "w", encoding="utf-8") as handle:
            handle.write(body)
        # Distinct mtime per variant: belt to run_acceptance's braces against a
        # .pyc cached for an earlier, same-sized variant (see the note above).
        _stamp_distinct_mtime(target_path)

        result = run_acceptance(card.acceptance, root, timeout)
        last_result = result
        log(
            f"mrph> acceptance {'passed' if result.passed else 'failed'} for "
            f"{variant_id!r} (exit {result.exit_code}, "
            f"`{card.acceptance}`)"
        )

        if result.passed:
            paths = [target_path]
            if card.variants > 1:
                # Keep the winner's suffixed file too, and clear any losing
                # suffixed files (from this or an earlier attempt) so only the
                # winner survives alongside the real target.
                winning_path = _output_path(card, variant_id, root)
                ensure_parent_dir(winning_path)
                with open(winning_path, "w", encoding="utf-8") as handle:
                    handle.write(body)
                paths.append(winning_path)
                for other_id in variant_ids:
                    if other_id == variant_id:
                        continue
                    other_path = _output_path(card, other_id, root)
                    if os.path.exists(other_path):
                        os.remove(other_path)
            return VerifyOutcome(
                passed=True,
                winning_custom_id=variant_id,
                paths=paths,
                attempts=attempts,
                result=result,
            )

        # Reject: undo this variant's write before trying the next.
        _restore_original(target_path, original)

    # No variant passed (or all responses were None): leave the target as we
    # found it and hand back the last failure as retry context.
    _restore_original(target_path, original)
    return VerifyOutcome(
        passed=False,
        winning_custom_id=None,
        paths=[],
        attempts=attempts,
        result=last_result,
    )
