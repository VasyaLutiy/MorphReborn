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
  fails, and the combined stdout+stderr is preserved (tail-truncated) as the
  error context a retry generation feeds back to the executor.
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
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from cards.generations import _output_path, _variant_ids, response_to_file_body
from cards.schema import MorphCard


# The combined stdout+stderr of an acceptance run is kept only as failure
# context for a retry; a runaway test log would otherwise bloat the next
# prompt, so we keep the tail (where the actual failure usually is).
_OUTPUT_TAIL_CAP = 4000


def _tail(output: str) -> str:
    """The last :data:`_OUTPUT_TAIL_CAP` characters of ``output``."""
    if len(output) > _OUTPUT_TAIL_CAP:
        return output[-_OUTPUT_TAIL_CAP:]
    return output


# -- the mechanical check ----------------------------------------------------


@dataclass
class AcceptanceResult:
    """The outcome of running one acceptance command.

    ``passed`` is true only on a clean exit 0. ``exit_code`` is the process
    return code, or ``None`` when the command timed out (``timed_out`` true).
    ``output`` is the combined stdout+stderr, tail-truncated to
    :data:`_OUTPUT_TAIL_CAP` characters -- it is the error context a failed
    card carries into its retry generation.
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
    tail-truncated to :data:`_OUTPUT_TAIL_CAP` characters.
    """
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        # exc.output holds the bytes/str captured before the timeout kill; it
        # may be empty, and is bytes when the process wrote non-decodable data.
        output = exc.output or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return AcceptanceResult(
            passed=False, exit_code=None, output=_tail(output), timed_out=True
        )

    return AcceptanceResult(
        passed=(completed.returncode == 0),
        exit_code=completed.returncode,
        output=_tail(completed.stdout or ""),
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
    deleted if a rejected variant left one behind.
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
        with open(target_path, "w", encoding="utf-8") as handle:
            handle.write(body)

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
