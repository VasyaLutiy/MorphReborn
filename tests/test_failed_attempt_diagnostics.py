"""Diagnostics for cards that were accepted only after failed acceptance tries.

A card that fails acceptance and is regenerated carries, once it finally
passes, the captured output of its LAST failed attempt in the new
``CardOutcome.earlier_failures`` field -- the evidence an operator needs to fix
a bad acceptance criterion, which ``acceptance_output`` cannot give (it is set
only when the card finally failed, and keeps exactly that meaning). These tests
drive :func:`cards.generations.run_deck` and
:func:`cards.generations.process_retry_batch` against a scripted backend and a
real acceptance command run in a temporary project root, and pin:

* a card accepted on a retry carries the earlier failure's output, while
  ``acceptance_output`` stays ``None``;
* a card accepted on its first attempt has ``earlier_failures`` ``None``;
* a card that failed for good still reports through ``acceptance_output`` as
  before, and ``earlier_failures`` stays ``None``.
"""

import os

from cards.acceptance import AcceptanceResult
from cards.generations import (
    CardOutcome,
    build_retry_cards,
    process_retry_batch,
    run_deck,
)
from cards.schema import MorphCard


# The checker installed into the temporary project root. It rejects any body
# that does not contain "good" and ECHOES THE BODY IT JUDGED into its output,
# so every failed attempt's captured output carries a marker unique to the body
# that produced it -- which is what lets the tests tell the LAST failed
# attempt's output from an earlier one's. The message goes to both streams, so
# the tests do not depend on which of stdout/stderr the acceptance runner
# captures.
_CHECKER_SOURCE = """import sys

body = open(sys.argv[1]).read()
message = "CHECK:" + body.strip()
sys.stdout.write(message + "\\n")
sys.stderr.write(message + "\\n")
sys.exit(0 if "good" in body else 1)
"""

# Bodies that fail the checker, each carrying a marker unique to its attempt.
FIRST_FAILURE_BODY = "VALUE = 'first-attempt-rejected'"
SECOND_FAILURE_BODY = "VALUE = 'second-attempt-rejected'"
THIRD_FAILURE_BODY = "VALUE = 'third-attempt-rejected'"
PASSING_BODY = "VALUE = 'good'"


class ScriptedBackend:
    """A duck-typed batch backend handing out pre-written results in order.

    ``batches`` is a list of ``{custom_id: response|None}`` maps, one per
    ``submit`` call in order: the generation batch first, then one entry per
    regeneration batch, under the ids ``run_deck`` and its retry loop mint
    (``<card>`` for a single-variant card, ``<card>.r<attempt>`` for its
    regenerations).
    """

    def __init__(self, batches):
        self._batches = list(batches)
        self._issued = []

    def submit(self, requests):
        batch_id = "batch-%d" % len(self._issued)
        self._issued.append(batch_id)
        return batch_id

    def status(self, batch_id):
        return "completed"

    def collect(self, batch_id):
        return self._batches[self._issued.index(batch_id)]


def _install_checker(root: str, target: str) -> str:
    """Write the checker into ``root``; return the acceptance command for it."""
    with open(os.path.join(root, "acceptance_check.py"), "w",
              encoding="utf-8") as handle:
        handle.write(_CHECKER_SOURCE)
    return "python3 acceptance_check.py %s" % target


def _card(custom_id: str, target: str, acceptance: str) -> MorphCard:
    """A single-variant card judged by ``acceptance``, in the backlog shape."""
    return MorphCard.from_dict({
        "custom_id": custom_id,
        "meta": {
            "intent": "generate",
            "target": target,
            "acceptance": acceptance,
        },
        "instruction": "Write the complete file %s." % target,
    })


def _response(body: str) -> str:
    """A model answer carrying ``body`` as one fenced python block."""
    return "Here is the file:\n\n```python\n" + body + "\n```\n"


def _run(backend, root: str, card):
    return run_deck(
        [card], backend, root=root, poll_interval=0.0, log=lambda _line: None,
        verify=True, acceptance_timeout=60.0, max_regenerations=2)


def test_card_accepted_on_retry_carries_the_earlier_failure_output(tmp_path):
    root = str(tmp_path)
    card = _card("c1", "out.py", _install_checker(root, "out.py"))
    backend = ScriptedBackend([
        {"c1": _response(FIRST_FAILURE_BODY)},      # generation: fails acceptance
        {"c1.r1": _response(PASSING_BODY)},         # regeneration 1: passes
    ])

    result = _run(backend, root, card)

    outcome = result.outcomes["c1"]
    assert outcome.status == "written"
    assert outcome.attempts == 2
    assert outcome.winning_variant == "c1.r1"
    assert outcome.earlier_failures is not None
    assert "first-attempt-rejected" in outcome.earlier_failures
    # acceptance_output keeps its exact old meaning: set on a terminal failure
    # only, so a card that eventually passed reports None there.
    assert outcome.acceptance_output is None
    assert outcome.paths == [os.path.join(root, "out.py")]
    with open(os.path.join(root, "out.py"), encoding="utf-8") as handle:
        assert "good" in handle.read()


def test_card_accepted_on_first_attempt_has_earlier_failures_none(tmp_path):
    root = str(tmp_path)
    card = _card("c1", "out.py", _install_checker(root, "out.py"))
    backend = ScriptedBackend([
        {"c1": _response(PASSING_BODY)},            # generation: passes at once
    ])

    result = _run(backend, root, card)

    outcome = result.outcomes["c1"]
    assert outcome.status == "written"
    assert outcome.attempts == 1
    assert outcome.earlier_failures is None
    assert outcome.acceptance_output is None


def test_earlier_failures_holds_the_last_failed_attempt_not_the_first(tmp_path):
    root = str(tmp_path)
    card = _card("c1", "out.py", _install_checker(root, "out.py"))
    backend = ScriptedBackend([
        {"c1": _response(FIRST_FAILURE_BODY)},      # generation: fails
        {"c1.r1": _response(SECOND_FAILURE_BODY)},  # regeneration 1: fails
        {"c1.r2": _response(PASSING_BODY)},         # regeneration 2: passes
    ])

    result = _run(backend, root, card)

    outcome = result.outcomes["c1"]
    assert outcome.status == "written"
    assert outcome.attempts == 3
    assert "second-attempt-rejected" in outcome.earlier_failures
    assert "first-attempt-rejected" not in outcome.earlier_failures
    assert outcome.acceptance_output is None


def test_card_failed_for_good_still_reports_acceptance_output(tmp_path):
    root = str(tmp_path)
    card = _card("c1", "out.py", _install_checker(root, "out.py"))
    backend = ScriptedBackend([
        {"c1": _response(FIRST_FAILURE_BODY)},      # generation: fails
        {"c1.r1": _response(SECOND_FAILURE_BODY)},  # regeneration 1: fails
        {"c1.r2": _response(THIRD_FAILURE_BODY)},   # regeneration 2: fails
    ])

    result = _run(backend, root, card)

    outcome = result.outcomes["c1"]
    assert outcome.status == "failed"
    assert outcome.attempts == 3
    # Unchanged semantics: the FINAL attempt's captured output sits on
    # acceptance_output, and the new field stays None -- there is no accepted
    # card whose earlier struggles it would report.
    assert outcome.acceptance_output is not None
    assert "third-attempt-rejected" in outcome.acceptance_output
    assert "second-attempt-rejected" not in outcome.acceptance_output
    assert outcome.earlier_failures is None


def test_card_without_acceptance_keeps_earlier_failures_none(tmp_path):
    root = str(tmp_path)
    card = MorphCard.from_dict({
        "custom_id": "plain",
        "meta": {"intent": "generate", "target": "out.py"},
        "instruction": "Write the complete file out.py.",
    })
    backend = ScriptedBackend([
        {"plain": _response(PASSING_BODY)},
    ])

    result = _run(backend, root, card)

    outcome = result.outcomes["plain"]
    assert outcome.status == "written"
    assert outcome.earlier_failures is None
    assert outcome.acceptance_output is None


def test_process_retry_batch_records_the_pending_result_as_earlier_failures(tmp_path):
    """The population point itself: the pending pair's result becomes the field.

    Drives :func:`cards.generations.process_retry_batch` directly, with a
    hand-built failing :class:`cards.acceptance.AcceptanceResult` as the
    previous attempt, so the EXACT value -- not just a marker of it -- can be
    asserted. This is the same judging the persisted-retry route
    (``cards.store.collect_generation``) goes through.
    """
    root = str(tmp_path)
    card = _card("c1", "out.py", _install_checker(root, "out.py"))
    failing = AcceptanceResult(
        passed=False, exit_code=1,
        output="CHECK:first-attempt-rejected", timed_out=False)
    retry_card = build_retry_cards(
        [(card, failing)], 1, 1, 1, 2, log=lambda _line: None)[0]

    outcomes = {}
    pending = process_retry_batch(
        [retry_card], [(card, failing)], {"c1.r1": _response(PASSING_BODY)},
        1, 1, 1, root, lambda _line: None, 60.0, 2, outcomes, set())

    assert pending == []
    outcome = outcomes["c1"]
    assert outcome.status == "written"
    assert outcome.attempts == 2
    assert outcome.earlier_failures == "CHECK:first-attempt-rejected"
    assert outcome.acceptance_output is None


def test_earlier_failures_defaults_to_none():
    """The field is additive: a bare outcome carries no earlier-failure text."""
    outcome = CardOutcome(custom_id="x", status="written")
    assert outcome.earlier_failures is None
