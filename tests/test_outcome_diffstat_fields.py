"""
The two git-mode reporting fields on :class:`cards.generations.CardOutcome`.

``commit`` (the sha of the commit a card's files became) and ``diffstat``
(what ``cards.repo.diffstat`` returned for that commit) are filled only for a
card that BECAME a commit -- a run in git mode, through the accepted-card
hook -- and stay ``None`` for everything else. Both default to ``None``, which
is the whole point: not one existing construction site of ``CardOutcome`` had
to change. These tests pin that contract from the outside:

* every argument combination the codebase already builds an outcome with still
  constructs, and both new fields come out ``None``;
* both fields accept and keep a sha and a list of per-file diffstat dicts;
* both fields are settable AFTER construction -- ``CardOutcome`` is a mutable
  dataclass, and the git layer fills them in once the commit exists, so a
  later card depends on that being possible.
"""

from cards.generations import CardOutcome


# A plausible 40-hex git sha.
SHA = "0f1e2d3c4b5a69788796a5b4c3d2e1f001234567"


def _diffstat():
    """A plausible ``cards.repo.diffstat`` return: one dict per changed file."""
    return [
        {"path": "gen_a.py", "insertions": 5, "deletions": 0},
        {"path": "util.py", "insertions": 1, "deletions": 2},
    ]


class TestConstructionUnchanged:
    """Every construction site that predates the fields still constructs,
    with both new fields None."""

    def test_the_two_positional_arguments_every_call_site_passes(self):
        # Every construction in cards/generations.py passes custom_id and
        # status positionally and everything else as keywords; the bare form
        # must keep constructing untouched.
        outcome = CardOutcome("a", "written")
        assert outcome.custom_id == "a"
        assert outcome.status == "written"
        assert outcome.commit is None
        assert outcome.diffstat is None

    def test_the_widest_argument_set_callers_used_before(self):
        # The largest construction in the codebase -- a card accepted on a
        # retry: paths, attempts, winning_variant and earlier_failures (plus
        # reason / acceptance_output on the failure sites). Every one of those
        # keywords must still be accepted, and the new fields must come out
        # None because this call does not mention them.
        outcome = CardOutcome(
            "a",
            "written",
            paths=["gen_a.py"],
            reason=None,
            attempts=2,
            winning_variant="a.r1.v1",
            acceptance_output=None,
            earlier_failures="1 failed, 0 passed",
        )
        assert outcome.custom_id == "a"
        assert outcome.status == "written"
        assert outcome.paths == ["gen_a.py"]
        assert outcome.reason is None
        assert outcome.attempts == 2
        assert outcome.winning_variant == "a.r1.v1"
        assert outcome.acceptance_output is None
        assert outcome.earlier_failures == "1 failed, 0 passed"
        assert outcome.commit is None
        assert outcome.diffstat is None

    def test_a_failure_site_with_reason_and_acceptance_output(self):
        outcome = CardOutcome("a", "failed",
                              reason="stale-context",
                              acceptance_output="the file moved under the card")
        assert outcome.status == "failed"
        assert outcome.reason == "stale-context"
        assert outcome.commit is None
        assert outcome.diffstat is None

    def test_a_skipped_site_with_attempts_zero(self):
        outcome = CardOutcome("b", "skipped", reason="a", attempts=0)
        assert outcome.status == "skipped"
        assert outcome.attempts == 0
        assert outcome.commit is None
        assert outcome.diffstat is None


class TestFieldsHoldValues:
    """Both fields accept and keep a sha and a list of diffstat dicts."""

    def test_commit_keeps_the_sha_it_was_given(self):
        outcome = CardOutcome("a", "written", commit=SHA)
        assert outcome.commit == SHA

    def test_diffstat_keeps_the_list_of_dicts_it_was_given(self):
        diffstat = _diffstat()
        outcome = CardOutcome("a", "written", diffstat=diffstat)
        # The list as given -- same object, real dicts, not stringified.
        assert outcome.diffstat is diffstat
        assert outcome.diffstat == diffstat
        assert all(isinstance(entry, dict) for entry in outcome.diffstat)

    def test_both_fields_at_once(self):
        outcome = CardOutcome("a", "written", commit=SHA, diffstat=_diffstat())
        assert outcome.commit == SHA
        assert len(outcome.diffstat) == 2

    def test_a_default_constructed_outcome_reports_them_none(self):
        outcome = CardOutcome("a", "failed")
        assert outcome.commit is None
        assert outcome.diffstat is None


class TestSettableAfterConstruction:
    """CardOutcome is a mutable dataclass: the accepted-card hook fills the
    fields in AFTER the card is accepted, so setting them post hoc must work."""

    def test_commit_can_be_set_after_construction(self):
        outcome = CardOutcome("a", "written")
        outcome.commit = SHA
        assert outcome.commit == SHA

    def test_diffstat_can_be_set_after_construction(self):
        outcome = CardOutcome("a", "written")
        diffstat = _diffstat()
        outcome.diffstat = diffstat
        assert outcome.diffstat == diffstat

    def test_filling_them_in_does_not_disturb_the_run_report_line(self):
        # The two fields are pure reporting additions: ``__str__`` renders the
        # outcome exactly as before they existed, whether or not they are set.
        outcome = CardOutcome("a", "written", paths=["gen_a.py"])
        rendered_before = str(outcome)
        outcome.commit = SHA
        outcome.diffstat = _diffstat()
        assert str(outcome) == rendered_before
        assert "gen_a.py" in rendered_before
