"""
Tests for ``cards/budget.py``: the three run limits, the ledger that
checks them, and the exception payload a run copies into the outcomes of
the cards it did not get to. Everything runs on a fake clock -- a
callable the test advances by hand -- so no test sleeps, and nothing
here touches a store, a provider, a subprocess or a network: the module
under test is pure logic and the tests hold it to that.
"""

import unittest
from dataclasses import FrozenInstanceError

from cards.budget import (
    STATUS_BUDGET_EXCEEDED,
    BudgetExceeded,
    BudgetLedger,
    RunBudget,
)


class FakeClock:
    """A stand-in for ``time.monotonic`` that the test advances by hand.

    Returns whatever moment it last advanced to; :meth:`advance` moves it
    forward without any sleeping, so a test walks a ledger through hours
    in microseconds.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.moment = start

    def __call__(self) -> float:
        return self.moment

    def advance(self, seconds: float) -> None:
        """Move the clock forward by ``seconds``, instantly."""
        self.moment += seconds


class StatusConstantTests(unittest.TestCase):
    """The status a budget-stopped run stamps onto the cards it never ran."""

    def test_is_the_literal_string(self) -> None:
        """The constant is spelled exactly as its consumer card expects."""
        self.assertEqual(STATUS_BUDGET_EXCEEDED, "budget-exceeded")


class RunBudgetShapeTests(unittest.TestCase):
    """RunBudget is a frozen dataclass whose fields all default to None."""

    def test_fields_default_to_none(self) -> None:
        """A bare RunBudget sets no limit of any kind."""
        budget = RunBudget()
        self.assertIsNone(budget.max_cards)
        self.assertIsNone(budget.max_regenerations)
        self.assertIsNone(budget.deadline_seconds)

    def test_frozen(self) -> None:
        """A budget cannot be quietly loosened once the run holds it."""
        budget = RunBudget(max_cards=3)
        with self.assertRaises(FrozenInstanceError):
            budget.max_cards = 4


class NoLimitTests(unittest.TestCase):
    """No budget at all, or a budget of all Nones: nothing ever refuses."""

    def test_none_budget_never_raises(self) -> None:
        """A ledger with no budget permits any count after any elapsed time."""
        clock = FakeClock()
        ledger = BudgetLedger(None, now=clock)
        clock.advance(10_000.0)
        self.assertIsNone(ledger.check(10_000, 10_000))
        self.assertEqual(ledger.elapsed, 10_000.0)

    def test_all_none_run_budget_never_raises(self) -> None:
        """An all-None RunBudget is, to the ledger, no limits at all."""
        clock = FakeClock()
        ledger = BudgetLedger(RunBudget(), now=clock)
        clock.advance(10_000.0)
        self.assertIsNone(ledger.check(10_000, 10_000))


class MaxCardsTests(unittest.TestCase):
    """The card limit on its own: at the limit passes, one past refuses."""

    BUDGET = RunBudget(max_cards=10)

    def test_at_the_limit_passes(self) -> None:
        """A prospective count exactly AT max_cards is allowed."""
        ledger = BudgetLedger(self.BUDGET, now=FakeClock())
        self.assertIsNone(ledger.check(10, 0))

    def test_one_past_the_limit_raises(self) -> None:
        """A prospective count one past max_cards refuses, naming it."""
        ledger = BudgetLedger(self.BUDGET, now=FakeClock())
        with self.assertRaises(BudgetExceeded) as caught:
            ledger.check(11, 0)
        self.assertEqual(caught.exception.limit, "max_cards")

    def test_other_dimensions_are_not_capped_by_it(self) -> None:
        """Without its own limit, neither time nor regenerations refuse."""
        clock = FakeClock()
        ledger = BudgetLedger(self.BUDGET, now=clock)
        clock.advance(10_000.0)
        self.assertIsNone(ledger.check(10, 10_000))


class MaxRegenerationsTests(unittest.TestCase):
    """The regeneration limit on its own: at the limit passes, one past
    refuses."""

    BUDGET = RunBudget(max_regenerations=5)

    def test_at_the_limit_passes(self) -> None:
        """A prospective regeneration count exactly AT the limit is allowed."""
        ledger = BudgetLedger(self.BUDGET, now=FakeClock())
        self.assertIsNone(ledger.check(0, 5))

    def test_one_past_the_limit_raises(self) -> None:
        """One regeneration past the limit refuses, naming it."""
        ledger = BudgetLedger(self.BUDGET, now=FakeClock())
        with self.assertRaises(BudgetExceeded) as caught:
            ledger.check(0, 6)
        self.assertEqual(caught.exception.limit, "max_regenerations")

    def test_other_dimensions_are_not_capped_by_it(self) -> None:
        """Without its own limit, neither time nor cards refuse."""
        clock = FakeClock()
        ledger = BudgetLedger(self.BUDGET, now=clock)
        clock.advance(10_000.0)
        self.assertIsNone(ledger.check(10_000, 5))


class DeadlineTests(unittest.TestCase):
    """The wall-clock limit on its own: at the deadline passes, past it
    refuses."""

    BUDGET = RunBudget(deadline_seconds=60.0)

    def test_at_the_deadline_passes(self) -> None:
        """Elapsed seconds exactly AT deadline_seconds are allowed."""
        clock = FakeClock()
        ledger = BudgetLedger(self.BUDGET, now=clock)
        clock.advance(60.0)
        self.assertIsNone(ledger.check(0, 0))

    def test_past_the_deadline_raises(self) -> None:
        """Elapsed seconds beyond deadline_seconds refuse, naming it."""
        clock = FakeClock()
        ledger = BudgetLedger(self.BUDGET, now=clock)
        clock.advance(60.5)
        with self.assertRaises(BudgetExceeded) as caught:
            ledger.check(0, 0)
        self.assertEqual(caught.exception.limit, "deadline_seconds")

    def test_counts_do_not_matter_to_it(self) -> None:
        """Within the deadline, any counts pass; the clock alone refuses."""
        clock = FakeClock()
        ledger = BudgetLedger(self.BUDGET, now=clock)
        clock.advance(1.0)
        self.assertIsNone(ledger.check(10_000, 10_000))


class FirstExceededTests(unittest.TestCase):
    """When several limits trip at once, the FIRST one exceeded is named.

    The order is fixed: :class:`RunBudget`'s field order -- ``max_cards``,
    then ``max_regenerations``, then ``deadline_seconds``.
    """

    def test_counts_at_both_limits_pass(self) -> None:
        """Two counts exactly at their limits are both allowed."""
        ledger = BudgetLedger(RunBudget(max_cards=10, max_regenerations=5),
                              now=FakeClock())
        self.assertIsNone(ledger.check(10, 5))

    def test_cards_are_checked_before_regenerations(self) -> None:
        """Both counts over their limits: the card limit is the one named."""
        ledger = BudgetLedger(RunBudget(max_cards=10, max_regenerations=5),
                              now=FakeClock())
        with self.assertRaises(BudgetExceeded) as caught:
            ledger.check(11, 6)
        self.assertEqual(caught.exception.limit, "max_cards")

    def test_regenerations_are_checked_before_deadline(self) -> None:
        """Regenerations over and time spent: the regeneration limit wins."""
        clock = FakeClock()
        ledger = BudgetLedger(RunBudget(max_regenerations=5,
                                        deadline_seconds=60.0), now=clock)
        clock.advance(120.0)
        with self.assertRaises(BudgetExceeded) as caught:
            ledger.check(1, 6)
        self.assertEqual(caught.exception.limit, "max_regenerations")

    def test_deadline_when_counts_are_within_limits(self) -> None:
        """Counts fine, time spent: the deadline is the one named."""
        clock = FakeClock()
        ledger = BudgetLedger(RunBudget(max_cards=10, max_regenerations=5,
                                        deadline_seconds=60.0), now=clock)
        clock.advance(61.0)
        with self.assertRaises(BudgetExceeded) as caught:
            ledger.check(1, 1)
        self.assertEqual(caught.exception.limit, "deadline_seconds")


class ExceptionPayloadTests(unittest.TestCase):
    """What the run's top-level handler copies into the card outcomes.

    ``.limit``, ``.allowed``, ``.reached`` and the ``.reason`` sentence,
    for each of the three limits.
    """

    def test_card_limit_payload(self) -> None:
        """The card refusal carries the limit, both numbers, the sentence."""
        ledger = BudgetLedger(RunBudget(max_cards=100), now=FakeClock())
        with self.assertRaises(BudgetExceeded) as caught:
            ledger.check(101, 0)
        error = caught.exception
        self.assertEqual(error.limit, "max_cards")
        self.assertEqual(error.allowed, 100)
        self.assertEqual(error.reached, 101)
        self.assertEqual(error.reason,
                         "max_cards exceeded: allowed 100, reached 101")

    def test_regeneration_limit_payload(self) -> None:
        """The regeneration refusal carries limit, numbers and sentence."""
        ledger = BudgetLedger(RunBudget(max_regenerations=8), now=FakeClock())
        with self.assertRaises(BudgetExceeded) as caught:
            ledger.check(0, 9)
        error = caught.exception
        self.assertEqual(error.limit, "max_regenerations")
        self.assertEqual(error.allowed, 8)
        self.assertEqual(error.reached, 9)
        self.assertEqual(error.reason,
                         "max_regenerations exceeded: allowed 8, reached 9")

    def test_deadline_payload(self) -> None:
        """The deadline refusal carries the seconds budgeted and elapsed."""
        clock = FakeClock()
        ledger = BudgetLedger(RunBudget(deadline_seconds=60.0), now=clock)
        clock.advance(61.5)
        with self.assertRaises(BudgetExceeded) as caught:
            ledger.check(0, 0)
        error = caught.exception
        self.assertEqual(error.limit, "deadline_seconds")
        self.assertEqual(error.allowed, 60.0)
        self.assertEqual(error.reached, 61.5)
        self.assertEqual(
            error.reason,
            "deadline_seconds exceeded: allowed 60.0, reached 61.5")

    def test_str_is_the_reason(self) -> None:
        """str(exc) is the same sentence the outcome's reason field gets."""
        ledger = BudgetLedger(RunBudget(max_cards=1), now=FakeClock())
        with self.assertRaises(BudgetExceeded) as caught:
            ledger.check(2, 0)
        self.assertEqual(str(caught.exception), caught.exception.reason)


class ElapsedTests(unittest.TestCase):
    """``elapsed`` measures from construction, by the injected clock."""

    def test_zero_at_construction(self) -> None:
        """A ledger just built has spent no seconds."""
        ledger = BudgetLedger(RunBudget(), now=FakeClock())
        self.assertEqual(ledger.elapsed, 0.0)

    def test_grows_with_the_fake_clock(self) -> None:
        """Each advance of the fake clock is a later elapsed reading."""
        clock = FakeClock()
        ledger = BudgetLedger(RunBudget(), now=clock)
        clock.advance(30.0)
        self.assertEqual(ledger.elapsed, 30.0)
        clock.advance(1.5)
        self.assertEqual(ledger.elapsed, 31.5)

    def test_measured_from_construction_not_from_clock_zero(self) -> None:
        """Time spent before the ledger existed is not charged to it."""
        clock = FakeClock()
        clock.advance(500.0)
        ledger = BudgetLedger(RunBudget(), now=clock)
        clock.advance(30.0)
        self.assertEqual(ledger.elapsed, 30.0)


if __name__ == "__main__":
    unittest.main()
