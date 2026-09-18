"""
The limits an unattended run gives itself: a budget, the ledger that
checks it while the run runs, and the outcome status the cards that
never got to run receive. Pure logic -- stdlib only, no I/O, no clock a
test cannot replace, and no import from ``cards.store``,
``cards.generations``, ``flows`` or ``processors`` at all: every one of
those must be able to import THIS module, so this module imports none of
them back.

WHY this exists: a run tonight has no limit of any kind. A regeneration
loop that cannot converge spends money in silence, and the only
wall-clock bound in the system is the six-hour WAIT budget of
``cards/cli_wait.py`` -- which bounds the waiting around a provider
queue, not the run itself. An unattended run needs three limits it stops
itself at: how many cards it will make (:attr:`RunBudget.max_cards`),
how many regenerations it will attempt
(:attr:`RunBudget.max_regenerations`), and how long it may run at all
(:attr:`RunBudget.deadline_seconds`).

And stopping has to be ORDINARY. Nothing here surfaces to the operator
as an error: when a ledger's :meth:`BudgetLedger.check` refuses, the run
catches :class:`BudgetExceeded` once, at the top, gives every card it
did not get to an outcome stamped :data:`STATUS_BUDGET_EXCEEDED` with
the exception's ``.reason`` -- one short sentence naming the limit and
both numbers -- and the run is archived like any other. A budgeted stop
is a finished run whose cards say why they did not run, not a crash.

The clock is a test seam, in the idiom of ``cards/cli_wait.py``:
:class:`BudgetLedger` takes a ``now`` callable defaulting to
``time.monotonic``, reads the start moment from it once at construction,
and takes every later reading from the same callable -- so a test
advances a fake clock instead of sleeping.
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional, Union


@dataclass(frozen=True)
class RunBudget:
    """The three limits a run may give itself; each ``None`` means no limit.

    Every field is Optional and defaults to ``None``, so ``RunBudget()``
    is an unlimited run and an operator budgets only the dimensions that
    worry them:

    ``max_cards`` -- the most cards the run may spend: it stops itself
    before it would have generated card number ``max_cards + 1``.

    ``max_regenerations`` -- the same bound on regeneration attempts, the
    dimension where a loop that cannot converge would otherwise spin and
    spend in silence.

    ``deadline_seconds`` -- wall-clock seconds the run may last, measured
    from the :class:`BudgetLedger`'s construction by its injected clock.

    Frozen so a running budget cannot be quietly loosened midway: the
    ledger reads these fields and never writes them.
    """

    max_cards: Optional[int] = None
    max_regenerations: Optional[int] = None
    deadline_seconds: Optional[float] = None


# The card-outcome status a run gives the cards it did not get to when a
# budget stops it: the run catches BudgetExceeded at the top, stamps every
# card that never ran with this status and the exception's ``.reason``, and
# archives the run like any other. The constant lives HERE, where the budget
# is defined, and not at the outcome-writing site, because its consumer is
# another card -- producer and consumer must share one spelling, and this
# module is the one both of them may import.
STATUS_BUDGET_EXCEEDED = "budget-exceeded"


class BudgetExceeded(Exception):
    """A run's budget says the run may not go on.

    Raised by :meth:`BudgetLedger.check` and by nothing else -- nothing
    else in this module holds a budget. Built from the limit's name, what
    it allowed and what was reached::

        BudgetExceeded("max_cards", 100, 101)

    and carries the three, plus the sentence that names them:

    ``limit`` -- exactly one of :class:`RunBudget`'s three field names:
    ``"max_cards"``, ``"max_regenerations"`` or ``"deadline_seconds"``.

    ``allowed`` -- the value of that limit: what the run was permitted.

    ``reached`` -- the prospective count, or the elapsed seconds, that
    the refusing check found.

    ``reason`` -- one short sentence naming the limit AND both numbers,
    e.g. ``"max_cards exceeded: allowed 100, reached 101"``; ``str(exc)``
    is that same sentence. It is fit to be copied straight into a card
    outcome's ``reason`` field: the run catches this exception at the
    top, gives every card it did not get to the status
    :data:`STATUS_BUDGET_EXCEEDED` with this sentence as its reason, and
    finishes ordinarily -- nothing is raised at the operator.
    """

    def __init__(self, limit: str, allowed: Union[int, float],
                 reached: Union[int, float]):
        self.limit = limit
        self.allowed = allowed
        self.reached = reached
        self.reason = f"{limit} exceeded: allowed {allowed}, reached {reached}"
        super().__init__(self.reason)


class BudgetLedger:
    """What a run keeps while it runs: its budget, a clock, and one answer.

    Constructed once, when the run starts, with the run's
    :class:`RunBudget` -- or ``None``, for a run with no limits at all --
    and an injected ``now`` in the idiom of ``cards/cli_wait.py``: the
    default is ``time.monotonic``, and tests hand in a fake clock and
    advance it instead of sleeping. The start moment is read from ``now``
    HERE, at construction, so :attr:`elapsed` and the deadline limit
    measure from the moment the run began, not from the first check.

    The two members:

    ``check(cards, regenerations)`` answers whether the run may go on.
    Its arguments are PROSPECTIVE: what the run WOULD have spent if it
    continued -- the totals after the card about to be generated, the
    regeneration about to be attempted. The rule is therefore strictly
    greater: a count exactly AT its limit is allowed (the run may spend
    up to and including it) and one past it refuses, and likewise elapsed
    seconds must EXCEED ``deadline_seconds`` to refuse. On refusal it
    raises :class:`BudgetExceeded`, which the run catches once at the
    top; on permission it returns ``None``, and the loop goes on.

    The limits are tested in a FIXED order -- ``max_cards``, then
    ``max_regenerations``, then ``deadline_seconds``: the field order of
    :class:`RunBudget` -- and the exception names the FIRST one exceeded,
    so a check that trips two limits at once diagnoses the same one every
    time.

    A ``None`` limit never refuses, and a ``None`` budget never refuses
    anything: ``BudgetLedger(None)`` is a ledger that always says yes, so
    the run's loop can call ``check`` unconditionally and stay ignorant
    of whether an operator gave a budget at all.
    """

    def __init__(self, budget: Optional[RunBudget],
                 now: Callable[[], float] = time.monotonic):
        self._budget = budget
        self._now = now
        # The start moment, read once. Every elapsed second afterwards is
        # ``now()`` minus this, by the SAME callable, so a fake clock that
        # never advances yields a ledger frozen at zero and one that jumps
        # an hour in a step yields an hour-old ledger just as fast.
        self._started_at = now()

    @property
    def elapsed(self) -> float:
        """Seconds since construction, by the injected clock.

        What the deadline limit is compared against, and what
        ``.reached`` carries when that limit refuses.
        """
        return self._now() - self._started_at

    def check(self, cards: int, regenerations: int) -> None:
        """Raise :class:`BudgetExceeded` if the run may not go on.

        ``cards`` and ``regenerations`` are the PROSPECTIVE totals -- what
        the run would have spent if it continued past this check -- so the
        comparisons are strictly greater: a count exactly at its limit
        passes, one past it refuses, and elapsed seconds must exceed
        ``deadline_seconds``. Returns ``None`` when every limit permits; a
        ``None`` limit, or a ``None`` budget, permits everything. The
        limits are tested in the fixed order of :class:`RunBudget`'s
        fields and the FIRST one exceeded is the one the exception names.
        """
        budget = self._budget
        if budget is None:
            return
        if budget.max_cards is not None and cards > budget.max_cards:
            raise BudgetExceeded("max_cards", budget.max_cards, cards)
        if (budget.max_regenerations is not None
                and regenerations > budget.max_regenerations):
            raise BudgetExceeded("max_regenerations",
                                 budget.max_regenerations, regenerations)
        deadline = budget.deadline_seconds
        if deadline is not None:
            seconds = self.elapsed
            if seconds > deadline:
                raise BudgetExceeded("deadline_seconds", deadline, seconds)
