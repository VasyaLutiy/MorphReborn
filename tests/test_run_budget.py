"""
Budget tests: a run stops itself at its limits, and stops ORDINARILY.

``cards/budget.py`` holds the three limits (:class:`RunBudget`) and the ledger
that judges them (:class:`BudgetLedger`); the wiring that consults the ledger is
``cards.generations.run_deck``, which asks it once per generation boundary --
before anything of that generation is submitted -- and turns a refusal into an
ordinary finish rather than an error.

These tests drive that wiring end to end, in the harness style of
``tests/test_generations.py``: a fake backend, no network, no real sleeping, a
temporary copy of the fixture project as the run's root. Each limit is
exercised on its own, against a three-card chain that degrades into three
generations of one card, plus the control that proves ``budget=None`` is
today's behaviour exactly.

What makes a budgeted stop ORDINARY, and what every test here asserts:
``run_deck`` RETURNS its :class:`DeckResult` and raises nothing; the cards that
never ran carry status ``cards.budget.STATUS_BUDGET_EXCEEDED`` with a reason
naming the limit and its number; the cards that DID run keep the outcomes they
earned, files included; and the fake backend was asked for no batch after the
refusal -- nothing was spent past the line. The deadline test never sleeps: it
injects a fake clock, and the fake backend advances that clock by one batch's
worth of seconds per submit, so the deadline trips exactly where the boundary
check sits -- between generations.
"""

import os
import shutil
import tempfile
import unittest

from cards.budget import STATUS_BUDGET_EXCEEDED, RunBudget
from cards.generations import DeckResult, run_deck
from cards.schema import MorphCard


MINIPROJECT = os.path.join("tests", "fixtures", "miniproject")


def _card(custom_id, target, depends_on=None, context_slice=None, acceptance=None):
    """A card fixture in the flat dict shape ``MorphCard.from_dict`` accepts."""
    data = {
        "custom_id": custom_id,
        "intent": "generate",
        "target": target,
        "instruction": "do it",
        "context_slice": context_slice or ["util.py"],
    }
    if depends_on:
        data["depends_on"] = depends_on
    if acceptance:
        data["acceptance"] = acceptance
    return MorphCard.from_dict(data)


def _code_block(body):
    """A fenced response whose single code block carries ``body``."""
    return f"```python\n{body}\n```"


class FakeBatchBackend:
    """Records every submit and replays scripted results.

    ``scripts`` maps a (variant) custom_id to the response text ``collect``
    returns for it; an explicit ``None`` models a failed request, and an id
    absent from the map falls back to ``default_response``. ``clock`` and
    ``tick`` are the deadline test's seam: when ``clock`` is given, every
    ``submit`` advances it by ``tick`` seconds -- the stand-in for the wall
    time a real provider batch costs, read by the run's injected clock -- so
    the deadline trips BETWEEN generations, exactly where the boundary check
    sits, and no test ever sleeps.
    """

    def __init__(self, scripts=None, default_response=None, clock=None, tick=0.0):
        self.scripts = dict(scripts or {})
        self.default_response = default_response
        self._clock = clock
        self._tick = tick
        self.submissions = []          # list of the request lists, per submit
        self.submitted_ids = []        # flat list of every custom_id submitted
        self._counter = 0

    def submit(self, requests):
        self.submissions.append(requests)
        for request in requests:
            self.submitted_ids.append(request["custom_id"])
        if self._clock is not None:
            self._clock[0] += self._tick
        self._counter += 1
        return f"fake-batch-{self._counter}"

    def status(self, batch_id):
        return "completed"

    def collect(self, batch_id):
        index = int(batch_id.rsplit("-", 1)[1]) - 1
        results = {}
        for request in self.submissions[index]:
            custom_id = request["custom_id"]
            results[custom_id] = self.scripts.get(custom_id, self.default_response)
        return results


class BudgetRunTests(unittest.TestCase):
    """Each limit on its own, against a three-card chain in three generations."""

    def setUp(self):
        # A private, writable copy of the fixture: morphs land here, not in the
        # repo tree.
        self.tmp = tempfile.mkdtemp(prefix="morph-budget-")
        self.root = os.path.join(self.tmp, "miniproject")
        shutil.copytree(MINIPROJECT, self.root,
                        ignore=shutil.ignore_patterns("node_modules"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _read(self, name):
        with open(os.path.join(self.root, name), "r", encoding="utf-8") as handle:
            return handle.read()

    def _chain(self):
        """a -> b -> c: three generations of one card each."""
        return [
            _card("a", "gen_a.py"),
            _card("b", "gen_b.py", depends_on=["a"], context_slice=["gen_a.py"]),
            _card("c", "gen_c.py", depends_on=["b"], context_slice=["gen_b.py"]),
        ]

    def _scripts(self):
        return {
            "a": _code_block("A_OK = 1"),
            "b": _code_block("B_OK = 2"),
            "c": _code_block("C_OK = 3"),
        }

    # -- max_cards -----------------------------------------------------------

    def test_max_cards_stops_before_the_generation_that_would_overspend(self):
        # max_cards=2. Generation 1 leaves the run at one card spent, generation
        # 2 at two -- exactly AT the limit, which is allowed (the ledger's rule
        # is strictly greater). Generation 3 would make it three, and the
        # boundary check refuses BEFORE anything of it is submitted.
        cards = self._chain()
        backend = FakeBatchBackend(scripts=self._scripts())

        result = run_deck(cards, backend, root=self.root, poll_interval=0,
                          budget=RunBudget(max_cards=2))

        self.assertIsInstance(result, DeckResult)   # returned, nothing raised
        self.assertEqual(result.outcomes["a"].status, "written")
        self.assertEqual(result.outcomes["b"].status, "written")
        self.assertEqual(result.outcomes["c"].status, STATUS_BUDGET_EXCEEDED)
        # The reason names the limit and its number.
        self.assertIn("max_cards", result.outcomes["c"].reason)
        self.assertIn("2", result.outcomes["c"].reason)
        # Nothing was spent after the refusal: exactly two batches asked for,
        # and the third card never reached the backend.
        self.assertEqual(len(backend.submissions), 2)
        self.assertNotIn("c", backend.submitted_ids)
        # The cards that DID run keep the outcomes they earned.
        self.assertEqual(self._read("gen_a.py"), "A_OK = 1\n")
        self.assertEqual(self._read("gen_b.py"), "B_OK = 2\n")
        self.assertEqual(result.outcomes["a"].paths,
                         [os.path.join(self.root, "gen_a.py")])

    # -- max_regenerations ---------------------------------------------------

    def test_max_regenerations_stop_before_the_next_generation(self):
        # The first generation's card fails acceptance twice and passes on its
        # second retry -- two regeneration batches, both spent INSIDE generation
        # 1, where the budget deliberately does not look. Generation 2's
        # boundary check counts them (len(batch_ids) - generations submitted)
        # against a lower max_regenerations and refuses: b and c never run --
        # and they are budget-exceeded, NOT skipped, because it was the ledger
        # that stopped the run and not a failed dependency.
        cards = [
            _card("a", "gen_a.py", acceptance="grep -q GOOD gen_a.py"),
            _card("b", "gen_b.py", depends_on=["a"], context_slice=["gen_a.py"]),
            _card("c", "gen_c.py", depends_on=["b"], context_slice=["gen_b.py"]),
        ]
        backend = FakeBatchBackend(scripts={
            "a": _code_block("BAD = 1"),
            "a.r1": _code_block("ALSO_BAD = 1"),
            "a.r2": _code_block("GOOD = 1"),
        })

        result = run_deck(cards, backend, root=self.root, poll_interval=0,
                          budget=RunBudget(max_regenerations=1))

        self.assertIsInstance(result, DeckResult)
        # The card that burned the regenerations finished ordinarily...
        self.assertEqual(result.outcomes["a"].status, "written")
        self.assertEqual(result.outcomes["a"].attempts, 3)
        self.assertEqual(result.outcomes["a"].winning_variant, "a.r2")
        self.assertEqual(self._read("gen_a.py"), "GOOD = 1\n")
        # ... and the unrun cards carry the budget's stamp.
        self.assertEqual(result.outcomes["b"].status, STATUS_BUDGET_EXCEEDED)
        self.assertEqual(result.outcomes["c"].status, STATUS_BUDGET_EXCEEDED)
        self.assertIn("max_regenerations", result.outcomes["b"].reason)
        self.assertIn("1", result.outcomes["b"].reason)
        # One generation batch plus the two retry batches the budget let happen
        # inside generation 1 -- and nothing for generation 2.
        self.assertEqual(len(backend.submissions), 3)
        self.assertNotIn("b", backend.submitted_ids)
        self.assertNotIn("c", backend.submitted_ids)

    # -- deadline_seconds ----------------------------------------------------

    def test_deadline_stops_the_run_between_generations(self):
        # The injected clock reads 0 when the ledger is built; every submit
        # costs a fake 60 seconds. Generation 1 is inside the 10-second
        # deadline; the boundary check before generation 2 reads an hour spent
        # and refuses. No real waiting anywhere.
        clock = [0.0]
        cards = self._chain()
        backend = FakeBatchBackend(scripts=self._scripts(), clock=clock, tick=60.0)

        result = run_deck(cards, backend, root=self.root, poll_interval=0,
                          budget=RunBudget(deadline_seconds=10.0),
                          now=lambda: clock[0])

        self.assertIsInstance(result, DeckResult)
        self.assertEqual(result.outcomes["a"].status, "written")
        self.assertEqual(self._read("gen_a.py"), "A_OK = 1\n")
        self.assertEqual(result.outcomes["b"].status, STATUS_BUDGET_EXCEEDED)
        self.assertEqual(result.outcomes["c"].status, STATUS_BUDGET_EXCEEDED)
        self.assertIn("deadline_seconds", result.outcomes["b"].reason)
        self.assertIn("10", result.outcomes["b"].reason)
        self.assertEqual(len(backend.submissions), 1)
        self.assertNotIn("b", backend.submitted_ids)

    # -- the control ---------------------------------------------------------

    def test_no_budget_runs_to_the_end_exactly_as_before(self):
        cards = self._chain()
        backend = FakeBatchBackend(scripts=self._scripts())

        result = run_deck(cards, backend, root=self.root, poll_interval=0,
                          budget=None)

        self.assertIsInstance(result, DeckResult)
        self.assertEqual([outcome.status for outcome in result.outcomes.values()],
                         ["written", "written", "written"])
        self.assertEqual(len(backend.submissions), 3)
        self.assertEqual(result.generations, [["a"], ["b"], ["c"]])
        self.assertEqual(self._read("gen_a.py"), "A_OK = 1\n")
        self.assertEqual(self._read("gen_b.py"), "B_OK = 2\n")
        self.assertEqual(self._read("gen_c.py"), "C_OK = 3\n")


if __name__ == "__main__":
    unittest.main()
