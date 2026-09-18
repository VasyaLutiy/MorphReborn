"""
Tests for the operator's three budget flags on ``mrph run``: the last wire
in the chain that starts at ``cards/budget.py``.

``run_deck`` stops a run at its budget's limits, but an operator sending a
deck off for the night had no way to SAY what those limits are. These tests
pin the wire that gives them the say: three flags on the ``run`` subcommand
alone, folded by the handler into one ``cards.budget.RunBudget``, carried by
``cli_cycle.run`` and handed to ``run_deck`` unchanged. Only the wiring is
under test here -- the ledger, the refusal and the stamped outcomes are
``cards.budget`` behaviour and have their own tests:

* all three flags together arrive as one ``RunBudget`` with each value on
  its own field, as the ``budget=`` keyword of ``cli_cycle.run``;
* no flags at all arrive as a budget that limits nothing -- ``None``, or a
  ``RunBudget`` whose every field is ``None``;
* each flag alone leaves the other two unlimited;
* a non-numeric value never reaches a handler: it is refused at the parse
  into the one usage-error document, exit code 4, exactly one line on
  stdout, with argparse's own usage and diagnosis on stderr;
* ``cli_cycle.run`` passes the budget it received straight on to
  ``run_deck`` as a keyword.

No test goes near a network or a provider. The argv-side tests replace
``cards.cli_cycle.run`` with a recording spy, so nothing of the run executes
at all and no deck is even needed. The passthrough test runs the real
``cli_cycle.run`` -- a one-card deck added through ``deck add``, a stub
backend injected through the documented ``backend=`` seam, ``nogit=True`` --
and stops it at ``run_deck`` itself, the exact seam where the budget changes
hands: the spy raises a sentinel as soon as it has read the call, because
everything the run would do after that point is other modules' behaviour and
deliberately out of scope here.
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from cards import cli_cycle
from cards.budget import RunBudget
from cards.cli import main


class _ReachedRunDeck(Exception):
    """Raised by the run_deck spy once it has read the call it was given.

    The passthrough test's business ends at the moment the budget changes
    hands: everything ``cli_cycle.run`` would do after ``run_deck`` returns
    -- the archive, the report, the notification -- is other modules'
    behaviour, so the spy stops the run right here instead of faking a deck
    result for machinery this test does not exercise.
    """


class _StubBackend:
    """A batch backend that answers instantly and never leaves the process.

    Injected through ``cli_cycle.run``'s documented ``backend=`` seam, so the
    passthrough test never consults the processor registry; and since the
    spy has replaced ``run_deck``, the proxy built around this backend is
    never asked to do anything either. It only has to be duck-shaped.
    """

    def submit(self, *args, **kwargs):
        return "batch-stub-0001"

    def status(self, *args, **kwargs):
        return "completed"

    def collect(self, *args, **kwargs):
        return []


def _card(custom_id: str, target: str, instruction: str) -> dict:
    """A minimal valid generate card -- ``MorphCard.from_dict`` accepts it."""
    return {
        "custom_id": custom_id,
        "intent": "generate",
        "target": target,
        "instruction": instruction,
    }


def _recording_spy(captured: dict):
    """A stand-in for ``cli_cycle.run`` that records its call and answers.

    Returns the ``(payload, exit_code)`` pair a ``run_cli`` handler must
    return, so the invocation completes with exit code 0 and the one document
    on stdout; the test reads the call back out of ``captured``.
    """

    def _spy(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return ({"spied": True}, 0)

    return _spy


class TestBudgetFlags(unittest.TestCase):
    """The three flags, wired from argv to ``run_deck`` and nowhere else."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

    # -- helpers -------------------------------------------------------------

    def _run(self, argv):
        """Drive ``main`` with both streams captured.

        Returns ``(code, out, err)``: the exit code and the two captured
        strings. The redirections are per call, so each invocation sees
        exactly what its process would have.
        """
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def _document(self, out: str, what: str) -> dict:
        """Parse stdout as the ONE JSON document, failing with its content.

        Every caller of :meth:`_run` wants this check; failing here names
        the command under test and shows the offending stream, which a bare
        ``json.loads`` traceback would not.
        """
        try:
            return json.loads(out)
        except ValueError as exc:
            self.fail(f"{what}: stdout was not one parseable JSON document "
                      f"({exc}); stdout was {out!r}")

    def _write_deck_file(self, name: str, cards: list) -> str:
        """Write ``cards`` as a JSON array under the temp root; return path."""
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(cards, handle)
        return path

    # -- argv to RunBudget ----------------------------------------------------

    def test_all_three_flags_reach_run_as_one_run_budget(self):
        captured = {}
        with mock.patch("cards.cli_cycle.run", _recording_spy(captured)):
            code, out, _err = self._run([
                "run",
                "--max-cards", "2",
                "--max-regenerations", "1",
                "--deadline", "900",
                "--root", self.root,
            ])
        self.assertEqual(code, 0, "the stubbed run must answer success")
        self._document(out, "run with all three budget flags")
        self.assertIn("budget", captured["kwargs"],
                      "the budget must travel as the budget= keyword")
        budget = captured["kwargs"]["budget"]
        self.assertIsInstance(budget, RunBudget,
                              "the three values must arrive as one RunBudget")
        self.assertEqual(
            budget,
            RunBudget(max_cards=2, max_regenerations=1,
                      deadline_seconds=900.0),
            "every flag must land on its own RunBudget field")

    def test_run_without_flags_limits_nothing(self):
        captured = {}
        with mock.patch("cards.cli_cycle.run", _recording_spy(captured)):
            code, _out, _err = self._run(["run", "--root", self.root])
        self.assertEqual(code, 0, "the stubbed run must answer success")
        budget = captured["kwargs"]["budget"]
        if budget is None:
            return  # None limits nothing: an acceptable shape
        self.assertIsInstance(budget, RunBudget)
        self.assertIsNone(budget.max_cards,
                          "no --max-cards flag must mean no card limit")
        self.assertIsNone(budget.max_regenerations,
                          "no --max-regenerations flag must mean no "
                          "regeneration limit")
        self.assertIsNone(budget.deadline_seconds,
                          "no --deadline flag must mean no deadline")

    def test_each_flag_stands_alone_and_the_others_stay_unlimited(self):
        cases = [
            (["--max-cards", "5"], "max_cards", 5),
            (["--max-regenerations", "3"], "max_regenerations", 3),
            (["--deadline", "60"], "deadline_seconds", 60.0),
        ]
        for flags, field, value in cases:
            with self.subTest(flags=flags):
                captured = {}
                with mock.patch("cards.cli_cycle.run",
                                _recording_spy(captured)):
                    code, _out, _err = self._run(
                        ["run", *flags, "--root", self.root])
                self.assertEqual(code, 0,
                                 "the stubbed run must answer success")
                budget = captured["kwargs"]["budget"]
                self.assertIsInstance(budget, RunBudget)
                self.assertEqual(getattr(budget, field), value,
                                 f"{field} must carry the flag's value")
                for other in ("max_cards", "max_regenerations",
                              "deadline_seconds"):
                    if other != field:
                        self.assertIsNone(
                            getattr(budget, other),
                            f"{other} must stay unlimited when only "
                            f"{field} is given")

    def test_a_non_numeric_budget_value_is_a_usage_failure(self):
        for flag, bad in [("--max-cards", "two"),
                          ("--max-regenerations", "many"),
                          ("--deadline", "soon")]:
            with self.subTest(flag=flag, bad=bad):
                code, out, err = self._run(
                    ["run", flag, bad, "--root", self.root])
                self.assertEqual(code, 4,
                                 f"{flag} {bad!r} must exit 4 (usage)")
                self.assertEqual(out.count("\n"), 1,
                                 "the usage failure must leave stdout at "
                                 "exactly one line")
                payload = self._document(out, f"run {flag} {bad!r}")
                self.assertEqual(payload["error"]["code"], 4,
                                 "the error document must carry the usage "
                                 "code")
                self.assertEqual(payload["error"]["kind"], "UsageError",
                                 "a non-numeric value is a parse failure")
                self.assertIn(flag, payload["error"]["message"],
                              "the diagnosis must name the flag that was "
                              "refused")
                self.assertTrue(
                    err.strip(),
                    "argparse's own usage and diagnosis belong on stderr")

    # -- RunBudget to run_deck -------------------------------------------------

    def test_run_hands_its_budget_unchanged_to_run_deck(self):
        path = self._write_deck_file("one-card.json", [
            _card("make-greeting", "greeting.py",
                  "Create greeting.py exposing hello()."),
        ])
        code, _out, _err = self._run(
            ["deck", "add", "--file", path, "--root", self.root])
        self.assertEqual(code, 0, "the fixture deck must add cleanly")

        captured = {}

        def _spy_deck(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            raise _ReachedRunDeck()

        with mock.patch("cards.cli_cycle.run_deck", _spy_deck):
            with self.assertRaises(_ReachedRunDeck):
                cli_cycle.run(self.root, nogit=True,
                              backend=_StubBackend(),
                              budget=RunBudget(max_cards=1))

        self.assertIn("budget", captured["kwargs"],
                      "run_deck must receive the budget as a keyword")
        self.assertEqual(captured["kwargs"]["budget"],
                         RunBudget(max_cards=1),
                         "the budget must arrive at run_deck unchanged")


if __name__ == "__main__":
    unittest.main()
