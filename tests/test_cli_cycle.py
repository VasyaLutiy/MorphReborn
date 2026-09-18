"""
End-to-end tests for :func:`cards.cli_cycle.run` -- ``mrph run`` in one call.

No network, no provider, no processor configuration, no real sleeping, and
nowhere near this repository's own ``.morph/``: every test runs against a
``tempfile.TemporaryDirectory`` root, with ``nogit=True`` (a tempdir holds no
branch to begin with, and the git layer must stay out of these assertions), a
FAKE batch backend injected through ``backend``, and a fake clock injected
through ``sleep``/``now``. The fake backend completes every batch on its
first poll, so the injected clock doubles as an alarm: if any code under test
tried to wait, the recorded naps would say so instead of the suite hanging on
the real one.

The morph-card fixtures are spelled in the exact shape
``cards.schema.MorphCard.from_dict`` accepts: ``intent`` is the literal
``"generate"`` -- it is NOT a description of what the card does, the prose
lives in ``instruction`` -- ``custom_id`` matches ``^[A-Za-z0-9._-]+$``, and
a card names ``target``, never ``targets`` as well. The acceptance criteria
are the one-word commands ``true`` and ``false``, whose exit codes decide the
pass/fail split under whatever invocation the acceptance runner chooses, so
these tests depend on nothing but the exit code.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from typing import Callable, Dict, List, Optional, Tuple

# So the suite also runs under a bare interpreter, where tests/ is the import
# root and the project root is not yet on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cards import cli_cycle
from cards.cli_json import EXIT_INCOMPLETE, EXIT_OK
from cards.hazards import HazardError
from cards.store import DeckStore, list_runs


# -- fixtures ----------------------------------------------------------------


def _card(custom_id: str, target: str, instruction: str,
          acceptance: Optional[str] = None,
          depends_on: Optional[List[str]] = None) -> dict:
    """One morph-card dict in the nested shape ``MorphCard.from_dict`` accepts.

    ``intent`` is the literal ``"generate"``; the prose is ``instruction``.
    ``custom_id`` sticks to the filesystem-safe character set the schema
    demands. The card names ``target`` and never ``targets`` -- the schema
    refuses a card that names both.
    """
    meta: Dict[str, object] = {"intent": "generate", "target": target}
    if acceptance is not None:
        meta["acceptance"] = acceptance
    if depends_on is not None:
        meta["depends_on"] = list(depends_on)
    return {"custom_id": custom_id, "meta": meta, "instruction": instruction}


def _fenced(body: str) -> str:
    """A model's answer for one file: one fenced block holding ``body``."""
    return "```python\n" + body + "\n```"


class FakeBackend:
    """A batch backend with no provider behind it: every batch is done at once.

    Duck-typed to the three methods ``cards.generations.run_deck`` calls. It
    remembers every batch it was handed -- so a test can assert how many
    batches a run really paid for -- and answers ``collect`` from the bodies
    the test stated, keyed by request ``custom_id``, including the
    ``.r1``/``.r2`` ids a regeneration attempt carries, so a retry gets a
    real answer to fail its acceptance with.
    """

    def __init__(self, bodies: Dict[str, str]) -> None:
        self.bodies = dict(bodies)
        self.requests: List[List[dict]] = []
        self.batch_ids: List[str] = []

    def submit(self, requests: List[dict]) -> str:
        self.requests.append([dict(request) for request in requests])
        batch_id = "fake-%04d" % len(self.requests)
        self.batch_ids.append(batch_id)
        return batch_id

    def status(self, batch_id: str) -> str:
        return "completed"

    def collect(self, batch_id: str) -> Dict[str, Optional[str]]:
        results: Dict[str, Optional[str]] = {}
        batch = self.requests[self.batch_ids.index(batch_id)]
        for request in batch:
            custom_id = request.get("custom_id")
            results[custom_id] = self.bodies.get(custom_id)
        return results


class RunCycleTest(unittest.TestCase):
    """``mrph run`` end to end: tempdir root, fake backend, fake clock."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.naps: List[float] = []
        self._tick = 0.0

        def fake_sleep(seconds: float) -> None:
            self.naps.append(seconds)

        def fake_now() -> float:
            self._tick += 1.0
            return self._tick

        self.sleep = fake_sleep
        self.now = fake_now

    # -- helpers -------------------------------------------------------------


    def _add_cards(self, *card_dicts: dict) -> None:
        """Put cards in the backlog through the store's own, validating API."""
        DeckStore(self.root).add_cards(list(card_dicts))

    def _run(self, fake: FakeBackend,
             log: Optional[Callable[[str], None]] = None) -> Tuple[dict, int]:
        """One headless run against this test's root, backend and clock."""
        return cli_cycle.run(root=self.root, nogit=True, backend=fake,
                             log=log, sleep=self.sleep, now=self.now)

    def _assert_json_round_trips(self, payload: dict, note: str) -> None:
        self.assertEqual(json.loads(json.dumps(payload)), payload, msg=note)

    # -- the contract --------------------------------------------------------


    def test_empty_backlog_is_a_completed_decision(self) -> None:
        """No deck at all: code 0, a payload that says so, nothing run."""
        fake = FakeBackend({})
        payload, code = self._run(fake)

        self.assertEqual(code, EXIT_OK, msg=(
            "an empty backlog is not an error: the call must return "
            "EXIT_OK (0)"))
        self.assertIs(payload.get("empty"), True, msg=(
            "the payload must say the backlog was empty, via an 'empty' key "
            "that is True"))
        self.assertEqual(
            payload["counts"], {"written": 0, "failed": 0, "skipped": 0},
            msg="an empty backlog ran no cards, so every count must be zero")
        self.assertEqual(fake.requests, [], msg=(
            "an empty backlog must submit no batch"))
        self.assertFalse(
            os.path.isdir(os.path.join(self.root, ".morph", "runs")), msg=(
                "an empty backlog starts no run, so no run archive may exist"))
        self._assert_json_round_trips(
            payload, "the empty-backlog payload must survive json.dumps")

    def test_one_card_with_passing_acceptance_is_written_and_exits_zero(
            self) -> None:
        self._add_cards(
            _card("alpha", "alpha.py", "Write alpha.py.", acceptance="true"))
        fake = FakeBackend({"alpha": _fenced("VALUE = 1")})
        payload, code = self._run(fake)

        self.assertEqual(code, EXIT_OK, msg=(
            "every card ended written, so the call must return EXIT_OK (0)"))
        self.assertEqual(payload["counts"]["written"], 1, msg=(
            "the report's counts must record the one written card"))
        self.assertEqual(payload["outcomes"]["alpha"]["status"], "written",
                         msg="the card's outcome in the report must be "
                             "'written'")
        self.assertEqual(payload["outcomes"]["alpha"]["attempts"], 1, msg=(
            "a card accepted on its first attempt must report one attempt"))
        self.assertTrue(os.path.isfile(os.path.join(self.root, "alpha.py")),
                        msg="an accepted card's target must exist on disk")
        with open(os.path.join(self.root, "alpha.py"), "r",
                  encoding="utf-8") as handle:
            written = handle.read()
        self.assertEqual(written, "VALUE = 1\n", msg=(
            "the written file must hold the fenced body with the fence "
            "stripped"))
        self.assertEqual(len(fake.requests), 1, msg=(
            "one one-card generation must be exactly one batch"))
        self.assertEqual(self.naps, [], msg=(
            "the fake completes on the first poll, so nothing may have slept "
            "-- not even through the injected clock"))
        self._assert_json_round_trips(
            payload, "the run report payload must survive json.dumps")

    def test_card_whose_acceptance_fails_ends_failed_and_exits_one(
            self) -> None:
        self._add_cards(
            _card("beta", "beta.py", "Write beta.py.", acceptance="false"))
        body = _fenced("VALUE = 2")
        fake = FakeBackend({"beta": body, "beta.r1": body, "beta.r2": body})
        payload, code = self._run(fake)

        self.assertEqual(code, EXIT_INCOMPLETE, msg=(
            "a card that ended failed must make the call return "
            "EXIT_INCOMPLETE (1)"))
        self.assertEqual(payload["outcomes"]["beta"]["status"], "failed", msg=(
            "a card whose acceptance failed through every retry must end "
            "'failed'"))
        self.assertEqual(payload["outcomes"]["beta"]["attempts"], 3, msg=(
            "the original attempt plus two regenerations (the default "
            "max_regenerations) must be recorded as 3 attempts"))
        self.assertEqual(payload["counts"]["failed"], 1, msg=(
            "the report's counts must record the failed card"))
        self.assertEqual(len(fake.requests), 3, msg=(
            "the failing card must be resubmitted up to the limit: one "
            "original batch plus two regeneration batches"))
        self._assert_json_round_trips(
            payload, "the run report payload must survive json.dumps")

    def test_dependency_deck_runs_in_two_generations(self) -> None:
        self._add_cards(
            _card("first", "first.py", "Write first.py."),
            _card("second", "second.py", "Write second.py.",
                  depends_on=["first"]))
        fake = FakeBackend({"first": _fenced("FIRST = 1"),
                            "second": _fenced("SECOND = 2")})
        payload, code = self._run(fake)

        self.assertEqual(code, EXIT_OK, msg=(
            "both cards ended written, so the call must return EXIT_OK (0)"))
        self.assertEqual(payload["generations"], [["first"], ["second"]], msg=(
            "the dependent card must sit in its own, later generation, and "
            "the report must record both generations"))
        self.assertEqual(payload["counts"]["written"], 2, msg=(
            "both cards must be counted written"))
        self.assertEqual(payload["outcomes"]["first"]["status"], "written",
                         msg="the independent card must be written")
        self.assertEqual(payload["outcomes"]["second"]["status"], "written",
                         msg="the dependent card must be written after it")
        self.assertTrue(os.path.isfile(os.path.join(self.root, "first.py")),
                        msg="the first generation's file must exist on disk")
        self.assertTrue(os.path.isfile(os.path.join(self.root, "second.py")),
                        msg="the second generation's file must exist on disk")
        self.assertEqual(len(fake.requests), 2, msg=(
            "two generations must mean two batches, the second compiled after "
            "the first wrote its file"))
        self._assert_json_round_trips(
            payload, "the run report payload must survive json.dumps")

    def test_write_write_conflict_refuses_before_anything_happens(
            self) -> None:
        self._add_cards(
            _card("writer-a", "shared.py", "Write shared.py your way."),
            _card("writer-b", "shared.py", "Write shared.py my way."))
        fake = FakeBackend({"writer-a": _fenced("A = 1"),
                            "writer-b": _fenced("B = 2")})

        with self.assertRaises(HazardError, msg=(
                "two cards of one generation writing one file must be "
                "refused by the preflight, as a HazardError")):
            self._run(fake)

        self.assertEqual(fake.requests, [], msg=(
            "a refused deck must never reach the provider: no batch may have "
            "been submitted"))
        self.assertFalse(os.path.isfile(os.path.join(self.root, "shared.py")),
                         msg="a refused run must have written nothing")
        self.assertFalse(
            os.path.isdir(os.path.join(self.root, ".morph", "runs")), msg=(
                "a refused run must have archived nothing"))
        self.assertFalse(
            os.path.isfile(os.path.join(self.root, ".morph", "state.json")),
            msg="a refused run must have got no further than the preflight: "
                "no run state, and therefore no branch, may have been "
                "recorded")
        self.assertEqual(len(DeckStore(self.root).load_cards()), 2, msg=(
            "a refused deck must come back exactly as its author left it: "
            "the backlog is untouched"))

    def test_finished_run_is_archived_and_listed(self) -> None:
        self._add_cards(_card("solo", "solo.py", "Write solo.py."))
        fake = FakeBackend({"solo": _fenced("SOLO = 1")})
        payload, code = self._run(fake)  # log=None: the quiet path must work

        self.assertEqual(code, EXIT_OK, msg=(
            "the one card ended written, so the call must return EXIT_OK (0)"))
        store = DeckStore(self.root)
        runs = list_runs(store)
        self.assertEqual(len(runs), 1, msg=(
            "the finished run must be archived exactly once"))
        self.assertEqual(runs[0].deck_id, payload["deck_id"], msg=(
            "the report list_runs finds must be THIS run's, under the deck "
            "id the payload carries"))
        self.assertEqual(runs[0].counts["written"], 1, msg=(
            "the archived report must carry the same counts the printed one "
            "does"))
        directory = store.run_dir(payload["deck_id"])
        self.assertTrue(os.path.isfile(os.path.join(directory, "report.json")),
                        msg="the archive must contain the run's report.json")
        self.assertTrue(os.path.isfile(os.path.join(directory, "deck.json")),
                        msg="the archive must contain the deck as executed")
        self._assert_json_round_trips(
            payload, "the run report payload must survive json.dumps")

    def test_log_receives_the_progress_and_stdout_stays_empty(self) -> None:
        self._add_cards(
            _card("alpha", "alpha.py", "Write alpha.py.", acceptance="true"))
        fake = FakeBackend({"alpha": _fenced("VALUE = 1")})
        lines: List[str] = []
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            payload, code = self._run(fake, log=lines.append)

        self.assertEqual(code, EXIT_OK, msg=(
            "the run itself must succeed, so the assertions below are about "
            "logging, not about a failure"))
        self.assertEqual(captured.getvalue(), "", msg=(
            "the run must print nothing itself: stdout belongs to the "
            "caller's one JSON document"))
        self.assertTrue(lines, msg=(
            "the log callable must receive the run's progress lines"))
        self.assertTrue(any("alpha" in line for line in lines), msg=(
            "the progress lines must name the card they are about"))
        self.assertEqual(self.naps, [], msg=(
            "nothing may have slept, on the real clock or the injected one"))


if __name__ == "__main__":
    unittest.main()
