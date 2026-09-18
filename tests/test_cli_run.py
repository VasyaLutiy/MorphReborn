"""
Tests for :mod:`cards.cli_run` -- the headless ``submit`` / ``collect`` pair.

Every test drives the real store machinery (preflight, composition, run state,
archive) against a FAKE batch backend injected through the ``backend``
argument, with ``nogit=True`` throughout so no git command is ever run, inside
a ``tempfile.TemporaryDirectory`` root so this project's own ``.morph/`` is
never touched. ``sleep`` and ``now`` are injected everywhere a clock is taken,
so no test sleeps for real and no wait can outlive its assertion; nothing
here reaches the network, and the fake backend is the whole provider.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest

from cards import cli_run
from cards.cli_wait import WaitTimeout
from cards.hazards import HazardError
from cards.store import DeckStore


class FakeClock:
    """An injected clock: ``sleep`` advances the fake moment, ``now`` reads it.

    Both methods have the signatures :mod:`cards.cli_wait` expects, so they
    pass as ``sleep=`` / ``now=`` wherever the module takes a clock; the
    recorded ``sleeps`` let a test assert that waiting went through this seam
    and not through ``time.sleep``.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.moment = start
        self.sleeps = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.moment += seconds

    def now(self) -> float:
        return self.moment


class FakeBackend:
    """A duck-typed batch backend with scripted answers and no I/O.

    ``statuses`` is the sequence ``status()`` answers, in order; the LAST
    entry repeats forever, so "in_progress, then completed" is scripted as
    ``["in_progress", "completed"]`` and a batch that never finishes as
    ``["in_progress"]``. ``results`` is what ``collect()`` returns for any
    batch -- ``{custom_id: text or None}``, with ``None`` meaning a variant
    that produced nothing. ``submit`` records every request list it is
    handed, so a test can assert what was -- or was never -- sent to a
    provider.
    """

    def __init__(self) -> None:
        self.statuses = ["completed"]
        self.results = {}
        self.submissions = []
        self.batch_ids = []
        self.status_calls = 0
        self.collect_calls = 0
        self._counter = 0

    def submit(self, requests):
        self.submissions.append([dict(request) for request in requests])
        self._counter += 1
        batch_id = "batch-{}".format(self._counter)
        self.batch_ids.append(batch_id)
        return batch_id

    def status(self, batch_id):
        self.status_calls += 1
        if len(self.statuses) > 1:
            return self.statuses.pop(0)
        return self.statuses[0]

    def collect(self, batch_id):
        self.collect_calls += 1
        return dict(self.results)


class CliRunTest(unittest.TestCase):
    """The headless step subcommands, against a fake backend and a fake clock."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.clock = FakeClock()
        self.backend = FakeBackend()
        self.lines = []

    # -- helpers ------------------------------------------------------------

    def add_card(self, custom_id="card-1", target="hello.txt",
                 instruction="Write a greeting file.", **meta):
        """Append one valid card to the temp root's backlog."""
        payload = {"intent": "generate", "target": target}
        payload.update(meta)
        DeckStore(self.root).add_card({
            "custom_id": custom_id,
            "meta": payload,
            "instruction": instruction,
        })

    def submit_deck(self, **kwargs):
        """``cli_run.submit`` with the standard test seams filled in."""
        kwargs.setdefault("nogit", True)
        kwargs.setdefault("backend", self.backend)
        kwargs.setdefault("log", self.lines.append)
        return cli_run.submit(self.root, **kwargs)

    def collect_deck(self, **kwargs):
        """``cli_run.collect`` with the standard test seams filled in."""
        kwargs.setdefault("backend", self.backend)
        kwargs.setdefault("sleep", self.clock.sleep)
        kwargs.setdefault("now", self.clock.now)
        kwargs.setdefault("log", self.lines.append)
        return cli_run.collect(self.root, **kwargs)

    def assert_survives_json(self, payload):
        """Every returned dict must encode as one JSON document, as-is."""
        try:
            json.dumps(payload)
        except (TypeError, ValueError) as error:
            self.fail("the returned payload must survive json.dumps: "
                      f"{error!r}")

    # -- submit -------------------------------------------------------------

    def test_submit_one_card_sends_batch_and_names_the_card(self):
        self.add_card()
        payload, code = self.submit_deck()
        self.assertEqual(code, 0,
                         "a submit that sent a batch must exit EXIT_OK (0)")
        self.assertTrue(payload["submitted"],
                        "one backlog card must go out as a submitted batch")
        self.assertEqual(payload["cards"], ["card-1"],
                         "the payload must name the submitted card by id")
        self.assertFalse(payload["done"],
                         "a deck whose generation is in flight is not done")
        self.assertEqual(payload["batch_id"], "batch-1",
                         "the payload must carry the id the backend returned")
        self.assertEqual(len(self.backend.submissions), 1,
                         "exactly one batch must have been submitted")
        self.assertEqual(
            [request["custom_id"] for request in self.backend.submissions[0]],
            ["card-1"],
            "the batch request must carry the card's own custom_id")
        self.assert_survives_json(payload)

    def test_submit_on_an_empty_deck_reports_done(self):
        # Deliberately no ``log``: the None default must be safe -- the
        # store's lines dropped -- rather than falling back to print.
        payload, code = cli_run.submit(self.root, nogit=True,
                                       backend=self.backend)
        self.assertEqual(code, 0,
                         "an empty deck is finished work, not a failure")
        self.assertTrue(payload["done"], "an empty deck must report done")
        self.assertFalse(payload["submitted"],
                         "an empty deck must not submit anything")
        self.assertEqual(self.backend.submissions, [],
                         "no batch may be submitted for an empty deck")
        self.assert_survives_json(payload)

    def test_submit_log_goes_to_the_caller_and_not_stdout(self):
        self.add_card()
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            payload, code = self.submit_deck()
        self.assertEqual(captured.getvalue(), "",
                         "submit must write nothing to stdout: the "
                         "one-document contract belongs to the dispatcher")
        self.assertEqual(code, 0, "the logged submit must still succeed")
        self.assertTrue(any("submitting" in line for line in self.lines),
                        "the store's progress lines must land in the "
                        "caller's log list")
        self.assertTrue(any("git" in line for line in self.lines),
                        "the nogit announcement must reach the caller's "
                        "log too")
        self.assert_survives_json(payload)

    def test_submit_refuses_two_cards_writing_one_file(self):
        self.add_card("writer-a", "shared.txt", "Write shared.txt.")
        self.add_card("writer-b", "shared.txt",
                      "Write shared.txt differently.")
        with self.assertRaises(
                HazardError,
                msg="two cards of one generation writing one file must be "
                    "refused as a HazardError"):
            cli_run.submit(self.root, nogit=True, backend=self.backend,
                           log=self.lines.append)
        self.assertEqual(self.backend.submissions, [],
                         "the refusal must come before any money: no batch "
                         "may be submitted")
        self.assertFalse(
            os.path.exists(os.path.join(self.root, ".morph", "state.json")),
            "the refusal must stop the run before any run state exists")

    def test_submit_returns_code_1_when_cards_are_skipped(self):
        self.add_card("card-1", "hello.txt", "Write hello.")
        self.add_card("card-2", "other.txt", "Write other.",
                      depends_on=["card-1"])
        self.submit_deck()
        self.backend.statuses = ["completed"]
        self.backend.results = {"card-1": None}
        self.collect_deck()
        payload, code = self.submit_deck()
        self.assertEqual(code, 1,
                         "a submit that only records skips exits "
                         "EXIT_INCOMPLETE (1)")
        self.assertFalse(payload["submitted"],
                         "a generation with nothing runnable sends nothing")
        self.assertEqual(
            payload["skipped"],
            [{"custom_id": "card-2", "blocked_by": "card-1"}],
            "the skipped card must be reported with the dependency that "
            "blocked it")
        self.assertEqual(len(self.backend.submissions), 1,
                         "no second batch may be sent for a skipped "
                         "generation")
        self.assert_survives_json(payload)

    # -- collect ------------------------------------------------------------

    def test_collect_without_wait_on_a_running_batch_is_news_not_failure(self):
        self.add_card()
        self.submit_deck()
        self.backend.statuses = ["in_progress"]
        payload, code = self.collect_deck()
        self.assertEqual(code, 0,
                         "a poll that finds the batch still running is "
                         "news, not a failure")
        self.assertTrue(payload["in_progress"],
                        "a running batch must report in_progress")
        self.assertEqual(payload["phase"], "submitted",
                         "the run must still be in the submitted phase")
        self.assertFalse(payload["retry"]["in_flight"],
                         "a first-attempt batch has no regeneration in "
                         "flight")
        self.assertEqual(self.backend.collect_calls, 0,
                         "a batch that is still running must not be "
                         "collected")
        self.assert_survives_json(payload)

    def test_collect_with_wait_loops_until_the_batch_completes(self):
        self.add_card()
        self.submit_deck()
        self.backend.statuses = ["in_progress", "completed"]
        self.backend.results = {"card-1": "hello world"}
        payload, code = self.collect_deck(wait=True, timeout=3600.0)
        self.assertEqual(code, 0,
                         "a settled generation with every card written "
                         "exits EXIT_OK (0)")
        self.assertFalse(payload["in_progress"],
                         "the wait must have ended with the batch settled")
        self.assertEqual(payload["phase"], "done",
                         "the deck's only generation is settled")
        self.assertEqual(payload["outcomes"]["card-1"]["status"], "written",
                         "the card's outcome must be the written one the "
                         "judge recorded")
        self.assertTrue(os.path.exists(os.path.join(self.root, "hello.txt")),
                        "the written card's file must be on disk under the "
                        "run root")
        self.assertEqual(self.backend.status_calls, 2,
                         "the wait must have polled twice: in_progress, "
                         "then completed")
        self.assertTrue(self.clock.sleeps,
                        "the loop must sleep between polls through the "
                        "injected clock, never through time.sleep")
        self.assert_survives_json(payload)

    def test_collect_returns_code_1_when_a_card_fails(self):
        self.add_card()
        self.submit_deck()
        self.backend.statuses = ["completed"]
        self.backend.results = {"card-1": None}
        payload, code = self.collect_deck()
        self.assertEqual(code, 1,
                         "a settled generation carrying a failed card exits "
                         "EXIT_INCOMPLETE (1)")
        self.assertFalse(payload["in_progress"],
                         "a completed batch must settle in this call")
        self.assertEqual(payload["outcomes"]["card-1"]["status"], "failed",
                         "a variant answer of None must record the card as "
                         "failed")
        self.assert_survives_json(payload)

    def test_collect_wait_timeout_propagates_and_leaves_run_collectable(self):
        self.add_card()
        self.submit_deck()
        self.backend.statuses = ["in_progress"]  # never finishes
        with self.assertRaises(
                WaitTimeout,
                msg="an expired wait must surface as WaitTimeout, never be "
                    "swallowed"):
            self.collect_deck(wait=True, timeout=50.0)
        self.assertGreaterEqual(
            self.clock.moment, 50.0,
            "the injected clock must have been driven past the timeout")
        state = DeckStore(self.root).load_state()
        self.assertEqual(state["phase"], "submitted",
                         "an expired wait must leave the run collectable: "
                         "the phase is still submitted")
        self.assertEqual(state["batch_id"], "batch-1",
                         "the in-flight batch id must survive the expired "
                         "wait untouched")


if __name__ == "__main__":
    unittest.main()
