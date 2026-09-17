"""
Phase 5 acceptance: the split-step (/submit -> /collect) cycle, end to end.

This is the development plan's Phase 5 acceptance scenario, driven
NON-interactively: we never spin up the ConsoleBot loop, we call the store's
``submit_generation`` / ``collect_generation`` directly -- the same functions the
CLI transitions wrap. Everything runs against a throwaway copy of
``tests/fixtures/miniproject`` in a tempdir, so morphs are written into the copy,
never the repo tree. No network and no real SDKs.

Three scenarios:

* the headline case -- a 2-independent-card deck through ``LocalBatchBackend``
  over a fake 2-processor registry (one generation, both files written), with the
  ``/deck`` status rendering asserted written/written before and after;
* a dependent card -- a 2-generation deck through the split-step path, proving the
  second generation is compiled only after the first's morph is on disk;
* an acceptance-verified card -- the retry path exercised through
  ``collect_generation`` (first variant fails acceptance, the regeneration
  passes on the next /collect);
* a PERSISTED regeneration -- one retry batch however often /collect is called,
  picked up by a fresh store after a restart, and a lost local retry recovered.
"""

import os
import re
import shutil
import tempfile
import time
import unittest

from cards.generations import run_deck
from cards.store import (
    DeckStore,
    StoreError,
    build_deck_status,
    collect_generation,
    record_run,
    recover_orphaned_local_batch,
    submit_generation,
)
from processors.batch import LocalBatchBackend


MINIPROJECT = os.path.join("tests", "fixtures", "miniproject")


def _code_block(body):
    return f"```python\n{body}\n```"


# -- fakes -------------------------------------------------------------------


class _FakeRegistry:
    """A stand-in registry: ``LocalBatchBackend`` only ever calls ``run``.

    Mirrors the fake in ``tests/test_batch_backends.py``. Every processor echoes
    a plain (non-fenced) response, which the writer stores verbatim.
    """

    def __init__(self, failing_ids=()):
        self.failing_ids = set(failing_ids)

    def run(self, processor_id, messages):
        if processor_id in self.failing_ids:
            raise RuntimeError("processor exploded")
        return f"response from {processor_id}"


class _FakeBatchBackend:
    """A scripted submit/status/collect backend (pattern from test_generations).

    ``scripts`` maps a (variant/retry) custom_id to the text ``collect`` returns;
    ids absent from the map fall back to ``default_response``. ``status`` always
    reports ``completed``, so a single ``collect`` poll suffices.
    """

    def __init__(self, scripts=None, default_response=None):
        self.scripts = scripts or {}
        self.default_response = default_response
        self.submissions = []
        self._counter = 0

    def submit(self, requests):
        self.submissions.append(requests)
        self._counter += 1
        return f"fake-batch-{self._counter}"

    def status(self, batch_id):
        return "completed"

    def collect(self, batch_id):
        index = int(batch_id.rsplit("-", 1)[1]) - 1
        return {
            request["custom_id"]: self.scripts.get(request["custom_id"], self.default_response)
            for request in self.submissions[index]
        }


class _SlowBatchBackend(_FakeBatchBackend):
    """A scripted backend whose batches finish only when explicitly released.

    ``_FakeBatchBackend`` reports every batch ``completed`` at once, which cannot
    express the state the idempotence of ``/collect`` is about: a batch sitting
    in a provider's queue while the operator (or a polling loop) runs ``/collect``
    again and again. Here a batch is ``in_progress`` until :meth:`finish` -- or
    until :meth:`fail`, the provider-side wholesale failure.
    """

    def __init__(self, scripts=None, default_response=None):
        super().__init__(scripts=scripts, default_response=default_response)
        self.finished = set()
        self.failed = set()

    def status(self, batch_id):
        if batch_id in self.failed:
            return "failed"
        return "completed" if batch_id in self.finished else "in_progress"

    def finish(self, batch_id):
        self.finished.add(batch_id)

    def fail(self, batch_id):
        self.failed.add(batch_id)


class _E2EBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-e2e-")
        self.root = os.path.join(self.tmp, "miniproject")
        shutil.copytree(MINIPROJECT, self.root,
                        ignore=shutil.ignore_patterns("node_modules"))
        self.store = DeckStore(project_root=self.root)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _exists(self, name):
        return os.path.exists(os.path.join(self.root, name))

    def _read(self, name):
        with open(os.path.join(self.root, name), "r", encoding="utf-8") as handle:
            return handle.read()

    def _add(self, custom_id, target, **meta):
        instruction = meta.pop("instruction", "do it")
        self.store.add_card({
            "custom_id": custom_id,
            "meta": {"intent": meta.pop("intent", "generate"), "target": target, **meta},
            "instruction": instruction,
        })

    def _collect_until_done(self, backend, store=None, **kwargs):
        """Re-run /collect until the batch finishes (models the CLI's re-poll).

        ``LocalBatchBackend`` drains on worker threads, so an early status poll
        may still read ``in_progress``; the user (here, this loop) re-runs
        /collect. State is untouched while in progress, so this is safe.
        """
        for _ in range(200):
            result = collect_generation(store or self.store, backend, root=self.root,
                                        poll_interval=0, **kwargs)
            if not result.in_progress:
                return result
            time.sleep(0.02)
        self.fail("collect never completed")


class NightlyAcceptanceTests(_E2EBase):
    def test_two_independent_cards_submit_then_collect(self):
        self._add("card-a", "gen_a.py", context_slice=["util.py"], instruction="make a")
        self._add("card-b", "gen_b.py", context_slice=["util.py"], instruction="make b")

        # Before any run: both pending, one preview generation.
        before = build_deck_status(self.store)
        self.assertEqual(dict(before.card_status), {"card-a": "pending", "card-b": "pending"})
        self.assertEqual(before.generations, [["card-a", "card-b"]])

        registry = _FakeRegistry()
        backend = LocalBatchBackend(registry, ["node-a", "node-b"])

        submitted = submit_generation(self.store, backend, root=self.root,
                                      backend_label="node-a+node-b", log=lambda _l: None)
        self.assertTrue(submitted.submitted)
        self.assertEqual(submitted.generation_number, 1)
        self.assertEqual(submitted.total_generations, 1)
        self.assertEqual(set(submitted.card_ids), {"card-a", "card-b"})

        # Mid-flight the store reports the cards in flight and refuses a re-submit.
        mid = build_deck_status(self.store)
        self.assertEqual(mid.phase, "submitted")
        self.assertEqual(dict(mid.card_status), {"card-a": "in_flight", "card-b": "in_flight"})
        with self.assertRaises(Exception):
            submit_generation(self.store, backend, root=self.root)

        collected = self._collect_until_done(backend)
        self.assertFalse(collected.in_progress)
        self.assertEqual(collected.phase, "done")
        self.assertEqual(collected.total_generations, 1)
        self.assertEqual(set(collected.outcomes), {"card-a", "card-b"})
        for outcome in collected.outcomes.values():
            self.assertEqual(outcome.status, "written")

        # Both morphs are on disk and the run is finished.
        self.assertTrue(self._exists("gen_a.py"))
        self.assertTrue(self._exists("gen_b.py"))
        self.assertEqual(self.store.load_state()["phase"], "done")

        # The /deck-style rendering now shows written/written.
        after = build_deck_status(self.store)
        self.assertEqual(dict(after.card_status), {"card-a": "written", "card-b": "written"})

    def test_state_persists_across_a_fresh_store_between_submit_and_collect(self):
        # Simulate a CLI restart between /submit and /collect: a *new* DeckStore
        # over the same project reads the in-flight run from state.json. (The
        # LocalBatchBackend instance is reused, as it must be -- its batch lives
        # in memory; only the orchestrator's bookkeeping is what survives.)
        self._add("card-a", "gen_a.py", context_slice=["util.py"], instruction="make a")
        backend = LocalBatchBackend(_FakeRegistry(), ["node-a"])

        submit_generation(self.store, backend, root=self.root, log=lambda _l: None)

        resumed = DeckStore(project_root=self.root)
        self.assertEqual(resumed.load_state()["phase"], "submitted")
        result = None
        for _ in range(200):
            result = collect_generation(resumed, backend, root=self.root, poll_interval=0)
            if not result.in_progress:
                break
            time.sleep(0.02)
        self.assertFalse(result.in_progress)
        self.assertEqual(resumed.load_state()["phase"], "done")
        self.assertEqual(build_deck_status(resumed).card_status[0], ("card-a", "written"))


class DependentCardTests(_E2EBase):
    def test_two_generations_through_split_step(self):
        self._add("card-a", "gen_a.py", context_slice=["util.py"], instruction="make a")
        self._add("card-b", "gen_b.py", depends_on=["card-a"],
                  context_slice=["gen_a.py"], instruction="make b")

        fresh = "FRESH_FROM_A = 42"
        backend = _FakeBatchBackend(scripts={
            "card-a": _code_block(fresh),
            "card-b": _code_block("B_OK = 1"),
        })

        # Generation 1: card-a only.
        sub1 = submit_generation(self.store, backend, root=self.root, log=lambda _l: None)
        self.assertEqual(sub1.card_ids, ["card-a"])
        self.assertEqual(sub1.total_generations, 2)
        col1 = collect_generation(self.store, backend, root=self.root, poll_interval=0)
        self.assertEqual(col1.phase, "idle")
        self.assertEqual(col1.outcomes["card-a"].status, "written")
        self.assertEqual(self._read("gen_a.py"), fresh + "\n")

        # Generation 2: card-b, compiled AFTER card-a's morph landed -- its
        # context slice must embed the fresh content.
        sub2 = submit_generation(self.store, backend, root=self.root, log=lambda _l: None)
        self.assertEqual(sub2.card_ids, ["card-b"])
        b_requests = backend.submissions[1]
        b_text = "".join(message["content"]
                         for request in b_requests
                         for message in request["messages"])
        self.assertIn(fresh, b_text)

        col2 = collect_generation(self.store, backend, root=self.root, poll_interval=0)
        self.assertEqual(col2.phase, "done")
        self.assertEqual(col2.outcomes["card-b"].status, "written")

        after = build_deck_status(self.store)
        self.assertEqual(dict(after.card_status), {"card-a": "written", "card-b": "written"})

    def test_failed_dependency_skips_dependent_across_the_split_step(self):
        self._add("card-a", "gen_a.py", context_slice=["util.py"], instruction="make a")
        self._add("card-b", "gen_b.py", depends_on=["card-a"],
                  context_slice=["gen_a.py"], instruction="make b")

        backend = _FakeBatchBackend(scripts={"card-a": None})  # a produces nothing

        submit_generation(self.store, backend, root=self.root, log=lambda _l: None)
        col1 = collect_generation(self.store, backend, root=self.root, poll_interval=0)
        self.assertEqual(col1.outcomes["card-a"].status, "failed")
        # a failed but there is still generation 2 to resolve.
        self.assertEqual(col1.phase, "idle")

        # Submitting generation 2 finds card-b blocked -> skipped, nothing sent,
        # and the run finishes.
        sub2 = submit_generation(self.store, backend, root=self.root, log=lambda _l: None)
        self.assertFalse(sub2.submitted)
        self.assertTrue(sub2.done)
        self.assertEqual(sub2.skipped, [("card-b", "card-a")])

        after = build_deck_status(self.store)
        self.assertEqual(dict(after.card_status), {"card-a": "failed", "card-b": "skipped"})
        self.assertEqual(self.store.load_state()["phase"], "done")
        # Only card-a was ever handed to the backend.
        self.assertEqual(len(backend.submissions), 1)


class AcceptanceRetryTests(_E2EBase):
    def test_retry_path_through_collect_generation(self):
        # card-c has an acceptance command; the first variant fails it, the
        # inline retry (a second batch submitted by collect_generation) passes.
        self._add("card-c", "gen_c.py", context_slice=["util.py"],
                  acceptance="grep -q PASS gen_c.py", instruction="make c")

        backend = _FakeBatchBackend(scripts={
            "card-c": _code_block("VALUE = 1"),       # no PASS -> acceptance fails
            "card-c.r1": _code_block("PASS = 1"),      # retry -> acceptance passes
        })

        submit_generation(self.store, backend, root=self.root, log=lambda _l: None)

        # The first /collect finds the acceptance failure, submits the
        # regeneration, records it and RETURNS -- it does not poll it inline.
        first = collect_generation(self.store, backend, root=self.root,
                                   poll_interval=0, acceptance_timeout=30)
        self.assertTrue(first.in_progress)
        self.assertTrue(first.retry_in_flight)
        self.assertEqual(first.retry_attempt, 1)
        self.assertEqual(first.retry_card_ids, ["card-c"])
        self.assertFalse(self._exists("gen_c.py"))

        # The second /collect picks that very batch up.
        result = self._collect_until_done(backend, acceptance_timeout=30)

        self.assertEqual(result.phase, "done")
        outcome = result.outcomes["card-c"]
        self.assertEqual(outcome.status, "written")
        self.assertEqual(outcome.attempts, 2)      # original + one retry
        self.assertIn("PASS", self._read("gen_c.py"))
        # Two batches were submitted: the original and the retry.
        self.assertEqual(len(backend.submissions), 2)
        self.assertEqual(build_deck_status(self.store).card_status[0], ("card-c", "written"))

    def test_acceptance_failure_exhausts_retries_and_fails(self):
        self._add("card-c", "gen_c.py", context_slice=["util.py"],
                  acceptance="grep -q PASS gen_c.py", instruction="make c")

        # Nothing ever contains PASS -> every attempt fails acceptance.
        backend = _FakeBatchBackend(default_response=_code_block("VALUE = 1"))

        submit_generation(self.store, backend, root=self.root, log=lambda _l: None)
        result = self._collect_until_done(backend, acceptance_timeout=30,
                                          max_regenerations=2)

        self.assertEqual(result.phase, "done")
        self.assertEqual(result.outcomes["card-c"].status, "failed")
        # original + 2 retries = 3 attempts, 3 batches.
        self.assertEqual(result.outcomes["card-c"].attempts, 3)
        self.assertEqual(len(backend.submissions), 3)


class PersistedRegenerationTests(_E2EBase):
    """A regeneration is submitted ONCE, recorded, and survives a restart.

    The defect these pin down, all three measured on a real cloud deck: a
    ``/collect`` that met an acceptance failure used to submit the retry batch
    and poll it INLINE, in memory, persisting nothing. So one ``/collect`` blocked
    for an hour in silence; a second CLI session polling the same deck submitted
    its OWN retry for the same card (four paid batches for one card); and a
    process that died mid-retry orphaned a paid batch nobody could collect.
    """

    def _add_failing_card(self, store=None):
        (store or self.store).add_card({
            "custom_id": "card-c",
            "meta": {"intent": "generate", "target": "gen_c.py",
                     "context_slice": ["util.py"],
                     "acceptance": "grep -q PASS gen_c.py"},
            "instruction": "make c",
        })

    def _collect_to_first_retry(self, backend, store=None, **kwargs):
        """Run /collect until a regeneration is recorded; return that result."""
        for _ in range(200):
            result = collect_generation(store or self.store, backend, root=self.root,
                                        poll_interval=0, acceptance_timeout=30, **kwargs)
            if result.retry_in_flight:
                return result
            if not result.in_progress:
                self.fail("the generation finished without a regeneration")
            time.sleep(0.02)
        self.fail("no regeneration was ever recorded")

    def test_one_retry_batch_however_often_collect_is_called(self):
        # The idempotence claim, stated as money: ten /collect calls while a
        # regeneration is in flight submit nothing and change nothing.
        self._add_failing_card()
        backend = _SlowBatchBackend(scripts={
            "card-c": _code_block("VALUE = 1"),     # no PASS -> acceptance fails
            "card-c.r1": _code_block("PASS = 1"),   # the regeneration passes
        })

        submit_generation(self.store, backend, root=self.root,
                          backend_label="fake", log=lambda _l: None)
        backend.finish("fake-batch-1")

        first = self._collect_to_first_retry(backend)
        self.assertTrue(first.in_progress)
        self.assertEqual(first.retry_attempt, 1)
        self.assertEqual(first.retry_limit, 2)
        self.assertEqual(first.retry_card_ids, ["card-c"])
        self.assertEqual(len(backend.submissions), 2)

        recorded = self.store.load_state()
        self.assertEqual(recorded["phase"], "submitted")
        self.assertEqual(recorded["batch_id"], first.retry_batch_id)
        self.assertEqual(recorded["retries"]["card-c"]["attempt"], 1)
        self.assertEqual(recorded["retries"]["card-c"]["batch_id"], first.retry_batch_id)
        self.assertEqual(recorded["retries"]["card-c"]["backend_label"], "fake")

        for _ in range(10):
            polled = collect_generation(self.store, backend, root=self.root,
                                        poll_interval=0, acceptance_timeout=30)
            self.assertTrue(polled.in_progress)
            self.assertTrue(polled.retry_in_flight)
            self.assertEqual(polled.retry_attempt, 1)
        # Not one extra batch, and not one byte of state moved.
        self.assertEqual(len(backend.submissions), 2)
        self.assertEqual(self.store.load_state(), recorded)

        # Released, the recorded batch is collected -- by the same call that
        # would have submitted a replacement in the broken version.
        backend.finish(first.retry_batch_id)
        done = collect_generation(self.store, backend, root=self.root,
                                  poll_interval=0, acceptance_timeout=30)
        self.assertFalse(done.in_progress)
        self.assertEqual(done.phase, "done")
        self.assertEqual(done.outcomes["card-c"].status, "written")
        self.assertEqual(done.outcomes["card-c"].attempts, 2)
        self.assertEqual(len(backend.submissions), 2)
        self.assertEqual(self.store.load_state()["retries"], {})

    def test_a_fresh_store_picks_up_a_recorded_cloud_retry(self):
        # The CLI restart: the retry batch is on a provider's server, and all
        # that has to survive is the record naming it.
        self._add_failing_card()
        backend = _SlowBatchBackend(scripts={
            "card-c": _code_block("VALUE = 1"),
            "card-c.r1": _code_block("PASS = 1"),
        })

        submit_generation(self.store, backend, root=self.root,
                          backend_label="fake", log=lambda _l: None)
        backend.finish("fake-batch-1")
        first = self._collect_to_first_retry(backend)

        resumed = DeckStore(project_root=self.root)
        self.assertEqual(resumed.load_state()["retries"]["card-c"]["batch_id"],
                         first.retry_batch_id)
        # Still in flight for the new session too -- and still free to poll.
        waiting = collect_generation(resumed, backend, root=self.root,
                                     poll_interval=0, acceptance_timeout=30)
        self.assertTrue(waiting.retry_in_flight)
        self.assertEqual(len(backend.submissions), 2)

        backend.finish(first.retry_batch_id)
        done = collect_generation(resumed, backend, root=self.root,
                                  poll_interval=0, acceptance_timeout=30)
        self.assertEqual(done.phase, "done")
        self.assertEqual(done.outcomes["card-c"].status, "written")
        self.assertEqual(done.outcomes["card-c"].attempts, 2)
        self.assertIn("PASS", self._read("gen_c.py"))
        self.assertEqual(dict(build_deck_status(resumed).card_status),
                         {"card-c": "written"})

    def test_the_regeneration_carries_the_previous_error_across_a_restart(self):
        # The retry card's instruction must still quote what went wrong, even
        # though the VerifyOutcome that produced it died with the old process --
        # which is why the acceptance output is part of the recorded retry.
        marker = "MISSING_PASS_MARKER"
        self.store.add_card({
            "custom_id": "card-c",
            "meta": {"intent": "generate", "target": "gen_c.py",
                     "context_slice": ["util.py"],
                     "acceptance": f"grep -q PASS gen_c.py || {{ echo {marker}; exit 1; }}"},
            "instruction": "make c",
        })
        backend = _SlowBatchBackend(default_response=_code_block("VALUE = 1"))

        submit_generation(self.store, backend, root=self.root, log=lambda _l: None)
        backend.finish("fake-batch-1")
        first = self._collect_to_first_retry(backend)

        def request_text(submission):
            return "".join(message["content"]
                           for request in submission
                           for message in request["messages"])

        self.assertIn(marker, request_text(backend.submissions[1]))
        self.assertIn(marker, self.store.load_state()["retries"]["card-c"]["acceptance_output"])

        # Restart, then let attempt 1 fail too: attempt 2's instruction is built
        # from the RECORDED error context, not from anything still in memory.
        resumed = DeckStore(project_root=self.root)
        backend.finish(first.retry_batch_id)
        second = collect_generation(resumed, backend, root=self.root,
                                    poll_interval=0, acceptance_timeout=30)
        self.assertEqual(second.retry_attempt, 2)
        self.assertEqual(resumed.load_state()["retries"]["card-c"]["attempt"], 2)
        self.assertIn(marker, request_text(backend.submissions[2]))

    def test_a_retry_batch_that_fails_after_a_restart_keeps_the_recorded_error(self):
        # The only consumer of the recorded error context that memory cannot
        # supply: the retry batch itself fails wholesale, so there is no fresh
        # acceptance result to record -- the card must still say what went wrong
        # on the attempt before the process died.
        marker = "MISSING_PASS_MARKER"
        self.store.add_card({
            "custom_id": "card-c",
            "meta": {"intent": "generate", "target": "gen_c.py",
                     "context_slice": ["util.py"],
                     "acceptance": f"grep -q PASS gen_c.py || {{ echo {marker}; exit 1; }}"},
            "instruction": "make c",
        })
        backend = _SlowBatchBackend(default_response=_code_block("VALUE = 1"))

        submit_generation(self.store, backend, root=self.root, log=lambda _l: None)
        backend.finish("fake-batch-1")
        first = self._collect_to_first_retry(backend)

        resumed = DeckStore(project_root=self.root)
        backend.fail(first.retry_batch_id)
        done = collect_generation(resumed, backend, root=self.root,
                                  poll_interval=0, acceptance_timeout=30)

        self.assertEqual(done.phase, "done")
        outcome = done.outcomes["card-c"]
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.attempts, 2)
        self.assertIn(marker, outcome.acceptance_output)

    def test_the_final_outcome_is_the_one_nightly_would_have_recorded(self):
        # The /deck view must not be able to tell the two routes apart.
        self._add_failing_card()
        scripts = {"card-c": _code_block("VALUE = 1"), "card-c.r1": _code_block("PASS = 1")}

        split_backend = _FakeBatchBackend(scripts=dict(scripts))
        submit_generation(self.store, split_backend, root=self.root, log=lambda _l: None)
        split = self._collect_until_done(split_backend, acceptance_timeout=30)

        # The same deck through /nightly, in its own copy of the project.
        nightly_root = os.path.join(self.tmp, "nightly")
        shutil.copytree(MINIPROJECT, nightly_root,
                        ignore=shutil.ignore_patterns("node_modules"))
        nightly_store = DeckStore(project_root=nightly_root)
        self._add_failing_card(nightly_store)
        nightly = run_deck(nightly_store.load_cards(), _FakeBatchBackend(scripts=dict(scripts)),
                           root=nightly_root, poll_interval=0, log=lambda _l: None)

        def comparable(outcome, root):
            return (outcome.status, outcome.attempts, outcome.winning_variant,
                    [os.path.relpath(path, root) for path in outcome.paths])

        self.assertEqual(comparable(split.outcomes["card-c"], self.root),
                         comparable(nightly.outcomes["card-c"], nightly_root))

    def test_a_lost_local_retry_is_recovered_and_only_it_is_re_sent(self):
        # A local batch is worker threads: a restart kills it, retry or not. The
        # card goes back to the attempt it was pending instead of wedging the
        # deck -- and its generation-mate, already written, is NOT paid for twice.
        self._add("card-ok", "gen_ok.py", context_slice=["util.py"], instruction="make ok")
        self._add_failing_card()

        backend = LocalBatchBackend(_FakeRegistry(), ["node-a"])
        submit_generation(self.store, backend, root=self.root,
                          backend_label="node-a", log=lambda _l: None)
        first = self._collect_to_first_retry(backend)
        self.assertTrue(first.retry_batch_id.startswith("local-"))
        self.assertTrue(self._exists("gen_ok.py"))

        # The CLI restarts; the retry batch is gone with the process.
        resumed = DeckStore(project_root=self.root)
        self.assertEqual(resumed.load_state()["phase"], "submitted")
        with self.assertRaises(StoreError):
            submit_generation(resumed, backend, root=self.root, log=lambda _l: None)

        self.assertTrue(recover_orphaned_local_batch(resumed))
        after = resumed.load_state()
        self.assertEqual(after["phase"], "idle")
        self.assertEqual(after["retries"], {})
        self.assertEqual(dict(build_deck_status(resumed).card_status),
                         {"card-ok": "written", "card-c": "pending"})

        # And the deck runs again -- re-sending the pending card ONLY.
        again = submit_generation(resumed, LocalBatchBackend(_FakeRegistry(), ["node-a"]),
                                  root=self.root, backend_label="node-a",
                                  log=lambda _l: None)
        self.assertTrue(again.submitted)
        self.assertEqual(again.card_ids, ["card-c"])


class NightlyPersistenceTests(_E2EBase):
    """``/nightly`` must leave the run behind, exactly as ``/collect`` does."""

    def test_a_nightly_run_is_visible_to_the_next_deck_view(self):
        self._add("card-a", "gen_a.py", context_slice=["util.py"], instruction="make a")
        self._add("card-b", "gen_b.py", depends_on=["card-a"],
                  context_slice=["util.py"], instruction="make b")

        backend = LocalBatchBackend(_FakeRegistry(), ["node-a"])
        result = run_deck(self.store.load_cards(), backend, root=self.root,
                          poll_interval=0, log=lambda _l: None)
        record_run(self.store, result, backend_label="node-a")

        self.assertTrue(self._exists("gen_a.py"))
        self.assertTrue(self._exists("gen_b.py"))

        # The next /deck -- through a fresh store, as a later CLI session sees it.
        view = build_deck_status(DeckStore(project_root=self.root))
        self.assertEqual(view.phase, "done")
        self.assertEqual(dict(view.card_status),
                         {"card-a": "written", "card-b": "written"})
        self.assertEqual(view.generations, [["card-a"], ["card-b"]])
        outcomes = DeckStore(project_root=self.root).load_outcomes()
        self.assertIn(os.path.join(self.root, "gen_a.py"), outcomes["card-a"].paths)


class NewPackageTargetTests(_E2EBase):
    """A card may create the package it targets -- and used to kill the run.

    Every card this project had ever run wrote into the project root, so the
    writers' bare ``open(path, "w")`` never met a missing directory. The first
    card targeting ``<new package>/<module>.py`` raised ``FileNotFoundError``
    inside ``collect_generation``, out of the transition, and took the CLI with
    it. These two drive the store-level path the ``/collect`` transition wraps.
    """

    def test_nested_target_is_written_not_raised(self):
        self._add("card-n", "pkg/sub/mod.py", context_slice=["util.py"],
                  instruction="make the module")

        backend = _FakeBatchBackend(scripts={"card-n": _code_block("N_OK = 1")})

        submit_generation(self.store, backend, root=self.root, log=lambda _l: None)
        result = collect_generation(self.store, backend, root=self.root,
                                    poll_interval=0)

        self.assertEqual(result.phase, "done")
        self.assertEqual(result.outcomes["card-n"].status, "written")
        self.assertTrue(self._exists(os.path.join("pkg", "sub", "mod.py")))
        self.assertEqual(self._read(os.path.join("pkg", "sub", "mod.py")), "N_OK = 1\n")

    def test_verified_nested_target_is_written_through_the_acceptance_path(self):
        # The same target with an acceptance command, so verify_card (not
        # _write_variants) is the writer that meets the missing directory.
        self._add("card-n", "pkg/mod.py", context_slice=["util.py"],
                  acceptance="grep -q PASS pkg/mod.py", instruction="make it")

        backend = _FakeBatchBackend(scripts={"card-n": _code_block("PASS = 1")})

        submit_generation(self.store, backend, root=self.root, log=lambda _l: None)
        result = collect_generation(self.store, backend, root=self.root,
                                    poll_interval=0, acceptance_timeout=30)

        self.assertEqual(result.outcomes["card-n"].status, "written")
        self.assertTrue(self._exists(os.path.join("pkg", "mod.py")))
        self.assertEqual(build_deck_status(self.store).card_status[0],
                         ("card-n", "written"))


class LocalBatchRestartTests(_E2EBase):
    """The escape from a local batch that died with the CLI process."""

    def test_orphaned_local_batch_is_recovered_and_the_deck_runs_again(self):
        self._add("card-a", "gen_a.py", context_slice=["util.py"], instruction="make a")

        fired = LocalBatchBackend(_FakeRegistry(), ["node-a"])
        submitted = submit_generation(self.store, fired, root=self.root,
                                      backend_label="node-a", log=lambda _l: None)
        self.assertTrue(submitted.batch_id.startswith("local-"))

        # The CLI restarts: a fresh store, and the backend instance is gone.
        resumed = DeckStore(project_root=self.root)
        with self.assertRaises(StoreError):
            submit_generation(resumed, LocalBatchBackend(_FakeRegistry(), ["node-a"]),
                              root=self.root, log=lambda _l: None)

        self.assertTrue(recover_orphaned_local_batch(resumed))
        self.assertEqual(dict(build_deck_status(resumed).card_status),
                         {"card-a": "pending"})

        # And the generation can simply be sent again.
        backend = LocalBatchBackend(_FakeRegistry(), ["node-a"])
        again = submit_generation(resumed, backend, root=self.root,
                                  backend_label="node-a", log=lambda _l: None)
        self.assertTrue(again.submitted)
        self.assertEqual(again.card_ids, ["card-a"])
        collected = self._collect_until_done(backend, store=resumed)
        self.assertEqual(collected.outcomes["card-a"].status, "written")
        self.assertTrue(self._exists("gen_a.py"))



if __name__ == "__main__":
    unittest.main()
