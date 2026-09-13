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
  ``collect_generation`` (first variant fails acceptance, the inline retry
  passes).
"""

import os
import re
import shutil
import tempfile
import time
import unittest

from cards.store import (
    DeckStore,
    build_deck_status,
    collect_generation,
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

    def _collect_until_done(self, backend, **kwargs):
        """Re-run /collect until the batch finishes (models the CLI's re-poll).

        ``LocalBatchBackend`` drains on worker threads, so an early status poll
        may still read ``in_progress``; the user (here, this loop) re-runs
        /collect. State is untouched while in progress, so this is safe.
        """
        for _ in range(200):
            result = collect_generation(self.store, backend, root=self.root,
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
        result = collect_generation(self.store, backend, root=self.root,
                                    poll_interval=0, acceptance_timeout=30)

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
        result = collect_generation(self.store, backend, root=self.root,
                                    poll_interval=0, acceptance_timeout=30,
                                    max_regenerations=2)

        self.assertEqual(result.phase, "done")
        self.assertEqual(result.outcomes["card-c"].status, "failed")
        # original + 2 retries = 3 attempts, 3 batches.
        self.assertEqual(result.outcomes["card-c"].attempts, 3)
        self.assertEqual(len(backend.submissions), 3)


if __name__ == "__main__":
    unittest.main()
