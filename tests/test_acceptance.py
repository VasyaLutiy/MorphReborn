"""
Phase 4 tests: the acceptance runner, best-of-N variant selection, rollback of
rejected morphs, and regeneration of failed cards into retry generations.

No network. ``run_deck`` is driven through the same :class:`FakeBatchBackend`
pattern as ``tests/test_generations.py`` -- scripted responses keyed by (variant,
retry) custom_id, with an explicit ``None`` modelling a failed request.
Acceptance commands are portable shell one-liners (``grep`` / ``python3 -c``)
that judge the file actually written to disk, so the verifier runs for real
against a tempdir workspace.
"""

import os
import subprocess
import tempfile
import unittest

from cards.acceptance import AcceptanceResult, run_acceptance, verify_card
from cards.generations import run_deck
from cards.schema import MorphCard


def _card(custom_id, target, depends_on=None, context_slice=None, variants=1,
          intent="generate", instruction="do it", acceptance=None):
    return MorphCard(
        custom_id=custom_id,
        intent=intent,
        target=target,
        instruction=instruction,
        context_slice=context_slice or [],
        depends_on=depends_on or [],
        variants=variants,
        acceptance=acceptance,
    )


def _code_block(body):
    """A fenced response whose single code block carries ``body``."""
    return f"```python\n{body}\n```"


# A response body that clears / fails the ``_accept`` check below.
PASS_BODY = _code_block("PASS = 1")
FAIL_BODY = _code_block("nope = 1")


def _accept(target):
    """Acceptance command: pass iff ``target`` contains PASS, else print a marker.

    The ``||`` group makes the command exit non-zero *and* emit
    ``ACCEPTANCE_FAILED_MARKER`` on stdout when the marker is absent, so a
    failing run has real output to feed back into the retry instruction.
    """
    return f"grep -q PASS {target} || {{ echo ACCEPTANCE_FAILED_MARKER; exit 1; }}"


# -- fake backend (mirrors tests/test_generations.py) ------------------------


class FakeBatchBackend:
    """Records every submit and replays scripted results.

    ``scripts`` maps a (variant/retry) custom_id to the response text ``collect``
    returns; an explicit ``None`` models a failed request, and an absent id falls
    back to ``default_response``. ``fail`` makes ``status`` report ``"failed"``
    for every batch.
    """

    def __init__(self, scripts=None, default_response=None, fail=False):
        self.scripts = scripts or {}
        self.default_response = default_response
        self.fail = fail
        self.submissions = []          # list of the request lists, per submit
        self.submitted_ids = []        # flat list of every custom_id submitted
        self._counter = 0

    def submit(self, requests):
        self.submissions.append(requests)
        for request in requests:
            self.submitted_ids.append(request["custom_id"])
        self._counter += 1
        return f"fake-batch-{self._counter}"

    def status(self, batch_id):
        return "failed" if self.fail else "completed"

    def collect(self, batch_id):
        index = int(batch_id.rsplit("-", 1)[1]) - 1
        results = {}
        for request in self.submissions[index]:
            custom_id = request["custom_id"]
            if custom_id in self.scripts:
                results[custom_id] = self.scripts[custom_id]
            else:
                results[custom_id] = self.default_response
        return results

    def instruction_of(self, custom_id):
        """The user instruction submitted for a (retry) custom_id, or None."""
        for submission in self.submissions:
            for request in submission:
                if request["custom_id"] == custom_id:
                    return "".join(
                        message["content"] for message in request["messages"]
                    )
        return None


# -- run_acceptance ----------------------------------------------------------


class RunAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="morph-acc-")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def test_exit_zero_passes(self):
        result = run_acceptance("python3 -c \"pass\"", self.root)
        self.assertIsInstance(result, AcceptanceResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.exit_code, 0)
        self.assertFalse(result.timed_out)

    def test_nonzero_fails_with_combined_output(self):
        command = (
            "python3 -c \"import sys; print('OUT_LINE'); "
            "sys.stderr.write('ERR_LINE'); sys.exit(2)\""
        )
        result = run_acceptance(command, self.root)
        self.assertFalse(result.passed)
        self.assertEqual(result.exit_code, 2)
        self.assertFalse(result.timed_out)
        # stdout and stderr are both captured (combined).
        self.assertIn("OUT_LINE", result.output)
        self.assertIn("ERR_LINE", result.output)

    def test_timeout_fails(self):
        result = run_acceptance(
            "python3 -c \"import time; time.sleep(30)\"", self.root, timeout=0.5
        )
        self.assertFalse(result.passed)
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.exit_code)

    def test_output_is_tail_truncated(self):
        # Emit far more than the 4000-char cap; only the tail is kept.
        command = "python3 -c \"print('z' * 10000)\""
        result = run_acceptance(command, self.root)
        self.assertTrue(result.passed)
        self.assertLessEqual(len(result.output), 4000)


# -- verify_card + run_deck integration --------------------------------------


class VerifyCardDeckTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="morph-acc-deck-")
        # A file so context slices resolve without walking the whole project.
        self._write("util.py", "U = 1\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, name, content):
        with open(os.path.join(self.root, name), "w", encoding="utf-8") as handle:
            handle.write(content)

    def _read(self, name):
        with open(os.path.join(self.root, name), "r", encoding="utf-8") as handle:
            return handle.read()

    def _exists(self, name):
        return os.path.exists(os.path.join(self.root, name))

    # -- the plan's named scenarios ------------------------------------------

    def test_three_variants_only_second_passes(self):
        card = _card(
            "m", "out.py", context_slice=["util.py"], variants=3,
            acceptance=_accept("out.py"),
        )
        backend = FakeBatchBackend(scripts={
            "m.v1": FAIL_BODY,
            "m.v2": PASS_BODY,
            "m.v3": PASS_BODY,   # would also pass, but v2 wins first
        })

        result = run_deck([card], backend, root=self.root, poll_interval=0)

        outcome = result.outcomes["m"]
        self.assertEqual(outcome.status, "written")
        self.assertEqual(outcome.winning_variant, "m.v2")
        self.assertEqual(outcome.attempts, 1)          # first-attempt success
        # The real target holds the winner's content.
        self.assertEqual(self._read("out.py"), "PASS = 1\n")
        # Winning suffixed file kept; losing suffixed files absent.
        self.assertTrue(self._exists("out.m.v2.py"))
        self.assertFalse(self._exists("out.m.v1.py"))
        self.assertFalse(self._exists("out.m.v3.py"))
        self.assertIn(os.path.join(self.root, "out.py"), outcome.paths)
        self.assertIn(os.path.join(self.root, "out.m.v2.py"), outcome.paths)
        # A single batch: no retry was needed.
        self.assertEqual(len(backend.submissions), 1)

    def test_all_variants_fail_regenerates_then_exhausts(self):
        # a's variants all fail acceptance; b depends on a.
        cards = [
            _card("a", "a.py", context_slice=["util.py"], variants=2,
                  acceptance=_accept("a.py")),
            _card("b", "b.py", depends_on=["a"], context_slice=["util.py"]),
        ]
        # a.py exists first, so we can assert it is restored after all attempts.
        self._write("a.py", "ORIGINAL\n")
        backend = FakeBatchBackend(default_response=FAIL_BODY)  # every attempt fails

        result = run_deck(cards, backend, root=self.root, poll_interval=0,
                          max_regenerations=2)

        outcome = result.outcomes["a"]
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.attempts, 3)          # 1 original + 2 retries
        self.assertIn("ACCEPTANCE_FAILED_MARKER", outcome.acceptance_output)
        # The retry batch's instruction embeds the acceptance error output.
        retry_instruction = backend.instruction_of("a.r1.v1")
        self.assertIsNotNone(retry_instruction)
        self.assertIn("ACCEPTANCE_FAILED_MARKER", retry_instruction)
        self.assertIn("a.py", retry_instruction)       # the command is named
        # Three attempts => three batches submitted for a (original + r1 + r2).
        self.assertIn("a.r1.v1", backend.submitted_ids)
        self.assertIn("a.r2.v1", backend.submitted_ids)
        # Original target state restored; no variant files left behind.
        self.assertEqual(self._read("a.py"), "ORIGINAL\n")
        self.assertFalse(self._exists("a.a.v1.py"))
        self.assertFalse(self._exists("a.a.v2.py"))
        # b's dependency finally failed -> b skipped, never submitted (cascade).
        self.assertEqual(result.outcomes["b"].status, "skipped")
        self.assertEqual(result.outcomes["b"].reason, "a")
        self.assertNotIn("b", backend.submitted_ids)
        self.assertFalse(self._exists("b.py"))

    # -- retry that succeeds --------------------------------------------------

    def test_retry_succeeds_and_dependent_sees_fresh_content(self):
        cards = [
            _card("a", "a.py", context_slice=["util.py"], acceptance=_accept("a.py")),
            _card("b", "b.py", depends_on=["a"], context_slice=["a.py"]),
        ]
        backend = FakeBatchBackend(scripts={
            "a": FAIL_BODY,        # first attempt fails acceptance
            "a.r1": PASS_BODY,     # retry passes
            "b": _code_block("B_OK = 1"),
        })

        result = run_deck(cards, backend, root=self.root, poll_interval=0)

        outcome = result.outcomes["a"]
        self.assertEqual(outcome.status, "written")
        self.assertEqual(outcome.attempts, 2)          # original + one retry
        self.assertEqual(outcome.winning_variant, "a.r1")
        self.assertEqual(self._read("a.py"), "PASS = 1\n")
        # b runs after a's retry and its context embeds a's fresh (retry) content.
        self.assertEqual(result.outcomes["b"].status, "written")
        b_instruction = backend.instruction_of("b")
        self.assertIn("PASS = 1", b_instruction)
        self.assertEqual(self._read("b.py"), "B_OK = 1\n")

    # -- rollback -------------------------------------------------------------

    def test_failing_patch_card_leaves_original_bytes_intact(self):
        self._write("p.py", "ORIGINAL PATCH\n")
        card = _card("p", "p.py", intent="patch", context_slice=["util.py"],
                     acceptance=_accept("p.py"))
        backend = FakeBatchBackend(scripts={"p": FAIL_BODY})

        result = run_deck([card], backend, root=self.root, poll_interval=0,
                          max_regenerations=0)

        self.assertEqual(result.outcomes["p"].status, "failed")
        # Rejected morph rolled back to the exact original bytes.
        self.assertEqual(self._read("p.py"), "ORIGINAL PATCH\n")

    def test_failing_generate_card_leaves_no_file_behind(self):
        card = _card("g", "g.py", context_slice=["util.py"], acceptance=_accept("g.py"))
        backend = FakeBatchBackend(scripts={"g": FAIL_BODY})

        result = run_deck([card], backend, root=self.root, poll_interval=0,
                          max_regenerations=0)

        self.assertEqual(result.outcomes["g"].status, "failed")
        # An originally-absent target is deleted again after rollback.
        self.assertFalse(self._exists("g.py"))

    # -- coexistence ----------------------------------------------------------

    def test_verified_and_unverified_cards_coexist_in_one_generation(self):
        cards = [
            _card("v", "v.py", context_slice=["util.py"], acceptance=_accept("v.py")),
            _card("u", "u.py", context_slice=["util.py"]),   # no acceptance
        ]
        backend = FakeBatchBackend(scripts={
            "v": PASS_BODY,
            "u": _code_block("U_OK = 1"),
        })

        result = run_deck(cards, backend, root=self.root, poll_interval=0)

        # Both submitted together, both written; the verified one records a
        # winning variant, the unverified one keeps Phase 3 semantics.
        self.assertEqual(len(backend.submissions), 1)
        self.assertEqual(result.outcomes["v"].status, "written")
        self.assertEqual(result.outcomes["v"].winning_variant, "v")
        self.assertEqual(result.outcomes["u"].status, "written")
        self.assertIsNone(result.outcomes["u"].winning_variant)
        self.assertEqual(self._read("v.py"), "PASS = 1\n")
        self.assertEqual(self._read("u.py"), "U_OK = 1\n")

    # -- timeout as failure ---------------------------------------------------

    def test_timeout_counts_as_failure_and_block_reaches_retry(self):
        card = _card("t", "t.py", context_slice=["util.py"],
                     acceptance="python3 -c \"import time; time.sleep(30)\"")
        backend = FakeBatchBackend(default_response=_code_block("whatever = 1"))

        result = run_deck([card], backend, root=self.root, poll_interval=0,
                          acceptance_timeout=0.3, max_regenerations=1)

        outcome = result.outcomes["t"]
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.attempts, 2)          # original + one retry
        # A timeout has (possibly empty) output, but the error block itself --
        # naming the acceptance command -- still lands in the retry instruction.
        retry_instruction = backend.instruction_of("t.r1")
        self.assertIsNotNone(retry_instruction)
        self.assertIn("failed its acceptance check", retry_instruction)

    # -- verify_card directly -------------------------------------------------

    def test_verify_card_all_none_responses_fail_without_running(self):
        card = _card("z", "z.py", context_slice=["util.py"], variants=2,
                     acceptance=_accept("z.py"))
        outcome = verify_card(card, {"z.v1": None, "z.v2": None}, self.root, 5.0,
                              log=lambda _msg: None)
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.attempts, 0)          # nothing ran
        self.assertIsNone(outcome.result)
        self.assertFalse(self._exists("z.py"))


if __name__ == "__main__":
    unittest.main()
