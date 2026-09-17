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
from cards.generations import response_to_file_body, run_deck
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

    # -- a target in a directory that does not exist yet ----------------------

    def test_verify_card_creates_the_targets_missing_directories(self):
        # Creating a new package is the most ordinary thing a card can ask for,
        # and no card had ever done it: every target so far lived in the project
        # root, so the writers' bare open() never met a missing directory.
        card = _card("n", "pkg/sub/mod.py", context_slice=["util.py"],
                     acceptance=_accept("pkg/sub/mod.py"))

        outcome = verify_card(card, {"n": PASS_BODY}, self.root, 30.0,
                              log=lambda _msg: None)

        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.winning_custom_id, "n")
        self.assertTrue(self._exists(os.path.join("pkg", "sub", "mod.py")))
        self.assertEqual(self._read(os.path.join("pkg", "sub", "mod.py")),
                         "PASS = 1\n")

    def test_multi_variant_winner_keeps_its_suffixed_file_in_the_new_directory(self):
        card = _card("n", "pkg/mod.py", context_slice=["util.py"], variants=2,
                     acceptance=_accept("pkg/mod.py"))

        outcome = verify_card(card, {"n.v1": FAIL_BODY, "n.v2": PASS_BODY},
                              self.root, 30.0, log=lambda _msg: None)

        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.winning_custom_id, "n.v2")
        # Target and the winner's suffixed file, both inside the created package.
        self.assertTrue(self._exists(os.path.join("pkg", "mod.py")))
        self.assertTrue(self._exists(os.path.join("pkg", "mod.n.v2.py")))
        self.assertFalse(self._exists(os.path.join("pkg", "mod.n.v1.py")))
        self.assertEqual(sorted(outcome.paths), sorted([
            os.path.join(self.root, "pkg", "mod.py"),
            os.path.join(self.root, "pkg", "mod.n.v2.py"),
        ]))

    def test_failing_card_in_a_new_directory_rolls_the_file_back(self):
        # Rollback deletes the file the rejected variant created; the directory
        # created for it is deliberately left behind (see ensure_parent_dir).
        card = _card("n", "pkg/mod.py", context_slice=["util.py"],
                     acceptance=_accept("pkg/mod.py"))

        outcome = verify_card(card, {"n": FAIL_BODY}, self.root, 30.0,
                              log=lambda _msg: None)

        self.assertFalse(outcome.passed)
        self.assertFalse(self._exists(os.path.join("pkg", "mod.py")))
        self.assertTrue(os.path.isdir(os.path.join(self.root, "pkg")))


# -- the stale-bytecode trap -------------------------------------------------


# Three variant bodies of IDENTICAL byte size: only the second computes a sum.
# Equal size is the point -- see BytecodeCacheTrapTests.
_ADD_MINUS = _code_block("def add(a, b):\n    return a - b")
_ADD_PLUS = _code_block("def add(a, b):\n    return a + b")
_ADD_TIMES = _code_block("def add(a, b):\n    return a * b")


class BytecodeCacheTrapTests(unittest.TestCase):
    """Acceptance must judge the variant it just wrote, not a cached ancestor.

    CPython validates a cached ``.pyc`` against (source mtime truncated to WHOLE
    SECONDS, source size). ``verify_card`` writes its variants to the real target
    milliseconds apart, so two variants of the SAME byte size are
    indistinguishable to the import system: without a guard, variant 2's
    acceptance run silently imports variant 1's bytecode, best-of-N rejects a
    correct morph and the card burns its retries. The guards are in
    :func:`run_acceptance` (never write bytecode during acceptance) and in
    :func:`verify_card` (a distinct whole-second mtime per write).
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="morph-acc-pyc-")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def test_acceptance_runs_with_bytecode_writing_disabled(self):
        result = run_acceptance(
            "python3 -c \"import sys; print(sys.dont_write_bytecode)\"", self.root)
        self.assertTrue(result.passed)
        self.assertIn("True", result.output)

    def test_acceptance_does_not_leak_into_the_parent_environment(self):
        before = os.environ.get("PYTHONDONTWRITEBYTECODE")
        run_acceptance("python3 -c \"pass\"", self.root)
        self.assertEqual(os.environ.get("PYTHONDONTWRITEBYTECODE"), before)

    def test_same_size_variants_are_judged_individually(self):
        # The acceptance command imports the target, so a stale .pyc from the
        # previous variant would answer instead of the file on disk.
        card = _card(
            "add", "stand_add.py", variants=3,
            acceptance="python3 -c \"import stand_add; "
                       "assert stand_add.add(2, 3) == 5\"",
        )
        bodies = {"add.v1": _ADD_MINUS, "add.v2": _ADD_PLUS, "add.v3": _ADD_TIMES}
        sizes = {len(response_to_file_body(body)) for body in bodies.values()}
        self.assertEqual(len(sizes), 1)   # the trap only springs on equal sizes

        outcome = verify_card(card, bodies, self.root, 30.0, log=lambda _msg: None)

        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.winning_custom_id, "add.v2")
        self.assertEqual(outcome.attempts, 2)
        with open(os.path.join(self.root, "stand_add.py"), encoding="utf-8") as handle:
            self.assertIn("a + b", handle.read())


def _file_block(path, body):
    """One file of a multi-file answer, in the format the directive asks for."""
    return f"FILE: {path}\n```python\n{body}\n```\n"


class ChangesetAtomicityTests(unittest.TestCase):
    """Phase 7: a card's files are accepted or rolled back as ONE set.

    The limitation this closes was measured, not theorised: every fix this
    project shipped by hand had to touch a module AND its test, so none of them
    could be expressed as a card. What makes the set safe to attempt is that a
    failure leaves nothing behind -- no half-written changeset for a human to
    find with ``git status`` and undo by hand.
    """

    ORIGINAL = "ORIGINAL = 1\n"

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-changeset-")
        self.root = self.tmp
        with open(os.path.join(self.root, "existing.py"), "w", encoding="utf-8") as handle:
            handle.write(self.ORIGINAL)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _files_on_disk(self):
        """Every FILE under the root, relative -- directories deliberately not.

        A directory created for a rejected write is left in place on purpose
        (see ``cards.generations.ensure_parent_dir``); an empty one is harmless.
        """
        found = set()
        for walk_root, _dirs, files in os.walk(self.root):
            for name in files:
                found.add(os.path.relpath(os.path.join(walk_root, name), self.root))
        return found

    def _read(self, name):
        with open(os.path.join(self.root, name), encoding="utf-8") as handle:
            return handle.read()

    def _card(self, targets, acceptance, variants=1):
        return MorphCard(
            custom_id="set",
            intent="generate",
            targets=targets,
            instruction="do it",
            variants=variants,
            acceptance=acceptance,
        )

    def _three_file_answer(self, marker="PASS"):
        return (_file_block("existing.py", f"{marker} = 1")
                + _file_block("fresh.py", f"{marker} = 2")
                + _file_block("pkg/deep.py", f"{marker} = 3"))

    def test_three_targets_all_written_when_acceptance_passes(self):
        card = self._card(
            ["existing.py", "fresh.py", "pkg/deep.py"],
            "grep -q PASS existing.py && grep -q PASS fresh.py "
            "&& grep -q PASS pkg/deep.py")

        outcome = verify_card(card, {"set": self._three_file_answer()},
                              self.root, 30.0, log=lambda _msg: None)

        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.winning_custom_id, "set")
        self.assertEqual(outcome.paths, [
            os.path.join(self.root, "existing.py"),
            os.path.join(self.root, "fresh.py"),
            os.path.join(self.root, "pkg", "deep.py"),
        ])
        self.assertEqual(self._read("existing.py"), "PASS = 1\n")
        self.assertEqual(self._read("fresh.py"), "PASS = 2\n")
        self.assertEqual(self._read(os.path.join("pkg", "deep.py")), "PASS = 3\n")

    def test_a_failing_acceptance_leaves_none_of_the_three_behind(self):
        before = self._files_on_disk()
        card = self._card(["existing.py", "fresh.py", "pkg/deep.py"],
                          "grep -q NEVER_PRESENT existing.py")

        outcome = verify_card(card, {"set": self._three_file_answer()},
                              self.root, 30.0, log=lambda _msg: None)

        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.attempts, 1)
        # The file that existed is back to its original bytes; the two that did
        # not exist are gone again. Nothing at all is left to clean up.
        self.assertEqual(self._read("existing.py"), self.ORIGINAL)
        self.assertEqual(self._files_on_disk(), before)

    def test_the_whole_set_is_written_before_acceptance_runs(self):
        # The card's ONE acceptance command judges the SET: a command that only
        # passes when every file is present proves they all land first.
        card = self._card(["existing.py", "fresh.py", "pkg/deep.py"],
                          "test -f fresh.py && test -f pkg/deep.py "
                          "&& grep -q PASS existing.py")

        outcome = verify_card(card, {"set": self._three_file_answer()},
                              self.root, 30.0, log=lambda _msg: None)

        self.assertTrue(outcome.passed)

    def test_a_response_missing_a_declared_target_writes_nothing(self):
        before = self._files_on_disk()
        card = self._card(["existing.py", "fresh.py"], "true")

        outcome = verify_card(card, {"set": _file_block("existing.py", "PASS = 1")},
                              self.root, 30.0, log=lambda _msg: None)

        # Corrupt, not a partial write: acceptance never ran (a `true` command
        # would have passed), and the tree is untouched.
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.attempts, 0)
        self.assertEqual(self._files_on_disk(), before)
        self.assertEqual(self._read("existing.py"), self.ORIGINAL)
        self.assertIn("fresh.py", outcome.result.output)

    def test_a_response_naming_an_undeclared_path_writes_nothing(self):
        before = self._files_on_disk()
        card = self._card(["existing.py"], "true")

        answer = (_file_block("existing.py", "PASS = 1")
                  + _file_block("somewhere_else.py", "SNEAKY = 1"))
        outcome = verify_card(card, {"set": answer}, self.root, 30.0,
                              log=lambda _msg: None)

        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.attempts, 0)
        self.assertEqual(self._files_on_disk(), before)
        self.assertIn("somewhere_else.py", outcome.result.output)

    def test_best_of_n_keeps_the_first_passing_set_and_no_suffixed_copies(self):
        card = self._card(["existing.py", "fresh.py"], "grep -q PASS existing.py",
                          variants=2)
        losing = (_file_block("existing.py", "NOPE = 1")
                  + _file_block("fresh.py", "LOSER = 2"))
        winning = (_file_block("existing.py", "PASS = 1")
                   + _file_block("fresh.py", "WINNER = 2"))

        outcome = verify_card(card, {"set.v1": losing, "set.v2": winning},
                              self.root, 30.0, log=lambda _msg: None)

        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.winning_custom_id, "set.v2")
        self.assertEqual(outcome.attempts, 2)
        self.assertEqual(self._read("fresh.py"), "WINNER = 2\n")
        # No per-variant debris: N variants times M files helps nobody.
        self.assertEqual(self._files_on_disk(), {"existing.py", "fresh.py"})


if __name__ == "__main__":
    unittest.main()
