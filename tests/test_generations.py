"""
Phase 3 tests: generation splitting and the generation execution loop.

No network. ``run_deck`` is driven through a :class:`FakeBatchBackend` that
records what was submitted and returns scripted responses keyed by (variant)
custom_id. Where file content matters, the deck runs against a *temp-dir copy*
of ``tests/fixtures/miniproject`` -- morphs are written into that copy, never
into the repo tree.
"""

import os
import shutil
import tempfile
import unittest

from cards.generations import (
    CardOutcome,
    DeckResult,
    is_truncated_response,
    response_to_file_body,
    run_deck,
    split_into_generations,
)
from cards.schema import MorphCard


MINIPROJECT = os.path.join("tests", "fixtures", "miniproject")


def _card(custom_id, target, depends_on=None, context_slice=None, variants=1,
          intent="generate", instruction="do it"):
    return MorphCard(
        custom_id=custom_id,
        intent=intent,
        target=target,
        instruction=instruction,
        context_slice=context_slice or [],
        depends_on=depends_on or [],
        variants=variants,
    )


def _code_block(body):
    """A fenced response whose single code block carries ``body``."""
    return f"```python\n{body}\n```"


# -- fake backend ------------------------------------------------------------


class FakeBatchBackend:
    """Records every submit and replays scripted results.

    ``scripts`` maps a (variant) custom_id to the response text ``collect``
    returns for it; an explicit ``None`` models a failed request, and an id
    absent from the map falls back to ``default_response``. When ``fail`` is
    True, ``status`` reports ``"failed"`` for every batch (and ``collect`` is
    never reached). ``log_ref``, when given, is snapshotted at each submit so a
    test can assert what had already been logged *before* the submit.
    """

    def __init__(self, scripts=None, default_response=None, fail=False, log_ref=None):
        self.scripts = scripts or {}
        self.default_response = default_response
        self.fail = fail
        self._log_ref = log_ref
        self.submissions = []          # list of the request lists, per submit
        self.submitted_ids = []        # flat list of every custom_id submitted
        self.log_snapshots = []        # copy of log_ref at each submit
        self._counter = 0

    def submit(self, requests):
        self.submissions.append(requests)
        for request in requests:
            self.submitted_ids.append(request["custom_id"])
        if self._log_ref is not None:
            self.log_snapshots.append(list(self._log_ref))
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


# -- splitting ---------------------------------------------------------------


class SplitIntoGenerationsTests(unittest.TestCase):
    @staticmethod
    def _ids(generations):
        return [[card.custom_id for card in generation] for generation in generations]

    def test_wide_independent_deck_is_one_generation(self):
        cards = [_card("a", "a.py"), _card("b", "b.py"), _card("c", "c.py")]
        self.assertEqual(self._ids(split_into_generations(cards)), [["a", "b", "c"]])

    def test_chain_degrades_to_one_generation_per_card(self):
        cards = [
            _card("a", "a.py"),
            _card("b", "b.py", depends_on=["a"]),
            _card("c", "c.py", depends_on=["b"]),
        ]
        self.assertEqual(self._ids(split_into_generations(cards)), [["a"], ["b"], ["c"]])

    def test_diamond_splits_into_three_generations(self):
        cards = [
            _card("a", "a.py"),
            _card("b", "b.py", depends_on=["a"]),
            _card("c", "c.py", depends_on=["a"]),
            _card("d", "d.py", depends_on=["b", "c"]),
        ]
        self.assertEqual(
            self._ids(split_into_generations(cards)),
            [["a"], ["b", "c"], ["d"]],
        )

    def test_deck_order_preserved_within_a_generation(self):
        # c is declared before b, though both sit in generation 1.
        cards = [
            _card("a", "a.py"),
            _card("c", "c.py", depends_on=["a"]),
            _card("b", "b.py", depends_on=["a"]),
            _card("d", "d.py", depends_on=["b", "c"]),
        ]
        self.assertEqual(
            self._ids(split_into_generations(cards)),
            [["a"], ["c", "b"], ["d"]],
        )

    def test_longest_path_wins_over_shortest(self):
        # d depends on both a (gen 0) and c (gen 2) -> d lands in gen 3.
        cards = [
            _card("a", "a.py"),
            _card("b", "b.py", depends_on=["a"]),
            _card("c", "c.py", depends_on=["b"]),
            _card("d", "d.py", depends_on=["a", "c"]),
        ]
        self.assertEqual(
            self._ids(split_into_generations(cards)),
            [["a"], ["b"], ["c"], ["d"]],
        )

    def test_stored_generation_field_is_ignored(self):
        # Author put a wrong generation hint on b; computed placement wins.
        cards = [
            MorphCard(custom_id="a", intent="generate", target="a.py",
                      instruction="x", generation=5),
            MorphCard(custom_id="b", intent="generate", target="b.py",
                      instruction="x", depends_on=["a"], generation=0),
        ]
        self.assertEqual(self._ids(split_into_generations(cards)), [["a"], ["b"]])

    def test_input_cards_not_mutated(self):
        cards = [
            _card("a", "a.py"),
            _card("b", "b.py", depends_on=["a"]),
        ]
        generations = split_into_generations(cards)
        # Same objects handed back, and their advisory fields untouched.
        self.assertIs(generations[0][0], cards[0])
        self.assertIs(generations[1][0], cards[1])
        self.assertEqual(cards[0].generation, 0)
        self.assertEqual(cards[1].generation, 0)
        self.assertEqual(cards[1].depends_on, ["a"])

    def test_empty_deck(self):
        self.assertEqual(split_into_generations([]), [])


# -- run_deck ----------------------------------------------------------------


class RunDeckTests(unittest.TestCase):
    def setUp(self):
        # A private, writable copy of the fixture: morphs land here, not in the
        # repo tree.
        self.tmp = tempfile.mkdtemp(prefix="morph-gen-")
        self.root = os.path.join(self.tmp, "miniproject")
        shutil.copytree(MINIPROJECT, self.root, ignore=shutil.ignore_patterns("node_modules"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _read(self, name):
        with open(os.path.join(self.root, name), "r", encoding="utf-8") as handle:
            return handle.read()

    def _exists(self, name):
        return os.path.exists(os.path.join(self.root, name))

    def test_happy_path_two_independent_cards(self):
        cards = [
            _card("a", "gen_a.py", context_slice=["util.py"]),
            _card("b", "gen_b.py", context_slice=["util.py"]),
        ]
        backend = FakeBatchBackend(scripts={
            "a": _code_block("A_BODY = 1"),
            "b": _code_block("B_BODY = 2"),
        })

        result = run_deck(cards, backend, root=self.root, poll_interval=0)

        self.assertIsInstance(result, DeckResult)
        # One generation, both cards submitted together.
        self.assertEqual(result.generations, [["a", "b"]])
        self.assertEqual(len(backend.submissions), 1)
        self.assertEqual(result.outcomes["a"].status, "written")
        self.assertEqual(result.outcomes["b"].status, "written")
        self.assertEqual(self._read("gen_a.py"), "A_BODY = 1\n")
        self.assertEqual(self._read("gen_b.py"), "B_BODY = 2\n")
        self.assertEqual(
            result.outcomes["a"].paths, [os.path.join(self.root, "gen_a.py")]
        )

    def test_dependent_card_sees_fresh_dependency_content(self):
        # b depends on a and reads a's freshly written target in its context.
        cards = [
            _card("a", "gen_a.py", context_slice=["util.py"]),
            _card("b", "gen_b.py", depends_on=["a"], context_slice=["gen_a.py"]),
        ]
        fresh = "FRESH_FROM_A = 42"
        backend = FakeBatchBackend(scripts={
            "a": _code_block(fresh),
            "b": _code_block("B_OK = 1"),
        })

        result = run_deck(cards, backend, root=self.root, poll_interval=0)

        # Two generations, two separate submits.
        self.assertEqual(result.generations, [["a"], ["b"]])
        self.assertEqual(len(backend.submissions), 2)
        # a is written before b is compiled.
        self.assertEqual(self._read("gen_a.py"), fresh + "\n")
        # b's submitted messages must embed a's FRESH content -> compile happened
        # after the write.
        b_requests = backend.submissions[1]
        b_text = "".join(
            message["content"]
            for request in b_requests
            for message in request["messages"]
        )
        self.assertIn(fresh, b_text)
        self.assertEqual(result.outcomes["b"].status, "written")

    def test_variants_write_one_file_each_and_skip_none(self):
        cards = [_card("m", "multi.py", context_slice=["util.py"], variants=3)]
        backend = FakeBatchBackend(scripts={
            "m.v1": _code_block("V1 = 1"),
            "m.v2": None,               # this variant failed
            "m.v3": _code_block("V3 = 3"),
        })

        result = run_deck(cards, backend, root=self.root, poll_interval=0)

        # A card with one surviving variant is still "written".
        self.assertEqual(result.outcomes["m"].status, "written")
        self.assertTrue(self._exists("multi.m.v1.py"))
        self.assertFalse(self._exists("multi.m.v2.py"))
        self.assertTrue(self._exists("multi.m.v3.py"))
        self.assertEqual(self._read("multi.m.v1.py"), "V1 = 1\n")
        self.assertEqual(self._read("multi.m.v3.py"), "V3 = 3\n")
        self.assertEqual(
            result.outcomes["m"].paths,
            [os.path.join(self.root, "multi.m.v1.py"),
             os.path.join(self.root, "multi.m.v3.py")],
        )

    def test_card_targeting_a_new_package_creates_its_directories(self):
        # The no-acceptance path (_write_variants). A card may target a module in
        # a package that does not exist yet -- the plainest way to grow a project
        # -- and the writer must create the tree rather than die on open().
        cards = [_card("n", "pkg/sub/mod.py", context_slice=["util.py"])]
        backend = FakeBatchBackend(scripts={"n": _code_block("N = 1")})

        result = run_deck(cards, backend, root=self.root, poll_interval=0)

        self.assertEqual(result.outcomes["n"].status, "written")
        self.assertEqual(self._read(os.path.join("pkg", "sub", "mod.py")), "N = 1\n")
        self.assertEqual(result.outcomes["n"].paths,
                         [os.path.join(self.root, "pkg", "sub", "mod.py")])

    def test_variants_of_a_new_package_card_land_in_that_package(self):
        cards = [_card("n", "pkg/mod.py", context_slice=["util.py"], variants=2)]
        backend = FakeBatchBackend(scripts={
            "n.v1": _code_block("V1 = 1"),
            "n.v2": _code_block("V2 = 2"),
        })

        result = run_deck(cards, backend, root=self.root, poll_interval=0)

        self.assertEqual(result.outcomes["n"].status, "written")
        self.assertTrue(self._exists(os.path.join("pkg", "mod.n.v1.py")))
        self.assertTrue(self._exists(os.path.join("pkg", "mod.n.v2.py")))

    def test_card_fails_when_all_variants_none(self):
        cards = [_card("m", "multi.py", context_slice=["util.py"], variants=2)]
        backend = FakeBatchBackend(scripts={"m.v1": None, "m.v2": None})

        result = run_deck(cards, backend, root=self.root, poll_interval=0)

        self.assertEqual(result.outcomes["m"].status, "failed")
        self.assertFalse(self._exists("multi.m.v1.py"))
        self.assertFalse(self._exists("multi.m.v2.py"))

    def test_failure_cascade_skips_transitive_dependents(self):
        # a fails (all None); b depends on a, c depends on b -> both skipped and
        # never submitted. Independent d still runs.
        cards = [
            _card("a", "gen_a.py", context_slice=["util.py"]),
            _card("d", "gen_d.py", context_slice=["util.py"]),
            _card("b", "gen_b.py", depends_on=["a"], context_slice=["util.py"]),
            _card("c", "gen_c.py", depends_on=["b"], context_slice=["util.py"]),
        ]
        backend = FakeBatchBackend(scripts={
            "a": None,
            "d": _code_block("D_OK = 1"),
        })

        result = run_deck(cards, backend, root=self.root, poll_interval=0)

        self.assertEqual(result.outcomes["a"].status, "failed")
        self.assertEqual(result.outcomes["d"].status, "written")
        self.assertEqual(result.outcomes["b"].status, "skipped")
        self.assertEqual(result.outcomes["b"].reason, "a")
        self.assertEqual(result.outcomes["c"].status, "skipped")
        self.assertEqual(result.outcomes["c"].reason, "b")
        # b and c were never handed to the backend.
        self.assertNotIn("b", backend.submitted_ids)
        self.assertNotIn("c", backend.submitted_ids)
        self.assertFalse(self._exists("gen_b.py"))
        self.assertFalse(self._exists("gen_c.py"))
        self.assertTrue(self._exists("gen_d.py"))

    def test_whole_batch_failed_fails_cards_and_skips_dependents(self):
        cards = [
            _card("a", "gen_a.py", context_slice=["util.py"]),
            _card("a2", "gen_a2.py", context_slice=["util.py"]),
            _card("b", "gen_b.py", depends_on=["a"], context_slice=["util.py"]),
        ]
        backend = FakeBatchBackend(fail=True)

        result = run_deck(cards, backend, root=self.root, poll_interval=0)

        # The whole first-generation batch failed -> every card in it failed.
        self.assertEqual(result.outcomes["a"].status, "failed")
        self.assertEqual(result.outcomes["a2"].status, "failed")
        # b's dependency failed -> b is skipped and never submitted.
        self.assertEqual(result.outcomes["b"].status, "skipped")
        self.assertEqual(result.outcomes["b"].reason, "a")
        self.assertNotIn("b", backend.submitted_ids)

    def test_generation_composition_logged_before_submit(self):
        cards = [
            _card("a", "gen_a.py", context_slice=["util.py"]),
            _card("b", "gen_b.py", context_slice=["util.py"]),
        ]
        log = []
        backend = FakeBatchBackend(
            scripts={"a": _code_block("A = 1"), "b": _code_block("B = 2")},
            log_ref=log,
        )

        run_deck(cards, backend, root=self.root, poll_interval=0, log=log.append)

        # At the moment submit was called, the composition line was already
        # logged.
        snapshot = backend.log_snapshots[0]
        self.assertTrue(
            any("submitting 2 card(s): a, b" in line for line in snapshot),
            snapshot,
        )
        self.assertTrue(any("[generation 1/1]" in line for line in snapshot))

    def test_deck_result_str_is_readable(self):
        cards = [
            _card("a", "gen_a.py", context_slice=["util.py"]),
            _card("b", "gen_b.py", depends_on=["a"], context_slice=["util.py"]),
        ]
        backend = FakeBatchBackend(scripts={"a": None, "b": _code_block("x=1")})

        result = run_deck(cards, backend, root=self.root, poll_interval=0)
        rendered = str(result)

        self.assertIn("deck run: 2 generation(s)", rendered)
        self.assertIn("a: failed", rendered)
        self.assertIn("b: skipped (blocked by a)", rendered)


class TruncatedResponseTests(unittest.TestCase):
    """A cut-off answer is a corrupt response, not a file body.

    Measured defect: a regeneration came back opening ```` ```python ```` and
    ending with ``---`` instead of a closing fence. The extraction regex needs
    the closing fence, matched nothing, and the "no fenced block -> verbatim"
    fallback wrote the literal ```` ```python ```` line to disk, so the card died
    of ``SyntaxError: invalid syntax`` three attempts running -- a paid batch
    each -- without ever telling the executor what was wrong.
    """

    def test_an_unclosed_fence_is_truncated(self):
        self.assertTrue(is_truncated_response("```python\nBROKEN = 1\n---"))
        self.assertTrue(is_truncated_response("```python\nBROKEN = 1"))
        # Two blocks, the second cut off: the file is still incomplete.
        self.assertTrue(is_truncated_response(
            "```python\nA = 1\n```\n```python\nB = 2"))

    def test_a_closed_fence_and_a_bare_answer_are_not(self):
        self.assertFalse(is_truncated_response(_code_block("OK = 1")))
        self.assertFalse(is_truncated_response("OK = 1\n"))
        self.assertFalse(is_truncated_response(""))
        self.assertFalse(is_truncated_response(None))
        # Prose that merely mentions a fence mid-line is not a delimiter.
        self.assertFalse(is_truncated_response("write it inside a ``` block"))

    def test_a_bare_answer_is_still_written_verbatim(self):
        # The behaviour that must NOT change: a model answering with plain code.
        self.assertEqual(response_to_file_body("OK = 1\n"), "OK = 1\n")


class TruncatedResponseRunDeckTests(unittest.TestCase):
    """The same defect through the deck loop, on both writer paths."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-cut-")
        self.root = os.path.join(self.tmp, "miniproject")
        shutil.copytree(MINIPROJECT, self.root, ignore=shutil.ignore_patterns("node_modules"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _exists(self, name):
        return os.path.exists(os.path.join(self.root, name))

    def _read(self, name):
        with open(os.path.join(self.root, name), "r", encoding="utf-8") as handle:
            return handle.read()

    def test_a_cut_off_answer_is_not_written_as_a_file(self):
        # No acceptance command: the card fails as if the response were missing,
        # rather than leaving a file that opens with ```python.
        cards = [_card("t", "gen_t.py", context_slice=["util.py"])]
        backend = FakeBatchBackend(scripts={"t": "```python\nBROKEN = 1\n---"})

        result = run_deck(cards, backend, root=self.root, poll_interval=0,
                          log=lambda _l: None)

        self.assertEqual(result.outcomes["t"].status, "failed")
        self.assertFalse(self._exists("gen_t.py"))

    def test_one_cut_off_variant_does_not_beat_a_whole_one(self):
        cards = [_card("m", "multi.py", context_slice=["util.py"], variants=2)]
        backend = FakeBatchBackend(scripts={
            "m.v1": "```python\nCUT = 1",          # cut off -> unusable
            "m.v2": _code_block("WHOLE = 2"),
        })

        result = run_deck(cards, backend, root=self.root, poll_interval=0,
                          log=lambda _l: None)

        self.assertEqual(result.outcomes["m"].status, "written")
        self.assertEqual(result.outcomes["m"].paths,
                         [os.path.join(self.root, "multi.m.v2.py")])
        self.assertFalse(self._exists("multi.m.v1.py"))

    def test_a_cut_off_answer_is_retried_and_told_that_it_was_cut_off(self):
        # With acceptance: the variant is rejected WITHOUT running acceptance
        # (there is no file to run it against), and the regeneration's
        # instruction says why -- the fact the three wasted attempts never had.
        cards = [_card("t", "gen_t.py", context_slice=["util.py"])]
        cards[0].acceptance = "grep -q PASS gen_t.py"
        backend = FakeBatchBackend(scripts={
            "t": "```python\nPASS = 1\n# ... cut off here",
            "t.r1": _code_block("PASS = 1"),
        })

        result = run_deck(cards, backend, root=self.root, poll_interval=0,
                          log=lambda _l: None)

        self.assertEqual(result.outcomes["t"].status, "written")
        self.assertEqual(result.outcomes["t"].attempts, 2)
        self.assertEqual(self._read("gen_t.py"), "PASS = 1\n")

        retry_text = "".join(message["content"]
                             for request in backend.submissions[1]
                             for message in request["messages"])
        self.assertIn("cut off mid-file", retry_text)


if __name__ == "__main__":
    unittest.main()
