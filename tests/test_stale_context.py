"""
The runtime half of file ownership: a card judged against files that moved.

``tests/test_hazards.py`` covers the deck-level gate, which refuses a deck that
PLANS a collision. This covers what no deck check can know: the tree changing
between the moment a card was compiled and the moment its answer is judged --
a sibling's accepted morph, a retry landing next to one, a hand edit during the
hour a batch sits in a provider's queue. Also here: a card that cannot be
compiled at all, which used to take the whole run with it.

Offline. Every test runs against a throwaway tempdir; the acceptance commands
are one-line ``python3 -c`` imports of the file under test.
"""

import os
import shutil
import tempfile
import unittest

from cards.compiler import CompiledInputs
from cards.generations import REASON_COMPILE, REASON_STALE, run_deck
from cards.schema import MorphCard
from cards.store import DeckStore, collect_generation, submit_generation


BASE = "def base():\n    return 1\n"
WITH_A = BASE + "\n\ndef helper_a():\n    return 'a'\n"
WITH_B = BASE + "\n\ndef helper_b():\n    return 'b'\n"
WITH_BOTH = WITH_A + "\n\ndef helper_b():\n    return 'b'\n"


def _block(body):
    return f"```python\n{body}\n```"


class FakeBatchBackend:
    """Replays scripted responses and records every request it was given."""

    def __init__(self, scripts, default_response=None):
        self.scripts = scripts
        self.default_response = default_response
        self.submissions = []
        self._batches = {}

    def submit(self, requests):
        batch_id = f"batch-{len(self.submissions) + 1}"
        self.submissions.append(list(requests))
        self._batches[batch_id] = [request["custom_id"] for request in requests]
        return batch_id

    def status(self, _batch_id):
        return "completed"

    def collect(self, batch_id):
        return {custom_id: self.scripts.get(custom_id, self.default_response)
                for custom_id in self._batches[batch_id]}

    def instruction_for(self, custom_id):
        """The closing user message of the request with this custom_id."""
        for batch in self.submissions:
            for request in batch:
                if request["custom_id"] == custom_id:
                    return request["messages"][-1]["content"]
        raise AssertionError(f"{custom_id!r} was never submitted")


class _ProjectCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="morph-stale-")
        self._write("util.py", BASE)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, name, text):
        path = os.path.join(self.root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(name) else None
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def _read(self, name):
        with open(os.path.join(self.root, name), encoding="utf-8") as handle:
            return handle.read()

    @staticmethod
    def _card(custom_id, target, acceptance=None, slice_=None, depends_on=None,
              intent="patch"):
        return MorphCard(
            custom_id=custom_id, intent=intent, target=target,
            instruction="do it", context_slice=list(slice_ or []),
            acceptance=acceptance, depends_on=list(depends_on or []))


class CompiledInputsTests(_ProjectCase):
    """What a compilation read, and how it knows the answer is about to be stale."""

    def test_capture_records_the_slice_and_the_target(self):
        self._write("app.py", "import util\n")
        card = self._card("c", "util.py", slice_=["app.py"])
        captured = CompiledInputs.capture(card, self.root)
        self.assertEqual(captured.paths, ["app.py", "util.py"])

    def test_an_untouched_tree_has_changed_nothing(self):
        card = self._card("c", "util.py", slice_=["util.py"])
        self.assertEqual(CompiledInputs.capture(card, self.root).changed(self.root), [])

    def test_an_edited_input_is_named(self):
        card = self._card("c", "app.py", slice_=["util.py"])
        captured = CompiledInputs.capture(card, self.root)
        self._write("util.py", WITH_A)
        self.assertEqual(captured.changed(self.root), ["util.py"])

    def test_a_deleted_input_is_a_change(self):
        card = self._card("c", "app.py", slice_=["util.py"])
        captured = CompiledInputs.capture(card, self.root)
        os.remove(os.path.join(self.root, "util.py"))
        self.assertEqual(captured.changed(self.root), ["util.py"])

    def test_a_file_the_tree_merely_GREW_is_not_a_change(self):
        # The first acceptance run of a generation drops .pytest_cache/README.md
        # into the tree; that is not a change to anything a card was compiled
        # from, and every later card of the generation would otherwise be
        # declared stale by it.
        card = self._card("c", "util.py", slice_=["util.py"])
        captured = CompiledInputs.capture(card, self.root)
        self._write("README.md", "# unrelated\n")
        self.assertEqual(captured.changed(self.root), [])

    def test_an_empty_slice_declares_only_the_targets(self):
        # An empty slice sends the whole project, but DECLARES nothing: watching
        # the walk would declare every card of a generation stale as soon as the
        # first is accepted, buying a provider queue per generation to
        # regenerate answers that are almost always fine. What it still watches
        # is the card's own target -- the only file through which a card can
        # destroy another card's accepted work.
        self._write("app.py", "import util\n")
        card = self._card("c", "util.py")
        captured = CompiledInputs.capture(card, self.root)
        self.assertEqual(captured.paths, ["util.py"])
        self._write("app.py", "import util  # edited by a sibling\n")
        self.assertEqual(captured.changed(self.root), [])
        self._write("util.py", WITH_A)
        self.assertEqual(captured.changed(self.root), ["util.py"])

    def test_the_record_survives_a_round_trip_through_state(self):
        card = self._card("c", "util.py", slice_=["util.py"])
        captured = CompiledInputs.capture(card, self.root)
        restored = CompiledInputs.from_dict(captured.to_dict())
        self.assertEqual(restored.changed(self.root), [])
        self._write("util.py", WITH_A)
        self.assertEqual(restored.changed(self.root), ["util.py"])

    def test_a_missing_or_malformed_record_disables_the_check(self):
        # An in-flight batch submitted by an older version must stay
        # collectable: no record means no check, never a crash.
        self.assertEqual(CompiledInputs.from_dict(None).changed(self.root), [])
        self.assertEqual(CompiledInputs.from_dict({"digests": "?"}).digests, {})


class StaleAnswerTests(_ProjectCase):
    """Two cards of one generation writing one file, with the gate bypassed."""

    def _cards(self, acceptance_a=None, acceptance_b=None):
        return [
            self._card("card-a", "util.py", acceptance=acceptance_a,
                       slice_=["util.py"]),
            self._card("card-b", "util.py", acceptance=acceptance_b,
                       slice_=["util.py"]),
        ]

    def test_the_second_card_is_regenerated_instead_of_overwriting_the_first(self):
        backend = FakeBatchBackend({
            "card-a": _block(WITH_A),
            "card-b": _block(WITH_B),        # written against util.py WITHOUT helper_a
            "card-b.r1": _block(WITH_BOTH),  # recompiled, this one has seen it
        })
        notes = []
        result = run_deck(
            self._cards("python3 -c \"import util; util.helper_a()\"",
                        "python3 -c \"import util; util.helper_b()\""),
            backend, root=self.root, poll_interval=0, log=notes.append)

        self.assertEqual(result.outcomes["card-a"].status, "written")
        self.assertEqual(result.outcomes["card-b"].status, "written")
        self.assertEqual(result.outcomes["card-b"].attempts, 2)
        # The whole point: neither card's work was lost.
        body = self._read("util.py")
        self.assertIn("helper_a", body)
        self.assertIn("helper_b", body)
        self.assertTrue(any("discarded UNREAD" in note for note in notes),
                        notes)

    def test_the_stale_answer_is_never_read_and_never_written(self):
        # card-b's response would break card-a's acceptance if it landed.
        backend = FakeBatchBackend({
            "card-a": _block(WITH_A),
            "card-b": _block(WITH_B),
            "card-b.r1": _block(WITH_BOTH),
        })
        run_deck(self._cards("python3 -c \"import util; util.helper_a()\"",
                             "python3 -c \"import util; util.helper_b()\""),
                 backend, root=self.root, poll_interval=0, log=lambda _l: None)
        self.assertIn("helper_a", self._read("util.py"))

    def test_the_retry_prompt_does_not_blame_the_executor(self):
        # A stale answer failed no test -- nothing ran. Telling the model it
        # failed acceptance makes it "fix" code nothing found fault with.
        backend = FakeBatchBackend({
            "card-a": _block(WITH_A),
            "card-b": _block(WITH_B),
            "card-b.r1": _block(WITH_BOTH),
        })
        run_deck(self._cards("python3 -c \"import util; util.helper_a()\"",
                             "python3 -c \"import util; util.helper_b()\""),
                 backend, root=self.root, poll_interval=0, log=lambda _l: None)
        instruction = backend.instruction_for("card-b.r1")
        self.assertIn("discarded before acceptance could run", instruction)
        self.assertIn("no longer exists", instruction)
        self.assertNotIn("failed its acceptance check", instruction)

    def test_without_acceptance_a_stale_card_fails_rather_than_overwriting(self):
        # Nothing to regenerate against, so the only safe answer is to refuse
        # the write -- the previous behaviour was to write it silently.
        backend = FakeBatchBackend({"card-a": _block(WITH_A),
                                    "card-b": _block(WITH_B)})
        result = run_deck(self._cards(), backend, root=self.root,
                          poll_interval=0, log=lambda _line: None)
        self.assertEqual(result.outcomes["card-a"].status, "written")
        self.assertEqual(result.outcomes["card-b"].status, "failed")
        self.assertEqual(result.outcomes["card-b"].reason, REASON_STALE)
        body = self._read("util.py")
        self.assertIn("helper_a", body)
        self.assertNotIn("helper_b", body)

    def test_a_hand_edit_during_the_batch_is_caught_too(self):
        class EditingBackend(FakeBatchBackend):
            """Edits the tree while the batch is 'in the queue'."""

            def __init__(self, scripts, root):
                super().__init__(scripts)
                self.root = root

            def status(self, batch_id):
                with open(os.path.join(self.root, "util.py"), "w",
                          encoding="utf-8") as handle:
                    handle.write("def base():\n    return 99\n")
                return "completed"

        backend = EditingBackend({"card-a": _block(WITH_A)}, self.root)
        result = run_deck([self._card("card-a", "util.py", slice_=["util.py"])],
                          backend, root=self.root, poll_interval=0,
                          log=lambda _line: None)
        self.assertEqual(result.outcomes["card-a"].status, "failed")
        self.assertEqual(result.outcomes["card-a"].reason, REASON_STALE)
        self.assertEqual(self._read("util.py"), "def base():\n    return 99\n")


class CompileFailureTests(_ProjectCase):
    """One card that will not compile is one card, not the run."""

    def test_a_bad_slice_fails_its_own_card_and_spares_the_generation(self):
        cards = [
            self._card("good", "a.py", intent="generate", slice_=["util.py"]),
            self._card("bad", "b.py", intent="generate", slice_=["nope.py"]),
            self._card("after", "c.py", intent="generate", slice_=["util.py"],
                       depends_on=["bad"]),
        ]
        backend = FakeBatchBackend({}, default_response=_block("A = 1\n"))
        notes = []
        result = run_deck(cards, backend, root=self.root, poll_interval=0,
                          log=notes.append)

        self.assertEqual(result.outcomes["good"].status, "written")
        self.assertEqual(result.outcomes["bad"].status, "failed")
        self.assertEqual(result.outcomes["bad"].reason, REASON_COMPILE)
        self.assertIn("nope.py", result.outcomes["bad"].acceptance_output)
        # ...and its dependent is skipped, not compiled against a file that will
        # never be written.
        self.assertEqual(result.outcomes["after"].status, "skipped")
        self.assertEqual(result.outcomes["after"].reason, "bad")
        # The generation that carried the bad card still went to the provider.
        self.assertEqual([request["custom_id"] for request in backend.submissions[0]],
                         ["good"])

    def test_a_generation_where_nothing_compiles_does_not_submit_an_empty_batch(self):
        cards = [self._card("bad", "b.py", intent="generate", slice_=["nope.py"])]
        backend = FakeBatchBackend({}, default_response=_block("A = 1\n"))
        result = run_deck(cards, backend, root=self.root, poll_interval=0,
                          log=lambda _line: None)
        self.assertEqual(result.outcomes["bad"].status, "failed")
        self.assertEqual(backend.submissions, [])


class SplitStepTests(_ProjectCase):
    """The same two guarantees on the /submit -> /collect route.

    That route persists its batch and is collected by a LATER CLI session, so
    the record of what each card was compiled from has to survive
    ``.morph/state.json`` -- otherwise the guard would protect the blocking
    ``/nightly`` pass and quietly not protect the overnight one, which is the
    route that runs while nobody is watching.
    """

    def setUp(self):
        super().setUp()
        self.store = DeckStore(project_root=self.root)

    def _submit(self, cards, backend):
        self.store._save_cards(cards)
        return submit_generation(self.store, backend, root=self.root,
                                 backend_label="fake", use_git=False,
                                 log=lambda _line: None)

    def test_the_compiled_inputs_are_persisted_with_the_batch(self):
        backend = FakeBatchBackend({"card-a": _block(WITH_A)})
        self._submit([self._card("card-a", "util.py", slice_=["util.py"])],
                     backend)
        stored = DeckStore(project_root=self.root).load_state()["inputs"]
        self.assertEqual(sorted(stored["card-a"]["digests"]), ["util.py"])

    def test_an_edit_between_submit_and_collect_discards_the_answer(self):
        backend = FakeBatchBackend({"card-a": _block(WITH_A)})
        self._submit([self._card("card-a", "util.py", slice_=["util.py"])],
                     backend)
        # A whole CLI session later, with the file edited by hand meanwhile.
        self._write("util.py", "def base():\n    return 99\n")
        result = collect_generation(DeckStore(project_root=self.root), backend,
                                    root=self.root, log=lambda _line: None)
        outcome = result.outcomes["card-a"]
        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.reason, REASON_STALE)
        self.assertEqual(self._read("util.py"), "def base():\n    return 99\n")

    def test_a_card_that_will_not_compile_does_not_stop_the_submit(self):
        backend = FakeBatchBackend({}, default_response=_block("A = 1\n"))
        result = self._submit([
            self._card("good", "a.py", intent="generate", slice_=["util.py"]),
            self._card("bad", "b.py", intent="generate", slice_=["nope.py"]),
        ], backend)
        self.assertTrue(result.submitted)
        self.assertEqual(result.card_ids, ["good"])
        outcomes = DeckStore(project_root=self.root).load_outcomes()
        self.assertEqual(outcomes["bad"].status, "failed")
        self.assertEqual(outcomes["bad"].reason, REASON_COMPILE)

    def test_a_generation_that_compiles_to_nothing_sends_no_batch(self):
        backend = FakeBatchBackend({}, default_response=_block("A = 1\n"))
        result = self._submit(
            [self._card("bad", "b.py", intent="generate", slice_=["nope.py"])],
            backend)
        self.assertFalse(result.submitted)
        self.assertTrue(result.done)
        self.assertEqual(backend.submissions, [])
        self.assertEqual(
            DeckStore(project_root=self.root).load_outcomes()["bad"].reason,
            REASON_COMPILE)


if __name__ == "__main__":
    unittest.main()
