"""
File-ownership tests: what two concurrent cards do to one file.

The deck-level half of the guarantee (the runtime half is
``tests/test_stale_context.py``). Pure and offline except for the preflight
tests, which need a ``DeckStore`` and therefore a throwaway tempdir.
"""

import os
import shutil
import tempfile
import unittest

from cards.deck import DeckError
from cards.generations import split_into_generations
from cards.hazards import (
    KIND_IMPLICIT_READ,
    KIND_READ_WRITE,
    KIND_UNORDERED_READ,
    KIND_WRITE_WRITE,
    HazardError,
    check_deck,
    errors,
    find_hazards,
    repair_deck,
    warnings,
)
from cards.schema import MorphCard
from cards.store import DeckStore, preflight_deck


def _card(custom_id, target=None, targets=None, slice_=None, depends_on=None):
    kwargs = {
        "custom_id": custom_id,
        "intent": "patch",
        "instruction": "do it",
        "context_slice": list(slice_ or []),
        "depends_on": list(depends_on or []),
    }
    if targets is not None:
        kwargs["targets"] = list(targets)
    else:
        kwargs["target"] = target
    return MorphCard(**kwargs)


class WriteWriteTests(unittest.TestCase):
    """Two cards writing one file: the lost update the machine used to allow."""

    def test_same_generation_is_an_error(self):
        cards = [_card("a", "util.py"), _card("b", "util.py")]
        found = find_hazards(cards)
        self.assertEqual([hazard.kind for hazard in found], [KIND_WRITE_WRITE])
        self.assertEqual(found[0].paths, ("util.py",))
        self.assertEqual(errors(found), found)

    def test_the_run_is_refused_and_names_every_pair(self):
        cards = [_card("a", "util.py"), _card("b", "util.py"),
                 _card("c", "app.py"), _card("d", "app.py")]
        with self.assertRaises(HazardError) as caught:
            check_deck(cards)
        message = str(caught.exception)
        self.assertIn("'a'", message)
        self.assertIn("'d'", message)
        self.assertEqual(len(caught.exception.hazards), 2)

    def test_a_hazard_error_is_a_deck_error(self):
        # Every caller that already reports a broken deck reports this too.
        with self.assertRaises(DeckError):
            check_deck([_card("a", "util.py"), _card("b", "util.py")])

    def test_different_generations_are_a_legitimate_patch_chain(self):
        # Generation N+1 is compiled AFTER N's morphs are on disk, so the second
        # card rewrites a file it has actually seen.
        cards = [_card("a", "util.py"), _card("b", "util.py", depends_on=["a"])]
        self.assertEqual(find_hazards(cards), [])

    def test_spelling_does_not_hide_a_collision(self):
        cards = [_card("a", "./util.py"), _card("b", "util.py")]
        self.assertEqual([hazard.kind for hazard in find_hazards(cards)],
                         [KIND_WRITE_WRITE])

    def test_a_changeset_card_contends_on_every_file_it_writes(self):
        cards = [_card("a", targets=["util.py", "tests/test_util.py"]),
                 _card("b", "tests/test_util.py")]
        found = find_hazards(cards)
        self.assertEqual(found[0].kind, KIND_WRITE_WRITE)
        self.assertEqual(found[0].paths, ("tests/test_util.py",))


class ReadWriteTests(unittest.TestCase):
    """A card reading what a sibling writes: an error, but a repairable one."""

    def test_reading_a_siblings_target_is_an_error(self):
        cards = [_card("writer", "util.py"), _card("reader", "app.py",
                                                   slice_=["util.py"])]
        found = [hazard for hazard in find_hazards(cards)
                 if hazard.kind == KIND_READ_WRITE]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].paths, ("util.py",))
        # The writer is always ``left``, the reader always ``right``.
        self.assertEqual((found[0].left, found[0].right), ("writer", "reader"))
        self.assertTrue(errors(find_hazards(cards)))

    def test_repair_adds_the_edge_and_serializes_the_pair(self):
        cards = [_card("writer", "util.py"), _card("reader", "app.py",
                                                   slice_=["util.py"])]
        repaired, edges = repair_deck(cards, find_hazards(cards))
        self.assertEqual(len(edges), 1)
        self.assertEqual(repaired[1].depends_on, ["writer"])
        self.assertEqual([[card.custom_id for card in generation]
                          for generation in split_into_generations(repaired)],
                         [["writer"], ["reader"]])
        # And the repaired deck is clean.
        self.assertEqual(find_hazards(repaired), [])

    def test_repair_never_mutates_the_cards_it_was_given(self):
        cards = [_card("writer", "util.py"), _card("reader", "app.py",
                                                   slice_=["util.py"])]
        repair_deck(cards, find_hazards(cards))
        self.assertEqual(cards[1].depends_on, [])

    def test_a_repair_that_would_close_a_cycle_is_refused(self):
        cards = [_card("a", "one.py", slice_=["two.py"]),
                 _card("b", "two.py", slice_=["one.py"])]
        with self.assertRaises(HazardError) as caught:
            repair_deck(cards, find_hazards(cards))
        self.assertIn("cycle", str(caught.exception))

    def test_an_existing_edge_is_no_hazard(self):
        cards = [_card("writer", "util.py"),
                 _card("reader", "app.py", slice_=["util.py"],
                       depends_on=["writer"])]
        self.assertEqual(find_hazards(cards), [])

    def test_a_transitive_edge_is_enough(self):
        cards = [_card("writer", "util.py"),
                 _card("middle", "mid.py", depends_on=["writer"]),
                 _card("reader", "app.py", slice_=["util.py"],
                       depends_on=["middle"])]
        self.assertEqual(find_hazards(cards), [])


class WarningTests(unittest.TestCase):
    """The hazards that are real but must not refuse a deck."""

    def test_an_empty_slice_reads_every_siblings_target(self):
        # Both slices are empty, so the reading is mutual: two cards of one
        # generation each holding the other's target as it was before the batch.
        cards = [_card("writer", "util.py"), _card("reader", "app.py")]
        found = find_hazards(cards)
        self.assertEqual([hazard.kind for hazard in found],
                         [KIND_IMPLICIT_READ, KIND_IMPLICIT_READ])
        self.assertEqual({(hazard.left, hazard.right) for hazard in found},
                         {("writer", "reader"), ("reader", "writer")})
        self.assertEqual(warnings(found), found)
        self.assertEqual(errors(found), [])
        # A warning never stops a run...
        self.assertEqual(check_deck(cards), found)
        # ...unless the operator asks for a hard line.
        with self.assertRaises(HazardError):
            check_deck(cards, strict=True)

    def test_a_cross_generation_read_with_no_edge_warns_about_the_cascade(self):
        cards = [_card("writer", "util.py"),
                 _card("other", "one.py"),
                 _card("reader", "app.py", slice_=["util.py"],
                       depends_on=["other"])]
        found = [hazard for hazard in find_hazards(cards)
                 if hazard.kind == KIND_UNORDERED_READ]
        self.assertEqual(len(found), 1)
        self.assertIsNone(found[0].generation)
        self.assertEqual(check_deck(cards), find_hazards(cards))

    def test_a_write_write_pair_is_not_also_reported_as_a_read(self):
        # Both cards write util.py and both have empty slices; naming that file
        # twice in two voices is how a report stops being read.
        cards = [_card("a", "util.py"), _card("b", "util.py")]
        self.assertEqual([hazard.kind for hazard in find_hazards(cards)],
                         [KIND_WRITE_WRITE])


class PreflightTests(unittest.TestCase):
    """The gate itself: what a run does with a deck before it spends anything."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-hazards-")
        self.store = DeckStore(project_root=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _notes(self):
        return []

    def test_preflight_repairs_and_saves_the_repaired_backlog(self):
        cards = [_card("writer", "util.py"),
                 _card("reader", "app.py", slice_=["util.py"])]
        self.store._save_cards(cards)
        notes = []
        repaired = preflight_deck(self.store, cards, log=notes.append)
        self.assertEqual(repaired[1].depends_on, ["writer"])
        # Saved, not merely returned: the deck as executed must be the deck on
        # disk, or the archive would record a deck that never ran.
        self.assertEqual(
            DeckStore(project_root=self.tmp).load_cards()[1].depends_on,
            ["writer"])
        self.assertTrue(any("deck repair:" in line for line in notes))

    def test_preflight_refuses_a_write_write_deck_and_changes_nothing(self):
        cards = [_card("a", "util.py"), _card("b", "util.py")]
        self.store._save_cards(cards)
        with self.assertRaises(HazardError):
            preflight_deck(self.store, cards, log=lambda _line: None)
        self.assertEqual(
            [card.depends_on
             for card in DeckStore(project_root=self.tmp).load_cards()],
            [[], []])

    def test_a_refused_deck_is_not_repaired_on_the_way_out(self):
        # A run that will not start must not also have edited the backlog while
        # saying so -- the operator gets their deck back as they wrote it.
        cards = [_card("a", "util.py"),
                 _card("b", "util.py"),
                 _card("reader", "app.py", slice_=["util.py"])]
        self.store._save_cards(cards)
        with self.assertRaises(HazardError) as caught:
            preflight_deck(self.store, cards, log=lambda _line: None)
        # The report still names BOTH problems, not just the one that stopped it.
        self.assertIn("both write util.py", str(caught.exception))
        self.assertIn("'reader'", str(caught.exception))
        self.assertEqual(
            [card.depends_on
             for card in DeckStore(project_root=self.tmp).load_cards()],
            [[], [], []])

    def test_preflight_passes_a_clean_deck_through_untouched(self):
        cards = [_card("a", "one.py", slice_=["util.py"]),
                 _card("b", "two.py", slice_=["util.py"])]
        self.store._save_cards(cards)
        notes = []
        self.assertEqual(
            [card.custom_id
             for card in preflight_deck(self.store, cards, log=notes.append)],
            ["a", "b"])
        self.assertEqual(notes, [])

    def test_preflight_warns_without_refusing(self):
        cards = [_card("writer", "util.py"), _card("reader", "app.py")]
        self.store._save_cards(cards)
        notes = []
        preflight_deck(self.store, cards, log=notes.append)
        self.assertTrue(any("deck warning:" in line for line in notes))


class DeckCheckCommandTests(unittest.TestCase):
    """``/deck check``: the one deck property the generation view cannot show.

    Driven through the real transition, in a temporary cwd, the way
    ``tests/test_deck_clear.py`` drives its own argument: the branch reads no
    instance state, so a ``MorphBot`` shell built without ``__init__`` keeps the
    test free of ".env", of configured processors and of the console transport.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-check-")
        self.previous = os.getcwd()
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self.previous)
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _check():
        """Run ``/deck check`` and return what it sent."""
        import asyncio

        from flows.morph import MorphBot

        sent = []

        class _Bot:
            async def send_message(self, chat_id=None, text=None, **_kwargs):
                sent.append(text)

        class _Context:
            def __init__(self, bot):
                self.bot = bot

        bot = _Bot()
        action = {"update": {"effective_chat": {"id": 7}},
                  "text": "/deck check",
                  "context": _Context(bot)}
        transition = MorphBot.build_deck_transition(MorphBot.__new__(MorphBot))
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(transition(action))
        return "\n".join(sent)

    def test_an_empty_deck_says_so(self):
        self.assertIn("empty", self._check())

    def test_a_clean_deck_says_what_it_checked(self):
        DeckStore(".")._save_cards([_card("a", "one.py", slice_=["util.py"]),
                                    _card("b", "two.py", slice_=["util.py"])])
        self.assertIn("no file-ownership hazards", self._check())

    def test_a_contended_file_is_named_with_both_cards(self):
        DeckStore(".")._save_cards([_card("a", "util.py"), _card("b", "util.py")])
        text = self._check()
        self.assertIn("util.py", text)
        self.assertIn("'a'", text)
        self.assertIn("'b'", text)
        self.assertIn("stop a run", text)

    def test_a_warning_only_deck_says_nothing_stops_the_run(self):
        DeckStore(".")._save_cards([_card("a", "one.py"), _card("b", "two.py")])
        self.assertIn("None of these stop a run", self._check())


if __name__ == "__main__":
    unittest.main()
