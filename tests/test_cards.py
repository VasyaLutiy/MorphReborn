import json
import os
import tempfile
import unittest

from cards import (
    CardError,
    DeckError,
    MorphCard,
    load_deck,
    validate_deck,
)


class CardParsingTests(unittest.TestCase):

    def test_nested_form_parses_with_correct_defaults(self):
        card = MorphCard.from_dict({
            "custom_id": "gen-042",
            "meta": {"intent": "generate", "target": "scheduler.py"},
            "instruction": "Add priority levels.",
        })
        self.assertEqual(card.custom_id, "gen-042")
        self.assertEqual(card.intent, "generate")
        self.assertEqual(card.target, "scheduler.py")
        self.assertEqual(card.instruction, "Add priority levels.")
        # Defaults.
        self.assertEqual(card.context_slice, [])
        self.assertEqual(card.variants, 1)
        self.assertEqual(card.generation, 0)
        self.assertEqual(card.depends_on, [])
        self.assertIsNone(card.acceptance)
        self.assertIsNone(card.model)

    def test_flat_form_parses_identically_to_nested(self):
        nested = MorphCard.from_dict({
            "custom_id": "gen-042",
            "meta": {
                "intent": "patch",
                "target": "scheduler.py",
                "context_slice": ["scheduler.py"],
                "acceptance": "pytest tests/test_scheduler.py passes",
                "model": "claude-sonnet-5",
                "variants": 3,
                "generation": 1,
                "depends_on": [],
            },
            "instruction": "Add priorities.",
        })
        flat = MorphCard.from_dict({
            "custom_id": "gen-042",
            "intent": "patch",
            "target": "scheduler.py",
            "context_slice": ["scheduler.py"],
            "acceptance": "pytest tests/test_scheduler.py passes",
            "model": "claude-sonnet-5",
            "variants": 3,
            "generation": 1,
            "depends_on": [],
            "instruction": "Add priorities.",
        })
        self.assertEqual(nested, flat)

    def test_full_field_set_is_preserved(self):
        card = MorphCard.from_dict({
            "custom_id": "gen-042-scheduler-priorities",
            "meta": {
                "intent": "patch",
                "target": "scheduler.py",
                "context_slice": ["scheduler.py", "tests/test_scheduler.py"],
                "acceptance": "pytest tests/test_scheduler.py passes",
                "model": "claude-sonnet-5",
                "variants": 3,
                "generation": 1,
                "depends_on": ["gen-041"],
            },
            "instruction": "Add priority levels to queued jobs.",
        })
        self.assertEqual(card.context_slice, ["scheduler.py", "tests/test_scheduler.py"])
        self.assertEqual(card.acceptance, "pytest tests/test_scheduler.py passes")
        self.assertEqual(card.model, "claude-sonnet-5")
        self.assertEqual(card.variants, 3)
        self.assertEqual(card.generation, 1)
        self.assertEqual(card.depends_on, ["gen-041"])


class CardValidationTests(unittest.TestCase):

    def _card(self, **overrides):
        data = {
            "custom_id": "c1",
            "intent": "generate",
            "target": "out.py",
            "instruction": "do it",
        }
        data.update(overrides)
        return MorphCard.from_dict(data)

    def test_unknown_intent_rejected(self):
        with self.assertRaises(CardError) as ctx:
            self._card(intent="refactor")
        self.assertIn("intent", str(ctx.exception))

    def test_variants_zero_rejected(self):
        with self.assertRaises(CardError) as ctx:
            self._card(variants=0)
        self.assertIn("variants", str(ctx.exception))

    def test_variants_must_be_int_not_bool(self):
        with self.assertRaises(CardError):
            self._card(variants=True)

    def test_empty_custom_id_rejected(self):
        with self.assertRaises(CardError) as ctx:
            self._card(custom_id="")
        self.assertIn("custom_id", str(ctx.exception))

    def test_custom_id_with_slash_rejected(self):
        with self.assertRaises(CardError) as ctx:
            self._card(custom_id="foo/bar")
        self.assertIn("custom_id", str(ctx.exception))

    def test_custom_id_with_spaces_rejected(self):
        with self.assertRaises(CardError) as ctx:
            self._card(custom_id="foo bar")
        self.assertIn("custom_id", str(ctx.exception))

    def test_unknown_top_level_field_rejected(self):
        with self.assertRaises(CardError) as ctx:
            self._card(bogus="x")
        self.assertIn("bogus", str(ctx.exception))

    def test_unknown_meta_field_rejected(self):
        with self.assertRaises(CardError) as ctx:
            MorphCard.from_dict({
                "custom_id": "c1",
                "meta": {"intent": "generate", "target": "o.py", "bogus": 1},
                "instruction": "x",
            })
        self.assertIn("bogus", str(ctx.exception))

    def test_error_message_names_the_custom_id(self):
        with self.assertRaises(CardError) as ctx:
            self._card(custom_id="c-42", variants=0)
        self.assertIn("c-42", str(ctx.exception))

    def test_missing_target_rejected(self):
        with self.assertRaises(CardError) as ctx:
            MorphCard.from_dict({
                "custom_id": "c1",
                "meta": {"intent": "generate"},
                "instruction": "x",
            })
        self.assertIn("target", str(ctx.exception))

    def test_mixing_nested_and_flat_meta_rejected(self):
        with self.assertRaises(CardError):
            MorphCard.from_dict({
                "custom_id": "c1",
                "meta": {"intent": "generate", "target": "o.py"},
                "intent": "patch",
                "instruction": "x",
            })


class InstructionRequirementTests(unittest.TestCase):

    def test_todo_with_empty_instruction_accepted(self):
        card = MorphCard.from_dict({
            "custom_id": "todo-1",
            "meta": {"intent": "todo", "target": "TODO.md"},
        })
        self.assertEqual(card.instruction, "")
        self.assertEqual(card.intent, "todo")

    def test_generate_with_empty_instruction_rejected(self):
        with self.assertRaises(CardError) as ctx:
            MorphCard.from_dict({
                "custom_id": "gen-1",
                "meta": {"intent": "generate", "target": "o.py"},
            })
        self.assertIn("instruction", str(ctx.exception))

    def test_patch_with_empty_instruction_rejected(self):
        with self.assertRaises(CardError) as ctx:
            MorphCard.from_dict({
                "custom_id": "patch-1",
                "meta": {"intent": "patch", "target": "o.py"},
                "instruction": "",
            })
        self.assertIn("instruction", str(ctx.exception))


class DeckValidationTests(unittest.TestCase):

    def _card(self, custom_id, depends_on=None):
        return MorphCard(
            custom_id=custom_id,
            intent="generate",
            target=custom_id + ".py",
            instruction="do it",
            depends_on=depends_on or [],
        )

    def test_valid_deck_passes(self):
        validate_deck([self._card("a"), self._card("b", ["a"])])

    def test_duplicate_custom_id_rejected(self):
        with self.assertRaises(DeckError) as ctx:
            validate_deck([self._card("a"), self._card("a")])
        self.assertIn("a", str(ctx.exception))
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_unknown_dependency_rejected(self):
        with self.assertRaises(DeckError) as ctx:
            validate_deck([self._card("a", ["missing"])])
        self.assertIn("missing", str(ctx.exception))
        self.assertIn("a", str(ctx.exception))

    def test_self_dependency_rejected(self):
        with self.assertRaises(DeckError) as ctx:
            validate_deck([self._card("a", ["a"])])
        self.assertIn("a", str(ctx.exception))

    def test_cycle_rejected(self):
        cards = [
            self._card("a", ["c"]),
            self._card("b", ["a"]),
            self._card("c", ["b"]),
        ]
        with self.assertRaises(DeckError) as ctx:
            validate_deck(cards)
        message = str(ctx.exception).lower()
        self.assertIn("cycle", message)
        for member in ("a", "b", "c"):
            self.assertIn(member, message)

    def test_valid_diamond_dependency_accepted(self):
        cards = [
            self._card("a"),
            self._card("b", ["a"]),
            self._card("c", ["a"]),
            self._card("d", ["b", "c"]),
        ]
        validate_deck(cards)  # must not raise


class DeckLoadingTests(unittest.TestCase):

    def test_load_deck_round_trip(self):
        deck = [
            {
                "custom_id": "a",
                "meta": {"intent": "generate", "target": "a.py"},
                "instruction": "make a",
            },
            {
                "custom_id": "b",
                "intent": "patch",
                "target": "b.py",
                "instruction": "patch b",
                "depends_on": ["a"],
                "variants": 2,
            },
        ]
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(deck, handle)
            cards = load_deck(path)
        finally:
            os.remove(path)

        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0].custom_id, "a")
        self.assertEqual(cards[1].custom_id, "b")
        self.assertEqual(cards[1].depends_on, ["a"])
        self.assertEqual(cards[1].variants, 2)

    def test_load_deck_rejects_non_list(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"custom_id": "a"}, handle)
            with self.assertRaises(DeckError):
                load_deck(path)
        finally:
            os.remove(path)

    def test_load_deck_propagates_deck_errors(self):
        deck = [
            {"custom_id": "a", "intent": "generate", "target": "a.py",
             "instruction": "x", "depends_on": ["ghost"]},
        ]
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(deck, handle)
            with self.assertRaises(DeckError) as ctx:
                load_deck(path)
            self.assertIn("ghost", str(ctx.exception))
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
