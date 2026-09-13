import json
import os
import unittest

from cards.deck import load_deck
from cards.schema import MorphCard
from cards.compiler import (
    compile_card,
    compile_deck,
    serialize_anthropic,
    serialize_openai,
)
from context_folder_dialog import ContextFolderDialog


HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
GOLDEN = os.path.join(HERE, "golden")

# The compile root is intentionally RELATIVE to the invocation directory (the
# repo root, as the acceptance step and test_scheduler both assume). The root
# string is embedded in the context file paths, so a relative root keeps the
# golden files portable across machines; the golden files were generated with
# exactly this value.
MINIPROJECT = os.path.join("tests", "fixtures", "miniproject")

# The default models the golden files were serialized with. Kept in step with
# the generation step documented in tests/golden and the compiler report.
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-5"
OPENAI_DEFAULT_MODEL = "gpt-4o"


def _read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


class GoldenFileTests(unittest.TestCase):
    """The reference deck compiled against the fixture project must match the
    committed golden JSONL byte-for-byte, for both providers."""

    def setUp(self):
        self.cards = load_deck(os.path.join(FIXTURES, "deck.json"))
        self.requests = compile_deck(self.cards, root=MINIPROJECT)

    def test_anthropic_matches_golden(self):
        produced = serialize_anthropic(self.requests, default_model=ANTHROPIC_DEFAULT_MODEL)
        self.assertEqual(produced, _read(os.path.join(GOLDEN, "anthropic.jsonl")))

    def test_openai_matches_golden(self):
        produced = serialize_openai(self.requests, default_model=OPENAI_DEFAULT_MODEL)
        self.assertEqual(produced, _read(os.path.join(GOLDEN, "openai.jsonl")))

    def test_both_serializers_end_with_single_newline(self):
        anthropic = serialize_anthropic(self.requests, default_model=ANTHROPIC_DEFAULT_MODEL)
        openai = serialize_openai(self.requests, default_model=OPENAI_DEFAULT_MODEL)
        for text in (anthropic, openai):
            self.assertTrue(text.endswith("\n"))
            self.assertFalse(text.endswith("\n\n"))


class DeterminismTests(unittest.TestCase):
    def test_compiling_twice_is_byte_identical(self):
        cards = load_deck(os.path.join(FIXTURES, "deck.json"))

        first_requests = compile_deck(cards, root=MINIPROJECT)
        second_requests = compile_deck(cards, root=MINIPROJECT)

        self.assertEqual(
            serialize_anthropic(first_requests, default_model=ANTHROPIC_DEFAULT_MODEL),
            serialize_anthropic(second_requests, default_model=ANTHROPIC_DEFAULT_MODEL),
        )
        self.assertEqual(
            serialize_openai(first_requests, default_model=OPENAI_DEFAULT_MODEL),
            serialize_openai(second_requests, default_model=OPENAI_DEFAULT_MODEL),
        )

    def test_no_time_field_leaks_into_any_message(self):
        cards = load_deck(os.path.join(FIXTURES, "deck.json"))
        requests = compile_deck(cards, root=MINIPROJECT)
        for request in requests:
            for message in request["messages"]:
                self.assertNotIn("time", message)
                self.assertEqual(set(message), {"role", "content"})


class VariantTests(unittest.TestCase):
    def test_single_variant_keeps_bare_custom_id(self):
        card = MorphCard(
            custom_id="solo",
            intent="generate",
            target="out.py",
            instruction="do it",
            context_slice=["app.py"],
        )
        requests = compile_card(card, root=MINIPROJECT)
        self.assertEqual([r["custom_id"] for r in requests], ["solo"])

    def test_multiple_variants_are_suffixed_v1_upwards(self):
        card = MorphCard(
            custom_id="multi",
            intent="generate",
            target="out.py",
            instruction="do it",
            context_slice=["app.py"],
            variants=3,
        )
        requests = compile_card(card, root=MINIPROJECT)
        self.assertEqual(
            [r["custom_id"] for r in requests],
            ["multi.v1", "multi.v2", "multi.v3"],
        )

    def test_variants_share_the_identical_conversation(self):
        card = MorphCard(
            custom_id="multi",
            intent="generate",
            target="out.py",
            instruction="do it",
            context_slice=["app.py"],
            variants=3,
        )
        requests = compile_card(card, root=MINIPROJECT)
        self.assertEqual(requests[0]["messages"], requests[1]["messages"])
        self.assertEqual(requests[1]["messages"], requests[2]["messages"])


class ModelResolutionTests(unittest.TestCase):
    def test_card_model_overrides_default_in_anthropic(self):
        card = MorphCard(
            custom_id="pinned",
            intent="generate",
            target="out.py",
            instruction="do it",
            context_slice=["app.py"],
            model="claude-opus-4",
        )
        requests = compile_card(card, root=MINIPROJECT)
        line = serialize_anthropic(requests, default_model="claude-sonnet-5").strip()
        self.assertEqual(json.loads(line)["params"]["model"], "claude-opus-4")

    def test_card_model_overrides_default_in_openai(self):
        card = MorphCard(
            custom_id="pinned",
            intent="generate",
            target="out.py",
            instruction="do it",
            context_slice=["app.py"],
            model="claude-opus-4",
        )
        requests = compile_card(card, root=MINIPROJECT)
        line = serialize_openai(requests, default_model="gpt-4o").strip()
        self.assertEqual(json.loads(line)["body"]["model"], "claude-opus-4")

    def test_missing_card_model_falls_back_to_default(self):
        card = MorphCard(
            custom_id="unpinned",
            intent="generate",
            target="out.py",
            instruction="do it",
            context_slice=["app.py"],
        )
        requests = compile_card(card, root=MINIPROJECT)
        line = serialize_anthropic(requests, default_model="claude-sonnet-5").strip()
        self.assertEqual(json.loads(line)["params"]["model"], "claude-sonnet-5")


class WholeProjectModeTests(unittest.TestCase):
    def _context_paths(self, messages):
        """The embedded file paths, in emission order, from context messages."""
        paths = []
        prefix = 'Contents for another file "'
        for message in messages:
            content = message["content"]
            if message["role"] == "user" and content.startswith(prefix):
                paths.append(content[len(prefix):].split('"', 1)[0])
        return paths

    def test_empty_slice_walks_whole_project_sorted_excluding_node_modules(self):
        card = MorphCard(
            custom_id="whole",
            intent="generate",
            target="out.py",
            instruction="do it",
            context_slice=[],
        )
        requests = compile_card(card, root=MINIPROJECT)
        paths = self._context_paths(requests[0]["messages"])

        expected = [
            os.path.join(MINIPROJECT, "README.md"),
            os.path.join(MINIPROJECT, "app.py"),
            os.path.join(MINIPROJECT, "util.py"),
        ]
        self.assertEqual(paths, expected)
        self.assertTrue(all("node_modules" not in p for p in paths))
        self.assertEqual(paths, sorted(paths))


class FileListModeTests(unittest.TestCase):
    def test_missing_slice_file_raises_naming_the_path(self):
        card = MorphCard(
            custom_id="broken",
            intent="generate",
            target="out.py",
            instruction="do it",
            context_slice=["does_not_exist.py"],
        )
        with self.assertRaises(FileNotFoundError) as ctx:
            compile_card(card, root=MINIPROJECT)
        self.assertIn("does_not_exist.py", str(ctx.exception))


class PatchConversationShapeTests(unittest.TestCase):
    def test_patch_order_is_assistant_assistant_context_user(self):
        card = MorphCard(
            custom_id="patch",
            intent="patch",
            target="util.py",
            instruction="add subtract",
            context_slice=["app.py"],
        )
        requests = compile_card(card, root=MINIPROJECT)
        messages = requests[0]["messages"]

        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["content"], "Let's update the util.py file provided.")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertTrue(messages[1]["content"].startswith("Original file:"))
        # one context message (app.py), then the user instruction last.
        self.assertEqual([m["role"] for m in messages], ["assistant", "assistant", "user", "user"])
        self.assertEqual(messages[-1]["content"], "add subtract")

    def test_todo_has_fixed_prompt_and_no_project_context(self):
        card = MorphCard(
            custom_id="todo",
            intent="todo",
            target="util.py",
            context_slice=["app.py"],
        )
        requests = compile_card(card, root=MINIPROJECT)
        messages = requests[0]["messages"]

        self.assertEqual([m["role"] for m in messages], ["assistant", "assistant", "user"])
        self.assertIn("criticize this file contents", messages[-1]["content"])
        # todo never embeds project context, even when a slice is present.
        self.assertNotIn("Contents for another file", "".join(m["content"] for m in messages))


class ContextFolderDialogWalkModeTests(unittest.TestCase):
    """Smoke test: the pre-existing walk mode still behaves as before when no
    file_list is supplied (the constructor gained an optional argument only)."""

    def test_walk_mode_unchanged_reads_filtered_files(self):
        def keep_py(path):
            return path.endswith(".py") and "node_modules" not in path

        dialog = ContextFolderDialog(MINIPROJECT, filter_callback=keep_py)
        dialog.process([])

        contents = [m["content"] for m in dialog.conversation]
        self.assertEqual(len(contents), 2)  # app.py and util.py, not README or junk.js
        joined = "".join(contents)
        self.assertIn("app.py", joined)
        self.assertIn("util.py", joined)
        self.assertNotIn("junk.js", joined)
        # messages carry the transport time field in walk mode (unchanged).
        self.assertIn("time", dialog.conversation[0])


if __name__ == "__main__":
    unittest.main()
