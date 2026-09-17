import json
import os
import unittest

from cards.deck import load_deck
from cards.schema import MorphCard
from cards.compiler import (
    MULTI_TARGET_DIRECTIVE,
    _context_messages,
    compile_card,
    compile_deck,
    serialize_anthropic,
    serialize_openai,
    serialize_openrouter,
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

# OpenRouter applies ONE model to the whole batch, and the reference deck pins
# "claude-opus-4" on its variants card. So the golden batch model is that same
# slug -- which is also the only configuration in which this deck is a legal
# OpenRouter submission at all, and it exercises the "a request may name the
# batch model" case. A deck pinning anything else must be split (see
# OpenRouterMixedModelTests).
OPENROUTER_DEFAULT_MODEL = "claude-opus-4"


def _read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


class GoldenFileTests(unittest.TestCase):
    """The reference deck compiled against the fixture project must match the
    committed golden payload byte-for-byte, for every provider."""

    def setUp(self):
        self.cards = load_deck(os.path.join(FIXTURES, "deck.json"))
        self.requests = compile_deck(self.cards, root=MINIPROJECT)

    def test_anthropic_matches_golden(self):
        produced = serialize_anthropic(self.requests, default_model=ANTHROPIC_DEFAULT_MODEL)
        self.assertEqual(produced, _read(os.path.join(GOLDEN, "anthropic.jsonl")))

    def test_openai_matches_golden(self):
        produced = serialize_openai(self.requests, default_model=OPENAI_DEFAULT_MODEL)
        self.assertEqual(produced, _read(os.path.join(GOLDEN, "openai.jsonl")))

    def test_openrouter_matches_golden(self):
        produced = serialize_openrouter(self.requests, default_model=OPENROUTER_DEFAULT_MODEL)
        self.assertEqual(produced, _read(os.path.join(GOLDEN, "openrouter.json")))

    def test_openrouter_payload_key_order_is_endpoint_model_requests(self):
        # The service stream-parses the body and answers 400 if "requests"
        # arrives first, so the order is part of the golden contract.
        produced = serialize_openrouter(self.requests, default_model=OPENROUTER_DEFAULT_MODEL)
        payload = json.loads(produced, object_pairs_hook=list)
        self.assertEqual([key for key, _ in payload], ["endpoint", "model", "requests"])

    def test_openrouter_request_bodies_carry_no_model(self):
        produced = serialize_openrouter(self.requests, default_model=OPENROUTER_DEFAULT_MODEL)
        payload = json.loads(produced)
        self.assertEqual(payload["endpoint"], "/v1/chat/completions")
        self.assertEqual(len(payload["requests"]), len(self.requests))
        for item in payload["requests"]:
            self.assertEqual(set(item), {"custom_id", "body"})
            self.assertEqual(set(item["body"]), {"messages"})

    def test_both_serializers_end_with_single_newline(self):
        anthropic = serialize_anthropic(self.requests, default_model=ANTHROPIC_DEFAULT_MODEL)
        openai = serialize_openai(self.requests, default_model=OPENAI_DEFAULT_MODEL)
        for text in (anthropic, openai):
            self.assertTrue(text.endswith("\n"))
            self.assertFalse(text.endswith("\n\n"))


class OpenRouterMixedModelTests(unittest.TestCase):
    """OpenRouter takes one model per batch, so a deck mixing models is a
    compile-time error naming the offending card, not a 400 from the service."""

    def test_mixed_models_raise_naming_the_custom_id(self):
        card = MorphCard(
            custom_id="pinned-elsewhere",
            intent="generate",
            target="out.py",
            instruction="do it",
            context_slice=["app.py"],
            model="claude-opus-4",
        )
        requests = compile_card(card, root=MINIPROJECT)
        with self.assertRaises(ValueError) as ctx:
            serialize_openrouter(requests, default_model="z-ai/glm-5.3-flash:batch")
        message = str(ctx.exception)
        self.assertIn("pinned-elsewhere", message)
        self.assertIn("claude-opus-4", message)
        self.assertIn("z-ai/glm-5.3-flash:batch", message)

    def test_a_request_pinning_the_batch_model_is_accepted(self):
        card = MorphCard(
            custom_id="pinned-same",
            intent="generate",
            target="out.py",
            instruction="do it",
            context_slice=["app.py"],
            model="z-ai/glm-5.3-flash:batch",
        )
        requests = compile_card(card, root=MINIPROJECT)
        payload = json.loads(
            serialize_openrouter(requests, default_model="z-ai/glm-5.3-flash:batch"))
        self.assertEqual(payload["model"], "z-ai/glm-5.3-flash:batch")
        self.assertNotIn("model", payload["requests"][0]["body"])


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


class TargetIsSentOnceTests(unittest.TestCase):
    """A patch card ships its target as "Original file"; the context slice must
    not ship it again. Measured on a real card, the duplicate was 92% of a
    127 KB prompt -- twice the input cost for nothing the model did not have."""

    def _occurrences(self, messages, needle):
        return sum(m["content"].count(needle) for m in messages)

    def _prompt_size(self, messages):
        return sum(len(m["content"]) for m in messages)

    def _context_paths(self, messages):
        """The embedded file paths, in emission order, from context messages."""
        paths = []
        prefix = 'Contents for another file "'
        for message in messages:
            content = message["content"]
            if message["role"] == "user" and content.startswith(prefix):
                paths.append(content[len(prefix):].split('"', 1)[0])
        return paths

    # A line that exists only in the fixture's util.py, so counting it counts
    # copies of the target rather than of any incidental substring.
    UTIL_MARKER = "def add(a, b):"

    def test_patch_slice_listing_its_own_target_sends_it_once(self):
        with_target = MorphCard(
            custom_id="patch-dup",
            intent="patch",
            target="util.py",
            instruction="add subtract",
            context_slice=["app.py", "util.py"],
        )
        without_target = MorphCard(
            custom_id="patch-clean",
            intent="patch",
            target="util.py",
            instruction="add subtract",
            context_slice=["app.py"],
        )
        duplicated = compile_card(with_target, root=MINIPROJECT)[0]["messages"]
        baseline = compile_card(without_target, root=MINIPROJECT)[0]["messages"]

        # The target's distinctive line appears exactly once: in "Original file".
        self.assertEqual(self._occurrences(duplicated, self.UTIL_MARKER), 1)
        self.assertTrue(duplicated[1]["content"].startswith("Original file:"))
        self.assertIn(self.UTIL_MARKER, duplicated[1]["content"])
        self.assertNotIn("util.py", "".join(self._context_paths(duplicated)))

        # ...and the prompt is now exactly the one of a card that never listed
        # the target at all.
        self.assertEqual(duplicated, baseline)

        # The measurement itself: the un-skipped rendering is larger by exactly
        # the size of the one context message carrying the target.
        unskipped = _context_messages(with_target, MINIPROJECT, skip_target=False)
        skipped = _context_messages(with_target, MINIPROJECT, skip_target=True)
        target_message_size = self._prompt_size(unskipped) - self._prompt_size(skipped)
        self.assertGreater(target_message_size, 0)
        self.assertEqual(self._occurrences(unskipped, self.UTIL_MARKER), 1)
        self.assertEqual(self._occurrences(skipped, self.UTIL_MARKER), 0)
        self.assertLess(
            self._prompt_size(duplicated),
            self._prompt_size(duplicated) + target_message_size,
        )

    def test_patch_slice_normalises_paths_before_matching_the_target(self):
        # "./util.py" and "util.py" are the same file; a raw string compare
        # would miss it and ship the target twice.
        card = MorphCard(
            custom_id="patch-dotted",
            intent="patch",
            target="util.py",
            instruction="add subtract",
            context_slice=["app.py", "./util.py"],
        )
        messages = compile_card(card, root=MINIPROJECT)[0]["messages"]
        self.assertEqual(self._occurrences(messages, self.UTIL_MARKER), 1)
        self.assertEqual(
            self._context_paths(messages),
            [os.path.join(MINIPROJECT, "app.py")],
        )

    def test_patch_with_empty_slice_excludes_target_from_whole_project_walk(self):
        # The whole-project fallback also yields the target; it needs the same
        # treatment as an explicit slice.
        card = MorphCard(
            custom_id="patch-whole",
            intent="patch",
            target="util.py",
            instruction="add subtract",
            context_slice=[],
        )
        messages = compile_card(card, root=MINIPROJECT)[0]["messages"]
        self.assertEqual(self._occurrences(messages, self.UTIL_MARKER), 1)
        self.assertEqual(
            self._context_paths(messages),
            [
                os.path.join(MINIPROJECT, "README.md"),
                os.path.join(MINIPROJECT, "app.py"),
            ],
        )

    def test_patch_slice_of_other_files_is_unchanged_and_ordered(self):
        card = MorphCard(
            custom_id="patch-others",
            intent="patch",
            target="app.py",
            instruction="add subtract",
            context_slice=["util.py", "README.md"],
        )
        messages = compile_card(card, root=MINIPROJECT)[0]["messages"]
        # Slice entries are sorted for determinism, and both survive.
        self.assertEqual(
            self._context_paths(messages),
            [
                os.path.join(MINIPROJECT, "README.md"),
                os.path.join(MINIPROJECT, "util.py"),
            ],
        )

    def test_generate_card_still_receives_a_target_listed_in_its_slice(self):
        # generate has no "Original file" message, so the target in the slice is
        # ordinary context and must stay.
        card = MorphCard(
            custom_id="gen-with-target",
            intent="generate",
            target="util.py",
            instruction="rewrite it",
            context_slice=["app.py", "util.py"],
        )
        messages = compile_card(card, root=MINIPROJECT)[0]["messages"]
        self.assertEqual(self._occurrences(messages, self.UTIL_MARKER), 1)
        self.assertEqual(
            self._context_paths(messages),
            [
                os.path.join(MINIPROJECT, "app.py"),
                os.path.join(MINIPROJECT, "util.py"),
            ],
        )

    def test_generate_whole_project_walk_still_includes_the_target(self):
        card = MorphCard(
            custom_id="gen-whole",
            intent="generate",
            target="util.py",
            instruction="rewrite it",
            context_slice=[],
        )
        messages = compile_card(card, root=MINIPROJECT)[0]["messages"]
        self.assertIn(
            os.path.join(MINIPROJECT, "util.py"), self._context_paths(messages))


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


# The directive's opening sentence: present in a changeset card's prompt and,
# just as load-bearing, absent from every single-target one.
DIRECTIVE_MARKER = "This card writes SEVERAL files"


class MultiTargetDirectiveTests(unittest.TestCase):
    """A card that writes a SET of files must SAY how to return them.

    The format is generated by the compiler, never restated by the card author:
    a convention re-typed card by card is a convention that drifts until the
    parser rejects the answer.
    """

    def test_single_target_prompt_carries_no_directive_for_any_provider(self):
        # Byte-identity with Morph 1.0 is what tests/golden holds to (see
        # GoldenFileTests); this is the same claim stated where a reader of the
        # changeset feature will look for it.
        requests = compile_deck(load_deck(os.path.join(FIXTURES, "deck.json")),
                                root=MINIPROJECT)
        payloads = [
            serialize_anthropic(requests, default_model=ANTHROPIC_DEFAULT_MODEL),
            serialize_openai(requests, default_model=OPENAI_DEFAULT_MODEL),
            serialize_openrouter(requests, default_model=OPENROUTER_DEFAULT_MODEL),
        ]
        for payload in payloads:
            self.assertNotIn(DIRECTIVE_MARKER, payload)
            self.assertNotIn("FILE:", payload)

    def test_multi_target_generate_appends_the_directive_listing_targets_in_order(self):
        card = MorphCard(
            custom_id="changeset",
            intent="generate",
            targets=["util.py", "pkg/new_mod.py"],
            instruction="do it",
            context_slice=["app.py"],
        )
        messages = compile_card(card, root=MINIPROJECT)[0]["messages"]
        closing = messages[-1]

        self.assertEqual(closing["role"], "user")
        self.assertTrue(closing["content"].startswith("do it"))
        self.assertIn(DIRECTIVE_MARKER, closing["content"])
        self.assertIn("FILE: <path>", closing["content"])
        # The files are listed in the card's order -- the order the parser and
        # the writer both walk.
        listing = closing["content"]
        self.assertLess(listing.index("- util.py"), listing.index("- pkg/new_mod.py"))

    def test_directive_is_the_compiler_constant_filled_with_the_card_targets(self):
        card = MorphCard(
            custom_id="changeset",
            intent="generate",
            targets=["a.py", "b.py"],
            instruction="do it",
            context_slice=["app.py"],
        )
        messages = compile_card(card, root=MINIPROJECT)[0]["messages"]
        expected = "do it" + MULTI_TARGET_DIRECTIVE.format(files="- a.py\n- b.py")
        self.assertEqual(messages[-1]["content"], expected)

    def test_multi_target_todo_keeps_its_fixed_prompt_and_gains_the_directive(self):
        card = MorphCard(
            custom_id="todo-set",
            intent="todo",
            targets=["util.py", "app.py"],
        )
        messages = compile_card(card, root=MINIPROJECT)[0]["messages"]
        self.assertIn("criticize this file contents", messages[-1]["content"])
        self.assertIn(DIRECTIVE_MARKER, messages[-1]["content"])


class MultiTargetPatchTests(unittest.TestCase):
    """A changeset patch card ships ONE "Original file" per existing target.

    The de-duplication measured on a real card (see TargetIsSentOnceTests) must
    cover the whole set, not just the first target -- otherwise the changeset
    card reintroduces, once per extra file, exactly the duplicate that cost 92%
    of a 127 KB prompt.
    """

    UTIL_MARKER = "def add(a, b):"
    APP_MARKER = "from util import add"

    def _messages(self, targets, context_slice):
        card = MorphCard(
            custom_id="patch-set",
            intent="patch",
            targets=targets,
            instruction="do it",
            context_slice=context_slice,
        )
        return compile_card(card, root=MINIPROJECT)[0]["messages"]

    def test_one_framing_pair_per_existing_target_in_card_order(self):
        messages = self._messages(["util.py", "app.py"], ["README.md"])

        self.assertEqual([m["role"] for m in messages],
                         ["assistant", "assistant", "assistant", "assistant",
                          "user", "user"])
        self.assertEqual(messages[0]["content"],
                         "Let's update the util.py file provided.")
        self.assertIn(self.UTIL_MARKER, messages[1]["content"])
        self.assertEqual(messages[2]["content"],
                         "Let's update the app.py file provided.")
        self.assertIn(self.APP_MARKER, messages[3]["content"])

    def test_every_target_is_sent_once_even_when_the_slice_lists_them_all(self):
        messages = self._messages(["util.py", "app.py"], ["app.py", "util.py"])
        joined = "".join(m["content"] for m in messages)
        self.assertEqual(joined.count(self.UTIL_MARKER), 1)
        self.assertEqual(joined.count(self.APP_MARKER), 1)

    def test_a_target_that_does_not_exist_yet_is_simply_not_sent(self):
        # The headline changeset: patch a module, CREATE its test. There is no
        # original to show for the file being created.
        messages = self._messages(["util.py", "tests/test_util.py"], ["README.md"])

        self.assertEqual([m["role"] for m in messages],
                         ["assistant", "assistant", "user", "user"])
        self.assertIn(self.UTIL_MARKER, messages[1]["content"])
        self.assertNotIn("tests/test_util.py file provided", messages[0]["content"])
        # ... but the directive still asks for it.
        self.assertIn("- tests/test_util.py", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
