"""Tests for ``cards.cli_json``: the headless CLI's one-document contract.

The guarantee under test is absolute: whatever a handler does -- return a
payload, return a ``(payload, exit_code)`` pair, or raise anything at all --
the stream receives exactly one parseable JSON document and the caller
receives one exit code. So every test here checks parseability first (a
document a script cannot ``json.loads`` is exactly the failure the module
exists to abolish) and shape second, and the classification tests raise REAL
``cards`` exceptions, because the classify table is only as good as its
ordering against the real subclass graph: ``HazardError`` subclasses
``DeckError``, and ``json.JSONDecodeError`` subclasses ``ValueError``.
"""

import io
import json
import os
import sys
import unittest

# Make the project root importable however this file is run -- ``python -m
# unittest`` from the root, ``unittest discover``, or a runner that puts only
# this directory on the path: the package under test lives one level up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cards import cli_json
from cards.cli_json import (
    EXIT_INCOMPLETE,
    EXIT_OK,
    EXIT_REFUSED,
    EXIT_TRANSPORT,
    EXIT_USAGE,
    CliError,
    classify,
    emit,
    error_document,
    run_cli,
)
from cards.deck import DeckError
from cards.hazards import HazardError
from cards.schema import CardError
from cards.store import StoreError


class _FlushingStringIO(io.StringIO):
    """A ``StringIO`` that counts ``flush()`` calls, so a test can see one."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flush_calls = 0

    def flush(self):
        self.flush_calls += 1
        super().flush()


class ExitCodeConstantsTest(unittest.TestCase):
    """The five exit codes, with the values the module docstring promises."""

    def test_exit_codes_have_their_documented_values(self):
        """The vocabulary a script branches on must not drift undocumented."""
        self.assertEqual(cli_json.EXIT_OK, 0,
                         "EXIT_OK must be 0: done, every card accepted")
        self.assertEqual(cli_json.EXIT_INCOMPLETE, 1,
                         "EXIT_INCOMPLETE must be 1: finished but some cards "
                         "failed or were skipped")
        self.assertEqual(cli_json.EXIT_REFUSED, 2,
                         "EXIT_REFUSED must be 2: refused before spending "
                         "money")
        self.assertEqual(cli_json.EXIT_TRANSPORT, 3,
                         "EXIT_TRANSPORT must be 3: transport/provider "
                         "failure")
        self.assertEqual(cli_json.EXIT_USAGE, 4,
                         "EXIT_USAGE must be 4: usage error")


class EmitTest(unittest.TestCase):
    """emit: one document, one trailing newline, flushed, always parseable."""

    def test_compact_emit_is_a_single_parseable_line(self):
        """Compact is the form a script reads: one line, one newline, round-trips."""
        stream = _FlushingStringIO()
        payload = {"ok": True, "generation": 2, "cards": ["a", "b"]}
        emit(payload, stream)
        text = stream.getvalue()
        self.assertTrue(text.endswith("\n"),
                        "the document must end with exactly one newline")
        body = text[:-1]
        self.assertNotIn("\n", body,
                         "a compact document must be a single line")
        self.assertEqual(json.loads(body), payload,
                         "the compact document must parse back to the payload")
        self.assertEqual(stream.flush_calls, 1,
                         "emit must flush the stream before returning")

    def test_pretty_emit_is_indented_and_parseable(self):
        """Pretty is the same document for a human: indented, still round-trips."""
        stream = io.StringIO()
        payload = {"ok": True,
                   "cards": [{"custom_id": "a", "status": "written"}]}
        emit(payload, stream, pretty=True)
        text = stream.getvalue()
        self.assertTrue(text.endswith("\n"),
                        "the pretty document must end with exactly one newline")
        self.assertIn('\n  "', text,
                      "pretty must use indent=2 (two-space indentation)")
        self.assertEqual(json.loads(text), payload,
                         "the pretty document must parse back to the payload")

    def test_emit_degrades_an_unserialisable_value_to_its_string_form(self):
        """default=str is what keeps one odd value from destroying the contract."""
        stream = io.StringIO()
        emit({"detail": object()}, stream)
        document = json.loads(stream.getvalue())
        self.assertIsInstance(document["detail"], str,
                              "default=str must degrade an unexpected "
                              "non-serialisable value to its string form "
                              "rather than raising")


class ErrorDocumentTest(unittest.TestCase):
    """error_document: exactly the documented shape, and nothing else."""

    def test_error_document_is_exactly_error_code_kind_message(self):
        """The whole-dict comparison pins the shape: no extra keys at any depth."""
        document = error_document(EXIT_REFUSED, "StoreError",
                                  "refused before submit")
        self.assertEqual(
            document,
            {"error": {"code": EXIT_REFUSED, "kind": "StoreError",
                       "message": "refused before submit"}},
            "error_document must return exactly "
            "{'error': {'code', 'kind', 'message'}} and nothing else")


class ClassifyTest(unittest.TestCase):
    """classify: the exit-code table, in the order the subclass graph forces."""

    def test_the_refused_family_classifies_to_exit_2(self):
        """The deck/store family is refused BEFORE money is spent: exit 2."""
        for exc in (StoreError("a generation is already submitted"),
                    HazardError("two cards of one generation write one file"),
                    DeckError("duplicate custom_id in deck")):
            cli_error = classify(exc)
            self.assertEqual(
                cli_error.code, EXIT_REFUSED,
                f"{type(exc).__name__} must classify as refused (exit "
                f"{EXIT_REFUSED}), got {cli_error.code}")

    def test_the_usage_family_classifies_to_exit_4(self):
        """Malformed input from the caller is usage: exit 4."""
        for exc in (CardError("card 'x': intent is required"),
                    json.JSONDecodeError("Expecting value", "not json", 0),
                    FileNotFoundError("no such file: deck.json"),
                    IsADirectoryError("expected a file, found a directory"),
                    NotADirectoryError("a path component is a file"),
                    ValueError("bad value"),
                    TypeError("wrong type"),
                    KeyError("missing key")):
            cli_error = classify(exc)
            self.assertEqual(
                cli_error.code, EXIT_USAGE,
                f"{type(exc).__name__} must classify as usage (exit "
                f"{EXIT_USAGE}), got {cli_error.code}")

    def test_oserror_classifies_to_exit_3(self):
        """A failing machine or stream is transport, not the caller's fault."""
        cli_error = classify(OSError("connection reset by peer"))
        self.assertEqual(cli_error.code, EXIT_TRANSPORT,
                         "OSError must classify as transport (exit "
                         f"{EXIT_TRANSPORT}), got {cli_error.code}")

    def test_an_untabled_exception_classifies_to_exit_3(self):
        """The catch-all: transport, kind is the class name, message str(exc)."""
        cli_error = classify(RuntimeError("the provider is down"))
        self.assertEqual(cli_error.code, EXIT_TRANSPORT,
                         "an exception no table row names must classify as "
                         "transport -- the honest default for what we cannot "
                         "classify")
        self.assertEqual(cli_error.kind, "RuntimeError",
                         "the catch-all kind must be the exception's class "
                         "name")
        self.assertEqual(cli_error.message, "the provider is down",
                         "the message must be str(exc)")

    def test_hazarderror_is_tested_before_its_parent_deckerror(self):
        """Order IS the mapping: a hazard must not report as a mere deck error."""
        cli_error = classify(HazardError("read/write hazard"))
        self.assertEqual(cli_error.kind, "HazardError",
                         "HazardError must carry its own kind, which only "
                         "holds if the table tests it before DeckError")

    def test_jsondecodeerror_is_tested_before_valueerror(self):
        """Order IS the mapping: bad JSON must not report as a mere bad value."""
        cli_error = classify(json.JSONDecodeError("Expecting value", "", 0))
        self.assertEqual(cli_error.kind, "JSONDecodeError",
                         "JSONDecodeError must carry its own kind, which only "
                         "holds if the table tests it before ValueError")

    def test_a_cli_error_passes_through_unchanged(self):
        """Classification is idempotent: an already-classified error stands."""
        original = CliError(EXIT_USAGE, "UsageError", "unknown processor")
        self.assertIs(classify(original), original,
                      "a CliError must be returned unchanged, not "
                      "re-classified")

    def test_an_empty_message_falls_back_to_the_class_name(self):
        """A document whose message is an empty string says nothing at all."""
        cli_error = classify(RuntimeError())
        self.assertEqual(cli_error.code, EXIT_TRANSPORT,
                         "a bare RuntimeError must classify as transport (3)")
        self.assertEqual(cli_error.message, "RuntimeError",
                         "an empty str(exc) must fall back to the class name "
                         "so the document is never an empty message")


class RunCliTest(unittest.TestCase):
    """run_cli: the wrapper that carries the whole guarantee."""

    def test_a_dict_payload_is_emitted_and_exits_0(self):
        """A plain dict payload is the one document, and the code is 0."""
        stream = io.StringIO()
        payload = {"ok": True, "written": 3}
        code = run_cli(lambda: payload, stream)
        self.assertEqual(code, EXIT_OK,
                         "a handler returning a plain dict must exit 0")
        self.assertEqual(json.loads(stream.getvalue()), payload,
                         "the handler's payload must be the one document on "
                         "the stream")

    def test_a_payload_code_pair_returns_the_handlers_code(self):
        """The pair form lets a handler report incompleteness with its payload."""
        stream = io.StringIO()
        payload = {"written": 2, "failed": 1, "skipped": 1}
        code = run_cli(lambda: (payload, EXIT_INCOMPLETE), stream)
        self.assertEqual(code, EXIT_INCOMPLETE,
                         "run_cli must return the exit code the handler "
                         "paired with its payload")
        self.assertEqual(json.loads(stream.getvalue()), payload,
                         "the paired payload must be the one document on the "
                         "stream")

    def test_a_store_error_becomes_a_parseable_refused_document(self):
        """A real StoreError must exit 2 with exactly the error document."""
        stream = io.StringIO()

        def handler():
            raise StoreError("a generation is already submitted")

        code = run_cli(handler, stream)
        self.assertEqual(code, EXIT_REFUSED,
                         "a StoreError must exit refused (2)")
        # The parse itself is the parseability assertion: an unparseable
        # stream fails HERE, in the test, not in the caller's script.
        document = json.loads(stream.getvalue())
        self.assertEqual(
            document,
            error_document(EXIT_REFUSED, "StoreError",
                           "a generation is already submitted"),
            "the failure must be exactly the documented error document and "
            "the only thing on the stream")

    def test_a_plain_exception_becomes_a_parseable_transport_document(self):
        """An untabled exception must exit 3, classified, still one document."""
        stream = io.StringIO()

        def handler():
            raise Exception("the provider is down")

        code = run_cli(handler, stream)
        self.assertEqual(code, EXIT_TRANSPORT,
                         "an exception no table row names must exit "
                         "transport (3)")
        document = json.loads(stream.getvalue())
        self.assertEqual(document["error"]["kind"], "Exception",
                         "the catch-all kind must be the exception's class "
                         "name")
        self.assertEqual(document["error"]["message"], "the provider is down",
                         "the document must carry the exception's message")

    def test_keyboard_interrupt_is_reraised(self):
        """Ctrl-C is the process's, not the command's: re-raise, write nothing."""
        stream = io.StringIO()

        def handler():
            raise KeyboardInterrupt()

        try:
            run_cli(handler, stream)
        except KeyboardInterrupt:
            pass
        else:
            self.fail("run_cli must re-raise KeyboardInterrupt, not swallow "
                      "it into an exit code")
        self.assertEqual(stream.getvalue(), "",
                         "an interrupted run must leave nothing on the stream")

    def test_system_exit_is_reraised(self):
        """A sys.exit in flight belongs to its caller: re-raise, write nothing."""
        stream = io.StringIO()

        def handler():
            raise SystemExit(2)

        try:
            run_cli(handler, stream)
        except SystemExit:
            pass
        else:
            self.fail("run_cli must re-raise SystemExit, not swallow it into "
                      "an exit code")
        self.assertEqual(stream.getvalue(), "",
                         "a sys.exit in flight must leave nothing on the "
                         "stream")

    def test_pretty_passes_through_to_emit(self):
        """The pretty flag changes the spelling, never the code."""
        stream = io.StringIO()
        code = run_cli(lambda: {"ok": True}, stream, pretty=True)
        self.assertEqual(code, EXIT_OK,
                         "pretty must not change the exit code")
        self.assertIn('\n  "', stream.getvalue(),
                      "pretty must pass through to emit (indent=2)")


if __name__ == "__main__":
    unittest.main()
