"""
End-to-end tests for the headless entry point, :func:`cards.cli.main`.

Every test drives the CLI the way its caller does -- one ``main([...])``
call in this process -- and never through a subprocess: a subprocess would
couple the test to the interpreter, the package layout and the shell at
once, and a failure would not say which of the three broke. stdout and
stderr are captured with :class:`io.StringIO` over
:func:`contextlib.redirect_stdout` / :func:`contextlib.redirect_stderr`, and
every invocation gets a fresh :mod:`tempfile` directory as ``--root``, so no
test reads or writes this project's own ``.morph/``.

What the tests pin down: the one-document contract (exactly one parseable
JSON document on stdout, for a success and for EVERY failure), the exit-code
table (0 done, 2 refused, 4 usage), the payload shapes the handlers return,
the routing of the store's human log to stderr, and the rule that makes the
headless path headless -- ``flows.morph``, the console bot, never loads
behind ``main``.

The provider-facing commands are never allowed near a network: the tests
patch ``cards.cli_run.resolve_backend`` -- the one seam through which
``submit`` and ``collect`` obtain a backend -- and substitute a stub that
answers inside the process. That is the same bargain the handlers offer
(an injected backend is their documented test seam); ``main`` simply does
not expose a flag for it, so the patch stands in for the flag.
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from cards.cli import main


class _StubBackend:
    """A batch backend that answers instantly and never leaves the process.

    Shaped for the duck type the store expects (``submit``/``status``/
    ``collect``); ``submit`` answers a batch id, the value the submit path
    records. The collect test never reaches it -- the store refuses a run
    with nothing in flight first -- and the submit test only needs the
    submission to be accepted and logged.
    """

    def submit(self, *args, **kwargs):
        return "batch-stub-0001"

    def status(self, *args, **kwargs):
        return "completed"

    def collect(self, *args, **kwargs):
        return []


def _card(custom_id: str, target: str, instruction: str) -> dict:
    """A minimal valid generate card -- ``MorphCard.from_dict`` accepts it."""
    return {
        "custom_id": custom_id,
        "intent": "generate",
        "target": target,
        "instruction": instruction,
    }


class TestMain(unittest.TestCase):
    """The headless contract, exercised one ``main([...])`` call at a time."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

    # -- helpers -------------------------------------------------------------

    def _run(self, argv):
        """Drive ``main`` with both streams captured.

        Returns ``(code, out, err)``: the exit code and the two captured
        strings. The redirections are per call, so each invocation sees
        exactly what its process would have.
        """
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def _document(self, out: str, what: str) -> dict:
        """Parse stdout as the ONE JSON document, failing with its content.

        Every caller of :meth:`_run` wants this check; failing here names
        the command under test and shows the offending stream, which a bare
        ``json.loads`` traceback would not.
        """
        try:
            return json.loads(out)
        except ValueError as exc:
            self.fail(f"{what}: stdout was not one parseable JSON document "
                      f"({exc}); stdout was {out!r}")

    def _write_deck_file(self, name: str, cards: list) -> str:
        """Write ``cards`` as a JSON array under the temp root; return path."""
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(cards, handle)
        return path

    # -- the one-document contract -------------------------------------------

    def test_deck_status_on_an_empty_project_is_one_json_line(self):
        code, out, _err = self._run(["deck", "status", "--root", self.root])
        self.assertEqual(code, 0,
                         "deck status on an empty project must exit 0")
        self.assertEqual(out.count("\n"), 1,
                         "compact output must be exactly one line on stdout")
        payload = self._document(out, "deck status")
        self.assertIsInstance(payload, dict,
                              "the status payload must be a JSON object")

    def test_deck_status_pretty_output_still_parses(self):
        code, out, _err = self._run(
            ["deck", "status", "--pretty", "--root", self.root])
        self.assertEqual(code, 0,
                         "--pretty must not change the command's verdict")
        self.assertIn("\n", out,
                      "--pretty output must be indented over several lines")
        payload = self._document(out, "deck status --pretty")
        self.assertIsInstance(payload, dict,
                              "--pretty output must parse to the same shape")

    def test_no_subcommand_is_a_usage_error_document(self):
        code, out, _err = self._run([])
        self.assertEqual(code, 4,
                         "no subcommand at all must exit 4 (usage)")
        payload = self._document(out, "no subcommand")
        self.assertEqual(payload["error"]["code"], 4,
                         "the error document must carry the usage code")
        self.assertEqual(payload["error"]["kind"], "UsageError",
                         "a missing subcommand must be reported as UsageError")
        self.assertTrue(payload["error"]["message"],
                        "the error document must say what was wrong")

    def test_unknown_subcommand_is_a_usage_error_document(self):
        code, out, _err = self._run(["frobnicate"])
        self.assertEqual(code, 4,
                         "an unknown subcommand must exit 4 (usage)")
        payload = self._document(out, "unknown subcommand")
        self.assertEqual(payload["error"]["kind"], "UsageError",
                         "an unknown subcommand is a parse failure")
        self.assertIn("frobnicate", payload["error"]["message"],
                      "the diagnosis must name the command that was refused")

    def test_deck_add_with_a_missing_file_names_the_path(self):
        missing = os.path.join(self.root, "absent.json")
        code, out, _err = self._run(
            ["deck", "add", "--file", missing, "--root", self.root])
        self.assertEqual(code, 4,
                         "a missing --file path must exit 4 (usage)")
        payload = self._document(out, "deck add with a missing file")
        self.assertEqual(payload["error"]["kind"], "FileNotFoundError",
                         "a missing file must be reported as "
                         "FileNotFoundError")
        self.assertIn(missing, payload["error"]["message"],
                      "the error document must name the unreadable path")

    def test_deck_add_reports_the_ids_it_added(self):
        path = self._write_deck_file("fragment.json", [
            _card("write-utils", "utils.py",
                  "Create utils.py with the shared helpers."),
            _card("use-utils", "main.py",
                  "Create main.py importing the helpers from utils.py."),
        ])
        code, out, _err = self._run(
            ["deck", "add", "--file", path, "--root", self.root])
        self.assertEqual(code, 0, "adding a valid fragment must exit 0")
        payload = self._document(out, "deck add")
        self.assertEqual(payload["added"], ["write-utils", "use-utils"],
                         "the payload must list the added ids in file order")

    def test_deck_check_refuses_two_cards_on_one_file(self):
        path = self._write_deck_file("conflict.json", [
            _card("first-writer", "shared.py",
                  "Create shared.py with the first version."),
            _card("second-writer", "shared.py",
                  "Rewrite shared.py with the second version."),
        ])
        code, out, _err = self._run(
            ["deck", "add", "--file", path, "--root", self.root])
        self.assertEqual(code, 0,
                         "adding a contested fragment reports, never refuses")
        code, out, _err = self._run(["deck", "check", "--root", self.root])
        self.assertEqual(code, 2,
                         "two cards on one file must fail the check (exit 2)")
        payload = self._document(out, "deck check")
        self.assertGreaterEqual(payload["errors"], 1,
                                "the conflict must be counted as an error")
        self.assertTrue(payload["hazards"],
                        "the payload must list the hazards themselves")

    # -- the provider-facing steps, against a stub backend --------------------

    def test_collect_with_nothing_in_flight_is_refused(self):
        with mock.patch("cards.cli_run.resolve_backend",
                        return_value=(_StubBackend(), "stub")):
            code, out, _err = self._run(["collect", "--root", self.root])
        self.assertEqual(code, 2,
                         "collect with nothing in flight must be refused (2)")
        payload = self._document(out, "collect with nothing in flight")
        self.assertEqual(payload["error"]["kind"], "StoreError",
                         "nothing in flight must be the store's own refusal")

    def test_store_log_lands_on_stderr_and_stdout_stays_one_document(self):
        path = self._write_deck_file("one-card.json", [
            _card("make-greeting", "greeting.py",
                  "Create greeting.py exposing hello()."),
        ])
        code, out, _err = self._run(
            ["deck", "add", "--file", path, "--root", self.root])
        self.assertEqual(code, 0, "the fixture deck must add cleanly")
        with mock.patch("cards.cli_run.resolve_backend",
                        return_value=(_StubBackend(), "stub")):
            code, out, err = self._run(
                ["submit", "--root", self.root, "--nogit"])
        self.assertTrue(err.strip(),
                        "the store's progress log must reach stderr, "
                        "not stdout")
        self.assertEqual(out.count("\n"), 1,
                         "stdout must stay exactly one JSON document")
        self._document(out, "submit with a stub backend")

    # -- the headless rule -----------------------------------------------------

    def test_headless_path_never_imports_the_console_bot(self):
        sys.modules.pop("flows.morph", None)
        code, out, _err = self._run(["deck", "status", "--root", self.root])
        self.assertEqual(code, 0, "the probe command must itself succeed")
        self.assertNotIn("flows.morph", sys.modules,
                         "main must complete without loading the console bot")

    def test_report_with_no_archived_run_is_a_usage_error(self):
        code, out, _err = self._run(["report", "--root", self.root])
        self.assertEqual(code, 4,
                         "report with an empty archive must exit 4 (usage)")
        payload = self._document(out, "report with no archived runs")
        self.assertEqual(payload["error"]["kind"], "UsageError",
                         "asking for a run that does not exist is usage")


if __name__ == "__main__":
    unittest.main()
