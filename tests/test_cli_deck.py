"""
Tests for ``cards.cli_deck``: the four provider-free headless handlers.

Every test runs against a ``tempfile.TemporaryDirectory`` used as the project
root. This project's own ``.morph/`` is never touched: each test's backlog and
run archive are created under the temporary root and die with it. No provider,
no network and no git repository is needed either -- the archived runs the
``report`` tests read are made through :func:`cards.store.archive_run` with a
branchless run state (or written as exactly the files
:func:`cards.store.list_runs` reads), which is local JSON and nothing else.

The handlers are called as functions and asserted on through their RETURN
values -- the property the handler layer exists for. Nothing here captures
stdout and nothing spawns a process; the one-document/one-code contract that
wraps these handlers lives in ``cards.cli_json`` and is not re-proved here.
"""

import json
import os
import tempfile
import unittest

from cards import cli_deck
from cards.cli_json import EXIT_OK, EXIT_REFUSED, EXIT_USAGE, CliError
from cards.hazards import KIND_WRITE_WRITE, SEVERITY_ERROR
from cards.store import DeckStore, archive_run


def _card(custom_id: str, target: str, **meta_extra) -> dict:
    """A minimal valid card dict (nested form) whose target is ``target``."""
    meta = {"intent": "patch", "target": target}
    meta.update(meta_extra)
    return {"custom_id": custom_id, "meta": meta,
            "instruction": f"Rewrite {target}."}


class TempRootCase(unittest.TestCase):
    """A test case whose project root is a fresh temporary directory."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

    # -- helpers -------------------------------------------------------------

    def write_fragment(self, name: str, payload) -> str:
        """Write ``payload`` as JSON into the root; return the file's path."""
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def assertSerialisable(self, payload, what: str) -> None:
        """Assert the payload survives json.dumps -- what emit() must accept."""
        try:
            json.dumps(payload)
        except (TypeError, ValueError) as error:
            self.fail(f"{what} must survive json.dumps, but json.dumps "
                      f"raised: {error}")


class DeckAddTests(TempRootCase):
    """``deck_add``: read a fragment, add it atomically, report hazards."""

    def test_json_array_adds_every_card_and_reports_their_ids(self):
        path = self.write_fragment("two.json", [
            # Explicit slices on purpose: an EMPTY context_slice reads the
            # whole project, and each card would then report an implicit-read
            # warning against its sibling -- real, but not what this test is
            # about.
            _card("card-a", "src/alpha.py", context_slice=["docs/brief.md"]),
            _card("card-b", "src/beta.py", context_slice=["docs/brief.md"]),
        ])
        payload = cli_deck.deck_add(self.root, path)
        self.assertEqual(payload["added"], ["card-a", "card-b"],
                         msg="expected the added custom_ids, in the order "
                             "given")
        backlog = [card.custom_id
                   for card in DeckStore(self.root).load_cards()]
        self.assertEqual(backlog, ["card-a", "card-b"],
                         msg="expected BOTH cards of the array in the saved "
                             "backlog")
        self.assertEqual(payload["hazards"], [],
                         msg="expected no hazards from two cards on distinct "
                             "files with explicit slices")
        self.assertSerialisable(payload, "the deck_add payload")

    def test_single_object_adds_one_card(self):
        path = self.write_fragment("one.json", _card("solo", "docs/plan.md"))
        payload = cli_deck.deck_add(self.root, path)
        self.assertEqual(payload["added"], ["solo"],
                         msg="expected a single card object to be wrapped and "
                             "added as one card")
        backlog = DeckStore(self.root).load_cards()
        self.assertEqual(len(backlog), 1,
                         msg="expected exactly one card in the backlog")
        self.assertEqual(backlog[0].custom_id, "solo",
                         msg="expected the one added card to be the one given")
        self.assertEqual(payload["hazards"], [],
                         msg="expected no hazards from a lone card, which has "
                             "no sibling to contend with")
        self.assertSerialisable(payload, "the deck_add payload")

    def test_write_write_conflict_reported_not_raised(self):
        path = self.write_fragment("clash.json", [
            _card("card-a", "src/shared.py"),
            _card("card-b", "src/shared.py"),
        ])
        # The call itself is the "without raising" assertion: a hazard must
        # come back in the payload, not as an exception.
        payload = cli_deck.deck_add(self.root, path)
        self.assertEqual(payload["added"], ["card-a", "card-b"],
                         msg="expected BOTH conflicting cards added: hazards "
                             "are reported here, never refused")
        self.assertEqual(len(payload["hazards"]), 1,
                         msg="expected exactly one hazard: two cards, one "
                             "shared target (a path the reader also writes is "
                             "excluded, so no second hazard)")
        hazard = payload["hazards"][0]
        for key in ("kind", "severity", "left", "right", "paths", "message"):
            self.assertIn(key, hazard,
                          msg=f"expected the hazard object to carry {key!r}")
        self.assertEqual(hazard["kind"], KIND_WRITE_WRITE,
                         msg="expected a write-write hazard for two cards "
                             "targeting the same file")
        self.assertEqual(hazard["severity"], SEVERITY_ERROR,
                         msg="expected a write-write hazard to carry severity "
                             "'error'")
        self.assertEqual({hazard["left"], hazard["right"]},
                         {"card-a", "card-b"},
                         msg="expected the hazard to name both conflicting "
                             "cards")
        self.assertEqual(list(hazard["paths"]), ["src/shared.py"],
                         msg="expected the contested file named as the cards "
                             "spell it")
        self.assertTrue(hazard["message"],
                        msg="expected a non-empty message on the hazard")
        self.assertSerialisable(payload, "the deck_add payload")

    def test_hazard_against_a_preexisting_card_is_reported(self):
        DeckStore(self.root).add_cards([_card("card-a", "src/shared.py")])
        path = self.write_fragment("late.json",
                                   [_card("card-late", "src/shared.py")])
        payload = cli_deck.deck_add(self.root, path)
        self.assertEqual(payload["added"], ["card-late"],
                         msg="expected only the new card in 'added'")
        self.assertEqual(len(payload["hazards"]), 1,
                         msg="expected the new card to be checked against the "
                             "WHOLE backlog, not just the fragment")
        self.assertEqual({payload["hazards"][0]["left"],
                          payload["hazards"][0]["right"]},
                         {"card-a", "card-late"},
                         msg="expected the hazard to name the new card and "
                             "the pre-existing one")
        self.assertSerialisable(payload, "the deck_add payload")

    def test_missing_file_raises_file_not_found(self):
        missing = os.path.join(self.root, "no-such-fragment.json")
        with self.assertRaises(
                FileNotFoundError,
                msg="expected FileNotFoundError from deck_add on a missing "
                    "file"
        ):
            cli_deck.deck_add(self.root, missing)

    def test_malformed_json_raises_a_decode_error(self):
        path = os.path.join(self.root, "broken.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"custom_id": "oops",,,}')
        with self.assertRaises(
                json.JSONDecodeError,
                msg="expected json.JSONDecodeError from deck_add on malformed "
                    "JSON"
        ):
            cli_deck.deck_add(self.root, path)

    def test_fragment_that_is_neither_array_nor_object_is_usage(self):
        path = self.write_fragment("scalar.json", 42)
        with self.assertRaises(
                CliError,
                msg="expected a CliError for a deck fragment that is neither "
                    "an array nor a card object"
        ) as caught:
            cli_deck.deck_add(self.root, path)
        self.assertEqual(caught.exception.code, EXIT_USAGE,
                         msg="expected exit code 4 (usage) for a scalar "
                             "fragment")

    def test_a_malformed_card_adds_nothing(self):
        path = self.write_fragment("mixed.json", [
            _card("good-one", "src/one.py"),
            _card("bad one", "src/two.py"),   # space: fails CUSTOM_ID_RE
        ])
        with self.assertRaises(
                ValueError,
                msg="expected the malformed card to raise CardError (a "
                    "ValueError)"
        ):
            cli_deck.deck_add(self.root, path)
        self.assertEqual(DeckStore(self.root).load_cards(), [],
                         msg="expected the atomic addition to have written "
                             "nothing: all cards or none")


class DeckCheckTests(TempRootCase):
    """``deck_check``: the counts, the hazard list, and the code as verdict."""

    def test_clean_deck_returns_exit_ok_with_zero_errors(self):
        # A lone card has no sibling to contend with, so the deck is clean.
        DeckStore(self.root).add_cards([_card("card-a", "src/alpha.py")])
        payload, code = cli_deck.deck_check(self.root)
        self.assertEqual(code, EXIT_OK,
                         msg="expected exit code 0 for a deck with no error "
                             "hazards")
        self.assertEqual(payload["errors"], 0,
                         msg="expected errors == 0 on a clean deck")
        self.assertEqual(payload["warnings"], 0,
                         msg="expected warnings == 0 on a clean deck")
        self.assertEqual(payload["cards"], 1,
                         msg="expected 'cards' to be the backlog size")
        self.assertEqual(payload["hazards"], [],
                         msg="expected an empty hazards list on a clean deck")
        self.assertSerialisable(payload, "the deck_check payload")

    def test_write_write_conflict_returns_exit_refused_with_hazards(self):
        DeckStore(self.root).add_cards([
            _card("card-a", "src/shared.py"),
            _card("card-b", "src/shared.py"),
        ])
        payload, code = cli_deck.deck_check(self.root)
        self.assertEqual(code, EXIT_REFUSED,
                         msg="expected exit code 2 when an error hazard "
                             "exists: a check that cannot fail a script is "
                             "not a check")
        self.assertTrue(payload["hazards"],
                        msg="expected a non-empty hazards list for the "
                            "conflicting deck")
        self.assertEqual(len(payload["hazards"]), 1,
                         msg="expected exactly one hazard for the one shared "
                             "file")
        self.assertEqual(payload["errors"], 1,
                         msg="expected the one write-write error to be "
                             "counted")
        self.assertEqual(payload["hazards"][0]["kind"], KIND_WRITE_WRITE,
                         msg="expected the listed hazard to be the write-write "
                             "one")
        self.assertSerialisable(payload, "the deck_check payload")

    def test_warnings_alone_do_not_change_the_exit_code(self):
        # The reader's EMPTY context_slice reads the whole project, including
        # the writer's target: one implicit-read WARNING, no errors.
        DeckStore(self.root).add_cards([
            _card("card-w", "src/written.py",
                  context_slice=["docs/brief.md"]),
            _card("card-r", "src/reader.py"),
        ])
        payload, code = cli_deck.deck_check(self.root)
        self.assertEqual(payload["warnings"], 1,
                         msg="expected the implicit-read hazard counted as a "
                             "warning")
        self.assertEqual(payload["errors"], 0,
                         msg="expected no error hazard from an implicit read")
        self.assertEqual(code, EXIT_OK,
                         msg="expected exit code 0: warnings alone must not "
                             "change the code")
        self.assertSerialisable(payload, "the deck_check payload")


class DeckStatusTests(TempRootCase):
    """``deck_status``: the /deck view, straight through cli_views."""

    def test_empty_root_reports_empty_true_and_the_documented_keys(self):
        payload = cli_deck.deck_status(self.root)
        self.assertEqual(
            set(payload),
            {"empty", "phase", "current_generation", "generations", "cards"},
            msg="expected exactly the documented deck_status keys")
        self.assertIs(payload["empty"], True,
                      msg="expected empty=true for a root with no backlog")
        self.assertEqual(payload["phase"], "idle",
                         msg="expected phase 'idle' before any run started")
        self.assertEqual(payload["current_generation"], 0,
                         msg="expected generation index 0 for a fresh root")
        self.assertEqual(payload["cards"], [],
                         msg="expected no cards listed for an empty root")
        self.assertEqual(payload["generations"], [],
                         msg="expected no generations previewed for an empty "
                             "root")
        self.assertSerialisable(payload, "the deck_status payload")

    def test_after_adding_cards_every_card_is_listed_with_a_status(self):
        DeckStore(self.root).add_cards([
            _card("card-a", "src/alpha.py"),
            _card("card-b", "src/beta.py"),
        ])
        payload = cli_deck.deck_status(self.root)
        self.assertIs(payload["empty"], False,
                      msg="expected empty=false once the backlog holds cards")
        listed = [(entry["custom_id"], entry["status"])
                  for entry in payload["cards"]]
        self.assertEqual(listed,
                         [("card-a", "pending"), ("card-b", "pending")],
                         msg="expected every backlog card listed in deck "
                             "order, each with status 'pending'")
        covered = {cid for generation in payload["generations"]
                   for cid in generation}
        self.assertEqual(covered, {"card-a", "card-b"},
                         msg="expected the generation preview to cover every "
                             "backlog card")
        self.assertSerialisable(payload, "the deck_status payload")


class ReportTests(TempRootCase):
    """``report``: one archived run -- latest or by id -- refusing what is not."""

    def _archive_run(self, deck_id: str) -> None:
        """Archive one branchless run under ``deck_id`` via archive_run.

        The run state's ``branch`` is None, so
        :func:`cards.store.archive_run` writes the two archive files and
        attempts no commit: no git repository is needed for these tests.
        """
        store = DeckStore(self.root)
        cards = store.load_cards()
        state = {
            "deck_id": deck_id,
            "branch": None,
            "backend_label": None,
            "generations": [[card.custom_id for card in cards]],
            "batch_ids": [],
            "outcomes": {},
        }
        directory = archive_run(store, state, cards, root=self.root,
                                log=lambda line: None)
        self.assertIsNotNone(directory,
                             msg="expected the run to be archived (the "
                                 "directory did not already exist)")

    def test_no_archived_runs_raises_usage_clierror(self):
        with self.assertRaises(
                CliError,
                msg="expected a CliError when no run is archived at all"
        ) as caught:
            cli_deck.report(self.root)
        self.assertEqual(caught.exception.code, EXIT_USAGE,
                         msg="expected exit code 4 when nothing is archived")
        self.assertEqual(caught.exception.kind, "UsageError",
                         msg="expected kind 'UsageError' on the refusal")

    def test_finds_a_run_archived_under_a_given_deck_id(self):
        deck_id = "20260101-090000-aaaa1111"
        store = DeckStore(self.root)
        store.add_cards([_card("card-a", "src/alpha.py")])
        self._archive_run(deck_id)
        payload = cli_deck.report(self.root, deck_id)
        self.assertEqual(payload["deck_id"], deck_id,
                         msg="expected the report of the requested deck id")
        self.assertEqual(payload["generations"], [["card-a"]],
                         msg="expected the archived run's composition")
        self.assertEqual(payload["counts"],
                         {"written": 0, "failed": 0, "skipped": 0},
                         msg="expected the counts of a run with no recorded "
                             "outcomes, under the key report_to_dict adds")
        self.assertSerialisable(payload, "the report payload")

    def test_unknown_deck_id_names_the_archived_ids(self):
        archived_id = "20260101-090000-aaaa1111"
        self._archive_run(archived_id)
        with self.assertRaises(
                CliError,
                msg="expected a CliError for a deck_id the archive does not "
                    "hold"
        ) as caught:
            cli_deck.report(self.root, "20260202-090000-missing0")
        self.assertEqual(caught.exception.code, EXIT_USAGE,
                         msg="expected exit code 4 for an unknown deck id")
        self.assertIn(archived_id, caught.exception.message,
                      msg="expected the message to NAME the archived run id, "
                          "not just refuse")
        self.assertIn("missing0", caught.exception.message,
                      msg="expected the message to echo the id that was asked "
                          "for")

    def test_without_deck_id_returns_the_latest_archived_run(self):
        # Written the way list_runs reads them, with controlled completed_at
        # stamps, so "latest" is decided by the test and not by the clock.
        older_id = "20260101-090000-older001"
        newer_id = "20260102-090000-newer001"
        for deck_id, completed_at in (
                (older_id, "2026-01-01T09:00:00.000"),
                (newer_id, "2026-01-02T09:00:00.000"),
        ):
            run_dir = os.path.join(self.root, ".morph", "runs", deck_id)
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, "report.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"deck_id": deck_id,
                           "completed_at": completed_at}, handle)
        payload = cli_deck.report(self.root)
        self.assertEqual(payload["deck_id"], newer_id,
                         msg="expected the NEWEST archived run: list_runs "
                             "sorts by (completed_at, deck_id) descending, so "
                             "index 0 is the latest")
        self.assertSerialisable(payload, "the report payload")


if __name__ == "__main__":
    unittest.main()
