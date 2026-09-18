"""Tests for ``cards.cli_views``: the JSON shapes a headless ``mrph`` prints.

Each shape is a contract with a script that reads the output, so these tests
pin exact key sets and values: a key added, renamed or dropped must fail here
rather than silently change what a consumer reads. The objects serialised are
built real -- the dataclasses the run itself produces, imported from
``cards.store`` / ``cards.hazards`` / ``cards.generations`` -- because a
serialiser tested against a mock proves only that it serialises the mock.

The one delegation is tested AS delegation: ``outcome_to_dict`` must return
exactly what ``cards.store._outcome_to_dict`` returns for the same outcome.
That is the anti-duplication guarantee in executable form -- if the store's
shape ever grows a field, this module inherits it by construction, or this
test fails.
"""

import json
import unittest

from cards import cli_views
from cards.generations import CardOutcome, REASON_COMPILE
from cards.hazards import (
    KIND_READ_WRITE,
    KIND_UNORDERED_READ,
    KIND_WRITE_WRITE,
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    Hazard,
)
from cards.store import (
    PHASE_IDLE,
    PHASE_SUBMITTED,
    STATUS_FAILED,
    STATUS_IN_FLIGHT,
    STATUS_PENDING,
    STATUS_SKIPPED,
    STATUS_WRITTEN,
    CollectResult,
    DeckStatusView,
    RunReport,
    SubmitResult,
    _outcome_to_dict,
)


# -- fixtures: the real objects the run produces -----------------------------


def _written_outcome() -> CardOutcome:
    """A settled, accepted card, as a committing run records it."""
    return CardOutcome(
        custom_id="card-a",
        status=STATUS_WRITTEN,
        paths=["src/one.py"],
        reason=None,
        attempts=1,
        winning_variant="card-a#2",
        acceptance_output="3 passed",
        commit="0123456789abcdef0123456789abcdef01234567",
        diffstat=[" src/one.py | 4 ++--"],
    )


def _failed_outcome() -> CardOutcome:
    """A card that never passed acceptance, as a run records it."""
    return CardOutcome(
        custom_id="card-b",
        status=STATUS_FAILED,
        paths=[],
        reason=REASON_COMPILE,
        attempts=2,
        winning_variant=None,
        acceptance_output="CompileError: slice names src/two.py, which does not exist",
        commit=None,
        diffstat=None,
    )


def _write_write_hazard() -> Hazard:
    """Two cards of one generation writing one file: an ERROR."""
    return Hazard(
        kind=KIND_WRITE_WRITE,
        severity=SEVERITY_ERROR,
        left="card-a",
        right="card-b",
        paths=("src/shared.py",),
        generation=1,
    )


def _read_write_hazard() -> Hazard:
    """A card reading a file a sibling of its generation writes: an ERROR."""
    return Hazard(
        kind=KIND_READ_WRITE,
        severity=SEVERITY_ERROR,
        left="card-a",
        right="card-d",
        paths=("src/one.py",),
        generation=1,
    )


def _unordered_read_hazard() -> Hazard:
    """A cross-generation read with no connecting edge: a WARNING, no generation."""
    return Hazard(
        kind=KIND_UNORDERED_READ,
        severity=SEVERITY_WARNING,
        left="card-a",
        right="card-c",
        paths=("./src/one.py",),
        generation=None,
    )


def _status_view() -> DeckStatusView:
    """A mid-run snapshot: one card written, one in flight, one pending."""
    return DeckStatusView(
        empty=False,
        phase=PHASE_SUBMITTED,
        current_generation=1,
        generations=[["card-a"], ["card-b", "card-c"]],
        card_status=[
            ("card-a", STATUS_WRITTEN),
            ("card-b", STATUS_IN_FLIGHT),
            ("card-c", STATUS_PENDING),
        ],
    )


def _submit_result() -> SubmitResult:
    """A submit that sent generation 2 and skipped one blocked card."""
    return SubmitResult(
        submitted=True,
        done=False,
        generation_number=2,
        total_generations=3,
        batch_id="batch-1",
        card_ids=["card-b"],
        skipped=[("card-c", "card-a")],
    )


def _collect_result() -> CollectResult:
    """A collect that judged one card and submitted a regeneration for another."""
    return CollectResult(
        in_progress=True,
        generation_number=1,
        total_generations=2,
        phase=PHASE_SUBMITTED,
        outcomes={"card-a": _written_outcome()},
        retry_in_flight=True,
        retry_submitted=True,
        retry_attempt=1,
        retry_limit=2,
        retry_card_ids=["card-b"],
        retry_batch_id="batch-retry-1",
    )


def _report() -> RunReport:
    """A finished run: one card written, one failed."""
    return RunReport(
        deck_id="20260917-114233-9f1c4b02",
        completed_at="2026-09-17T11:42:33.123000",
        branch="morph/20260917-114233-9f1c4b02",
        backend_label="openrouter",
        generations=[["card-a"], ["card-b"]],
        batch_ids=["batch-1", "batch-retry-1"],
        outcomes={"card-a": _written_outcome(), "card-b": _failed_outcome()},
    )


# -- hazards -----------------------------------------------------------------


class HazardSerialisationTests(unittest.TestCase):
    """``hazard_to_dict`` / ``hazards_to_list``: the hazard report's shape."""

    def test_hazard_to_dict_carries_exactly_the_agreed_keys(self):
        actual = cli_views.hazard_to_dict(_write_write_hazard())
        self.assertEqual(
            set(actual),
            {"kind", "severity", "left", "right", "paths", "generation",
             "message"},
            "a printed hazard must carry exactly the seven agreed keys")

    def test_hazard_to_dict_carries_the_hazard_verbatim(self):
        hazard = _write_write_hazard()
        actual = cli_views.hazard_to_dict(hazard)
        self.assertEqual(actual["kind"], KIND_WRITE_WRITE,
                         "kind must pass through unchanged")
        self.assertEqual(actual["severity"], SEVERITY_ERROR,
                         "severity must pass through unchanged")
        self.assertEqual(actual["left"], "card-a",
                         "left must be the hazard's writer")
        self.assertEqual(actual["right"], "card-b",
                         "right must be the hazard's reader/second writer")
        self.assertEqual(actual["paths"], ["src/shared.py"],
                         "paths must be the hazard's tuple as a list")
        self.assertEqual(actual["generation"], 1,
                         "generation must be the pair's shared generation")
        self.assertEqual(actual["message"], hazard.message(),
                         "message must be hazard.message(), verbatim")
        self.assertTrue(actual["message"],
                        "message must be a non-empty diagnosis a script can "
                        "surface as-is")

    def test_hazard_paths_is_a_list_not_a_tuple(self):
        actual = cli_views.hazard_to_dict(_write_write_hazard())
        self.assertIsInstance(
            actual["paths"], list,
            "paths must serialise as a JSON array, not a Python tuple")

    def test_hazard_paths_are_printed_as_the_cards_spell_them(self):
        actual = cli_views.hazard_to_dict(_unordered_read_hazard())
        self.assertEqual(
            actual["paths"], ["./src/one.py"],
            "paths must reach the reader exactly as spelled -- normalised "
            "only for comparison, never for display")

    def test_hazard_generation_none_prints_as_null(self):
        actual = cli_views.hazard_to_dict(_unordered_read_hazard())
        self.assertIn("generation", actual,
                      "the generation key must exist even when the pair "
                      "shares no generation")
        self.assertIsNone(actual["generation"],
                          "generation None (different generations) must "
                          "print as null, not as a missing key")

    def test_hazards_to_list_preserves_order_and_delegates(self):
        hazards = [_write_write_hazard(), _read_write_hazard(),
                   _unordered_read_hazard()]
        actual = cli_views.hazards_to_list(hazards)
        self.assertEqual(
            [entry["kind"] for entry in actual],
            [KIND_WRITE_WRITE, KIND_READ_WRITE, KIND_UNORDERED_READ],
            "hazards_to_list must preserve the input order -- find_hazards' "
            "order is deliberate")
        self.assertEqual(
            actual, [cli_views.hazard_to_dict(hazard) for hazard in hazards],
            "every element must be exactly hazard_to_dict of its own hazard")

    def test_hazards_to_list_of_nothing_is_an_empty_list(self):
        self.assertEqual(cli_views.hazards_to_list([]), [],
                         "no hazards must print as an empty list, not null")


# -- outcomes ----------------------------------------------------------------


class OutcomeSerialisationTests(unittest.TestCase):
    """``outcome_to_dict`` / ``outcomes_to_dict``: the stored outcome shape."""

    def test_outcome_to_dict_is_exactly_the_store_shape_written(self):
        outcome = _written_outcome()
        self.assertEqual(
            cli_views.outcome_to_dict(outcome), _outcome_to_dict(outcome),
            "outcome_to_dict must return exactly what cards.store."
            "_outcome_to_dict returns for the same outcome -- the "
            "anti-duplication guarantee, in executable form")

    def test_outcome_to_dict_is_exactly_the_store_shape_failed(self):
        outcome = _failed_outcome()
        self.assertEqual(
            cli_views.outcome_to_dict(outcome), _outcome_to_dict(outcome),
            "the delegation must hold for every outcome shape, failed included")

    def test_outcome_to_dict_carries_exactly_the_stored_keys(self):
        actual = cli_views.outcome_to_dict(_written_outcome())
        self.assertEqual(
            set(actual),
            {"custom_id", "status", "paths", "reason", "attempts",
             "winning_variant", "acceptance_output", "commit", "diffstat"},
            "the printed outcome is the stored one: exactly the nine keys "
            "cards.store._outcome_to_dict writes into state.json and the "
            "run archive")

    def test_outcome_to_dict_carries_the_outcome_verbatim(self):
        outcome = _written_outcome()
        actual = cli_views.outcome_to_dict(outcome)
        self.assertEqual(actual["custom_id"], "card-a",
                         "custom_id must pass through unchanged")
        self.assertEqual(actual["status"], STATUS_WRITTEN,
                         "status must pass through unchanged")
        self.assertEqual(actual["paths"], ["src/one.py"],
                         "paths must pass through as a list")
        self.assertEqual(actual["attempts"], 1,
                         "attempts must pass through unchanged")
        self.assertEqual(actual["winning_variant"], "card-a#2",
                         "winning_variant must pass through unchanged")
        self.assertEqual(actual["commit"], outcome.commit,
                         "commit must be carried: provenance reaches the "
                         "printed form")
        self.assertEqual(actual["diffstat"], outcome.diffstat,
                         "diffstat must be carried: the measured change "
                         "reaches the printed form")

    def test_outcomes_to_dict_maps_every_entry_unchanged(self):
        outcomes = {"card-a": _written_outcome(), "card-b": _failed_outcome()}
        actual = cli_views.outcomes_to_dict(outcomes)
        self.assertEqual(set(actual), set(outcomes),
                         "every custom_id must appear as a key, unchanged")
        for custom_id, outcome in outcomes.items():
            self.assertEqual(
                actual[custom_id], cli_views.outcome_to_dict(outcome),
                f"the entry for {custom_id!r} must be outcome_to_dict of "
                f"its own outcome")

    def test_outcomes_to_dict_of_nothing_is_an_empty_object(self):
        self.assertEqual(cli_views.outcomes_to_dict({}), {},
                         "no outcomes must print as an empty object, not null")


# -- run state ---------------------------------------------------------------


class DeckStatusSerialisationTests(unittest.TestCase):
    """``deck_status_to_dict``: the ``/deck`` snapshot's shape."""

    def test_carries_exactly_the_agreed_keys(self):
        actual = cli_views.deck_status_to_dict(_status_view())
        self.assertEqual(
            set(actual),
            {"empty", "phase", "current_generation", "generations", "cards"},
            "a printed deck status must carry exactly the five agreed keys")

    def test_carries_the_view_verbatim(self):
        actual = cli_views.deck_status_to_dict(_status_view())
        self.assertIs(actual["empty"], False,
                      "empty must be the view's flag, printed as a JSON false")
        self.assertEqual(actual["phase"], PHASE_SUBMITTED,
                         "phase must pass through unchanged")
        self.assertEqual(actual["current_generation"], 1,
                         "current_generation must pass through unchanged")
        self.assertEqual(actual["generations"],
                         [["card-a"], ["card-b", "card-c"]],
                         "generations must be lists of custom_ids in "
                         "composition order")

    def test_card_status_pairs_become_objects_in_the_same_order(self):
        actual = cli_views.deck_status_to_dict(_status_view())
        self.assertEqual(
            actual["cards"],
            [{"custom_id": "card-a", "status": STATUS_WRITTEN},
             {"custom_id": "card-b", "status": STATUS_IN_FLIGHT},
             {"custom_id": "card-c", "status": STATUS_PENDING}],
            "card_status pairs must print as objects with custom_id and "
            "status keys, in the same order")
        self.assertEqual(
            [card["custom_id"] for card in actual["cards"]],
            ["card-a", "card-b", "card-c"],
            "the printed cards must keep card_status's order: a consumer "
            "pairing positions relies on it")

    def test_an_empty_deck_prints_empty_collections(self):
        view = DeckStatusView(empty=True, phase=PHASE_IDLE,
                              current_generation=0, generations=[],
                              card_status=[])
        actual = cli_views.deck_status_to_dict(view)
        self.assertIs(actual["empty"], True,
                      "an empty backlog must print empty true")
        self.assertEqual(actual["generations"], [],
                         "no generations must print as an empty list")
        self.assertEqual(actual["cards"], [],
                         "no cards must print as an empty list")


class SubmitSerialisationTests(unittest.TestCase):
    """``submit_to_dict``: what one ``/submit`` did."""

    def test_carries_exactly_the_agreed_keys(self):
        actual = cli_views.submit_to_dict(_submit_result())
        self.assertEqual(
            set(actual),
            {"submitted", "done", "generation", "total", "batch_id", "cards",
             "skipped"},
            "a printed submit result must carry exactly the seven agreed keys")

    def test_generation_and_total_are_the_renamed_numbers(self):
        result = _submit_result()
        actual = cli_views.submit_to_dict(result)
        self.assertEqual(actual["generation"], result.generation_number,
                         "generation must be the result's generation_number "
                         "(1-based)")
        self.assertEqual(actual["total"], result.total_generations,
                         "total must be the result's total_generations")
        self.assertIs(actual["submitted"], True,
                      "submitted must pass through as a JSON true")
        self.assertIs(actual["done"], False,
                      "done must pass through as a JSON false")
        self.assertEqual(actual["batch_id"], "batch-1",
                         "batch_id must pass through unchanged")
        self.assertEqual(actual["cards"], ["card-b"],
                         "cards must be the submitted ids, in order")

    def test_skipped_pairs_become_objects(self):
        result = SubmitResult(
            submitted=False, done=True, generation_number=0,
            total_generations=3, batch_id=None, card_ids=[],
            skipped=[("card-c", "card-a"), ("card-d", "card-b")])
        actual = cli_views.submit_to_dict(result)
        self.assertEqual(
            actual["skipped"],
            [{"custom_id": "card-c", "blocked_by": "card-a"},
             {"custom_id": "card-d", "blocked_by": "card-b"}],
            "each skipped pair must print as an object naming custom_id and "
            "blocked_by, in order")

    def test_a_run_that_sent_nothing_prints_null_batch_and_zero_generation(self):
        result = SubmitResult(submitted=False, done=True,
                              generation_number=0, total_generations=3,
                              batch_id=None)
        actual = cli_views.submit_to_dict(result)
        self.assertIsNone(actual["batch_id"],
                          "nothing sent must print batch_id null")
        self.assertEqual(actual["generation"], 0,
                         "nothing sent must print generation 0, per the contract")
        self.assertEqual(actual["cards"], [],
                         "nothing sent must print an empty cards list")
        self.assertEqual(actual["skipped"], [],
                         "no skips must print as an empty list")


class CollectSerialisationTests(unittest.TestCase):
    """``collect_to_dict``: what one ``/collect`` did; retry always present."""

    def test_carries_exactly_the_agreed_keys(self):
        actual = cli_views.collect_to_dict(_collect_result())
        self.assertEqual(
            set(actual),
            {"in_progress", "generation", "total", "phase", "outcomes",
             "retry"},
            "a printed collect result must carry exactly the six agreed keys")
        self.assertEqual(
            set(actual["retry"]),
            {"in_flight", "submitted", "attempt", "limit", "cards",
             "batch_id"},
            "the retry object must carry exactly the six agreed keys")

    def test_retry_object_is_present_even_when_empty(self):
        result = CollectResult(
            in_progress=True, generation_number=1, total_generations=2,
            phase=PHASE_SUBMITTED)
        actual = cli_views.collect_to_dict(result)
        self.assertIn("retry", actual,
                      "retry must always be present: a consumer must never "
                      "have to branch on a missing key")
        self.assertEqual(
            actual["retry"],
            {"in_flight": False, "submitted": False, "attempt": 0,
             "limit": 0, "cards": [], "batch_id": None},
            "a first-attempt batch must print the empty retry object")
        self.assertEqual(actual["outcomes"], {},
                         "no settled cards yet must print an empty outcomes "
                         "object")

    def test_carries_a_regeneration_in_flight(self):
        result = _collect_result()
        actual = cli_views.collect_to_dict(result)
        self.assertIs(actual["in_progress"], True,
                      "in_progress must pass through as a JSON true")
        self.assertEqual(actual["generation"], result.generation_number,
                         "generation must be the result's generation_number")
        self.assertEqual(actual["total"], result.total_generations,
                         "total must be the result's total_generations")
        self.assertEqual(actual["phase"], PHASE_SUBMITTED,
                         "phase must pass through unchanged")
        self.assertEqual(
            actual["outcomes"],
            {"card-a": _outcome_to_dict(_written_outcome())},
            "outcomes must print in the stored shape, via outcomes_to_dict")
        self.assertEqual(
            actual["retry"],
            {"in_flight": True, "submitted": True, "attempt": 1,
             "limit": 2, "cards": ["card-b"], "batch_id": "batch-retry-1"},
            "the retry fields must pass through under their printed names")

    def test_a_finished_generation_carries_its_outcomes(self):
        result = CollectResult(
            in_progress=False, generation_number=1, total_generations=2,
            phase=PHASE_IDLE,
            outcomes={"card-a": _written_outcome(),
                      "card-b": _failed_outcome()})
        actual = cli_views.collect_to_dict(result)
        self.assertIs(actual["in_progress"], False,
                      "a settled generation must print in_progress false")
        self.assertEqual(set(actual["outcomes"]), {"card-a", "card-b"},
                         "every reported outcome must appear under its "
                         "custom_id")
        self.assertEqual(actual["outcomes"]["card-b"],
                         _outcome_to_dict(_failed_outcome()),
                         "each outcome must print as the stored shape")
        self.assertIs(actual["retry"]["in_flight"], False,
                      "a generation settled without a regeneration must say so")


# -- the run archive ---------------------------------------------------------


class ReportSerialisationTests(unittest.TestCase):
    """``report_to_dict``: the archived run report, printed."""

    def test_is_the_archived_shape_plus_counts(self):
        report = _report()
        actual = cli_views.report_to_dict(report)
        self.assertEqual(
            actual, {**report.to_dict(), "counts": report.counts},
            "report_to_dict must be RunReport.to_dict() plus counts: the "
            "archived shape IS the printed shape")
        self.assertEqual(
            set(actual), set(report.to_dict()) | {"counts"},
            "the printed report must add exactly one key over the archive: "
            "counts")

    def test_carries_the_report_fields_verbatim(self):
        actual = cli_views.report_to_dict(_report())
        self.assertEqual(actual["deck_id"], "20260917-114233-9f1c4b02",
                         "deck_id must pass through unchanged")
        self.assertEqual(actual["completed_at"], "2026-09-17T11:42:33.123000",
                         "completed_at must pass through unchanged")
        self.assertEqual(actual["branch"], "morph/20260917-114233-9f1c4b02",
                         "branch must pass through unchanged")
        self.assertEqual(actual["backend_label"], "openrouter",
                         "backend_label must pass through unchanged")
        self.assertEqual(actual["generations"], [["card-a"], ["card-b"]],
                         "generations must pass through in composition order")
        self.assertEqual(actual["batch_ids"], ["batch-1", "batch-retry-1"],
                         "batch_ids must pass through in submission order")
        self.assertEqual(actual["outcomes"]["card-a"],
                         _outcome_to_dict(_written_outcome()),
                         "the archived outcomes must print in the stored shape")

    def test_counts_tally_the_outcomes(self):
        actual = cli_views.report_to_dict(_report())
        self.assertEqual(
            actual["counts"],
            {STATUS_WRITTEN: 1, STATUS_FAILED: 1, STATUS_SKIPPED: 0},
            "counts must tally the report's outcomes by status, skipped "
            "zero included")


# -- JSON as it stands -------------------------------------------------------


class JsonAcceptanceTests(unittest.TestCase):
    """Every shape must be JSON as it stands -- no custom encoder, anywhere."""

    def test_every_serialiser_output_is_accepted_by_json_dumps(self):
        samples = {
            "hazard_to_dict": cli_views.hazard_to_dict(_write_write_hazard()),
            "hazards_to_list": cli_views.hazards_to_list([
                _write_write_hazard(), _read_write_hazard(),
                _unordered_read_hazard()]),
            "outcome_to_dict": cli_views.outcome_to_dict(_written_outcome()),
            "outcomes_to_dict": cli_views.outcomes_to_dict(
                {"card-a": _written_outcome(), "card-b": _failed_outcome()}),
            "deck_status_to_dict": cli_views.deck_status_to_dict(
                _status_view()),
            "submit_to_dict": cli_views.submit_to_dict(_submit_result()),
            "collect_to_dict": cli_views.collect_to_dict(_collect_result()),
            "report_to_dict": cli_views.report_to_dict(_report()),
        }
        for name, payload in samples.items():
            try:
                encoded = json.dumps(payload)
            except (TypeError, ValueError) as error:
                self.fail(
                    f"{name}(...) must return a value json.dumps accepts "
                    f"without a custom encoder; it raised {error!r} instead")
            self.assertIsInstance(
                encoded, str,
                f"{name}(...) must serialise to a JSON string")


if __name__ == "__main__":
    unittest.main()
