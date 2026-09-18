"""
The JSON shapes a headless ``mrph`` prints: one agreed shape per entity.

A script driving ``mrph`` without a terminal reads what it prints, so every
entity the commands can emit needs exactly ONE agreed shape -- the same keys,
the same renames, the same treatment of ``None``, every run. This module is
where those shapes are defined. It is pure serialisation: it computes nothing,
decides nothing and performs no I/O. Each function takes an object the rest of
``cards`` already produces and returns the dict ``json.dumps`` accepts without
a custom encoder.

WHY it is thin where it is thin. Where ``cards.store`` already serialises
something, this module DELEGATES to that serialiser and never builds a second
shape. A second serialiser for an entity that has one is a defect, not a
convenience: two shapes drift apart, and the day they do, the run archive
under ``.morph/runs/`` stops matching what the live command prints -- a script
written against one silently misreads the other. So ``outcome_to_dict`` is a
public name over ``cards.store._outcome_to_dict`` (the shape already written
into ``state.json`` and every archived ``report.json``, ``commit`` and
``diffstat`` included), and ``report_to_dict`` is ``RunReport.to_dict`` plus
the counts. The archived shape IS the printed shape, by construction rather
than by vigilance.

The renames are part of the contract, not carelessness. A tuple of pairs is a
programmer's shape: the printed form turns ``DeckStatusView.card_status`` into
``cards`` (the same order, objects instead of pairs) and ``SubmitResult``'s
``generation_number`` / ``total_generations`` into ``generation`` / ``total``.
``collect_to_dict`` always emits its ``retry`` object, filled or empty: a
polling script answers "is a regeneration in flight" by reading
``retry.in_flight`` -- never by asking whether the key exists.

Imports stop at ``cards.store`` and ``cards.hazards`` (plus ``CardOutcome``
from ``cards.generations``, for the type hints). This module sits BELOW the
CLI: ``cards.cli_json`` and the command modules render these dicts, so nothing
here may import them -- and nothing here imports from ``flows``. Stdlib-only.
"""

from typing import Dict, List, Mapping, Sequence

from cards.generations import CardOutcome
from cards.hazards import Hazard
from cards.store import (
    CollectResult,
    DeckStatusView,
    RunReport,
    SubmitResult,
    _outcome_to_dict,
)


# -- hazards -----------------------------------------------------------------


def hazard_to_dict(hazard: Hazard) -> Dict[str, object]:
    """One :class:`~cards.hazards.Hazard` as the headless report prints it.

    Fields straight off the hazard. ``paths`` is its tuple as a ``list`` (a
    JSON array -- a tuple is not one), spelled exactly as the cards spell the
    files: normalised only for comparison inside :mod:`cards.hazards`, never
    for display. ``generation`` is the attribute as-is, ``None`` included --
    for :data:`~cards.hazards.KIND_UNORDERED_READ` "no shared generation" IS
    the finding, and it must reach the reader as ``null``, not as a missing
    key. ``message`` is the hazard's own one-line diagnosis, so a script can
    surface it verbatim instead of re-deriving prose from the fields.
    """
    return {
        "kind": hazard.kind,
        "severity": hazard.severity,
        "left": hazard.left,
        "right": hazard.right,
        "paths": list(hazard.paths),
        "generation": hazard.generation,
        "message": hazard.message(),
    }


def hazards_to_list(hazards: Sequence[Hazard]) -> List[Dict[str, object]]:
    """A sequence of hazards as a JSON array, order preserved.

    :func:`cards.hazards.find_hazards` returns its list in a stable, deliberate
    order (deck order, at most one hazard per pair per direction); printing
    re-orders nothing, so a consumer diffing two runs' reports is diffing the
    decks, not the printer.
    """
    return [hazard_to_dict(hazard) for hazard in hazards]


# -- outcomes ----------------------------------------------------------------


def outcome_to_dict(outcome: CardOutcome) -> Dict[str, object]:
    """One :class:`~cards.generations.CardOutcome` as the headless report prints it.

    Delegates WHOLESALE to ``cards.store._outcome_to_dict`` and adds nothing.
    That function is the one shape an outcome has -- already written into
    ``state.json`` and every archived ``report.json``, ``commit`` and
    ``diffstat`` included -- and this wrapper exists for one reason only: it is
    private, and callers outside ``cards.store`` need a public name to reach
    the same shape by. Repeating its field list here would be a second
    serialiser waiting to drift away from the archive.
    """
    return _outcome_to_dict(outcome)


def outcomes_to_dict(
    outcomes: Mapping[str, CardOutcome],
) -> Dict[str, Dict[str, object]]:
    """A ``custom_id -> CardOutcome`` mapping as one JSON object.

    Keys pass through unchanged (they are the deck's own custom_ids), values
    through :func:`outcome_to_dict` -- the printed outcomes are the stored
    outcomes, key for key.
    """
    return {custom_id: outcome_to_dict(outcome)
            for custom_id, outcome in outcomes.items()}


# -- run state ---------------------------------------------------------------


def deck_status_to_dict(view: DeckStatusView) -> Dict[str, object]:
    """A :class:`~cards.store.DeckStatusView` as ``/deck`` prints it.

    ``generations`` is the composition as lists of custom_ids. ``cards`` is the
    view's ``card_status`` -- a list of ``(custom_id, status)`` pairs, a
    programmer's shape -- turned into objects in the SAME order, so a consumer
    that pairs the two arrays by position is pairing with the truth.
    """
    return {
        "empty": view.empty,
        "phase": view.phase,
        "current_generation": view.current_generation,
        "generations": [list(generation) for generation in view.generations],
        "cards": [{"custom_id": custom_id, "status": status}
                  for custom_id, status in view.card_status],
    }


def submit_to_dict(result: SubmitResult) -> Dict[str, object]:
    """A :class:`~cards.store.SubmitResult` as ``/submit`` prints it.

    ``generation`` is the result's ``generation_number`` -- 1-based, and ``0``
    when nothing was sent because every remaining card was skipped or the run
    was already finished -- and ``total`` its ``total_generations``. The
    ``skipped`` ``(custom_id, blocking_dep)`` pairs become objects: a pair's
    meaning is positional and dies at the JSON boundary, an object's meaning is
    in its keys.
    """
    return {
        "submitted": result.submitted,
        "done": result.done,
        "generation": result.generation_number,
        "total": result.total_generations,
        "batch_id": result.batch_id,
        "cards": list(result.card_ids),
        "skipped": [{"custom_id": custom_id, "blocked_by": blocking}
                    for custom_id, blocking in result.skipped],
    }


def collect_to_dict(result: CollectResult) -> Dict[str, object]:
    """A :class:`~cards.store.CollectResult` as ``/collect`` prints it.

    ``outcomes`` goes through :func:`outcomes_to_dict` -- the stored shape,
    again. ``retry`` is ALWAYS present, filled or empty: a polling script asks
    one question, "is a regeneration in flight", and must answer it by reading
    ``retry.in_flight``, never by asking whether the key exists. The empty
    form (``in_flight`` false, ``attempt`` 0, no cards, ``batch_id`` ``null``)
    is what a first-attempt batch prints.
    """
    return {
        "in_progress": result.in_progress,
        "generation": result.generation_number,
        "total": result.total_generations,
        "phase": result.phase,
        "outcomes": outcomes_to_dict(result.outcomes),
        "retry": {
            "in_flight": result.retry_in_flight,
            "submitted": result.retry_submitted,
            "attempt": result.retry_attempt,
            "limit": result.retry_limit,
            "cards": list(result.retry_card_ids),
            "batch_id": result.retry_batch_id,
        },
    }


# -- the run archive ---------------------------------------------------------


def report_to_dict(report: RunReport) -> Dict[str, object]:
    """A :class:`~cards.store.RunReport` as the run summary prints it.

    ``report.to_dict()`` plus one key, ``counts`` -- the written/failed/skipped
    tally the report already carries as a property. Same reason
    :func:`outcome_to_dict` delegates rather than re-implements: the archived
    ``report.json`` is written from the same ``to_dict``
    (:func:`cards.store.archive_run`), so the printed report is the archived
    report, and a script that learned one shape can read the other.
    """
    printed = report.to_dict()
    printed["counts"] = report.counts
    return printed
