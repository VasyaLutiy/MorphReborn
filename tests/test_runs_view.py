"""Tests for cards/runs_view.py, the pure archived-run-report formatter.

Self-contained by design: every report dict is built inline, so these tests
need no git, no temporary repository and no filesystem.  They pin the printed
line shape, the three mandatory rules (deletion-heavy mark, no-commit dashes,
reports written before ``commit``/``diffstat`` existed), generation ordering,
and the ``None``-count skipping that keeps binary files out of the sums.
"""

import copy
import os
import sys

# Make the repo root importable no matter how pytest is invoked, so this test
# file stays self-contained.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cards.runs_view import format_run_report  # noqa: E402

#: The exact mark a deletion-heavy line must end with.
MARK = "   !! removed more than added"


def make_outcome(custom_id, *, status="written", paths=(), **extra):
    """Build an outcome dict carrying the fields a real report has."""
    outcome = {
        "custom_id": custom_id,
        "status": status,
        "paths": list(paths),
        "reason": None,
        "attempts": 1,
        "winning_variant": custom_id,
    }
    outcome.update(extra)
    return outcome


def make_report(outcomes, generations):
    """Build a report dict: ``generations`` order, ``outcomes`` keyed by id."""
    return {"generations": generations, "outcomes": outcomes}


def test_happy_path_line_shape():
    report = make_report(
        {
            "card-a": make_outcome(
                "card-a",
                commit="0123456789abcdef",
                diffstat=[
                    {"path": "./cards/a.py", "insertions": 12, "deletions": 3},
                    {"path": "./tests/test_a.py", "insertions": 4, "deletions": 1},
                ],
            ),
            "card-b": make_outcome(
                "card-b",
                commit="fedcba98765432109876",
                diffstat=[
                    {"path": "./cards/b.py", "insertions": 1, "deletions": 0},
                ],
            ),
        },
        generations=[["card-a", "card-b"]],
    )

    assert format_run_report(report) == [
        "card-a   written   +16 -4   0123456789   ./cards/a.py, ./tests/test_a.py",
        "card-b   written   +1 -0   fedcba9876   ./cards/b.py",
    ]


def test_lines_follow_flattened_generation_order():
    report = make_report(
        {
            "gen1-card": make_outcome("gen1-card"),
            "gen2-a": make_outcome("gen2-a"),
            "gen2-b": make_outcome("gen2-b"),
        },
        generations=[["gen1-card"], ["gen2-a", "gen2-b"]],
    )

    lines = format_run_report(report)

    assert [line.split("   ")[0] for line in lines] == [
        "gen1-card",
        "gen2-a",
        "gen2-b",
    ]


def test_deletions_exceeding_insertions_get_marked_but_still_listed():
    report = make_report(
        {
            "net-negative": make_outcome(
                "net-negative",
                commit="a" * 40,
                diffstat=[{"path": "./cards/x.py", "insertions": 2, "deletions": 9}],
            ),
            "net-positive": make_outcome(
                "net-positive",
                commit="b" * 40,
                diffstat=[{"path": "./cards/y.py", "insertions": 9, "deletions": 2}],
            ),
            "balanced": make_outcome(
                "balanced",
                commit="c" * 40,
                diffstat=[{"path": "./cards/z.py", "insertions": 5, "deletions": 5}],
            ),
        },
        generations=[["net-negative", "net-positive", "balanced"]],
    )

    lines = format_run_report(report)

    # The deletion-heavy card is still listed normally: a mark, not a refusal.
    assert lines[0] == (
        "net-negative   written   +2 -9   aaaaaaaaaa   ./cards/x.py" + MARK
    )
    assert lines[1] == "net-positive   written   +9 -2   bbbbbbbbbb   ./cards/y.py"
    assert lines[2] == "balanced   written   +5 -5   cccccccccc   ./cards/z.py"


def test_mark_uses_sums_across_all_diffstat_entries():
    report = make_report(
        {
            "multi": make_outcome(
                "multi",
                commit="d" * 40,
                diffstat=[
                    {"path": "./cards/m1.py", "insertions": 1, "deletions": 3},
                    {"path": "./cards/m2.py", "insertions": 1, "deletions": 2},
                ],
            ),
        },
        generations=[["multi"]],
    )

    assert format_run_report(report) == [
        "multi   written   +2 -5   dddddddddd   ./cards/m1.py, ./cards/m2.py" + MARK
    ]


def test_no_commit_prints_dashes_and_does_not_raise():
    report = make_report(
        {
            "nogit-card": make_outcome("nogit-card", paths=["./cards/nogit.py"]),
            # A card that changed nothing: no commit, and explicit Nones.
            "empty-card": make_outcome(
                "empty-card", commit=None, diffstat=None, paths=[]
            ),
        },
        generations=[["nogit-card", "empty-card"]],
    )

    lines = format_run_report(report)  # must not raise

    assert lines == [
        "nogit-card   written   -   -   ./cards/nogit.py",
        "empty-card   written   -   -",
    ]


def test_old_report_without_commit_or_diffstat_formats_like_no_commit():
    # Shape of the real archived report of run 20260917-174646-f718dfe2,
    # written before `commit` and `diffstat` existed: the outcomes carry
    # neither key, and one card failed with no paths at all.
    report = {
        "generations": [
            [
                "collect-names-the-rejection",
                "keep-the-failed-attempt-output",
                "deck-clear-command",
                "preflight-the-batch-slug",
                "demo-script-truth",
            ]
        ],
        "outcomes": {
            "collect-names-the-rejection": {
                "custom_id": "collect-names-the-rejection",
                "status": "written",
                "paths": [
                    "./cards/store.py",
                    "./tests/test_collect_rejection_reason.py",
                ],
                "reason": None,
                "attempts": 2,
                "winning_variant": "collect-names-the-rejection.r1.v1",
            },
            "keep-the-failed-attempt-output": {
                "custom_id": "keep-the-failed-attempt-output",
                "status": "written",
                "paths": [
                    "./cards/generations.py",
                    "./tests/test_failed_attempt_diagnostics.py",
                ],
                "reason": None,
                "attempts": 2,
                "winning_variant": "keep-the-failed-attempt-output.r1.v1",
            },
            "deck-clear-command": {
                "custom_id": "deck-clear-command",
                "status": "written",
                "paths": ["./flows/morph.py", "./tests/test_deck_clear.py"],
                "reason": None,
                "attempts": 2,
                "winning_variant": "deck-clear-command.r1",
            },
            "preflight-the-batch-slug": {
                "custom_id": "preflight-the-batch-slug",
                "status": "written",
                "paths": [
                    "./processors/batch.py",
                    "./tests/test_batch_slug_preflight.py",
                ],
                "reason": None,
                "attempts": 1,
                "winning_variant": "preflight-the-batch-slug",
            },
            "demo-script-truth": {
                "custom_id": "demo-script-truth",
                "status": "failed",
                "paths": [],
                "reason": None,
                "attempts": 3,
                "winning_variant": None,
            },
        },
    }

    lines = format_run_report(report)  # must not raise

    assert lines == [
        "collect-names-the-rejection   written   -   -   ./cards/store.py, ./tests/test_collect_rejection_reason.py",
        "keep-the-failed-attempt-output   written   -   -   ./cards/generations.py, ./tests/test_failed_attempt_diagnostics.py",
        "deck-clear-command   written   -   -   ./flows/morph.py, ./tests/test_deck_clear.py",
        "preflight-the-batch-slug   written   -   -   ./processors/batch.py, ./tests/test_batch_slug_preflight.py",
        "demo-script-truth   failed   -   -",
    ]
    # Even the failed card with no paths leaves no trailing whitespace.
    assert all(line == line.rstrip() for line in lines)


def test_custom_id_missing_from_outcomes_does_not_raise():
    report = make_report(
        {
            "real-card": make_outcome(
                "real-card",
                commit="e" * 40,
                diffstat=[{"path": "./cards/real.py", "insertions": 3, "deletions": 1}],
            ),
        },
        generations=[["ghost-card", "real-card"]],
    )

    lines = format_run_report(report)  # must not raise

    assert lines == [
        "ghost-card   missing   -   -",
        "real-card   written   +3 -1   eeeeeeeeee   ./cards/real.py",
    ]


def test_binary_none_counts_are_skipped_in_the_sums():
    report = make_report(
        {
            "mixed": make_outcome(
                "mixed",
                commit="1" * 40,
                diffstat=[
                    {"path": "./cards/code.py", "insertions": 7, "deletions": 2},
                    {"path": "./assets/logo.png", "insertions": None, "deletions": None},
                ],
            ),
            "all-binary": make_outcome(
                "all-binary",
                commit="2" * 40,
                diffstat=[
                    {"path": "./assets/hero.png", "insertions": None, "deletions": None},
                ],
            ),
        },
        generations=[["mixed", "all-binary"]],
    )

    assert format_run_report(report) == [
        "mixed   written   +7 -2   1111111111   ./cards/code.py, ./assets/logo.png",
        "all-binary   written   +0 -0   2222222222   ./assets/hero.png",
    ]


def test_commit_without_diffstat_falls_back_to_outcome_paths():
    report = make_report(
        {
            "odd-card": make_outcome(
                "odd-card", commit="3" * 40, paths=["./cards/odd.py"]
            ),
        },
        generations=[["odd-card"]],
    )

    assert format_run_report(report) == [
        "odd-card   written   +0 -0   3333333333   ./cards/odd.py"
    ]


def test_degenerate_reports_do_not_raise():
    assert format_run_report({}) == []
    assert format_run_report({"generations": [], "outcomes": {}}) == []
    assert format_run_report({"generations": [[]], "outcomes": {}}) == []
    assert format_run_report({"generations": [["ghost"]], "outcomes": None}) == [
        "ghost   missing   -   -"
    ]


def test_formatting_does_not_mutate_the_report():
    report = make_report(
        {
            "card": make_outcome(
                "card",
                commit="f" * 40,
                diffstat=[{"path": "./cards/f.py", "insertions": 1, "deletions": 1}],
            ),
        },
        generations=[["card"]],
    )
    snapshot = copy.deepcopy(report)

    format_run_report(report)

    assert report == snapshot
