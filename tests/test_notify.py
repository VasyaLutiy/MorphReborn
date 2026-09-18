"""Tests for ``cards.notify`` -- the one line a finished run pushes out.

The transport under test is a FAKE one and must stay that way: every command
handed to :func:`cards.notify.send_notification` here is a local shell
one-liner that writes its stdin into a file under
``tempfile.TemporaryDirectory``. Nothing leaves this machine and no real
channel is ever touched.

The run report fixture is a COPY OF A REAL ARTIFACT -- the ``report.json``
of run ``20260918-150640-34c1caaf`` under ``.morph/runs/`` -- pasted as a
literal and rebuilt through ``RunReport.from_dict``, so the test depends on
no file on disk and never constructs a ``RunReport`` field by field.
Variants (a named branch, an unparseable id or timestamp) are deep copies of
that literal with one field overridden, never fresh constructions.

Assertions aim at the NUMBERS the message must carry -- and the
``<N> written`` form -- not at the wording around them.
"""

import contextlib
import copy
import io
import os
import shlex
import tempfile
import unittest
from unittest import mock

from cards.notify import build_run_message, send_notification
from cards.store import RunReport


# The archived report.json of run 20260918-150640-34c1caaf, verbatim. Its
# deck id head (20260918-150640) and completed_at (2026-09-18T17:16:41.998)
# are a real start/finish pair: 2 h 10 m 1.998 s apart, i.e. 130 whole
# minutes -- the number the minutes assertions below expect.
ARCHIVED_REPORT_JSON = {
    "deck_id": "20260918-150640-34c1caaf",
    "completed_at": "2026-09-18T17:16:41.998",
    "branch": None,
    "backend_label": "glm",
    "generations": [
        ["cli-run-cycle"],
        ["cli-main-dispatch"],
        ["cli-contract-judge"],
    ],
    "batch_ids": [
        "batch-1789733202-LDWizYrtXfCYduwNyaZq",
        "batch-1789734858-BbCJ3hK4NhiLmIdOtdmo",
        "batch-1789736556-W3LvWq7YzHFzxVq7VcQL",
        "batch-1789739480-lY0hPIxRpDvHFQLM6sa1",
    ],
    "outcomes": {
        "cli-run-cycle": {
            "custom_id": "cli-run-cycle",
            "status": "written",
            "paths": [
                "./cards/cli_cycle.py",
                "./tests/test_cli_cycle.py",
            ],
            "reason": None,
            "attempts": 1,
            "winning_variant": "cli-run-cycle",
            "acceptance_output": None,
            "commit": None,
            "diffstat": None,
        },
        "cli-main-dispatch": {
            "custom_id": "cli-main-dispatch",
            "status": "written",
            "paths": [
                "./cards/cli.py",
                "./bin/mrph",
                "./tests/test_cli_main.py",
            ],
            "reason": None,
            "attempts": 1,
            "winning_variant": "cli-main-dispatch",
            "acceptance_output": None,
            "commit": None,
            "diffstat": None,
        },
        "cli-contract-judge": {
            "custom_id": "cli-contract-judge",
            "status": "written",
            "paths": [
                "./tests/test_cli_contract.py",
            ],
            "reason": None,
            "attempts": 2,
            "winning_variant": "cli-contract-judge.r1",
            "acceptance_output": None,
            "commit": None,
            "diffstat": None,
        },
    },
}


def _report(**overrides: object) -> RunReport:
    """The archived report, rebuilt through ``from_dict``, fields overridden."""
    data = copy.deepcopy(ARCHIVED_REPORT_JSON)
    data.update(overrides)
    return RunReport.from_dict(data)


def _sink_command(directory: str, name: str = "sink.txt") -> str:
    """A local one-liner that writes its stdin into a file under *directory*."""
    return "cat > {0}".format(shlex.quote(os.path.join(directory, name)))


def _read_sink(directory: str, name: str = "sink.txt") -> str:
    """What the sink one-liner received on its stdin."""
    with open(os.path.join(directory, name), encoding="utf-8") as handle:
        return handle.read()


class BuildRunMessageTest(unittest.TestCase):
    """``build_run_message``: every number, on one line, from the report alone."""

    def test_message_carries_every_number(self) -> None:
        report = _report()
        message = build_run_message(report)
        counts = report.counts
        self.assertIn("20260918-150640-34c1caaf", message)
        self.assertIn("{0} written".format(counts["written"]), message)
        self.assertIn("{0} failed".format(counts["failed"]), message)
        self.assertIn("{0} skipped".format(counts["skipped"]), message)
        self.assertIn("3 generations", message)
        # one line, always: channels deliver one line as one message
        self.assertNotIn("\n", message)

    def test_counts_appear_in_scan_order_number_before_word(self) -> None:
        report = _report()
        message = build_run_message(report)
        counts = report.counts
        written_at = message.index("{0} written".format(counts["written"]))
        failed_at = message.index("{0} failed".format(counts["failed"]))
        skipped_at = message.index("{0} skipped".format(counts["skipped"]))
        self.assertLess(written_at, failed_at)
        self.assertLess(failed_at, skipped_at)

    def test_minutes_from_real_deck_id_and_completed_at(self) -> None:
        # Start: the deck id head 20260918-150640. Finish: the archived
        # completed_at 2026-09-18T17:16:41.998. The gap is 2 h 10 m 1.998 s,
        # so 130 whole minutes -- computed from the report alone, no clock
        # consulted.
        self.assertIn("130 min", build_run_message(_report()))

    def test_none_branch_says_no_branch(self) -> None:
        # the archived run owned no branch, and the line says so in words
        self.assertIn("no branch", build_run_message(_report()))

    def test_named_branch_is_named(self) -> None:
        message = build_run_message(_report(branch="feature/night-run"))
        self.assertIn("feature/night-run", message)
        self.assertNotIn("no branch", message)

    def test_unparseable_times_cost_the_minutes_not_the_message(self) -> None:
        counts = _report().counts
        message = build_run_message(_report(deck_id="not-a-deck-id"))
        self.assertIn("? min", message)
        message = build_run_message(_report(completed_at="not-a-timestamp"))
        self.assertIn("? min", message)
        # the numbers the report does still give reach the line
        self.assertIn("{0} written".format(counts["written"]), message)


class SendNotificationTest(unittest.TestCase):
    """``send_notification`` against a fake, strictly local transport."""

    def test_unset_env_var_is_a_silent_false(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("MORPH_NOTIFY_CMD", None)
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), \
                    contextlib.redirect_stderr(stderr):
                self.assertFalse(send_notification("run finished"))
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")

    def test_empty_command_is_a_silent_false(self) -> None:
        # empty however it arrives: an empty variable, or an empty or blank
        # command passed straight in -- none of it raises or prints
        with mock.patch.dict(os.environ):
            os.environ["MORPH_NOTIFY_CMD"] = ""
            self.assertFalse(send_notification("run finished"))
        self.assertFalse(send_notification("run finished", command=""))
        self.assertFalse(send_notification("run finished", command="   "))

    def test_failing_command_is_false_not_an_exception(self) -> None:
        # the sink one-liner, then a non-zero exit: the channel ran and
        # answered no, and that is a False, not a raise
        with tempfile.TemporaryDirectory() as tmp:
            command = "{0} ; exit 1".format(_sink_command(tmp))
            self.assertFalse(
                send_notification("run finished", command=command)
            )

    def test_missing_command_is_false_not_an_exception(self) -> None:
        self.assertFalse(
            send_notification(
                "run finished",
                command="mrph-no-such-notify-command-on-any-path",
            )
        )

    def test_command_receives_message_on_its_stdin(self) -> None:
        message = (
            "mrph run 20260918-150640-34c1caaf (no branch): "
            "3 written, 0 failed, 0 skipped; 3 generations, 130 min"
        )
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(
                send_notification(message, command=_sink_command(tmp))
            )
            self.assertEqual(_read_sink(tmp), message)

    def test_env_var_is_read_at_call_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ):
                os.environ["MORPH_NOTIFY_CMD"] = _sink_command(tmp, "first.txt")
                self.assertTrue(send_notification("first run"))
                # changed between calls, no argument passed either time: the
                # variable is consulted per call, not once at import
                os.environ["MORPH_NOTIFY_CMD"] = _sink_command(tmp, "second.txt")
                self.assertTrue(send_notification("second run"))
            self.assertEqual(_read_sink(tmp, "first.txt"), "first run")
            self.assertEqual(_read_sink(tmp, "second.txt"), "second run")

    def test_hanging_command_times_out_to_false(self) -> None:
        # a channel that never answers costs the notification, not the run:
        # the timeout is patched short so the test stays quick
        with mock.patch("cards.notify._NOTIFY_TIMEOUT_SECONDS", 0.25):
            self.assertFalse(
                send_notification("run finished", command="sleep 5")
            )


if __name__ == "__main__":
    unittest.main()
