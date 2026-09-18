"""
The notification ``cards.cli_cycle.run`` sends when a run ends: wiring tests.

``cards/notify.py`` can turn a finished run into one line with the numbers in
it and push that line through whatever channel ``$MORPH_NOTIFY_CMD`` names --
and ``mrph run`` is the path that must speak, because it is the unattended
path, the run that finishes at 03:40 with nobody watching. These tests prove
the wiring and nothing else: a finished run's line reaches the configured
channel with the run's deck id and its written/failed/skipped counts in it;
the very same line reaches ``log``; and neither a missing channel nor a
refusing one moves the payload or the exit code by a pixel -- no channel is
the normal case, and the notification is the last thing the run does, so it
must not be able to fail the run.

The transport is FAKE and nothing leaves the machine: ``$MORPH_NOTIFY_CMD``
points at a local shell one-liner -- ``cat`` writing its stdin into a file
inside the temporary directory -- so the bytes the channel received sit on
disk, assertable. The variable is set (or explicitly removed, for the
no-channel case) through a helper that restores the previous environment in
an ``addCleanup``, whatever the test does: a leaked channel would silently
re-channel every test that runs after the one that leaked it.

Otherwise this is the ``tests/test_cli_cycle.py`` harness: a
``tempfile.TemporaryDirectory`` root, ``nogit=True`` (a tempdir holds no
branch to begin with, and the git layer must stay out of these assertions),
a FAKE batch backend injected through ``backend``, and a fake clock injected
through ``sleep``/``now`` that doubles as an alarm -- the fake completes every
batch on its first poll, so a recorded nap means somebody waited who had no
business waiting. The morph-card fixtures are the same shape too: ``intent``
is the literal ``"generate"``, ``custom_id`` matches ``^[A-Za-z0-9._-]+$``, a
card names ``target`` and never ``targets``, and the acceptance criteria are
the one-word commands ``true`` and ``false``, whose exit codes decide the
pass/fail split under whatever invocation the acceptance runner chooses.
"""

import contextlib
import io
import json
import os
import re
import shlex
import sys
import tempfile
import unittest
from typing import Callable, Dict, List, Optional, Tuple

# So the suite also runs under a bare interpreter, where tests/ is the import
# root and the project root is not yet on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cards import cli_cycle
from cards.cli_json import EXIT_INCOMPLETE, EXIT_OK
from cards.store import DeckStore

#: The variable the wiring consults at run end, spelled as the literal the
#: contract names: a rename in ``cards.notify`` must show up as a failure
#: here, not as a silently channel-less suite.
ENV_VAR = "MORPH_NOTIFY_CMD"


# -- fixtures ----------------------------------------------------------------


def _card(custom_id: str, target: str, instruction: str,
          acceptance: Optional[str] = None,
          depends_on: Optional[List[str]] = None) -> dict:
    """One morph-card dict in the nested shape ``MorphCard.from_dict`` accepts.

    ``intent`` is the literal ``"generate"``; the prose is ``instruction``.
    ``custom_id`` sticks to the filesystem-safe character set the schema
    demands. The card names ``target`` and never ``targets`` -- the schema
    refuses a card that names both.
    """
    meta: Dict[str, object] = {"intent": "generate", "target": target}
    if acceptance is not None:
        meta["acceptance"] = acceptance
    if depends_on is not None:
        meta["depends_on"] = list(depends_on)
    return {"custom_id": custom_id, "meta": meta, "instruction": instruction}


def _fenced(body: str) -> str:
    """A model's answer for one file: one fenced block holding ``body``."""
    return "```python\n" + body + "\n```"


def _solo_deck() -> List[dict]:
    """The one-card, passing deck the comparison tests run, fresh per call.

    Fresh dicts per call because two roots must run the SAME deck without
    sharing one card's dict between two stores.
    """
    return [_card("solo", "solo.py", "Write solo.py.", acceptance="true")]


class FakeBackend:
    """A batch backend with no provider behind it: every batch is done at once.

    Duck-typed to the three methods ``cards.generations.run_deck`` calls. It
    remembers every batch it was handed -- so a test can assert how many
    batches a run really paid for -- and answers ``collect`` from the bodies
    the test stated, keyed by request ``custom_id``, including the
    ``.r1``/``.r2`` ids a regeneration attempt carries, so a retry gets a
    real answer to fail its acceptance with.
    """

    def __init__(self, bodies: Dict[str, str]) -> None:
        self.bodies = dict(bodies)
        self.requests: List[List[dict]] = []
        self.batch_ids: List[str] = []

    def submit(self, requests: List[dict]) -> str:
        self.requests.append([dict(request) for request in requests])
        batch_id = "fake-%04d" % len(self.requests)
        self.batch_ids.append(batch_id)
        return batch_id

    def status(self, batch_id: str) -> str:
        return "completed"

    def collect(self, batch_id: str) -> Dict[str, Optional[str]]:
        results: Dict[str, Optional[str]] = {}
        batch = self.requests[self.batch_ids.index(batch_id)]
        for request in batch:
            custom_id = request.get("custom_id")
            results[custom_id] = self.bodies.get(custom_id)
        return results


class NotifyWiringTest(unittest.TestCase):
    """The end-of-run notification: tempdir root, fake backend, fake channel."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.naps: List[float] = []
        self._tick = 0.0

        def fake_sleep(seconds: float) -> None:
            self.naps.append(seconds)

        def fake_now() -> float:
            self._tick += 1.0
            return self._tick

        self.sleep = fake_sleep
        self.now = fake_now

    # -- helpers -------------------------------------------------------------


    def _add_cards(self, *card_dicts: dict) -> None:
        """Put cards in the backlog through the store's own, validating API."""
        DeckStore(self.root).add_cards(list(card_dicts))

    def _fresh_root(self) -> str:
        """A second temporary morph root, cleaned up whenever the test ends."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return tmp.name

    def _notify_path(self, root: str) -> str:
        """Where the fake channel writes: a file in that run's tempdir root."""
        return os.path.join(root, "notify-received.txt")

    def _cat_command(self, path: str) -> str:
        """The fake channel: a local one-liner writing its stdin into ``path``.

        POSIX ``cat`` under ``shell=True`` -- the same shell assumption the
        acceptance commands ``true``/``false`` already make. Nothing leaves
        the machine: the channel's destination is a file in the temporary
        directory.
        """
        return "cat > {0}".format(shlex.quote(path))

    def _set_notify_cmd(self, value: Optional[str]) -> None:
        """Point ``$MORPH_NOTIFY_CMD`` at ``value``; restore it afterwards.

        ``None`` REMOVES the variable -- the no-channel case -- even if the
        ambient environment carried one, so a developer's own configuration
        cannot turn a no-channel test into a channel test by accident. The
        previous environment comes back in an ``addCleanup``, whatever the
        test does: a leaked variable would re-channel every later test.
        """
        had = ENV_VAR in os.environ
        original = os.environ.get(ENV_VAR)
        if value is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = value

        def restore() -> None:
            if had:
                os.environ[ENV_VAR] = original
            else:
                os.environ.pop(ENV_VAR, None)

        self.addCleanup(restore)

    def _run(self, fake: FakeBackend,
             log: Optional[Callable[[str], None]] = None,
             root: Optional[str] = None) -> Tuple[dict, int]:
        """One headless run against a root, this test's clock, the ambient env."""
        return cli_cycle.run(root=self.root if root is None else root,
                             nogit=True, backend=fake, log=log,
                             sleep=self.sleep, now=self.now)

    def _assert_json_round_trips(self, payload: dict, note: str) -> None:
        self.assertEqual(json.loads(json.dumps(payload)), payload, msg=note)

    # -- the wiring ----------------------------------------------------------


    def test_finished_run_writes_its_line_to_the_configured_channel(self) -> None:
        """The channel gets one line: this run's deck id, its mixed counts."""
        self._add_cards(
            _card("alpha", "alpha.py", "Write alpha.py.", acceptance="true"),
            _card("beta", "beta.py", "Write beta.py.", acceptance="false"))
        body = _fenced("VALUE = 2")
        fake = FakeBackend({"alpha": _fenced("VALUE = 1"),
                            "beta": body, "beta.r1": body, "beta.r2": body})
        received = self._notify_path(self.root)
        self._set_notify_cmd(self._cat_command(received))

        payload, code = self._run(fake)

        self.assertEqual(code, EXIT_INCOMPLETE, msg=(
            "the deck's premise: one card written and one failed through "
            "every retry, so the counts the line must carry are real, mixed "
            "counts -- and the notification fires on the incomplete split "
            "too, not only on success"))
        self.assertTrue(os.path.isfile(received), msg=(
            "a finished run must have pushed its line through the configured "
            "channel: the file the channel writes must exist"))
        with open(received, "r", encoding="utf-8") as handle:
            received_line = handle.read()
        self.assertEqual(len(received_line.splitlines()), 1, msg=(
            "the notification is ONE line, no newline inside it: it travels "
            "through channels that deliver one line as one message"))
        self.assertFalse(received_line.endswith("\n"), msg=(
            "the message carries no newline of its own"))
        pattern = (
            r"\Amrph run (?P<deck>\S+) \(no branch\): "
            r"1 written, 1 failed, 0 skipped; "
            r"\d+ generations?, (?P<minutes>\d+|\?) min\Z"
        )
        match = re.match(pattern, received_line)
        self.assertIsNotNone(match, msg=(
            "the line must be the run-summary shape with THIS run's counts "
            "in it; got {0!r}".format(received_line)))
        assert match is not None  # narrowed by the assertion above
        self.assertEqual(match.group("deck"), payload["deck_id"], msg=(
            "the line must carry this run's deck id"))
        self.assertEqual(self.naps, [], msg=(
            "the fake completes on the first poll and the channel is a local "
            "pipe: nothing may have slept, on the real clock or the injected "
            "one"))

    def test_the_same_line_reaches_the_log(self) -> None:
        """The log carries the very line the channel was handed."""
        self._add_cards(*_solo_deck())
        fake = FakeBackend({"solo": _fenced("SOLO = 1")})
        received = self._notify_path(self.root)
        self._set_notify_cmd(self._cat_command(received))
        lines: List[str] = []
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            payload, code = self._run(fake, log=lines.append)

        self.assertEqual(code, EXIT_OK, msg=(
            "the run itself must succeed, so the assertions below are about "
            "where the summary went, not about a failure"))
        with open(received, "r", encoding="utf-8") as handle:
            received_line = handle.read()
        self.assertTrue(received_line, msg=(
            "the channel must have received the run's line at all"))
        self.assertIn(received_line, lines, msg=(
            "the log must carry the very line the channel was handed: the "
            "operator's terminal deserves the summary it just sent elsewhere"))
        self.assertIn(payload["deck_id"], received_line, msg=(
            "the summary names the run it is about"))
        self.assertEqual(captured.getvalue(), "", msg=(
            "nothing may leak to stdout -- not the summary, not the "
            "channel's output: stdout belongs to the caller's one JSON "
            "document"))

    def test_no_channel_configured_changes_nothing(self) -> None:
        """$MORPH_NOTIFY_CMD unset -- the normal case: same payload, same code."""
        # Two identical decks in two roots: one runs with NO channel
        # configured, one with a working channel. The notification must not
        # be able to tell the two apart from the outside.
        self._set_notify_cmd(None)
        self._add_cards(*_solo_deck())
        lines: List[str] = []
        quiet_payload, quiet_code = self._run(
            FakeBackend({"solo": _fenced("SOLO = 1")}), log=lines.append)

        channel_root = self._fresh_root()
        DeckStore(channel_root).add_cards(_solo_deck())
        received = self._notify_path(channel_root)
        self._set_notify_cmd(self._cat_command(received))
        channel_payload, channel_code = self._run(
            FakeBackend({"solo": _fenced("SOLO = 1")}), root=channel_root)

        self.assertTrue(os.path.isfile(received), msg=(
            "premise: the working channel really received its line, so the "
            "comparison below is against a run that did speak"))
        self.assertEqual(quiet_code, EXIT_OK, msg=(
            "no channel configured is not an error: the run must complete "
            "with the exit code it always had, raising nothing"))
        self.assertEqual(channel_code, quiet_code, msg=(
            "the channel's presence must not move the exit code"))
        self.assertEqual(set(quiet_payload), set(channel_payload), msg=(
            "the notification adds no key to the payload and removes none: "
            "the no-channel payload has exactly the keys a run always had"))
        self.assertEqual(quiet_payload["counts"], channel_payload["counts"],
                         msg="the counts must not move either")
        self.assertEqual(quiet_payload["counts"]["written"], 1, msg=(
            "and the deck's one card is still written"))
        self.assertEqual(quiet_payload["generations"],
                         channel_payload["generations"], msg=(
            "the recorded generations must not move either"))
        self.assertTrue(any(line.startswith("mrph run ") for line in lines),
                        msg=(
            "even with no channel, the log still gets the run's summary: the "
            "terminal's line does not depend on the channel"))
        self._assert_json_round_trips(
            quiet_payload, "the no-channel payload must survive json.dumps")

    def test_a_refusing_channel_changes_nothing(self) -> None:
        """A channel that exits non-zero costs the notification, not the run."""
        self._set_notify_cmd("false")  # the channel that always refuses: exit 1
        self._add_cards(*_solo_deck())
        refusing_payload, refusing_code = self._run(
            FakeBackend({"solo": _fenced("SOLO = 1")}))

        channel_root = self._fresh_root()
        DeckStore(channel_root).add_cards(_solo_deck())
        received = self._notify_path(channel_root)
        self._set_notify_cmd(self._cat_command(received))
        working_payload, working_code = self._run(
            FakeBackend({"solo": _fenced("SOLO = 1")}), root=channel_root)

        self.assertTrue(os.path.isfile(received), msg=(
            "premise: the working channel really received its line, so the "
            "comparison below is against a run that did speak"))
        self.assertEqual(refusing_code, EXIT_OK, msg=(
            "a channel that exits non-zero must not move the run's exit "
            "code, and nothing may be raised"))
        self.assertEqual(working_code, refusing_code, msg=(
            "a refusing channel and a working channel must leave the run "
            "indistinguishable from the outside"))
        self.assertEqual(set(refusing_payload), set(working_payload), msg=(
            "a refusing channel adds no key to the payload and removes none"))
        self.assertEqual(refusing_payload["counts"],
                         working_payload["counts"], msg=(
            "the counts must not move either"))


if __name__ == "__main__":
    unittest.main()
