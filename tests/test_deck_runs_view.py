"""``/deck runs <deck-id>``: reading one archived run, card by card.

Three things the argument promises, one test each:

* bare ``/deck runs`` is unchanged -- with nothing archived it still answers
  with the one listing line it always has, and it never reaches for a report
  that was not asked for;
* ``/deck runs <deck-id>`` over a real archive prints one indented line per
  card, in generation order, carrying exactly the numbers, sha and paths that
  ``cards.runs_view.format_run_report`` derives -- all of the per-card
  formatting lives there, the flow only reads the archived report and prints;
* an unknown identifier answers with one line naming the id and pointing at
  bare ``/deck runs`` -- and raises nothing: a typo in a deck id is a
  question, not a failure.

A fourth test pins the /help line, so the new argument stays discoverable.

The transitions are driven the way tests/test_deck_clear.py drives them, on
the loop ``asyncio.get_event_loop()`` returns -- never through
``asyncio.run()``, which unsets the policy's current loop on the way out
and would leave every later ``get_event_loop()`` in the process (the
helpers in test_flows_transitions.py, notably) raising "There is no
current event loop".
"""

import asyncio
import json
import os
import shutil
import tempfile
import unittest

from flows.morph import MorphBot


# A deck id shaped the way the store names its archives, and the archived
# report it would leave behind: .morph/runs/<deck-id>/report.json -- the file
# cards.runs_view.format_run_report is documented against. Two cards, in
# generation order: one that became a commit (its line carries the diff
# numbers and the truncated sha), one with no commit (a nogit run, say; its
# line carries dashes where those would go).
DECK_ID = "20260101-000000-abcdef01"

REPORT = {
    "deck_id": DECK_ID,
    "completed_at": "2026-01-01T00:10:00.000000",
    "branch": "morph/" + DECK_ID,
    "generations": [["gen-alpha"], ["gen-beta"]],
    "outcomes": {
        "gen-alpha": {
            "custom_id": "gen-alpha",
            "status": "written",
            "paths": ["alpha.py"],
            "reason": None,
            "attempts": 1,
            "winning_variant": 0,
            "commit": "0123456789abcdef",
            "diffstat": [{"path": "alpha.py", "insertions": 10, "deletions": 2}],
        },
        "gen-beta": {
            "custom_id": "gen-beta",
            "status": "written",
            "paths": ["alpha.py"],
            "reason": None,
            "attempts": 1,
            "winning_variant": 0,
        },
    },
}

# The lines the flow owes for those two outcomes -- the formatter's five
# fields, three-space joined, indented one level -- pinned here so the test
# fails if either the flow stops printing them or the formatter's contract
# drifts.
ALPHA_LINE = "    gen-alpha   written   +10 -2   0123456789   alpha.py"
BETA_LINE = "    gen-beta   written   -   -   alpha.py"


class _RecordingBot:
    """Stands in for the console bot and keeps every message it is handed."""

    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id=None, text=None, reply_markup=None):
        self.messages.append(text)


class _Context:
    """The slice of the console context the transitions read: the bot."""

    def __init__(self, bot):
        self.bot = bot


def _run(coro):
    """Drive one transition coroutine on the suite's event loop.

    WHY not ``asyncio.run()``: on the way out it calls
    ``asyncio.set_event_loop(None)``, and from then on every
    ``asyncio.get_event_loop()`` in the main thread raises -- which is how
    one test file can break tests in another. Implicit creation via
    ``get_event_loop()`` is the convention the rest of the suite runs on, so
    it is the convention here; the fallback only fires if something before
    these tests has already unset the loop, and it heals the policy instead
    of inheriting the breakage.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _bare_bot():
    """A MorphBot shell built without ``__init__``.

    The /deck transition reads no instance state -- no registry, no
    scheduler, no backend -- and the help transition is a static method, so
    skipping the constructor keeps these tests independent of ".env", of
    configured processors and of the console transport, while still running
    the real transition code.
    """
    return MorphBot.__new__(MorphBot)


def _action(text):
    """An ``action`` shaped the way the console bot hands one over, with the
    bot that will record what the transition sends."""
    bot = _RecordingBot()
    return bot, {
        "update": {"effective_chat": {"id": 7}},
        "text": text,
        "context": _Context(bot),
    }


def _archive_the_run():
    """Write one archived run into the current working directory: the report
    the ``/deck runs <deck-id>`` branch reads."""
    path = os.path.join(".morph", "runs", DECK_ID, "report.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(REPORT, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


class DeckRunsViewTests(unittest.TestCase):
    """``/deck runs`` with and without an id, against a temporary cwd."""

    def setUp(self):
        self._previous_cwd = os.getcwd()
        self.root = tempfile.mkdtemp(prefix="mrph-deck-runs-view-")
        os.chdir(self.root)

    def tearDown(self):
        os.chdir(self._previous_cwd)
        shutil.rmtree(self.root, ignore_errors=True)

    def test_bare_deck_runs_is_unchanged(self):
        # Nothing archived (an empty runs/ directory is what a project that
        # has never run has): the bare listing answers with the line it has
        # always answered with.
        os.makedirs(os.path.join(".morph", "runs"), exist_ok=True)
        bot, action = _action("/deck runs")
        _run(_bare_bot().build_deck_transition()(action))

        self.assertEqual(1, len(bot.messages))
        text = bot.messages[0]
        self.assertTrue(text.startswith("mrph>"))
        self.assertIn("No archived runs yet", text)
        # ... and the new branch's answer is nowhere in it: no id was given,
        # so nothing was "not found" and no card-by-card report was printed.
        self.assertNotIn("was not found", text)
        self.assertNotIn("card by card", text)

    def test_deck_runs_with_id_prints_one_indented_line_per_card(self):
        _archive_the_run()

        bot, action = _action(f"/deck runs {DECK_ID}")
        _run(_bare_bot().build_deck_transition()(action))

        self.assertEqual(1, len(bot.messages))
        text = bot.messages[0]
        self.assertTrue(text.startswith("mrph>"))
        self.assertIn(DECK_ID, text)
        # One line per card, in generation order, indented under the
        # heading -- and the fields are the formatter's, verbatim.
        card_lines = [line for line in text.splitlines()
                      if line.strip().startswith("gen-")]
        self.assertEqual([ALPHA_LINE, BETA_LINE], card_lines)
        # The not-found answer is nowhere in it: the run WAS found.
        self.assertNotIn("was not found", text)

    def test_deck_runs_unknown_id_answers_without_raising(self):
        # One run IS archived -- the id below is a typo away from it, the
        # everyday case the not-found line exists for.
        _archive_the_run()

        bot, action = _action("/deck runs no-such-run")
        # The call itself is half the assertion: nothing may propagate out of
        # the transition for an id that has no archive.
        _run(_bare_bot().build_deck_transition()(action))

        self.assertEqual(1, len(bot.messages))
        line = bot.messages[0]
        self.assertTrue(line.startswith("mrph>"))
        self.assertIn("no-such-run", line)
        self.assertIn("was not found", line)
        # ... and it points at the listing that shows what there is.
        self.assertIn("/deck runs", line)

    def test_help_names_the_runs_id_argument(self):
        bot, action = _action("/help")
        _run(MorphBot.build_help_transition()(action))

        text = "\n".join(bot.messages)
        # The new argument is discoverable next to the bare listing it
        # extends, in the voice of the /deck line's other arguments.
        self.assertIn("/deck runs <deck-id>", text)
        self.assertIn("/deck runs", text)
