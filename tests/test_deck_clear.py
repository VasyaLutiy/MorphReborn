"""``/deck clear``: emptying the backlog from the CLI.

Three things the argument promises, one test each:

* a clear with no run in flight empties the backlog -- .morph/deck.json is
  gone -- and leaves .morph/state.json byte for byte alone, answering with
  the one line that says so;
* a clear while a generation is in flight (run phase ``submitted``) is
  refused with a line that names /deck reset, and nothing changes: the
  batch's results can only land on cards that still exist;
* the /help text names the new argument, so it can be discovered.

The transitions are driven the way the rest of this suite drives them, on
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

from cards.store import DeckStore
from flows.morph import MorphBot


# Two cards in the nested backlog shape ``cards.deck.load_deck`` reads: a
# generator and a patch depending on it, so what gets cleared (or refused)
# is a real two-card backlog and not a stub.
ALPHA = {
    "custom_id": "gen-alpha",
    "meta": {"intent": "generate", "target": "alpha.py"},
    "instruction": "Write alpha.py with one function.",
}
BETA = {
    "custom_id": "gen-beta",
    "meta": {"intent": "patch", "target": "alpha.py", "depends_on": ["gen-alpha"]},
    "instruction": "Add a second function to alpha.py.",
}

# The run states the two clear outcomes are tried against: "idle" is safe to
# clear the backlog under; "submitted" is a batch sitting in a queue, the
# case the refusal exists for.
IDLE_STATE = {
    "phase": "idle",
    "generation_index": 0,
    "generations": [["gen-alpha"], ["gen-beta"]],
    "batch_id": None,
    "backend_label": None,
    "submitted_ids": [],
    "retries": {},
    "outcomes": {},
    "deck_id": "20260101-000000-abcdef01",
    "branch": None,
    "batch_ids": [],
}
IN_FLIGHT_STATE = dict(IDLE_STATE, phase="submitted", batch_id="batch-test-1",
                       backend_label="k80", submitted_ids=["gen-alpha"],
                       batch_ids=["batch-test-1"])


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

    The clear branch of the /deck transition reads no instance state -- no
    registry, no scheduler, no backend -- and the help transition is a
    static method, so skipping the constructor keeps these tests independent
    of ".env", of configured processors and of the console transport, while
    still running the real transition code.
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


def _write_json(relative_path, payload):
    os.makedirs(os.path.dirname(relative_path), exist_ok=True)
    with open(relative_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _read_bytes(relative_path):
    with open(relative_path, "rb") as handle:
        return handle.read()


class DeckClearTests(unittest.TestCase):
    """``/deck clear`` against a real .morph/ directory in a temporary cwd."""

    def setUp(self):
        self._previous_cwd = os.getcwd()
        self.root = tempfile.mkdtemp(prefix="mrph-deck-clear-")
        os.chdir(self.root)

    def tearDown(self):
        os.chdir(self._previous_cwd)
        shutil.rmtree(self.root, ignore_errors=True)

    def test_clear_empties_the_backlog_and_leaves_the_run_state_untouched(self):
        _write_json(os.path.join(".morph", "deck.json"), [ALPHA, BETA])
        _write_json(os.path.join(".morph", "state.json"), IDLE_STATE)
        state_before = _read_bytes(os.path.join(".morph", "state.json"))

        bot, action = _action("/deck clear")
        _run(_bare_bot().build_deck_transition()(action))

        # The backlog is gone -- the file removed, the store reading an empty
        # deck -- and the answer is the one line naming what happened to each
        # of the two files.
        self.assertFalse(os.path.exists(os.path.join(".morph", "deck.json")))
        self.assertEqual([], DeckStore(".").load_cards())
        self.assertEqual(1, len(bot.messages))
        line = bot.messages[0]
        self.assertTrue(line.startswith("mrph>"))
        self.assertIn("Backlog emptied", line)
        self.assertIn(".morph/deck.json", line)
        self.assertIn("run state is untouched", line)
        # ... and the run state is, in fact, untouched: byte for byte.
        self.assertEqual(state_before,
                         _read_bytes(os.path.join(".morph", "state.json")))

    def test_clear_refuses_while_a_generation_is_in_flight(self):
        _write_json(os.path.join(".morph", "deck.json"), [ALPHA, BETA])
        _write_json(os.path.join(".morph", "state.json"), IN_FLIGHT_STATE)
        deck_before = _read_bytes(os.path.join(".morph", "deck.json"))
        state_before = _read_bytes(os.path.join(".morph", "state.json"))

        bot, action = _action("/deck clear")
        _run(_bare_bot().build_deck_transition()(action))

        # The refusal names the way out: /deck reset returns the cards to
        # pending without touching the batch, and the clear works after it.
        self.assertEqual(1, len(bot.messages))
        line = bot.messages[0]
        self.assertTrue(line.startswith("mrph>"))
        self.assertIn("in flight", line)
        self.assertIn("/deck reset", line)
        # ... and nothing changed, because the batch's results can only land
        # on cards that still exist: both files survive byte for byte.
        self.assertEqual(deck_before,
                         _read_bytes(os.path.join(".morph", "deck.json")))
        self.assertEqual(state_before,
                         _read_bytes(os.path.join(".morph", "state.json")))
        cards = DeckStore(".").load_cards()
        self.assertEqual(["gen-alpha", "gen-beta"], [card.custom_id for card in cards])
        self.assertEqual("submitted", DeckStore(".").load_state()["phase"])

    def test_help_names_the_clear_argument(self):
        bot, action = _action("/help")
        _run(MorphBot.build_help_transition()(action))

        text = "\n".join(bot.messages)
        # The new argument is discoverable, and the two it sits alongside are
        # still named next to it.
        self.assertIn("/deck clear", text)
        self.assertIn("/deck reset", text)
        self.assertIn("/deck runs", text)
