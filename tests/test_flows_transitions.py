"""
Phase 7 regression: one bad card must not kill the CLI.

``/submit``, ``/collect`` and ``/nightly`` used to catch only ``CardError`` /
``DeckError`` / ``StoreError``. Anything else -- a ``FileNotFoundError`` from a
card targeting a package that did not exist yet, a provider transport error, a
bug inside a morph body -- propagated out of the transition, through the state
machine, and terminated the whole ``mrph`` process. An operator collecting an
overnight deck lost the session to a single card.

The transitions are driven WITHOUT a ConsoleBot: ``build_*_transition`` are
ordinary methods returning a closure over ``self``, so a stub carrying the three
attributes they touch (``_active_backend``, ``resolve_batch_backend``,
``report_unexpected``) is enough, and the ``action`` dict is the two keys the
closures read. The store work is real -- a throwaway copy of
``tests/fixtures/miniproject`` as the current directory, because the transitions
build their ``DeckStore(".")`` from the process cwd.
"""

import asyncio
import os
import shutil
import tempfile
import unittest

from cards.store import DeckStore, submit_generation
from flows.morph import MorphBot


MINIPROJECT = os.path.join("tests", "fixtures", "miniproject")


class _Exploding:
    """A batch backend that fails the way a provider or a morph body does.

    Not a ``StoreError``/``CardError``: the point is the exception class nobody
    anticipated. ``submit`` succeeds or explodes depending on ``on``, so the same
    fake serves the /submit path (explode on submit) and the /collect path
    (submit fine, explode on the status poll).
    """

    def __init__(self, on="status"):
        self.on = on
        self.submitted = []

    def _boom(self, where):
        if self.on == where:
            raise RuntimeError("provider connection reset")

    def submit(self, requests):
        self._boom("submit")
        self.submitted.append(requests)
        return "fake-batch-1"

    def status(self, batch_id):
        self._boom("status")
        return "completed"

    def collect(self, batch_id):
        self._boom("collect")
        return {request["custom_id"]: "```python\nOK = 1\n```"
                for request in self.submitted[0]}


class _FakeChatBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **_kwargs):
        self.messages.append(text)


class _FakeContext:
    def __init__(self):
        self.bot = _FakeChatBot()


class _StubMorphBot:
    """The slice of ``MorphBot`` the three deck transitions actually use."""

    report_unexpected = staticmethod(MorphBot.report_unexpected)

    def __init__(self, backend, label="fake-processor"):
        self._active_backend = backend
        self._label = label

    def resolve_batch_backend(self, text):
        return self._active_backend, self._label


def _action():
    context = _FakeContext()
    return {
        "update": {"effective_chat": {"id": 1}},
        "context": context,
        "text": None,
    }


class UnexpectedFailureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-flows-")
        self.root = os.path.join(self.tmp, "miniproject")
        shutil.copytree(MINIPROJECT, self.root,
                        ignore=shutil.ignore_patterns("node_modules"))
        self.previous_cwd = os.getcwd()
        os.chdir(self.root)
        self.store = DeckStore(".")
        self.store.add_card({
            "custom_id": "card-a",
            "meta": {"intent": "generate", "target": "gen_a.py",
                     "context_slice": ["util.py"]},
            "instruction": "make a",
        })
        self.nested_calls = []

    def tearDown(self):
        os.chdir(self.previous_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def _nested(self, action):
        self.nested_calls.append(action)

    def _run(self, transition):
        action = _action()
        asyncio.get_event_loop().run_until_complete(transition(action))
        return action["context"].bot.messages

    # -- /collect -------------------------------------------------------------

    def test_collect_reports_an_unexpected_failure_and_keeps_the_session(self):
        backend = _Exploding(on="status")
        submit_generation(self.store, _Exploding(on="never"), root=".",
                          backend_label="fake-processor", log=lambda _l: None)
        bot = _StubMorphBot(backend)

        messages = self._run(MorphBot.build_collect_transition(bot, self._nested))

        # Reported, not raised, and the flow returned to the menu.
        self.assertEqual(len(self.nested_calls), 1)
        text = "\n".join(messages)
        self.assertIn("mrph>", text)
        self.assertIn("RuntimeError", text)
        self.assertIn("provider connection reset", text)
        self.assertNotIn("Traceback", text)
        # The user is told the batch is still collectable.
        self.assertIn("still in flight", text)
        self.assertIn("/collect", text)

    def test_collect_failure_leaves_the_run_state_untouched(self):
        submit_generation(self.store, _Exploding(on="never"), root=".",
                          backend_label="fake-processor", log=lambda _l: None)
        before = self.store.load_state()
        bot = _StubMorphBot(_Exploding(on="status"))

        self._run(MorphBot.build_collect_transition(bot, self._nested))

        after = self.store.load_state()
        self.assertEqual(after["phase"], "submitted")
        self.assertEqual(after, before)
        # Nothing was written for the card, and the backend is kept for a retry.
        self.assertFalse(os.path.exists("gen_a.py"))
        self.assertIsNotNone(bot._active_backend)

    def test_collect_retry_after_the_failure_completes_the_generation(self):
        # The whole point of not dying: the very next /collect works.
        healthy = _Exploding(on="never")
        submit_generation(self.store, healthy, root=".",
                          backend_label="fake-processor", log=lambda _l: None)

        failing = _StubMorphBot(_Exploding(on="status"))
        self._run(MorphBot.build_collect_transition(failing, self._nested))

        retrying = _StubMorphBot(healthy)
        messages = self._run(MorphBot.build_collect_transition(retrying, self._nested))

        self.assertIn("Collected generation 1/1", "\n".join(messages))
        self.assertEqual(self.store.load_state()["phase"], "done")
        self.assertTrue(os.path.exists("gen_a.py"))

    # -- /submit --------------------------------------------------------------

    def test_submit_reports_an_unexpected_failure_and_stays_pending(self):
        # Nothing is "active" yet -- a fresh run -- but /submit still resolves a
        # backend from the command line, and that one explodes on submit.
        bot = _StubMorphBot(None)
        bot.resolve_batch_backend = lambda _text: (_Exploding(on="submit"),
                                                   "fake-processor")

        messages = self._run(MorphBot.build_submit_transition(bot, self._nested))

        self.assertEqual(len(self.nested_calls), 1)
        text = "\n".join(messages)
        self.assertIn("RuntimeError", text)
        self.assertIn("/submit can be retried", text)
        self.assertEqual(self.store.load_state()["phase"], "idle")
        self.assertIsNone(bot._active_backend)

    # -- /nightly -------------------------------------------------------------

    def test_nightly_reports_an_unexpected_failure_and_records_nothing(self):
        bot = _StubMorphBot(None)
        bot.resolve_batch_backend = lambda _text: (_Exploding(on="submit"),
                                                   "fake-processor")

        messages = self._run(MorphBot.build_nightly_transition(bot, self._nested))

        self.assertEqual(len(self.nested_calls), 1)
        text = "\n".join(messages)
        self.assertIn("RuntimeError", text)
        self.assertIn("the deck is unchanged", text)
        # No run was recorded: the deck view still reads as never started.
        self.assertEqual(self.store.load_state()["phase"], "idle")


if __name__ == "__main__":
    unittest.main()
