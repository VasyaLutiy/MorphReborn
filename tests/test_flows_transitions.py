"""
Transition-level regressions for the deck commands.

Phase 7: one bad card must not kill the CLI.

``/submit``, ``/collect`` and ``/nightly`` used to catch only ``CardError`` /
``DeckError`` / ``StoreError``. Anything else -- a ``FileNotFoundError`` from a
card targeting a package that did not exist yet, a provider transport error, a
bug inside a morph body -- propagated out of the transition, through the state
machine, and terminated the whole ``mrph`` process. An operator collecting an
overnight deck lost the session to a single card.

Two later defects are pinned down in the same style, at the bottom of the file:
``/collect wait`` (a cloud queue runs 10-40 minutes, and a ``/collect`` that
prints nothing for an hour cannot be told from a hung one) and the interactive
save path's handling of an answer cut off inside a code fence (which used to be
written to disk verbatim, ```` ```python ```` line and all).

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
from unittest import mock

from cards.cli_wait import ResilientBackend
from cards.store import DeckStore, collect_generation, submit_generation
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
    # Phase 7: the transitions surface the store's git commentary (the branch a
    # run opened, the commit each accepted card became) out of the log they
    # otherwise discard.
    git_notes = staticmethod(MorphBot.git_notes)
    deck_notes = staticmethod(MorphBot.deck_notes)
    split_run_flags = staticmethod(MorphBot.split_run_flags)

    def __init__(self, backend, label="fake-processor"):
        self._active_backend = backend
        self._label = label

    def resolve_batch_backend(self, text):
        return self._active_backend, self._label


def _action(text=None):
    context = _FakeContext()
    return {
        "update": {"effective_chat": {"id": 1}},
        "context": context,
        "text": text,
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

    def _run(self, transition, text=None):
        action = _action(text)
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
        self.assertIn("No outcome was recorded", text)
        # No run was recorded: no card has an outcome and the deck view still
        # reads as not-yet-submitted. (Phase 7: the run was OPENED before
        # run_deck -- that is what a branch has to be -- so the composition is
        # now in state.json; what must not be there is a result.)
        state = self.store.load_state()
        self.assertEqual(state["phase"], "idle")
        self.assertEqual(state["outcomes"], {})
        self.assertFalse(os.path.exists(os.path.join(".morph", "runs")))


class _QueuedBackend:
    """A cloud batch that sits in a queue before it completes.

    ``pending_polls`` is how many ``status`` calls each batch answers
    ``in_progress`` before reporting ``completed`` -- the thing ``/collect wait``
    exists for, and the thing a backend that is instantly ``completed`` cannot
    express. Responses are scripted by (variant) custom_id, as elsewhere.
    """

    def __init__(self, scripts=None, default_response=None, pending_polls=0):
        self.scripts = scripts or {}
        self.default_response = default_response or "```python\nOK = 1\n```"
        self.pending_polls = pending_polls
        self.polls = {}
        self.submissions = []

    def submit(self, requests):
        self.submissions.append(requests)
        return f"cloud-batch-{len(self.submissions)}"

    def status(self, batch_id):
        self.polls[batch_id] = self.polls.get(batch_id, 0) + 1
        if self.polls[batch_id] <= self.pending_polls:
            return "in_progress"
        return "completed"

    def collect(self, batch_id):
        index = int(batch_id.rsplit("-", 1)[1]) - 1
        return {
            request["custom_id"]: self.scripts.get(request["custom_id"],
                                                   self.default_response)
            for request in self.submissions[index]
        }


class _BlinkingBackend:
    """A queue whose first status poll dies the way a provider does.

    One ``ConnectionResetError`` on the first poll, normal answers from then
    on -- the transport blink that used to tear a whole ``/collect wait``
    down and lose the run behind it. Responses are the default fenced block,
    as in :class:`_QueuedBackend`.
    """

    def __init__(self):
        self.polls = 0
        self.submissions = []

    def submit(self, requests):
        self.submissions.append(requests)
        return "cloud-batch-1"

    def status(self, batch_id):
        self.polls += 1
        if self.polls == 1:
            raise ConnectionResetError(104, "Connection reset by peer")
        return "completed"

    def collect(self, batch_id):
        return {request["custom_id"]: "```python\nOK = 1\n```"
                for request in self.submissions[0]}


class _FakeClock:
    """The clock :class:`cards.cli_wait.ResilientBackend` waits on, faked.

    ``sleep`` records the backoff delay and advances ``now`` instead of
    sleeping, so a wait that backs off costs the suite no time at all.
    """

    def __init__(self):
        self.moment = 0.0
        self.slept = []

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.moment += seconds

    def now(self):
        return self.moment


class CollectWaitTests(unittest.TestCase):
    """``/collect`` polls once; ``/collect wait`` polls until the work lands."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-wait-")
        self.root = os.path.join(self.tmp, "miniproject")
        shutil.copytree(MINIPROJECT, self.root,
                        ignore=shutil.ignore_patterns("node_modules"))
        self.previous_cwd = os.getcwd()
        os.chdir(self.root)
        self.store = DeckStore(".")
        self.nested_calls = []

    def tearDown(self):
        os.chdir(self.previous_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def _nested(self, action):
        self.nested_calls.append(action)

    def _add(self, custom_id, target, **meta):
        self.store.add_card({
            "custom_id": custom_id,
            "meta": {"intent": "generate", "target": target,
                     "context_slice": ["util.py"], **meta},
            "instruction": "make it",
        })

    def _collect(self, backend, text=None):
        bot = _StubMorphBot(backend)
        action = _action(text)
        asyncio.get_event_loop().run_until_complete(
            MorphBot.build_collect_transition(bot, self._nested)(action))
        return action["context"].bot.messages

    def test_bare_collect_still_polls_exactly_once(self):
        self._add("card-a", "gen_a.py")
        backend = _QueuedBackend(pending_polls=5)
        submit_generation(self.store, backend, root=".", backend_label="fake",
                          log=lambda _l: None)

        messages = self._collect(backend, text="/collect")

        self.assertEqual(backend.polls["cloud-batch-1"], 1)
        self.assertIn("still in progress", "\n".join(messages))
        self.assertEqual(self.store.load_state()["phase"], "submitted")
        self.assertEqual(len(self.nested_calls), 1)

    def test_collect_wait_polls_until_the_generation_lands(self):
        self._add("card-a", "gen_a.py")
        backend = _QueuedBackend(pending_polls=2)
        submit_generation(self.store, backend, root=".", backend_label="fake",
                          log=lambda _l: None)

        with mock.patch("flows.morph.COLLECT_WAIT_POLL_SECONDS", 0):
            messages = self._collect(backend, text="/collect wait")

        text = "\n".join(messages)
        # One progress line per unfinished poll, each naming what is being
        # waited for, which batch, and for how long.
        waiting = [line for line in messages if line.startswith("mrph> Waiting for")]
        self.assertEqual(len(waiting), 2)
        self.assertIn("generation 1/1", waiting[0])
        self.assertIn("cloud-batch-1", waiting[0])
        self.assertIn("elapsed", waiting[0])
        self.assertIn("Collected generation 1/1", text)
        self.assertEqual(self.store.load_state()["phase"], "done")
        self.assertTrue(os.path.exists("gen_a.py"))
        self.assertEqual(len(self.nested_calls), 1)

    def test_collect_wait_carries_on_across_a_regeneration(self):
        # The wait must not stop at the retry batch -- that is the hour of
        # silence it was built to replace.
        self._add("card-c", "gen_c.py", acceptance="grep -q PASS gen_c.py")
        backend = _QueuedBackend(pending_polls=1, scripts={
            "card-c": "```python\nVALUE = 1\n```",     # fails acceptance
            "card-c.r1": "```python\nPASS = 1\n```",   # the regeneration passes
        })
        submit_generation(self.store, backend, root=".", backend_label="fake",
                          log=lambda _l: None)

        with mock.patch("flows.morph.COLLECT_WAIT_POLL_SECONDS", 0):
            messages = self._collect(backend, text="/collect wait")

        text = "\n".join(messages)
        self.assertIn("card-c failed acceptance -- regeneration 1/2 submitted", text)
        self.assertIn("regeneration 1/2 of card-c", text)
        self.assertIn("cloud-batch-2", text)
        self.assertIn("Collected generation 1/1", text)
        self.assertIn("card-c: written", text)
        # Exactly two batches: the generation and ONE regeneration.
        self.assertEqual(len(backend.submissions), 2)
        self.assertEqual(self.store.load_state()["phase"], "done")

    def test_bare_collect_reports_the_regeneration_it_submitted(self):
        self._add("card-c", "gen_c.py", acceptance="grep -q PASS gen_c.py")
        backend = _QueuedBackend(scripts={"card-c": "```python\nVALUE = 1\n```"})
        submit_generation(self.store, backend, root=".", backend_label="fake",
                          log=lambda _l: None)

        messages = self._collect(backend, text="/collect")

        text = "\n".join(messages)
        self.assertIn("regeneration 1/2 for card-c submitted", text)
        self.assertIn("/collect again", text)
        self.assertIn("costs nothing", text)
        self.assertEqual(len(backend.submissions), 2)

    def test_collect_wait_gives_up_at_the_timeout_and_leaves_the_batch(self):
        self._add("card-a", "gen_a.py")
        backend = _QueuedBackend(pending_polls=5)
        submit_generation(self.store, backend, root=".", backend_label="fake",
                          log=lambda _l: None)

        with mock.patch("flows.morph.COLLECT_WAIT_POLL_SECONDS", 0), \
                mock.patch("flows.morph.COLLECT_WAIT_TIMEOUT_SECONDS", 0):
            messages = self._collect(backend, text="/collect wait")

        text = "\n".join(messages)
        self.assertIn("Gave up waiting", text)
        self.assertIn("still in flight", text)
        # Nothing was collected and the batch is untouched: one poll, no writes.
        self.assertEqual(backend.polls["cloud-batch-1"], 1)
        self.assertEqual(self.store.load_state()["phase"], "submitted")
        self.assertFalse(os.path.exists("gen_a.py"))
        self.assertEqual(len(self.nested_calls), 1)

    def test_collect_wait_reports_an_unexpected_failure_instead_of_looping(self):
        # The wait now wraps its backend in cards.cli_wait.ResilientBackend, so
        # a poll that fails EVERY time is retried until the wait's budget is
        # spent. The budget is pinned to zero here -- a spent budget gives up
        # after the first attempt -- and the assertions stay what they were:
        # the operator is told the failure by name, the run state is untouched.
        self._add("card-a", "gen_a.py")
        submit_generation(self.store, _Exploding(on="never"), root=".",
                          backend_label="fake", log=lambda _l: None)

        with mock.patch("flows.morph.COLLECT_WAIT_POLL_SECONDS", 0), \
                mock.patch("flows.morph.COLLECT_WAIT_TIMEOUT_SECONDS", 0):
            messages = self._collect(_Exploding(on="status"), text="/collect wait")

        text = "\n".join(messages)
        self.assertIn("RuntimeError", text)
        self.assertIn("still in flight", text)
        self.assertEqual(self.store.load_state()["phase"], "submitted")
        self.assertEqual(len(self.nested_calls), 1)

    def test_collect_wait_survives_a_blinked_connection(self):
        # One failed GET in the small hours used to tear the whole wait down:
        # here the FIRST status poll raises ConnectionResetError and every poll
        # after it answers normally. The wrapper is built by the real class with
        # an injected fake clock -- the backoff delay is recorded, not slept --
        # so the wait lands the generation instead of reporting the failure, and
        # the suite pays no time for it.
        self._add("card-a", "gen_a.py")
        backend = _BlinkingBackend()
        submit_generation(self.store, backend, root=".", backend_label="fake",
                          log=lambda _l: None)

        clock = _FakeClock()

        def factory(inner, **kwargs):
            return ResilientBackend(inner, sleep=clock.sleep, now=clock.now,
                                    **kwargs)

        with mock.patch("flows.morph.COLLECT_WAIT_POLL_SECONDS", 0), \
                mock.patch("flows.morph.ResilientBackend", factory):
            messages = self._collect(backend, text="/collect wait")

        text = "\n".join(messages)
        self.assertIn("Collected generation 1/1", text)
        self.assertNotIn("Unexpected failure", text)
        # The blinked poll left its line instead of silence: the wrapper's log
        # reaches the chat.
        self.assertIn("batch status: ConnectionResetError", text)
        self.assertIn("retrying in 5s", text)
        # The backoff ran on the injected clock.
        self.assertEqual(clock.slept, [5.0])
        self.assertTrue(os.path.exists("gen_a.py"))
        self.assertEqual(self.store.load_state()["phase"], "done")
        self.assertEqual(len(self.nested_calls), 1)


# -- the interactive save path (/generate, /patch) ----------------------------


class _Job:
    def __init__(self, assigned):
        self.id = 7
        self.assigned = assigned
        self.was_queued = False


class _StubScheduler:
    """Launches immediately and records the release, like the real pool."""

    def __init__(self):
        self.released = []

    def attach_launch(self, job, launch):
        launch()

    def release(self, processor_ids):
        self.released.append(list(processor_ids))


class _JobContext:
    """An ``action["context"]`` that carries a job, as ``context_get`` reads it."""

    def __init__(self, job):
        self.bot = _FakeChatBot()
        self._job = job

    def get(self, name):
        if name == "job":
            return self._job
        raise KeyError(name)


class _SaveStubBot:
    """The slice of ``MorphBot`` that ``morph_and_save`` actually touches."""

    context_get = staticmethod(MorphBot.context_get)
    output_file_name = staticmethod(MorphBot.output_file_name)
    response_to_file_body = staticmethod(MorphBot.response_to_file_body)

    def __init__(self, responses):
        self.responses = responses
        self.registry = None
        self.scheduler = _StubScheduler()
        self.messages = []

    async def run_morphers(self, registry, processor_ids, dialog):
        return dict(self.responses)

    async def send_message(self, chat_id, text, **_kwargs):
        self.messages.append(text)


class TruncatedInteractiveSaveTests(unittest.TestCase):
    """A cut-off answer must be reported, not written out with its fence.

    ``/generate`` and ``/patch`` share ``morph_and_save``. An answer that opened
    ```` ```python ```` and was cut off before the closing fence used to fall
    through to "no fenced block -> write it verbatim", handing the user a file
    whose first line is ```` ```python ```` and which does not parse.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-cut-")
        self.previous_cwd = os.getcwd()
        os.chdir(self.tmp)
        self.nested_calls = []

    def tearDown(self):
        os.chdir(self.previous_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def _nested(self, action):
        self.nested_calls.append(action)

    def _save(self, responses, processor_ids, file_name="out.py"):
        bot = _SaveStubBot(responses)
        action = {
            "update": {"effective_chat": {"id": 1}},
            "context": _JobContext(_Job(processor_ids)),
            "text": None,
        }
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            MorphBot.morph_and_save(bot, action, None, file_name, self._nested))
        # ``morph_and_save`` returns to the menu immediately and finishes the
        # write in a background task; drain it before asserting.
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        loop.run_until_complete(asyncio.gather(*pending))
        return bot

    def test_a_cut_off_answer_saves_nothing_and_says_so(self):
        bot = self._save({"p1": "```python\nOK = 1\n# cut off here"}, ["p1"])

        self.assertFalse(os.path.exists("out.py"))
        self.assertIn("cut off mid-file", "\n".join(bot.messages))
        self.assertEqual(bot.scheduler.released, [["p1"]])

    def test_a_whole_answer_is_still_saved(self):
        bot = self._save({"p1": "```python\nOK = 1\n```"}, ["p1"])

        self.assertTrue(os.path.exists("out.py"))
        with open("out.py", encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "OK = 1\n")
        self.assertIn("was saved", "\n".join(bot.messages))

    def test_one_cut_off_processor_does_not_hide_the_others(self):
        bot = self._save({"p1": "```python\nCUT = 1",
                          "p2": "```python\nWHOLE = 2\n```"}, ["p1", "p2"])

        text = "\n".join(bot.messages)
        self.assertFalse(os.path.exists("out.p1.py"))
        self.assertTrue(os.path.exists("out.p2.py"))
        self.assertIn("Saved (parallel)", text)
        self.assertIn("cut off mid-file, nothing saved for: p1", text)


if __name__ == "__main__":
    unittest.main()

