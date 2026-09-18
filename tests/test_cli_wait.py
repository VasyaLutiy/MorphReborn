"""``cards/cli_wait.py``: waits that survive a blinking network, on a fake clock.

Every wait in the module runs off injected ``sleep``/``now`` callables, and
every test here exploits that: the fake sleep just advances a fake timeline
and records what was asked of it, so the waits that in production span six
hours play out in microseconds. No test sleeps for real, and the whole file
runs in well under a second.

One decision per test:

* the backoff schedule doubles and then holds at its cap forever;
* ``retry`` returns a first-try success untouched; rides out two transient
  failures and answers on the third attempt; re-raises a
  :class:`KeyboardInterrupt` -- and any caller-declared permanent class --
  immediately, without a single sleep; raises :class:`WaitTimeout` once the
  injected clock passes the deadline, with a message an operator can
  diagnose; and never sleeps past that deadline;
* :class:`ResilientBackend` retries a flaky ``status`` into a real answer,
  spends ONE deadline across calls -- the second call inherits the reduced
  budget, not a fresh one -- exposes ``remaining`` read-only, and delegates
  ``submit`` and unknown attributes to the wrapped backend;
* :func:`wait_until` makes its first call before any sleep, loops to done,
  and raises ``WaitTimeout`` on a clock that expires -- while still giving
  the step its one immediate chance even when the budget is already zero.
"""

import itertools
import unittest

from cards.cli_wait import (
    PERMANENT,
    ResilientBackend,
    WaitTimeout,
    backoff_delays,
    retry,
    wait_until,
)


class _Clock:
    """A fake monotonic clock whose ``sleep`` just moves the timeline.

    ``now`` has the shape of ``time.monotonic``; ``sleep`` records every
    requested delay and advances the clock by exactly that much, so a test
    can assert on the sleeps a wait took -- and on the clock time they left
    behind -- while the process never blocks.
    """

    def __init__(self, start=1000.0):
        self.time = float(start)
        self.sleeps = []

    def now(self):
        return self.time

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.time += seconds


class _Probe:
    """A bare callable that fails a fixed number of times, then succeeds.

    Records the clock time of every attempt. A ``failures`` of ``10**9``
    means "always fails" -- the shape of a provider that never answers.
    """

    def __init__(self, clock, failures, value=None, error=None):
        self._clock = clock
        self._failures = failures
        self._value = value
        self._error = error if error is not None else RuntimeError("transient")
        self.attempts = []

    def __call__(self):
        self.attempts.append(self._clock.now())
        if len(self.attempts) <= self._failures:
            raise self._error
        return self._value


class _FakeBackend:
    """A duck-typed batch backend with scripted per-method failures.

    Each of ``submit``/``status``/``collect`` fails the number of times named
    for it in ``failures`` with a transport-shaped ConnectionError -- the
    blink this module exists to survive -- then answers: ``submit`` returns
    ``"batch-1"``, ``status`` returns ``status_value``, ``collect`` returns a
    one-card result dict. Every attempt is timestamped on the fake clock, so
    a test can see WHEN the retries ran as well as how many. ``model`` and
    ``failure_reason`` are extra attributes for the ``__getattr__`` test.
    """

    def __init__(self, clock, failures=None, status_value="in_progress"):
        self._clock = clock
        self._failures = {"submit": 0, "status": 0, "collect": 0}
        if failures:
            self._failures.update(failures)
        self._status_value = status_value
        self.model = "qwen2.5-32b-instruct"
        self.attempts = {"submit": [], "status": [], "collect": []}

    def submit(self, requests):
        return self._attempt("submit", "batch-1")

    def status(self, batch_id):
        return self._attempt("status", self._status_value)

    def collect(self, batch_id):
        return self._attempt("collect", {"gen-a": "the collected text"})

    def failure_reason(self, batch_id):
        return "quota exceeded"

    def _attempt(self, name, value):
        self.attempts[name].append(self._clock.now())
        if len(self.attempts[name]) <= self._failures[name]:
            raise ConnectionError(f"{name}: connection reset by peer")
        return value


class BackoffDelaysTests(unittest.TestCase):
    """The schedule: double from the start, then hold at the cap forever."""

    def test_doubles_then_holds_at_cap(self):
        delays = list(itertools.islice(backoff_delays(5.0, 120.0), 8))
        self.assertEqual(
            [5.0, 10.0, 20.0, 40.0, 80.0, 120.0, 120.0, 120.0], delays,
            "the schedule doubles from the start (5, 10, 20, 40, 80), caps "
            "the next step (160 -> 120) and then yields the cap forever")

    def test_start_at_the_cap_yields_the_cap_from_the_first_step(self):
        delays = list(itertools.islice(backoff_delays(120.0, 120.0), 3))
        self.assertEqual(
            [120.0, 120.0, 120.0], delays,
            "a start already at the cap never yields anything below it")


class RetryTests(unittest.TestCase):
    """``retry``: one decision per test, all on the fake clock."""

    def test_first_try_success_returns_the_value_without_sleeping(self):
        clock = _Clock()
        probe = _Probe(clock, failures=0, value="batch-1")
        self.assertEqual(
            "batch-1",
            retry(probe, timeout=60.0, sleep=clock.sleep, now=clock.now),
            "the call's return value comes straight through")
        self.assertEqual(1, len(probe.attempts),
                         "a first-try success is not retried")
        self.assertEqual([], clock.sleeps,
                         "a first-try success sleeps not at all")

    def test_survives_two_transient_failures_and_answers_on_the_third(self):
        clock = _Clock()
        probe = _Probe(clock, failures=2, value="third time lucky",
                       error=ConnectionError("reset by peer"))
        reported = []
        value = retry(probe, timeout=600.0, start_delay=2.0, max_delay=100.0,
                      sleep=clock.sleep, now=clock.now, log=reported.append)
        self.assertEqual("third time lucky", value,
                         "the value from the third attempt is returned")
        self.assertEqual(3, len(probe.attempts),
                         "two transient failures were retried; the third "
                         "attempt answered")
        self.assertEqual([2.0, 4.0], clock.sleeps,
                         "the delay doubles between retries: 2s then 4s")
        self.assertEqual(2, len(reported),
                         "each transient failure is reported through log; "
                         "a success reports nothing")
        self.assertIn("ConnectionError", reported[0],
                      "the report names the exception class")

    def test_keyboard_interrupt_is_reraised_immediately(self):
        clock = _Clock()
        probe = _Probe(clock, failures=10 ** 9, error=KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt,
                               msg="a KeyboardInterrupt must never be retried"):
            retry(probe, timeout=600.0, sleep=clock.sleep, now=clock.now)
        self.assertEqual(1, len(probe.attempts),
                         "exactly one attempt: the interrupt re-raises at once")
        self.assertEqual([], clock.sleeps,
                         "no backoff sleep happens before the re-raise")

    def test_caller_declared_permanent_classes_are_not_retried(self):
        clock = _Clock()

        class NoBatchEndpoint(RuntimeError):
            """A configuration mistake: retrying a wrong model cannot help."""

        probe = _Probe(clock, failures=10 ** 9,
                       error=NoBatchEndpoint("slug has no :batch endpoint"))
        with self.assertRaises(
                NoBatchEndpoint,
                msg="a caller-declared permanent class re-raises immediately"):
            retry(probe, timeout=600.0, permanent=(NoBatchEndpoint,),
                  sleep=clock.sleep, now=clock.now)
        self.assertEqual(1, len(probe.attempts),
                         "a permanent error is not given a second attempt")
        self.assertEqual([], clock.sleeps,
                         "no backoff sleep happens before the re-raise")

    def test_permanent_names_the_unretryable_classes(self):
        for klass in (KeyboardInterrupt, SystemExit, MemoryError):
            self.assertIn(
                klass, PERMANENT,
                f"{klass.__name__} must never be retried: retrying cannot help")

    def test_times_out_once_the_injected_clock_passes_the_deadline(self):
        clock = _Clock(0.0)
        probe = _Probe(clock, failures=10 ** 9, error=ValueError("queue jammed"))
        with self.assertRaises(
                WaitTimeout,
                msg="a deadline spent on failures ends in WaitTimeout") as caught:
            retry(probe, timeout=45.0, start_delay=20.0, max_delay=20.0,
                  sleep=clock.sleep, now=clock.now, what="batch status")
        timeout_error = caught.exception
        self.assertEqual(45.0, timeout_error.elapsed,
                         "elapsed is the seconds actually waited, "
                         "deadline-exact")
        self.assertIn("batch status", str(timeout_error),
                      "the message names what was being waited for")
        self.assertIn("45.0s", str(timeout_error),
                      "the message names the elapsed seconds")
        self.assertIn("ValueError", str(timeout_error),
                      "the message carries the last exception's class")
        self.assertIn("queue jammed", str(timeout_error),
                      "the message carries the last exception's text -- "
                      "the diagnosis")
        self.assertEqual(4, len(probe.attempts),
                         "attempts continued until the clock reached the "
                         "deadline: t=0, 20, 40, 45")

    def test_never_sleeps_past_the_deadline(self):
        clock = _Clock(0.0)
        probe = _Probe(clock, failures=10 ** 9, error=ConnectionError("blink"))
        with self.assertRaises(WaitTimeout,
                               msg="the spent deadline ends the wait"):
            retry(probe, timeout=50.0, start_delay=30.0, max_delay=30.0,
                  sleep=clock.sleep, now=clock.now, what="the deck")
        self.assertEqual([30.0, 20.0], clock.sleeps,
                         "the second sleep is clamped to the 20 seconds "
                         "remaining -- its unclamped backoff step is 30")
        self.assertEqual(50.0, clock.time,
                         "the clock never moved past the deadline: "
                         "30 + 20 = 50, not 60")


class ResilientBackendTests(unittest.TestCase):
    """The proxy: retry each protocol call under ONE shared deadline."""

    def test_status_retries_a_flaky_backend_and_returns_the_status(self):
        clock = _Clock()
        backend = _FakeBackend(clock, failures={"status": 2})
        reported = []
        resilient = ResilientBackend(
            backend, timeout=3600.0, start_delay=5.0, max_delay=120.0,
            sleep=clock.sleep, now=clock.now, log=reported.append)
        self.assertEqual("in_progress", resilient.status("batch-1"),
                         "the wrapped status answer comes through the retry")
        self.assertEqual(3, len(backend.attempts["status"]),
                         "two transient failures were retried; the third "
                         "attempt answered")
        self.assertEqual([5.0, 10.0], clock.sleeps,
                         "the backoff doubled between the retries: "
                         "5s then 10s")
        self.assertEqual(3585.0, resilient.remaining,
                         "the shared budget shrank by exactly the seconds "
                         "the retry slept")
        self.assertIn("ConnectionError", "\n".join(reported),
                      "each transient failure was reported through log")

    def test_submit_delegates_through_retry_too(self):
        clock = _Clock()
        backend = _FakeBackend(clock, failures={"submit": 1})
        resilient = ResilientBackend(
            backend, timeout=600.0, start_delay=5.0, max_delay=120.0,
            sleep=clock.sleep, now=clock.now)
        self.assertEqual("batch-1", resilient.submit([{"custom_id": "gen-a"}]),
                         "the wrapped submit's batch id comes through the "
                         "retry")
        self.assertEqual(2, len(backend.attempts["submit"]),
                         "the one transient submit failure was retried once")

    def test_one_deadline_is_shared_across_calls(self):
        clock = _Clock(0.0)
        backend = _FakeBackend(clock, failures={"status": 1,
                                                "collect": 10 ** 9})
        resilient = ResilientBackend(
            backend, timeout=30.0, start_delay=10.0, max_delay=10.0,
            sleep=clock.sleep, now=clock.now)
        self.assertEqual(30.0, resilient.remaining,
                         "the deadline is fixed at construction: the full "
                         "budget")
        self.assertEqual("in_progress", resilient.status("batch-1"),
                         "the first call answers after one retried failure")
        self.assertEqual(20.0, resilient.remaining,
                         "the budget is NOT restored per call: status's "
                         "retry spent ten of the thirty seconds")
        with self.assertRaises(
                WaitTimeout,
                msg="collect fails forever, and the SHARED budget runs out") \
                as caught:
            resilient.collect("batch-1")
        self.assertEqual(20.0, caught.exception.elapsed,
                         "collect inherited the 20 seconds LEFT of the shared "
                         "budget, not a fresh 30-second timeout")
        self.assertEqual(3, len(backend.attempts["collect"]),
                         "collect retried only while budget remained: "
                         "attempts at t=10, 20, 30")
        self.assertEqual(0.0, resilient.remaining,
                         "after the timeout the shared budget is spent")

    def test_remaining_is_read_only(self):
        clock = _Clock()
        resilient = ResilientBackend(_FakeBackend(clock), timeout=60.0,
                                     sleep=clock.sleep, now=clock.now)
        with self.assertRaises(
                AttributeError,
                msg="the shared budget belongs to the wait; no call may "
                    "reset it"):
            resilient.remaining = 99.0

    def test_unknown_attributes_reach_the_wrapped_backend(self):
        clock = _Clock()
        backend = _FakeBackend(clock)
        resilient = ResilientBackend(backend, timeout=60.0,
                                     sleep=clock.sleep, now=clock.now)
        self.assertEqual("qwen2.5-32b-instruct", resilient.model,
                         "an attribute the proxy does not define falls "
                         "through to the wrapped backend")
        self.assertEqual("quota exceeded", resilient.failure_reason("batch-1"),
                         "a method the proxy does not define stays callable "
                         "through it, arguments intact")
        with self.assertRaises(
                AttributeError,
                msg="an attribute the WRAPPED backend also lacks still "
                    "raises AttributeError"):
            resilient.no_such_thing


class WaitUntilTests(unittest.TestCase):
    """The poll loop: first call immediate, then backoff, then the deadline."""

    def test_first_call_is_immediate_and_loops_until_done(self):
        clock = _Clock(500.0)
        states = [(False, "queued"), (False, "running"), (True, "deck done")]
        poll_times = []

        def step():
            poll_times.append(clock.now())
            return states.pop(0)

        value = wait_until(step, timeout=300.0, start_delay=5.0,
                           max_delay=100.0, sleep=clock.sleep, now=clock.now,
                           what="the deck")
        self.assertEqual("deck done", value,
                         "the value carried by the done step is returned")
        self.assertEqual([500.0, 505.0, 515.0], poll_times,
                         "the first poll ran at the start time -- before ANY "
                         "sleep -- and the next two followed the backoff "
                         "steps")
        self.assertEqual([5.0, 10.0], clock.sleeps,
                         "exactly one sleep per not-done poll, doubling: "
                         "5s then 10s")

    def test_raises_waittimeout_once_the_clock_expires(self):
        clock = _Clock(0.0)
        poll_times = []

        def step():
            poll_times.append(clock.now())
            return (False, "queued")

        with self.assertRaises(
                WaitTimeout,
                msg="a step that never reports done exhausts the deadline "
                    "into WaitTimeout") as caught:
            wait_until(step, timeout=12.0, start_delay=10.0, max_delay=10.0,
                       sleep=clock.sleep, now=clock.now, what="the deck")
        self.assertEqual(12.0, caught.exception.elapsed,
                         "elapsed is the whole deadline, spent polling")
        self.assertIn("the deck", str(caught.exception),
                      "the message names what was being waited for")
        self.assertEqual(3, len(poll_times),
                         "polls at t=0, 10 and 12 -- the step is still "
                         "consulted at the deadline before giving up")
        self.assertEqual([10.0, 2.0], clock.sleeps,
                         "the second sleep is clamped to the 2 seconds "
                         "remaining -- its unclamped backoff step is 10")

    def test_zero_timeout_still_gives_the_step_its_first_call(self):
        clock = _Clock(0.0)
        poll_times = []

        def step():
            poll_times.append(clock.now())
            return (False, "queued")

        with self.assertRaises(
                WaitTimeout,
                msg="an already-spent deadline still allows the immediate "
                    "first call, then raises"):
            wait_until(step, timeout=0.0, sleep=clock.sleep, now=clock.now,
                       what="the deck")
        self.assertEqual(1, len(poll_times),
                         "exactly one poll: the immediate first call")
        self.assertEqual([], clock.sleeps,
                         "no sleep happens once the deadline is already spent")


if __name__ == "__main__":
    unittest.main()
