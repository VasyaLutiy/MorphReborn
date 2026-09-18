"""
Waiting that survives a blinking network: caller-side retry and poll logic
for batch runs. Pure logic -- no store calls, no provider knowledge, no
import from ``flows``, and no import from ``processors`` at module load
(from neither at all, in fact: this module needs only the stdlib).

WHY this exists: a batch run is HOURS of waiting around a queue the provider
documents at 20-40 minutes. Today a single failed GET against the batch
provider -- one connection reset in the small hours -- tears down a wait
that has already been running for over an hour, and the run is lost: the
deck was submitted, the cards were fine, only the transport blinked. A
headless run that waits for hours MUST treat a transient transport error as
something to retry, not as the end.

The retry wrapper deliberately lives HERE, in the caller, and not inside a
provider backend: the backend stays dumb. The backends in
``processors/batch.py`` translate one protocol call into one answer and
raise on anything they do not understand; whether an error is worth another
attempt is a property of the WAIT -- how much budget is left, what the
operator asked to survive -- not of the provider, so the decision is made
once, here, for every backend at once.

The batch backend protocol these helpers wrap is duck-typed and has exactly
three methods::

    submit(requests)  -> batch_id
    status(batch_id)  -> "in_progress" | "completed" | "failed"
    collect(batch_id) -> {custom_id: response text or None}

:class:`ResilientBackend` wraps any such object and makes each of the three
survive transient failures under ONE deadline shared across the whole wait;
:func:`wait_until` polls a step -- such as a :class:`ResilientBackend`'s
``status`` -- until it reports done. Every schedule runs off injected
``sleep``/``now`` callables so tests advance a fake clock instead of
sleeping; the defaults are the real ``time.sleep`` and ``time.monotonic``.
"""

import time
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


class WaitTimeout(Exception):
    """The wait exhausted its deadline.

    Raised by :func:`retry` and :func:`wait_until` -- never by a backend,
    which has no clock -- when the seconds budgeted for the wait are spent.
    The message names WHAT was being waited for, the seconds actually waited
    and the last error seen, so an operator reading the traceback of a dead
    run gets a diagnosis ("gave up waiting for batch status after 10841.3s
    (last error: ConnectionResetError(104, 'Connection reset by peer'))")
    and not a bare class name; ``elapsed`` carries the same number for code
    that reports it structurally.
    """

    def __init__(self, message: str, elapsed: float):
        super().__init__(message)
        self.elapsed = elapsed


# Exception classes that must NEVER be retried, because retrying cannot help:
# a KeyboardInterrupt is the operator cancelling the wait, a SystemExit is
# the process leaving, and a MemoryError means the next attempt fails the
# same way. Every wait in this module re-raises these immediately, before
# any deadline is consulted; callers may add their own through each
# function's ``permanent`` argument.
PERMANENT: Tuple[Type[BaseException], ...] = (KeyboardInterrupt, SystemExit,
                                              MemoryError)


def backoff_delays(start: float = 5.0, cap: float = 120.0) -> Iterator[float]:
    """Yield ``start``, ``start*2``, ``start*4``, ... capped at ``cap`` forever.

    The schedule behind every wait in this module. The first retry comes
    quickly -- a blinked connection is usually back within seconds -- each
    further failure doubles the wait, and once the cap is reached the SAME
    cap repeats forever, so a long queue is still polled every two minutes
    and a completed batch is picked up promptly instead of an hour late.

    Deterministic and jitter-free on purpose: jitter exists so that many
    simultaneous clients do not synchronise their retries, which buys a
    single-client poller nothing and would make the waits unreproducible in
    tests. A ``start`` already at or above ``cap`` yields ``cap`` from the
    first step.
    """
    delay = float(start)
    while delay < cap:
        yield delay
        delay *= 2.0
    while True:
        yield cap


def retry(call: Callable[[], T], *, timeout: float,
          start_delay: float = 5.0, max_delay: float = 120.0,
          permanent: Tuple[Type[BaseException], ...] = (),
          sleep: Callable[[float], None] = time.sleep,
          now: Callable[[], float] = time.monotonic,
          log: Optional[Callable[[str], None]] = None,
          what: str = "call") -> T:
    """Call ``call()`` and return its value, retrying transient failures.

    The first attempt happens immediately, whatever the budget. On an
    exception that is an instance of :data:`PERMANENT` or of ``permanent``
    the exception is re-raised at once -- no sleep, no second attempt,
    because retrying a Ctrl-C or a caller-declared hopeless error cannot
    help. On ANY other exception the error is reported through ``log`` (a
    one-argument callable; ``None`` reports nothing) and, while the deadline
    has not passed, the next :func:`backoff_delays` delay is slept -- clamped
    to the time remaining, so no sleep ever runs past the deadline -- and the
    call is tried again. Once the deadline HAS passed, the wait gives up and
    raises :class:`WaitTimeout`, whose message names ``what``, the elapsed
    seconds and the last exception.

    ``timeout`` bounds the RETRIES, not a single call: ``call()`` itself is
    never interrupted, and a timeout of zero (or less) still gives it its one
    immediate attempt -- only a failure is converted into
    :class:`WaitTimeout`.

    ``sleep`` and ``now`` are injected so tests run a fake clock instantly;
    the defaults are the real ``time.sleep`` and ``time.monotonic``.
    """
    started_at = now()
    deadline = started_at + timeout
    delays = backoff_delays(start_delay, max_delay)
    hopeless = PERMANENT + tuple(permanent)
    while True:
        try:
            return call()
        except hopeless:
            raise
        except Exception as error:
            moment = now()
            if moment >= deadline:
                elapsed = moment - started_at
                raise WaitTimeout(
                    f"gave up waiting for {what} after {elapsed:.1f}s "
                    f"(last error: {error!r})",
                    elapsed=elapsed) from error
            delay = min(next(delays), deadline - moment)
            if log is not None:
                log(f"{what}: {type(error).__name__}: {error}; "
                    f"retrying in {delay:.0f}s")
            sleep(delay)


class ResilientBackend:
    """A batch backend wrapped so its calls survive transient failures.

    Wraps any object with the duck-typed three-method protocol from the
    module docstring and delegates ``submit``, ``status`` and ``collect``
    through :func:`retry` under the configured budget, so one connection
    reset in the small hours costs a five-second pause instead of an
    hour-old run. The wrapper adds no provider knowledge: the wrapped
    backend stays dumb, deciding nothing about retries.

    The timeout is the budget of the WHOLE wait, shared across calls: the
    deadline is recorded ONCE, here at construction (``now() + timeout``),
    and each call receives whatever is left of it. Six hours by default --
    the provider queue is 20-40 minutes, but a headless run waits around it
    for far longer, and the operator's promise is six hours of wall clock
    for the entire wait, not six fresh hours per call, which a deck that
    fails once per poll would stretch indefinitely. The :attr:`remaining`
    property reads what is left; assigning to it is an error, because the
    budget belongs to the wait, not to any one call.

    Any OTHER attribute -- ``model``, OpenRouter's ``failure_reason``,
    whatever a backend grows -- falls through to the wrapped backend via
    ``__getattr__``, so this object is a drop-in wherever a backend is
    taken. ``permanent`` passes through to :func:`retry` unchanged, for
    callers whose backends raise their own hopeless errors.

    ``sleep``, ``now`` and ``log`` are the test seams, exactly as in
    :func:`retry`.
    """

    def __init__(self, backend, *, timeout: float = 6 * 3600.0,
                 submit_timeout: float = 120.0,
                 start_delay: float = 5.0, max_delay: float = 120.0,
                 permanent: Tuple[Type[BaseException], ...] = (),
                 sleep: Callable[[float], None] = time.sleep,
                 now: Callable[[], float] = time.monotonic,
                 log: Optional[Callable[[str], None]] = None):
        self._backend = backend
        # WHY submit has its own, much shorter budget. Waiting is what
        # legitimately takes hours: the provider queue runs 20-40 minutes and a
        # poll that fails is worth retrying all night. SUBMISSION is not -- it
        # is answered or refused in one call, and an error that repeats there
        # is almost always a refusal wearing a transport costume: no
        # credentials, a bad slug, a malformed request. 18.09 `mrph run`
        # without an .env retried "Missing credentials" against the whole
        # six-hour budget and would have slept through an entire unattended
        # night. Classifying provider exceptions cannot be relied on to prevent
        # that -- in the openai SDK every transport error and every
        # configuration error share one base class -- so the guard is
        # structural instead: submission may spend at most this much of the
        # budget, whatever the error turns out to be. It never EXCEEDS the
        # shared deadline; it only caps it.
        self._submit_timeout = submit_timeout
        self._start_delay = start_delay
        self._max_delay = max_delay
        self._permanent = permanent
        self._sleep = sleep
        self._now = now
        self._log = log
        # ONE deadline for the WHOLE wait, fixed here and now. A per-call
        # deadline would hand every method a fresh budget and let a deck
        # that fails once per poll wait forever; the operator's promise is
        # wall clock for the whole run.
        self._deadline = now() + timeout

    @property
    def remaining(self) -> float:
        """Seconds left on the shared budget; zero or negative once spent."""
        return self._deadline - self._now()

    def _budget(self) -> float:
        """The time left, floored at zero: this call's retry timeout."""
        return max(self.remaining, 0.0)

    def _through_retry(self, call: Callable[[], T], what: str,
                       cap: Optional[float] = None) -> T:
        """Run one backend call under the shared budget's remaining time.

        ``cap`` shortens THIS call's slice of the budget without touching the
        shared deadline: the call gets ``min(remaining, cap)``. Only submission
        uses it -- see :attr:`_submit_timeout`.
        """
        budget = self._budget()
        if cap is not None:
            budget = min(budget, cap)
        return retry(call, timeout=budget,
                     start_delay=self._start_delay, max_delay=self._max_delay,
                     permanent=self._permanent, sleep=self._sleep,
                     now=self._now, log=self._log, what=what)

    def submit(self, requests: List[dict]) -> str:
        """Submit the deck through :func:`retry`; return the batch id.

        Capped at :attr:`_submit_timeout`, unlike the polling calls: a refusal
        that looks like a transport failure must not be allowed to eat the
        whole night.
        """
        return self._through_retry(
            lambda: self._backend.submit(requests), "batch submit",
            cap=self._submit_timeout)

    def status(self, batch_id: str) -> str:
        """Poll the batch status through :func:`retry`."""
        return self._through_retry(
            lambda: self._backend.status(batch_id), "batch status")

    def collect(self, batch_id: str) -> Dict[str, Optional[str]]:
        """Collect the deck's results through :func:`retry`."""
        return self._through_retry(
            lambda: self._backend.collect(batch_id), "batch collect")

    def __getattr__(self, name: str):
        """Anything else: the wrapped backend's own attribute.

        Only reached when normal lookup fails, so ``submit``, ``status``,
        ``collect`` and ``remaining`` never get here. Guarded against the
        not-yet-constructed case (an attribute read before ``__init__`` has
        stored ``_backend``) so the failure is a clean ``AttributeError``
        rather than an ``__getattr__`` recursion.
        """
        backend = self.__dict__.get("_backend")
        if backend is None:
            raise AttributeError(name)
        return getattr(backend, name)


def wait_until(step: Callable[[], Tuple[bool, T]], *, timeout: float,
               start_delay: float = 5.0, max_delay: float = 120.0,
               sleep: Callable[[float], None] = time.sleep,
               now: Callable[[], float] = time.monotonic,
               log: Optional[Callable[[str], None]] = None,
               what: str = "generation") -> T:
    """Poll ``step`` until it reports done, and return the value it found.

    The polling loop for a step that is itself already retry-protected -- a
    :class:`ResilientBackend`'s ``status``, say, whose transport errors never
    reach this loop. An exception out of ``step`` therefore propagates
    unchanged: anything that escapes a retry-protected step is either
    permanent or has already spent its own budget, and re-retrying it here
    would double every budget silently.

    ``step()`` returns a ``(done, value)`` pair; when ``done`` is true the
    loop returns ``value``. The FIRST call happens immediately, before any
    sleep, so a batch that is already completed is collected without waiting
    even one backoff delay. While the step reports not-done, the loop sleeps
    the next :func:`backoff_delays` delay -- clamped to the time remaining,
    so no sleep ever runs past the deadline -- polls again, and raises
    :class:`WaitTimeout` once the deadline has passed with the step still
    not done; the message names ``what`` and the elapsed seconds and carries
    the last not-done state, which is the diagnosis (``'in_progress'`` vs
    ``'failed'``) an operator needs. Each not-yet-done poll is reported
    through ``log`` when one is given, so a long wait leaves a trace of how
    long each state lasted.
    """
    started_at = now()
    deadline = started_at + timeout
    delays = backoff_delays(start_delay, max_delay)
    while True:
        done, value = step()
        if done:
            return value
        moment = now()
        if moment >= deadline:
            elapsed = moment - started_at
            raise WaitTimeout(
                f"gave up waiting for {what} after {elapsed:.1f}s "
                f"(last state: {value!r})",
                elapsed=elapsed)
        delay = min(next(delays), deadline - moment)
        if log is not None:
            log(f"{what}: not done after {moment - started_at:.1f}s "
                f"(state: {value!r}); next poll in {delay:.0f}s")
        sleep(delay)
