"""A provider-rejected batch must say why in the ``/collect`` log -- if it can.

``collect_generation`` polls one batch. A batch the provider REJECTED polls as
``"failed"``, which becomes ``results = None`` and every card of the generation
recorded failed -- until now without one word about why, even though the backend
often holds the reason (``OpenRouterBatchBackend.failure_reason`` surfaces e.g.
``HTTP 400: invalid batch inference job: job-submission-count for account
alex-79b5d6, in use: 16, quota: 16``). These tests pin the one-line diagnosis
now logged when the backend can speak, and that a backend which cannot -- no
``failure_reason`` method at all, or one that raises -- collects the failed
batch exactly as before: the diagnosis is an addition, never a policy.

The fixtures mirror ``tests/test_store.py``: an empty project directory per
test, a :class:`DeckStore` over it, and a log that is a plain list's
``append``.
"""

import pytest

from cards.store import DeckStore, collect_generation, submit_generation


# The shape of a real OpenRouter rejection (a full submission quota), so the
# "message verbatim" assertion is pinned against something an operator reads.
REJECTION_MESSAGE = (
    "HTTP 400: invalid batch inference job: job-submission-count for "
    "account alex-79b5d6, in use: 16, quota: 16")


class RejectedBatchBackend:
    """A batch backend whose every batch ends ``failed`` -- with a reason.

    ``submit`` answers at once and nothing is executed; ``status`` reports the
    batch failed, which is the path under test. ``collect`` is never reached for
    a failed batch (there are no results to download); it returns an empty map
    and counts its calls, so a surprise shows up in an assertion rather than as
    a crash. ``failure_reason`` is the optional diagnosis method the OpenRouter
    backend has and the others do not: it answers ``reason`` verbatim, or
    raises when the test asks for a broken reason service.
    """

    def __init__(self, reason: str = REJECTION_MESSAGE, raises: bool = False):
        self.batch_id = None
        self.collect_calls = 0
        self._reason = reason
        self._raises = raises

    def submit(self, requests):
        self.batch_id = "batch-rejected-0001"
        return self.batch_id

    def status(self, batch_id):
        return "failed"

    def collect(self, batch_id):
        self.collect_calls += 1
        return {}

    def failure_reason(self, batch_id):
        if self._raises:
            raise RuntimeError("the reason service is down")
        return self._reason


class ReasonlessFailedBackend:
    """The same failed batch, on a backend with NO ``failure_reason`` method.

    A separate class rather than a flag, because the case under test is
    ``getattr(backend, "failure_reason", None)`` finding nothing at all -- the
    situation of the local, OpenAI and Anthropic backends -- which a method
    that RETURNS ``None`` could not distinguish from "asked, and no reason".
    """

    def __init__(self):
        self.batch_id = None

    def submit(self, requests):
        self.batch_id = "batch-rejected-0002"
        return self.batch_id

    def status(self, batch_id):
        return "failed"

    def collect(self, batch_id):
        return {}


def _add_card(store):
    """One minimal generate card, so a run has a generation to submit."""
    store.add_card({
        "custom_id": "card-1",
        "intent": "generate",
        "target": "notes/hello.txt",
        "instruction": "write a greeting into notes/hello.txt",
    })


@pytest.fixture
def lines():
    """Every line the run logged, in order."""
    return []


@pytest.fixture
def log(lines):
    """The log callable the store functions take: ``list.append``."""
    return lines.append


@pytest.fixture
def root(tmp_path):
    """An empty project directory for one test."""
    return tmp_path


@pytest.fixture
def store(root):
    """A :class:`DeckStore` over the empty project directory."""
    return DeckStore(root)


def test_failed_batch_logs_the_provider_reason(store, root, log, lines):
    """The rejection line reaches the log: batch named, message verbatim."""
    _add_card(store)
    backend = RejectedBatchBackend()

    submit_generation(store, backend, root=root, log=log, use_git=False)
    result = collect_generation(store, backend, root=root, log=log)

    rejection_lines = [
        line for line in lines if "the provider rejected batch" in line
    ]
    assert rejection_lines == [
        f"mrph> the provider rejected batch {backend.batch_id}: "
        f"{REJECTION_MESSAGE}"
    ]
    # The diagnosis changed nothing else: the failed batch is still consumed
    # the way it always was -- the card recorded failed, the run finished.
    assert result.in_progress is False
    assert result.phase == "done"
    assert result.outcomes["card-1"].status == "failed"


def test_backend_without_failure_reason_still_collects(store, root, log, lines):
    """No ``failure_reason`` on the backend: collection proceeds untouched."""
    _add_card(store)
    backend = ReasonlessFailedBackend()

    submit_generation(store, backend, root=root, log=log, use_git=False)
    result = collect_generation(store, backend, root=root, log=log)

    assert result.in_progress is False
    assert result.outcomes["card-1"].status == "failed"
    # ...and nothing pretended to know a reason it was never given.
    assert not any("the provider rejected batch" in line for line in lines)


def test_raising_failure_reason_does_not_break_collection(store, root, log, lines):
    """A ``failure_reason`` that raises is swallowed: collection finishes."""
    _add_card(store)
    backend = RejectedBatchBackend(raises=True)

    submit_generation(store, backend, root=root, log=log, use_git=False)
    result = collect_generation(store, backend, root=root, log=log)

    assert result.in_progress is False
    assert result.outcomes["card-1"].status == "failed"
    assert not any("the provider rejected batch" in line for line in lines)
