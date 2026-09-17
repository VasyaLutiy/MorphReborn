"""
``OpenRouterBatchBackend.failure_reason``: the why behind a "failed" batch.

``status()`` reads only the batch object's status string, so a batch the
provider rejected -- GET answers with ``{"error": {"message": "HTTP 400:
invalid batch inference job: job-submission-count for account alex-79b5d6, in
use: 16, quota: 16"}}`` -- reported every card as failed with no reason at all,
and the operator had to curl the batch API by hand. These tests pin the
accessor that surfaces the message: it comes back as a string, a batch object
with no error yields ``None``, and the read-after-write grace window of
``OPENROUTER_SUBMIT_GRACE_SECONDS`` is honoured -- not visible yet is ``None``
and never a raise, while a genuine 404 outside the window stays fatal.

Self-contained by design: the transport / clock fakes here are local stand-ins
for the ones in ``tests/test_batch_backends.py`` (test modules must not import
from one another), driving the backend through the same ``transport=`` /
``now=`` constructor seams.
"""

import pytest

from processors.batch import (
    OPENROUTER_BATCH_BASE_URL,
    OPENROUTER_SUBMIT_GRACE_SECONDS,
    BatchNotReady,
    OpenRouterBatchBackend,
)

# A real id, as the live service hands them out -- deliberately NOT prefixed
# "local-", which is what keeps it out of the orphan recovery path.
OPENROUTER_BATCH_ID = "batch-1789576284-Ejahe4wq9AgVdp5xGdNm"

# Verbatim the shape from the field report: a deck over the account's
# job-submission quota, rejected at submission time.
REJECTED_MESSAGE = (
    "HTTP 400: invalid batch inference job: job-submission-count for "
    "account alex-79b5d6, in use: 16, quota: 16"
)


def _requests():
    """Two internal request dicts, neither pinning a model."""
    return [
        {"custom_id": "c1", "model": None,
         "messages": [{"role": "user", "content": "first"}]},
        {"custom_id": "c2", "model": None,
         "messages": [{"role": "user", "content": "second"}]},
    ]


class _FakeTransport:
    """Records every call and replays a queue of canned responses.

    The last response repeats, so a caller that hits the transport more than
    once need only declare the answer once.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, payload):
        self.calls.append((method, url, payload))
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


class _FakeClock:
    """A hand-cranked monotonic clock, so the grace window costs no wall time."""

    def __init__(self, start=1000.0):
        self.value = start

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def _batch_object(status, error=None):
    """The batch object shape the service returns from GET."""
    return {
        "id": OPENROUTER_BATCH_ID,
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "model": "z-ai/glm-5.3-flash-20260826",
        "completion_window": "24h",
        "status": status,
        "request_counts": {"total": 2, "completed": 0, "failed": 0},
        "results": None,
        "error": error,
    }


def _not_found_body(batch_id=None):
    """Verbatim the live service's 404 body, the one that killed a real run."""
    return {"error": {"message": f"Batch job {batch_id or OPENROUTER_BATCH_ID} "
                                 "not found.", "code": 404}}


def _backend(transport, clock=None):
    return OpenRouterBatchBackend(
        model="z-ai/glm-5.3-flash:batch", api_key="sk-or-xxx",
        transport=transport, now=clock)


def test_failure_reason_returns_the_provider_message_for_a_rejected_batch():
    error = {"message": REJECTED_MESSAGE, "code": 400}
    transport = _FakeTransport([(200, _batch_object("failed", error=error))])
    backend = _backend(transport)

    reason = backend.failure_reason(OPENROUTER_BATCH_ID)

    assert reason == REJECTED_MESSAGE
    # Exactly one retrieve: a GET of the batch object, nothing else.
    assert transport.calls == [
        ("GET",
         f"{OPENROUTER_BATCH_BASE_URL}/batches/{OPENROUTER_BATCH_ID}",
         None),
    ]


def test_failure_reason_is_none_when_the_batch_carries_no_error():
    # A healthy (or merely unfinished) batch object carries "error": null;
    # None must come back as None, not as the string "None".
    transport = _FakeTransport([(200, _batch_object("completed"))])
    assert _backend(transport).failure_reason(OPENROUTER_BATCH_ID) is None

    # Same when the key is absent entirely.
    body = _batch_object("in_progress")
    del body["error"]
    transport = _FakeTransport([(200, body)])
    assert _backend(transport).failure_reason(OPENROUTER_BATCH_ID) is None


def test_failure_reason_is_none_while_the_batch_is_not_visible_yet():
    # The production sequence: submit, then ask why before the
    # read-after-write window has closed. The service answers the retrieve
    # with a 404 because it cannot see its own write yet -- that is not a
    # failure, the batch may still be fine, so this is None and never a raise.
    transport = _FakeTransport([
        (202, _batch_object("validating")),
        (404, _not_found_body()),
    ])
    backend = _backend(transport, _FakeClock())

    batch_id = backend.submit(_requests())

    assert backend.failure_reason(batch_id) is None


def test_failure_reason_after_the_grace_period_lets_the_404_raise():
    # Outside the window the 404 is a genuine one -- a wrong or deleted id --
    # and must stay fatal: "no reason available" would paper over a batch that
    # does not exist at all.
    transport = _FakeTransport([
        (202, _batch_object("validating")),
        (404, _not_found_body()),
    ])
    clock = _FakeClock()
    backend = _backend(transport, clock)

    batch_id = backend.submit(_requests())
    clock.advance(OPENROUTER_SUBMIT_GRACE_SECONDS + 1)

    with pytest.raises(RuntimeError) as ctx:
        backend.failure_reason(batch_id)
    # pytest's ExceptionInfo exposes the raised exception as .value (the
    # .exception attribute belongs to unittest's assertRaises context).
    assert "404" in str(ctx.value)
    assert batch_id in str(ctx.value)
    # Not the "come back later" signal -- that would defeat the whole point.
    assert not isinstance(ctx.value, BatchNotReady)


def test_failure_reason_inside_the_window_still_lets_other_errors_raise():
    # The window forgives one status code for one id, not every failure: an
    # operator must still see a 500 with its body.
    transport = _FakeTransport([
        (202, _batch_object("validating")),
        (500, {"error": {"message": "internal error"}}),
    ])
    backend = _backend(transport, _FakeClock())

    batch_id = backend.submit(_requests())

    with pytest.raises(RuntimeError) as ctx:
        backend.failure_reason(batch_id)
    assert "500" in str(ctx.value)
    assert "internal error" in str(ctx.value)
    assert not isinstance(ctx.value, BatchNotReady)
