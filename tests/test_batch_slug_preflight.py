"""OpenRouter batch submissions rejected because the model has no batch API.

The OpenRouter catalogue lists ``:batch`` model slugs that have no batch
endpoint behind them -- measured on four Anthropic slugs -- so POSTing a deck
pinned to such a slug is answered with HTTP 400 and the sentence ``"Model
'anthropic/claude-sonnet-5:batch' does not have a :batch endpoint."``. Left to
the generic handler, that surfaced as ``OpenRouter batch submission failed with
HTTP 400`` and killed the whole generation for a reason that had nothing to do
with the cards.

``OpenRouterBatchBackend.submit`` now reads that sentence out of the rejected
body and raises an error naming the slug and saying a different model must be
configured, while every other rejection keeps the generic message verbatim and
a successful submission behaves exactly as before.

These tests drive the backend through the ``transport=`` constructor seam -- a
callable ``(method, url, payload) -> (status_code, parsed_json)`` -- with
canned responses, the same way ``tests/test_batch_backends.py`` does, so no
network and no SDK are involved.
"""

import json

import pytest

from processors.batch import OpenRouterBatchBackend

# A minimal deck in the compiler's internal request-dict shape. The fake
# transport never inspects the serialized payload; the serializer only needs
# well-formed requests (a None per-request model means "use the batch model").
DECK = [
    {"custom_id": "card-0", "model": None,
     "messages": [{"role": "user", "content": "morph this card"}]},
    {"custom_id": "card-1", "model": None,
     "messages": [{"role": "user", "content": "and this one"}]},
]


class _ScriptedTransport:
    """A canned transport: answers each call with the next (status, body)."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, method, url, payload):
        self.calls.append((method, url, payload))
        return self._responses.pop(0)


def _backend(transport, model="anthropic/claude-sonnet-5:batch"):
    return OpenRouterBatchBackend(
        model=model,
        api_key="sk-test",
        transport=transport,
        now=lambda: 0.0,
    )


def test_no_batch_endpoint_rejection_names_the_slug():
    body = {
        "error": {
            "message": "HTTP 400: invalid batch inference job: "
                       "Model 'anthropic/claude-sonnet-5:batch' does not have "
                       "a :batch endpoint."
        }
    }
    transport = _ScriptedTransport((400, body))

    with pytest.raises(RuntimeError) as excinfo:
        _backend(transport).submit(DECK)

    message = str(excinfo.value)
    # The offending slug is named, and the message says plainly what is wrong
    # with it and what to do instead.
    assert "anthropic/claude-sonnet-5:batch" in message
    assert "listed in the model catalogue" in message
    assert "no batch endpoint" in message
    assert "different model must be configured" in message


def test_unrelated_rejection_keeps_the_generic_message():
    body = {
        "error": {
            "message": "HTTP 400: invalid batch inference job: "
                       "job-submission-count for account alex-79b5d6, "
                       "in use: 16, quota: 16"
        }
    }
    transport = _ScriptedTransport((400, body))

    with pytest.raises(RuntimeError) as excinfo:
        _backend(transport, model="z-ai/glm-4.5:batch").submit(DECK)

    # Byte-for-byte the message submit has always raised for a rejected POST
    # that is not the no-batch-endpoint kind (the body is short, so _trim
    # returns it undumped-and-untrimmed).
    message = str(excinfo.value)
    assert message == (
        "OpenRouter batch submission failed with HTTP 400: "
        + json.dumps(body, ensure_ascii=False))
    # The configured slug is not blamed for a failure that is not about the
    # model, and no catalogue advice is bolted onto an unrelated fault.
    assert "z-ai/glm-4.5:batch" not in message
    assert "no batch endpoint" not in message


@pytest.mark.parametrize("status", [200, 202])
def test_successful_submission_is_unaffected(status):
    transport = _ScriptedTransport(
        (status, {"id": "batch_abc123"}),
        # A follow-up poll that 404s: inside the read-after-write grace window
        # this must still read as "not visible yet", i.e. in_progress, exactly
        # as it did before the slug preflight existed.
        (404, {"error": {"message": "Batch job batch_abc123 not found.",
                         "code": 404}}),
    )
    backend = _backend(transport)

    assert backend.submit(DECK) == "batch_abc123"

    # Exactly one POST to the batches endpoint, carrying the deck.
    assert len(transport.calls) == 1
    method, url, payload = transport.calls[0]
    assert method == "POST"
    assert url == "https://openrouter.ai/api/beta/batches"
    assert "card-0" in payload

    # The recorded submission still feeds the grace window on later polls.
    assert backend.status("batch_abc123") == "in_progress"


def test_marker_without_a_quoted_slug_falls_back_to_the_configured_model():
    body = {"error": {"message": "Model does not have a :batch endpoint."}}
    transport = _ScriptedTransport((400, body))

    with pytest.raises(RuntimeError) as excinfo:
        _backend(transport, model="z-ai/glm-4.5:batch").submit(DECK)

    # The phrase is recognized even when the sentence no longer quotes the
    # slug; the backend's own model is the setting that has to change either
    # way, so it is named instead of falling back to the generic message.
    message = str(excinfo.value)
    assert "z-ai/glm-4.5:batch" in message
    assert "no batch endpoint" in message
    assert "different model must be configured" in message
