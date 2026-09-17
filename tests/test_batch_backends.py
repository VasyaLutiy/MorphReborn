"""
Phase 2 tests: the batch backends (submit / status / collect).

No network and no real provider SDKs. The cloud backends are driven through
fake clients injected via ``client_factory``; the local backend runs against a
fake registry of fake processors. The suite must pass in an environment where
``anthropic`` is not installed at all (see the no-SDK-at-import constraint in
``processors/batch.py``).
"""

import io
import json
import os
import shutil
import socket
import tempfile
import unittest
import urllib.error
from types import SimpleNamespace
from unittest import mock

from cards.compiler import compile_deck, serialize_openai
from cards.schema import MorphCard
from cards.store import DeckStore, recover_orphaned_local_batch
from processors.batch import (
    OPENROUTER_HTTP_TIMEOUT_SECONDS,
    OPENROUTER_SUBMIT_GRACE_SECONDS,
    AnthropicBatchBackend,
    BatchNotReady,
    LocalBatchBackend,
    OpenAIBatchBackend,
    OpenRouterBatchBackend,
    _default_openrouter_transport,
)
from processors.registry import ProcessorRegistry


# Same relative compile root the compiler tests use: the root string is embedded
# in the context file paths, so a relative root keeps things portable.
MINIPROJECT = os.path.join("tests", "fixtures", "miniproject")


def _requests():
    """Two plain internal request dicts (the compiler's output shape)."""
    return [
        {"custom_id": "c1", "model": None,
         "messages": [{"role": "user", "content": "first"}]},
        {"custom_id": "c2", "model": "claude-opus-4",
         "messages": [{"role": "user", "content": "second"}]},
    ]


def _requests_for_one_model():
    """The same two requests, neither pinning a model.

    OpenRouter applies one model to the whole batch, so a deck bound for it
    leaves the per-request model unset (the serializer rejects a conflicting
    one; see ``tests/test_compiler.py``).
    """
    return [
        {"custom_id": "c1", "model": None,
         "messages": [{"role": "user", "content": "first"}]},
        {"custom_id": "c2", "model": None,
         "messages": [{"role": "user", "content": "second"}]},
    ]


# -- OpenAI fake client ------------------------------------------------------


class _FakeOpenAIFiles:
    def __init__(self, client):
        self._c = client

    def create(self, file, purpose):
        self._c.uploaded_file = file
        self._c.uploaded_purpose = purpose
        return SimpleNamespace(id="file-in-123")

    def content(self, file_id):
        self._c.downloaded_file_id = file_id
        return SimpleNamespace(text=self._c.output_text)


class _FakeOpenAIBatches:
    def __init__(self, client):
        self._c = client

    def create(self, input_file_id, endpoint, completion_window):
        self._c.create_kwargs = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
        }
        return SimpleNamespace(id="batch-abc")

    def retrieve(self, batch_id):
        self._c.retrieved = batch_id
        return SimpleNamespace(status=self._c.status, output_file_id="file-out-999")


class _FakeOpenAIClient:
    def __init__(self, status="completed", output_text=""):
        self.status = status
        self.output_text = output_text
        self.uploaded_file = None
        self.uploaded_purpose = None
        self.create_kwargs = None
        self.downloaded_file_id = None
        self.files = _FakeOpenAIFiles(self)
        self.batches = _FakeOpenAIBatches(self)


class OpenAIBatchBackendTests(unittest.TestCase):
    def test_submit_uploads_serializer_output_and_creates_batch(self):
        fake = _FakeOpenAIClient()
        backend = OpenAIBatchBackend(model="gpt-4o", client_factory=lambda: fake)
        requests = _requests()

        batch_id = backend.submit(requests)

        self.assertEqual(batch_id, "batch-abc")
        # The uploaded bytes must equal the golden-tested serializer output.
        expected = serialize_openai(requests, default_model="gpt-4o").encode("utf-8")
        self.assertEqual(fake.uploaded_file[1], expected)
        self.assertEqual(fake.uploaded_purpose, "batch")
        self.assertEqual(fake.create_kwargs["input_file_id"], "file-in-123")
        self.assertEqual(fake.create_kwargs["endpoint"], "/v1/chat/completions")
        self.assertEqual(fake.create_kwargs["completion_window"], "24h")

    def test_status_is_normalized(self):
        cases = {
            "validating": "in_progress",
            "in_progress": "in_progress",
            "finalizing": "in_progress",
            "completed": "completed",
            "failed": "failed",
            "expired": "failed",
            "cancelled": "failed",
        }
        for raw, expected in cases.items():
            fake = _FakeOpenAIClient(status=raw)
            backend = OpenAIBatchBackend(model="gpt-4o", client_factory=lambda f=fake: f)
            self.assertEqual(backend.status("batch-abc"), expected, raw)

    def test_collect_parses_success_and_error_lines(self):
        output = (
            '{"custom_id":"c1","response":{"status_code":200,'
            '"body":{"choices":[{"message":{"content":"hello"}}]}},"error":null}\n'
            '{"custom_id":"c2","response":null,"error":{"message":"boom"}}\n'
        )
        fake = _FakeOpenAIClient(status="completed", output_text=output)
        backend = OpenAIBatchBackend(model="gpt-4o", client_factory=lambda: fake)

        results = backend.collect("batch-abc")

        self.assertEqual(results, {"c1": "hello", "c2": None})
        self.assertEqual(fake.downloaded_file_id, "file-out-999")

    def test_collect_before_completed_raises(self):
        fake = _FakeOpenAIClient(status="in_progress")
        backend = OpenAIBatchBackend(model="gpt-4o", client_factory=lambda: fake)
        with self.assertRaises(BatchNotReady):
            backend.collect("batch-abc")


# -- Anthropic fake client ---------------------------------------------------


class _FakeAnthropicBatches:
    def __init__(self, client):
        self._c = client

    def create(self, requests):
        self._c.created_requests = requests
        return SimpleNamespace(id="msgbatch-1")

    def retrieve(self, batch_id):
        self._c.retrieved = batch_id
        return SimpleNamespace(processing_status=self._c.processing_status)

    def results(self, batch_id):
        return iter(self._c.results_entries)


class _FakeAnthropicMessages:
    def __init__(self, client):
        self.batches = _FakeAnthropicBatches(client)


class _FakeAnthropicClient:
    def __init__(self, processing_status="ended", results_entries=None):
        self.processing_status = processing_status
        self.results_entries = results_entries or []
        self.created_requests = None
        self.messages = _FakeAnthropicMessages(self)


def _succeeded(custom_id, text):
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(content=[SimpleNamespace(text=text)]),
        ),
    )


def _errored(custom_id):
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="errored"))


class AnthropicBatchBackendTests(unittest.TestCase):
    def test_submit_builds_request_entries_with_model_override(self):
        fake = _FakeAnthropicClient()
        backend = AnthropicBatchBackend(
            model="claude-sonnet-5", max_tokens=8192, client_factory=lambda: fake)
        requests = _requests()

        batch_id = backend.submit(requests)

        self.assertEqual(batch_id, "msgbatch-1")
        entries = fake.created_requests
        self.assertEqual([e["custom_id"] for e in entries], ["c1", "c2"])
        # c1 has no model hint -> the backend default; c2 overrides it.
        self.assertEqual(entries[0]["params"]["model"], "claude-sonnet-5")
        self.assertEqual(entries[1]["params"]["model"], "claude-opus-4")
        self.assertEqual(entries[0]["params"]["max_tokens"], 8192)
        self.assertEqual(entries[0]["params"]["messages"], requests[0]["messages"])

    def test_status_maps_processing_status(self):
        for raw, expected in (("in_progress", "in_progress"), ("ended", "completed")):
            fake = _FakeAnthropicClient(processing_status=raw)
            backend = AnthropicBatchBackend(
                model="claude-sonnet-5", client_factory=lambda f=fake: f)
            self.assertEqual(backend.status("msgbatch-1"), expected, raw)

    def test_collect_parses_succeeded_and_errored_entries(self):
        fake = _FakeAnthropicClient(
            processing_status="ended",
            results_entries=[_succeeded("c1", "answer"), _errored("c2")],
        )
        backend = AnthropicBatchBackend(
            model="claude-sonnet-5", client_factory=lambda: fake)

        results = backend.collect("msgbatch-1")

        self.assertEqual(results, {"c1": "answer", "c2": None})

    def test_collect_before_ended_raises(self):
        fake = _FakeAnthropicClient(processing_status="in_progress")
        backend = AnthropicBatchBackend(
            model="claude-sonnet-5", client_factory=lambda: fake)
        with self.assertRaises(BatchNotReady):
            backend.collect("msgbatch-1")


# -- OpenRouter fake transport -----------------------------------------------


# A real id, as the live service hands them out -- deliberately NOT prefixed
# "local-", which is what keeps it out of the orphan recovery path.
OPENROUTER_BATCH_ID = "batch-1789576284-Ejahe4wq9AgVdp5xGdNm"


class _FakeTransport:
    """Stand-in for :func:`processors.batch._default_openrouter_transport`.

    Records every call as ``(method, url, payload)`` and replays a queue of
    canned ``(status_code, body)`` responses -- the last one repeats, so a test
    that polls twice need only declare the answer once.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, payload):
        self.calls.append((method, url, payload))
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


def _batch_object(status, results=None):
    """The batch object shape the service returns from both POST and GET."""
    return {
        "id": OPENROUTER_BATCH_ID,
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "model": "z-ai/glm-5.3-flash-20260826",
        "completion_window": "24h",
        "status": status,
        "request_counts": {"total": 2, "completed": 0, "failed": 0},
        "results": results,
        "error": None,
    }


# Verbatim from the live service, the response that killed a real deck run.
def _not_found_body(batch_id=None):
    return {"error": {"message": f"Batch job {batch_id or OPENROUTER_BATCH_ID} "
                                 "not found.", "code": 404}}


class _FakeClock:
    """A hand-cranked monotonic clock, so a grace period costs no wall time."""

    def __init__(self, start=1000.0):
        self.value = start

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def _ok_result(custom_id, text):
    return {
        "id": f"gen-{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": "req-1",
            "body": {"choices": [{"message": {"content": text}}]},
        },
        "error": None,
    }


class OpenRouterBatchBackendTests(unittest.TestCase):
    def _backend(self, transport, model="z-ai/glm-5.3-flash:batch"):
        return OpenRouterBatchBackend(
            model=model, api_key="sk-or-xxx", transport=transport)

    def test_submit_posts_once_with_the_documented_key_order(self):
        transport = _FakeTransport([(202, _batch_object("validating"))])
        backend = self._backend(transport)

        batch_id = backend.submit(_requests_for_one_model())

        self.assertEqual(batch_id, OPENROUTER_BATCH_ID)
        self.assertEqual(len(transport.calls), 1)
        method, url, payload = transport.calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(url, "https://openrouter.ai/api/beta/batches")

        # The service stream-parses the body and returns 400 if "requests"
        # arrives before "endpoint" and "model". This is that trap, asserted.
        keys = [key for key, _ in json.loads(payload, object_pairs_hook=list)]
        self.assertEqual(keys, ["endpoint", "model", "requests"])

        body = json.loads(payload)
        self.assertEqual(body["endpoint"], "/v1/chat/completions")
        self.assertEqual(body["model"], "z-ai/glm-5.3-flash:batch")
        self.assertEqual([item["custom_id"] for item in body["requests"]], ["c1", "c2"])
        # Every request inherits the batch-level model; naming it per request
        # is at best redundant and at worst a rejected submission.
        for item in body["requests"]:
            self.assertNotIn("model", item["body"])

    def test_submit_sends_the_bearer_header(self):
        # The header lives in the real transport, so this exercises the default
        # one through a stubbed urlopen rather than the fake.
        captured = {}

        class _Response:
            status = 202

            def read(self):
                return json.dumps(_batch_object("validating")).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(request, timeout=None):
            captured["headers"] = dict(request.header_items())
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["data"] = request.data
            captured["timeout"] = timeout
            return _Response()

        backend = OpenRouterBatchBackend(
            model="z-ai/glm-5.3-flash:batch", api_key="sk-or-xxx")
        with mock.patch("urllib.request.urlopen", fake_urlopen):
            batch_id = backend.submit(_requests_for_one_model())

        self.assertEqual(batch_id, OPENROUTER_BATCH_ID)
        headers = {key.lower(): value for key, value in captured["headers"].items()}
        self.assertEqual(headers["authorization"], "Bearer sk-or-xxx")
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["url"], "https://openrouter.ai/api/beta/batches")
        self.assertEqual(captured["timeout"], OPENROUTER_HTTP_TIMEOUT_SECONDS)

    def test_submit_accepts_200_as_well_as_202(self):
        transport = _FakeTransport([(200, _batch_object("validating"))])
        backend = self._backend(transport)
        self.assertEqual(backend.submit(_requests_for_one_model()), OPENROUTER_BATCH_ID)

    def test_submit_failure_names_the_status_code_and_body(self):
        transport = _FakeTransport([
            (400, {"error": {"message": "requests must not precede model"}})])
        backend = self._backend(transport)

        with self.assertRaises(RuntimeError) as ctx:
            backend.submit(_requests_for_one_model())

        message = str(ctx.exception)
        self.assertIn("400", message)
        self.assertIn("requests must not precede model", message)

    def test_submit_honours_a_custom_base_url(self):
        transport = _FakeTransport([(202, _batch_object("validating"))])
        backend = OpenRouterBatchBackend(
            model="z-ai/glm-5.3-flash:batch", api_key="sk-or-xxx",
            base_url="https://proxy.internal/api/beta/", transport=transport)

        backend.submit(_requests_for_one_model())

        self.assertEqual(transport.calls[0][1], "https://proxy.internal/api/beta/batches")

    def test_status_is_normalized_for_every_documented_state(self):
        cases = {
            "validating": "in_progress",
            "in_progress": "in_progress",
            "finalizing": "in_progress",
            "completed": "completed",
            "failed": "failed",
            "expired": "failed",
            "cancelling": "failed",
            "cancelled": "failed",
        }
        for raw, expected in cases.items():
            transport = _FakeTransport([(200, _batch_object(raw))])
            backend = self._backend(transport)
            self.assertEqual(backend.status(OPENROUTER_BATCH_ID), expected, raw)
            self.assertEqual(
                transport.calls[0],
                ("GET", f"https://openrouter.ai/api/beta/batches/{OPENROUTER_BATCH_ID}", None))

    def test_collect_maps_inline_results_in_one_request(self):
        transport = _FakeTransport([(200, _batch_object("completed", results=[
            _ok_result("c1", "first answer"),
            _ok_result("c2", "second answer"),
        ]))])
        backend = self._backend(transport)

        results = backend.collect(OPENROUTER_BATCH_ID)

        self.assertEqual(results, {"c1": "first answer", "c2": "second answer"})
        # Results ride along with the status in one GET; a second call would be
        # a round trip for data already in hand.
        self.assertEqual(len(transport.calls), 1)

    def test_collect_maps_failed_items_to_none(self):
        errored = {
            "id": "gen-c2", "custom_id": "c2", "response": None,
            "error": {"message": "context length exceeded"},
        }
        server_error = {
            "id": "gen-c3", "custom_id": "c3",
            "response": {"status_code": 500, "request_id": "req-3", "body": {}},
            "error": None,
        }
        no_choices = {
            "id": "gen-c4", "custom_id": "c4",
            "response": {"status_code": 200, "request_id": "req-4",
                         "body": {"choices": []}},
            "error": None,
        }
        transport = _FakeTransport([(200, _batch_object("completed", results=[
            _ok_result("c1", "fine"), errored, server_error, no_choices,
        ]))])
        backend = self._backend(transport)

        results = backend.collect(OPENROUTER_BATCH_ID)

        self.assertEqual(
            results, {"c1": "fine", "c2": None, "c3": None, "c4": None})

    def test_collect_before_completed_raises(self):
        # results is null while the batch runs, which is exactly why collect
        # must refuse rather than return an empty mapping.
        transport = _FakeTransport([(200, _batch_object("in_progress"))])
        backend = self._backend(transport)

        with self.assertRaises(BatchNotReady) as ctx:
            backend.collect(OPENROUTER_BATCH_ID)

        self.assertIn(OPENROUTER_BATCH_ID, str(ctx.exception))
        self.assertIn("in_progress", str(ctx.exception))
        self.assertEqual(len(transport.calls), 1)

    # -- the read-after-write window (measured: +0s 404, +5s in_progress) -----

    def test_status_reads_a_404_right_after_submit_as_in_progress(self):
        # The exact production sequence: submit, poll immediately (404), poll
        # again seconds later (the write has landed).
        transport = _FakeTransport([
            (202, _batch_object("validating")),
            (404, _not_found_body()),
            (200, _batch_object("in_progress")),
        ])
        clock = _FakeClock()
        backend = OpenRouterBatchBackend(
            model="z-ai/glm-5.3-flash:batch", api_key="sk-or-xxx",
            transport=transport, now=clock)

        batch_id = backend.submit(_requests_for_one_model())

        self.assertEqual(backend.status(batch_id), "in_progress")
        clock.advance(5)
        self.assertEqual(backend.status(batch_id), "in_progress")
        self.assertEqual(len(transport.calls), 3)

    def test_collect_on_a_404_right_after_submit_raises_batch_not_ready(self):
        # BatchNotReady, not RuntimeError: "come back later" is what
        # flows/morph.py renders as "still in progress".
        transport = _FakeTransport([
            (202, _batch_object("validating")),
            (404, _not_found_body()),
        ])
        backend = OpenRouterBatchBackend(
            model="z-ai/glm-5.3-flash:batch", api_key="sk-or-xxx",
            transport=transport, now=_FakeClock())

        batch_id = backend.submit(_requests_for_one_model())

        with self.assertRaises(BatchNotReady) as ctx:
            backend.collect(batch_id)
        self.assertIn(batch_id, str(ctx.exception))

    def test_a_404_for_an_unknown_id_is_still_fatal(self):
        # Nothing was submitted through this backend, so a 404 is a genuine
        # one -- a wrong or deleted id -- and must not poll forever.
        transport = _FakeTransport([(404, _not_found_body("batch-nope"))])
        backend = self._backend(transport)

        with self.assertRaises(RuntimeError) as ctx:
            backend.status("batch-nope")

        self.assertNotIsInstance(ctx.exception, BatchNotReady)
        message = str(ctx.exception)
        self.assertIn("404", message)
        self.assertIn("batch-nope", message)

    def test_a_404_after_the_grace_period_is_fatal_again(self):
        transport = _FakeTransport([
            (202, _batch_object("validating")),
            (404, _not_found_body()),
        ])
        clock = _FakeClock()
        backend = OpenRouterBatchBackend(
            model="z-ai/glm-5.3-flash:batch", api_key="sk-or-xxx",
            transport=transport, now=clock)

        batch_id = backend.submit(_requests_for_one_model())
        clock.advance(OPENROUTER_SUBMIT_GRACE_SECONDS + 1)

        with self.assertRaises(RuntimeError) as ctx:
            backend.status(batch_id)
        self.assertNotIsInstance(ctx.exception, BatchNotReady)
        self.assertIn("404", str(ctx.exception))

    def test_a_non_404_inside_the_window_is_still_fatal(self):
        # The window forgives one status code, not every failure: an operator
        # must still see a 500 (or a 401, or a 429) with its body.
        transport = _FakeTransport([
            (202, _batch_object("validating")),
            (500, {"error": {"message": "internal error"}}),
        ])
        backend = OpenRouterBatchBackend(
            model="z-ai/glm-5.3-flash:batch", api_key="sk-or-xxx",
            transport=transport, now=_FakeClock())

        batch_id = backend.submit(_requests_for_one_model())

        with self.assertRaises(RuntimeError) as ctx:
            backend.status(batch_id)
        self.assertNotIsInstance(ctx.exception, BatchNotReady)
        self.assertIn("500", str(ctx.exception))
        self.assertIn("internal error", str(ctx.exception))


class _FakeHTTPResponse:
    """The context-manager shape ``urllib.request.urlopen`` returns."""

    def __init__(self, status=200, body=b'{"id": "batch-1"}'):
        self.status = status
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._body


class OpenRouterTransportTimeoutTests(unittest.TestCase):
    """The real transport must never be able to block forever.

    ``urlopen`` with no ``timeout`` inherits Python's default socket timeout of
    ``None``: a hung connection wedges the calling thread silently, inside a
    ``/collect`` that prints nothing while it waits. These tests pin the timeout
    to the ``urlopen`` call itself -- no network is touched, ``urlopen`` is
    replaced wholesale.
    """

    def test_the_timeout_reaches_urlopen(self):
        transport = _default_openrouter_transport("sk-or-xxx")

        with mock.patch("processors.batch.urllib.request.urlopen") as urlopen:
            urlopen.return_value = _FakeHTTPResponse()
            status_code, body = transport(
                "POST", "https://openrouter.ai/api/beta/batches", '{"x": 1}')

        self.assertEqual((status_code, body), (200, {"id": "batch-1"}))
        self.assertEqual(
            urlopen.call_args[1]["timeout"], OPENROUTER_HTTP_TIMEOUT_SECONDS)

    def test_the_timeout_reaches_urlopen_on_a_poll_too(self):
        # The GET path carries no payload; it must be bounded all the same --
        # polling is where a batch run spends nearly all of its wall time.
        transport = _default_openrouter_transport("sk-or-xxx")

        with mock.patch("processors.batch.urllib.request.urlopen") as urlopen:
            urlopen.return_value = _FakeHTTPResponse(body=b'{"status": "completed"}')
            transport("GET", "https://openrouter.ai/api/beta/batches/b1", None)

        self.assertEqual(
            urlopen.call_args[1]["timeout"], OPENROUTER_HTTP_TIMEOUT_SECONDS)

    def test_a_read_timeout_becomes_a_readable_error(self):
        transport = _default_openrouter_transport("sk-or-xxx")

        with mock.patch("processors.batch.urllib.request.urlopen") as urlopen:
            urlopen.side_effect = socket.timeout("timed out")
            with self.assertRaises(RuntimeError) as ctx:
                transport("GET", "https://openrouter.ai/api/beta/batches/b1", None)

        message = str(ctx.exception)
        self.assertIn("timed out after 120s", message)
        self.assertIn("GET https://openrouter.ai/api/beta/batches/b1", message)

    def test_a_connect_timeout_becomes_a_readable_error(self):
        # A connect that expires arrives wrapped in URLError, not raw.
        transport = _default_openrouter_transport("sk-or-xxx")

        with mock.patch("processors.batch.urllib.request.urlopen") as urlopen:
            urlopen.side_effect = urllib.error.URLError(socket.timeout("timed out"))
            with self.assertRaises(RuntimeError) as ctx:
                transport("POST", "https://openrouter.ai/api/beta/batches", "{}")

        self.assertIn("timed out after 120s", str(ctx.exception))

    def test_a_non_timeout_transport_error_is_left_alone(self):
        # DNS failures and refused connections are not this change's business:
        # they already fail fast and name their own cause.
        transport = _default_openrouter_transport("sk-or-xxx")

        with mock.patch("processors.batch.urllib.request.urlopen") as urlopen:
            urlopen.side_effect = urllib.error.URLError("name resolution failed")
            with self.assertRaises(urllib.error.URLError):
                transport("GET", "https://openrouter.ai/api/beta/batches/b1", None)

    def test_an_http_error_is_still_returned_not_raised(self):
        # HTTPError subclasses URLError; the new except clause must not swallow
        # the 4xx/5xx path that turns a status code into a readable message.
        transport = _default_openrouter_transport("sk-or-xxx")
        error = urllib.error.HTTPError(
            "https://openrouter.ai/api/beta/batches", 429, "Too Many Requests",
            {}, io.BytesIO(b'{"error": {"message": "rate limited"}}'))

        with mock.patch("processors.batch.urllib.request.urlopen") as urlopen:
            urlopen.side_effect = error
            status_code, body = transport(
                "POST", "https://openrouter.ai/api/beta/batches", "{}")

        self.assertEqual(status_code, 429)
        self.assertEqual(body["error"]["message"], "rate limited")


class OpenRouterOrphanRecoveryTests(unittest.TestCase):
    """An OpenRouter batch id must never be swept up by the local-batch repair.

    A cloud batch survives the CLI process: it is running on OpenRouter's
    servers for up to 24h and is genuinely collectable later, so returning its
    cards to ``pending`` would silently duplicate the work (and the spend).
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-openrouter-")
        self.store = DeckStore(project_root=self.tmp)
        self.store.add_card({
            "custom_id": "a",
            "meta": {"intent": "generate", "target": "a.py", "context_slice": []},
            "instruction": "do a",
        })

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_openrouter_batch_id_is_not_recovered(self):
        state = self.store.load_state()
        state["phase"] = "submitted"
        state["generations"] = [["a"]]
        state["batch_id"] = OPENROUTER_BATCH_ID
        state["backend_label"] = "glm"
        state["submitted_ids"] = ["a"]
        self.store.save_state(state)

        self.assertFalse(recover_orphaned_local_batch(self.store))

        state = self.store.load_state()
        self.assertEqual(state["phase"], "submitted")
        self.assertEqual(state["batch_id"], OPENROUTER_BATCH_ID)
        self.assertEqual(state["submitted_ids"], ["a"])


# -- Local emulation ---------------------------------------------------------


class _FakeRegistry:
    """A stand-in for :class:`processors.registry.ProcessorRegistry`.

    ``LocalBatchBackend`` only ever calls ``run(processor_id, messages)``.
    Processors in ``failing_ids`` raise, to exercise the per-request failure
    path.
    """

    def __init__(self, failing_ids=()):
        self.failing_ids = set(failing_ids)

    def run(self, processor_id, messages):
        if processor_id in self.failing_ids:
            raise RuntimeError("processor exploded")
        return f"response from {processor_id}"


class LocalBatchBackendTests(unittest.TestCase):
    def _deck_requests(self):
        cards = [
            MorphCard(custom_id="c1", intent="generate", target="a.py",
                      instruction="do a", context_slice=["app.py"]),
            MorphCard(custom_id="c2", intent="generate", target="b.py",
                      instruction="do b", context_slice=["util.py"]),
        ]
        return compile_deck(cards, root=MINIPROJECT)

    def test_drains_deck_and_collects_both_results(self):
        registry = _FakeRegistry()
        backend = LocalBatchBackend(registry, ["node-a", "node-b"])

        batch_id = backend.submit(self._deck_requests())
        results = backend.collect(batch_id)

        self.assertEqual(set(results), {"c1", "c2"})
        self.assertTrue(all(value is not None for value in results.values()))
        # Every slot is released once the deck has drained.
        self.assertEqual(backend.scheduler.busy_count(), 0)

    def test_processor_failure_yields_none_for_its_request_only(self):
        # node-b raises; rotation routes c1 -> node-a, c2 -> node-b.
        registry = _FakeRegistry(failing_ids=["node-b"])
        backend = LocalBatchBackend(registry, ["node-a", "node-b"])

        batch_id = backend.submit(self._deck_requests())
        results = backend.collect(batch_id)

        self.assertEqual(set(results), {"c1", "c2"})
        nones = [cid for cid, value in results.items() if value is None]
        succeeded = [value for value in results.values() if value is not None]
        self.assertEqual(len(nones), 1)
        self.assertEqual(len(succeeded), 1)
        self.assertTrue(succeeded[0].startswith("response from node-a"))
        self.assertEqual(backend.scheduler.busy_count(), 0)

    def test_status_reports_completed_after_collect(self):
        backend = LocalBatchBackend(_FakeRegistry(), ["node-a"])
        batch_id = backend.submit(self._deck_requests())
        backend.collect(batch_id)
        self.assertEqual(backend.status(batch_id), "completed")


# -- Registry wiring ---------------------------------------------------------


class RegistryBatchTests(unittest.TestCase):
    def test_anthropic_discovered_from_env(self):
        env = {
            "ANTHROPIC_API_KEY": "sk-ant-xxx",
            "ANTHROPIC_MODEL_NAME": "claude-opus-4",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            registry = ProcessorRegistry.from_env()
            self.assertIn("anthropic", registry.ids)
            config = registry.get("anthropic")
            self.assertEqual(config.kind, "anthropic")
            self.assertEqual(config.params["model"], "claude-opus-4")

    def test_anthropic_default_model_when_unset(self):
        env = {"ANTHROPIC_API_KEY": "sk-ant-xxx"}
        with mock.patch.dict(os.environ, env, clear=True):
            registry = ProcessorRegistry.from_env()
            self.assertEqual(registry.get("anthropic").params["model"], "claude-sonnet-5")

    def test_batch_returns_the_right_backend_per_type(self):
        env = {
            "MRPH_PROCESSORS": "gpt,claude,local",
            "MRPH_PROCESSOR_gpt_TYPE": "openai",
            "MRPH_PROCESSOR_gpt_API_KEY": "sk-openai",
            "MRPH_PROCESSOR_gpt_MODEL": "gpt-4o",
            "MRPH_PROCESSOR_claude_TYPE": "anthropic",
            "MRPH_PROCESSOR_claude_API_KEY": "sk-ant",
            "MRPH_PROCESSOR_claude_MODEL": "claude-sonnet-5",
            "MRPH_PROCESSOR_local_TYPE": "llama_cpp",
            "MRPH_PROCESSOR_local_ENDPOINT_URI": "http://192.168.0.14:8080/v1",
            "MRPH_PROCESSOR_local_MODEL": "k80-model",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            registry = ProcessorRegistry.from_env()
            self.assertIsInstance(registry.batch("gpt"), OpenAIBatchBackend)
            self.assertIsInstance(registry.batch("claude"), AnthropicBatchBackend)
            self.assertIsInstance(registry.batch("local"), LocalBatchBackend)

    def test_openrouter_is_configured_and_batched(self):
        env = {
            "MRPH_PROCESSOR_glm_TYPE": "openrouter",
            "MRPH_PROCESSOR_glm_API_KEY": "sk-or-xxx",
            "MRPH_PROCESSOR_glm_MODEL": "z-ai/glm-5.3-flash:batch",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            registry = ProcessorRegistry.from_env()
            self.assertIn("glm", registry.ids)
            self.assertEqual(
                registry.get("glm").describe(),
                '[glm] OpenRouter (model "z-ai/glm-5.3-flash:batch")')

            backend = registry.batch("glm")
            self.assertIsInstance(backend, OpenRouterBatchBackend)
            self.assertEqual(backend.model, "z-ai/glm-5.3-flash:batch")
            self.assertEqual(backend.base_url, "https://openrouter.ai/api/beta")

    def test_openrouter_base_url_overrides_the_batch_path(self):
        env = {
            "MRPH_PROCESSOR_glm_TYPE": "openrouter",
            "MRPH_PROCESSOR_glm_API_KEY": "sk-or-xxx",
            "MRPH_PROCESSOR_glm_MODEL": "z-ai/glm-5.3-flash:batch",
            "MRPH_PROCESSOR_glm_BASE_URL": "https://proxy.internal/api/beta",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            registry = ProcessorRegistry.from_env()
            self.assertEqual(
                registry.batch("glm").base_url, "https://proxy.internal/api/beta")

    def test_openrouter_needs_both_a_key_and_a_model(self):
        # The model slug picks the vendor AND the price tier, so there is no
        # sensible default to fall back to: an incomplete block is skipped.
        for missing in ("MRPH_PROCESSOR_glm_API_KEY", "MRPH_PROCESSOR_glm_MODEL"):
            env = {
                "MRPH_PROCESSOR_glm_TYPE": "openrouter",
                "MRPH_PROCESSOR_glm_API_KEY": "sk-or-xxx",
                "MRPH_PROCESSOR_glm_MODEL": "z-ai/glm-5.3-flash:batch",
            }
            del env[missing]
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(ProcessorRegistry.from_env().ids, [], missing)

    def test_batch_pool_spans_several_local_nodes(self):
        env = {
            "MRPH_PROCESSORS": "k80-a,k80-b",
            "MRPH_PROCESSOR_k80-a_TYPE": "llama_cpp",
            "MRPH_PROCESSOR_k80-a_ENDPOINT_URI": "http://192.168.0.14:8080/v1",
            "MRPH_PROCESSOR_k80-a_MODEL": "k80-model",
            "MRPH_PROCESSOR_k80-b_TYPE": "ollama",
            "MRPH_PROCESSOR_k80-b_ENDPOINT_URI": "http://192.168.0.15:11434",
            "MRPH_PROCESSOR_k80-b_MODEL": "k80-model",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            registry = ProcessorRegistry.from_env()
            pool = registry.batch_pool(["k80-a", "k80-b"])
            self.assertIsInstance(pool, LocalBatchBackend)
            self.assertEqual(len(pool.scheduler), 2)


if __name__ == "__main__":
    unittest.main()
