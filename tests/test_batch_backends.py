"""
Phase 2 tests: the batch backends (submit / status / collect).

No network and no real provider SDKs. The cloud backends are driven through
fake clients injected via ``client_factory``; the local backend runs against a
fake registry of fake processors. The suite must pass in an environment where
``anthropic`` is not installed at all (see the no-SDK-at-import constraint in
``processors/batch.py``).
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from cards.compiler import compile_deck, serialize_openai
from cards.schema import MorphCard
from processors.batch import (
    AnthropicBatchBackend,
    BatchNotReady,
    LocalBatchBackend,
    OpenAIBatchBackend,
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
