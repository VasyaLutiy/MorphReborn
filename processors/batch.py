"""
Batch backends: submit / status / collect, next to the synchronous ``run``.

Phase 2 of ``documentation/DEVELOPMENT_PLAN.md``. Where ``processors/registry.py``
runs one morph interactively, a *batch backend* runs a whole compiled deck
asynchronously -- the half-price, massively parallel execution path described in
``documentation/batch-orchestrator.md`` ("What batch APIs offer" and "What
survives from the current codebase"). Three backends implement one small
interface:

* :class:`OpenAIBatchBackend` -- the OpenAI Batch API (upload JSONL file,
  create a batch over ``/v1/chat/completions``, poll, download the output file).
* :class:`AnthropicBatchBackend` -- Anthropic Message Batches
  (``client.messages.batches``), the first Anthropic code path in the project.
* :class:`LocalBatchBackend` -- the "department minicomputer": no provider batch
  endpoint exists for a llama.cpp/Ollama node, so this emulates one by draining
  the deck through the existing :class:`scheduler.JobScheduler` pool, exactly the
  K80-nodes-as-overnight-batch story from the orchestrator design.

The unit shared with :mod:`cards.compiler` is the *internal request dict*:
``{"custom_id", "model" (may be None), "messages"}``. The SDK-backed backends
reuse the compiler's serializers so the wire format has a single source of
truth. No provider SDK is imported at module load time -- the ``openai`` and
``anthropic`` packages are imported lazily inside the default client factories,
so the test suite runs (and this module imports) in an environment where
``anthropic`` is not installed at all.
"""

import abc
import io
import json
import os
import threading
import uuid
from typing import Callable, Dict, List, Optional

from cards.compiler import serialize_openai
from scheduler import JobScheduler


class BatchNotReady(RuntimeError):
    """Raised by ``collect`` when a batch has not reached ``completed`` yet.

    The provider backends refuse to parse results before the batch has ended;
    the message names the batch id and its current normalized status so the
    caller knows to keep polling.
    """


class BatchBackend(abc.ABC):
    """The common submit / poll / collect contract for a deck executor.

    ``requests`` are the compiler's internal request dicts (see the module
    docstring). ``status`` is normalized to one of ``"in_progress"``,
    ``"completed"`` or ``"failed"`` across every provider, so the orchestrator's
    polling loop never has to know which backend it is talking to.
    """

    @abc.abstractmethod
    def submit(self, requests: List[dict]) -> str:
        """Submit a deck of request dicts; return a batch id to poll with."""

    @abc.abstractmethod
    def status(self, batch_id: str) -> str:
        """Return ``"in_progress"`` | ``"completed"`` | ``"failed"``."""

    @abc.abstractmethod
    def collect(self, batch_id: str) -> Dict[str, Optional[str]]:
        """Return ``custom_id -> response text`` (``None`` for an errored one)."""


# -- OpenAI Batch API --------------------------------------------------------


def _default_openai_factory(api_key: Optional[str], base_url: Optional[str]) -> Callable[[], object]:
    """A zero-arg factory that lazily builds an OpenAI client.

    The ``openai`` import lives inside the returned callable, never at module
    import time, so importing this module costs no SDK. Per-instance credentials
    win over the process-wide environment, mirroring
    :class:`processors.openai_processor.OpenAIProcessor`.
    """

    def factory():
        import openai

        kwargs = {}
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            kwargs["api_key"] = key
        if base_url:
            kwargs["base_url"] = base_url
        return openai.OpenAI(**kwargs)

    return factory


class OpenAIBatchBackend(BatchBackend):
    """Execute a deck through the OpenAI Batch API.

    The flow follows OpenAI's documented batch lifecycle: upload the deck as a
    JSONL file with ``purpose="batch"``, create a batch over
    ``/v1/chat/completions`` with a 24h completion window, poll
    ``batches.retrieve``, then download and parse the output file. The wire
    format is produced by :func:`cards.compiler.serialize_openai` so the batch
    file is byte-identical to the golden-tested serialization.

    ``client_factory`` is the test seam: a zero-arg callable returning the
    client. The default lazily imports the ``openai`` SDK.
    """

    def __init__(self, model: str, api_key: str = None, base_url: str = None,
                 client_factory: Callable[[], object] = None):
        self.model = model
        self._factory = client_factory or _default_openai_factory(api_key, base_url)
        self._client = None

    def client(self):
        """The lazily-built (and memoized) SDK client."""
        if self._client is None:
            self._client = self._factory()
        return self._client

    def submit(self, requests: List[dict]) -> str:
        jsonl = serialize_openai(requests, default_model=self.model)
        client = self.client()
        upload = client.files.create(
            file=("morph-batch.jsonl", jsonl.encode("utf-8")),
            purpose="batch",
        )
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return batch.id

    def status(self, batch_id: str) -> str:
        batch = self.client().batches.retrieve(batch_id)
        return self._normalize(batch.status)

    @staticmethod
    def _normalize(status: str) -> str:
        # OpenAI batch states: validating, in_progress, finalizing, completed,
        # failed, expired, cancelling, cancelled. Anything that is not a clean
        # completion or a terminal failure is still "work in progress".
        if status == "completed":
            return "completed"
        if status in ("failed", "expired", "cancelled", "canceled", "cancelling"):
            return "failed"
        return "in_progress"

    def collect(self, batch_id: str) -> Dict[str, Optional[str]]:
        current = self.status(batch_id)
        if current != "completed":
            raise BatchNotReady(
                f"OpenAI batch {batch_id!r} is {current!r}, not yet completed")

        client = self.client()
        batch = client.batches.retrieve(batch_id)
        output = client.files.content(batch.output_file_id)
        # ``files.content`` returns a binary-response wrapper exposing ``.text``.
        text = output.text

        results: Dict[str, Optional[str]] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            results[entry.get("custom_id")] = self._parse_line(entry)
        return results

    @staticmethod
    def _parse_line(entry: dict) -> Optional[str]:
        """One output line -> response text, or ``None`` for an errored one."""
        if entry.get("error"):
            return None
        response = entry.get("response")
        if not response:
            return None
        status_code = response.get("status_code")
        if status_code is not None and status_code != 200:
            return None
        body = response.get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        return message.get("content")


# -- Anthropic Message Batches -----------------------------------------------


def _default_anthropic_factory(api_key: Optional[str]) -> Callable[[], object]:
    """A zero-arg factory that lazily builds an Anthropic client.

    The ``anthropic`` import is inside the callable so the package is not
    required merely to import this module or run the test suite.
    """

    def factory():
        import anthropic

        kwargs = {}
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if key:
            kwargs["api_key"] = key
        return anthropic.Anthropic(**kwargs)

    return factory


class AnthropicBatchBackend(BatchBackend):
    """Execute a deck through Anthropic Message Batches.

    Request entries are built directly as objects (the dict shape of
    :func:`cards.compiler.serialize_anthropic`'s lines, but passed to the SDK
    rather than serialized to JSONL): ``{custom_id, params: {model, max_tokens,
    messages}}``. Polling reads ``processing_status`` (``in_progress`` while the
    batch runs, ``ended`` once every request is resolved -- there is no
    batch-level failure state, individual failures surface per result). Results
    come from the results iterator: an entry whose ``result.type`` is
    ``"succeeded"`` yields its first text block, anything else yields ``None``.

    ``client_factory`` is the test seam; the default lazily imports ``anthropic``.
    """

    def __init__(self, model: str, api_key: str = None, max_tokens: int = 8192,
                 client_factory: Callable[[], object] = None):
        self.model = model
        self.max_tokens = max_tokens
        self._factory = client_factory or _default_anthropic_factory(api_key)
        self._client = None

    def client(self):
        """The lazily-built (and memoized) SDK client."""
        if self._client is None:
            self._client = self._factory()
        return self._client

    def submit(self, requests: List[dict]) -> str:
        entries = []
        for request in requests:
            entries.append({
                "custom_id": request["custom_id"],
                "params": {
                    "model": request["model"] or self.model,
                    "max_tokens": self.max_tokens,
                    "messages": request["messages"],
                },
            })
        batch = self.client().messages.batches.create(requests=entries)
        return batch.id

    def status(self, batch_id: str) -> str:
        batch = self.client().messages.batches.retrieve(batch_id)
        # Anthropic exposes only "in_progress" / "ended"; "ended" means every
        # request is resolved (succeeded or errored), which is our "completed".
        if batch.processing_status == "ended":
            return "completed"
        return "in_progress"

    def collect(self, batch_id: str) -> Dict[str, Optional[str]]:
        current = self.status(batch_id)
        if current != "completed":
            raise BatchNotReady(
                f"Anthropic batch {batch_id!r} is {current!r}, not yet ended")

        results: Dict[str, Optional[str]] = {}
        for entry in self.client().messages.batches.results(batch_id):
            results[entry.custom_id] = self._extract(entry.result)
        return results

    @staticmethod
    def _extract(result) -> Optional[str]:
        """A per-request result object -> its text, or ``None`` if not succeeded."""
        if getattr(result, "type", None) != "succeeded":
            return None
        content = getattr(result.message, "content", None) or []
        if not content:
            return None
        return getattr(content[0], "text", None)


# -- Local emulation over the JobScheduler pool ------------------------------


class _LocalBatchState:
    """Bookkeeping for one in-flight local batch: results and completion."""

    def __init__(self, total: int):
        self._lock = threading.Lock()
        self._pending = total
        self.results: Dict[str, Optional[str]] = {}
        self.threads: List[threading.Thread] = []
        self._done = threading.Event()
        if total == 0:
            self._done.set()

    def record(self, custom_id: str, value: Optional[str]) -> None:
        with self._lock:
            self.results[custom_id] = value
            self._pending -= 1
            if self._pending == 0:
                self._done.set()

    def is_done(self) -> bool:
        return self._done.is_set()

    def wait(self) -> None:
        self._done.wait()
        for thread in list(self.threads):
            thread.join()


class LocalBatchBackend(BatchBackend):
    """Emulate a batch endpoint by draining the deck through the local pool.

    No llama.cpp/Ollama node offers a provider batch API, so this backend *is*
    the batch endpoint: it drives the same :class:`scheduler.JobScheduler` the
    interactive ``/generate`` fan-out uses, so a submitted deck spreads across
    the pool exactly the way concurrent interactive jobs do -- the K80 nodes
    become the overnight minicomputer of ``batch-orchestrator.md``.

    Each request becomes a rotation job (no pin). The scheduler assigns it a
    processor slot and fires our launch callback, which runs
    ``registry.run(processor_id, messages)`` on a worker thread (``registry.run``
    is blocking network I/O, so threads give real concurrency). When the worker
    finishes it releases the slot back to the scheduler, which immediately
    dispatches the next queued request. A request whose processor raises records
    ``None``. The request's ``model`` field is ignored: a local node's model is
    fixed by its own config, and ``registry.run`` already uses it.

    ``submit`` returns at once with a locally-generated id; ``status`` reflects
    live progress; ``collect`` joins the workers and returns the results.
    """

    def __init__(self, registry, processor_ids: List[str]):
        self._registry = registry
        self.scheduler = JobScheduler(processor_ids)
        # One lock serializes all scheduler mutations, since release() (called
        # from worker threads) dispatches queued jobs and is not itself
        # thread-safe.
        self._lock = threading.Lock()
        self._batches: Dict[str, _LocalBatchState] = {}

    def submit(self, requests: List[dict]) -> str:
        batch_id = f"local-{uuid.uuid4().hex}"
        state = _LocalBatchState(len(requests))
        self._batches[batch_id] = state
        for request in requests:
            self._enqueue(state, request)
        return batch_id

    def _enqueue(self, state: _LocalBatchState, request: dict) -> None:
        with self._lock:
            job = self.scheduler.submit(None)

            def launch(job=job, request=request):
                processor_id = job.assigned[0]
                worker = threading.Thread(
                    target=self._run_job,
                    args=(state, request, job, processor_id),
                    daemon=True,
                )
                state.threads.append(worker)
                worker.start()

            # Fires synchronously here if a slot was free, else when one frees.
            self.scheduler.attach_launch(job, launch)

    def _run_job(self, state, request, job, processor_id) -> None:
        try:
            value = self._registry.run(processor_id, request["messages"])
        except Exception:
            # A processor that raises fails only its own request; the rest of
            # the deck keeps draining.
            value = None
        state.record(request["custom_id"], value)
        with self._lock:
            self.scheduler.release(job.assigned)

    def status(self, batch_id: str) -> str:
        # Local execution has no batch-level failure state -- individual
        # failures surface as ``None`` in ``collect`` -- so a batch is either
        # still draining or done.
        return "completed" if self._batches[batch_id].is_done() else "in_progress"

    def collect(self, batch_id: str) -> Dict[str, Optional[str]]:
        state = self._batches[batch_id]
        state.wait()
        return dict(state.results)
