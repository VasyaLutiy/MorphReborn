"""
Batch backends: submit / status / collect, next to the synchronous ``run``.

Phase 2 of ``documentation/DEVELOPMENT_PLAN.md``. Where ``processors/registry.py``
runs one morph interactively, a *batch backend* runs a whole compiled deck
asynchronously -- the half-price, massively parallel execution path described in
``documentation/batch-orchestrator.md`` ("What batch APIs offer" and "What
survives from the current codebase"). Four backends implement one small
interface:

* :class:`OpenAIBatchBackend` -- the OpenAI Batch API (upload JSONL file,
  create a batch over ``/v1/chat/completions``, poll, download the output file).
* :class:`AnthropicBatchBackend` -- Anthropic Message Batches
  (``client.messages.batches``), the first Anthropic code path in the project.
* :class:`OpenRouterBatchBackend` -- the OpenRouter Batch API. No SDK and no
  file upload: the deck is POSTed inline as one JSON document and the results
  come back inline in the poll response, over ``urllib.request``.
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
``anthropic`` is not installed at all; OpenRouter needs no SDK whatsoever, only
``urllib`` from the stdlib.
"""

import abc
import io
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Callable, Dict, List, Optional, Tuple

from cards.compiler import serialize_openai, serialize_openrouter
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


# -- OpenRouter Batch API ----------------------------------------------------


# OpenRouter's batch API lives under /api/beta, NOT under the /api/v1 path their
# OpenAI-compatible synchronous API uses. One processor id therefore talks to two
# base paths: this one for decks, ``.../api/v1`` for interactive /generate.
OPENROUTER_BATCH_BASE_URL = "https://openrouter.ai/api/beta"

# How much of a failing response body an error message carries. Enough for an
# operator to see WHY a deck did not go out, short enough to stay readable.
_ERROR_BODY_LIMIT = 500

# OpenRouter's batch API is read-after-write inconsistent: a submission answers
# 202 with a batch id, but a GET of that id issued immediately after can answer
# 404 {"error": {"message": "Batch job <id> not found.", "code": 404}} before the
# write has propagated. Measured against the live service on one id:
# ``+0s -> HTTP 404``, ``+5s -> in_progress``, ``+20s -> in_progress``. Inside
# this grace period a 404 for the id WE just submitted therefore means "not
# visible yet", not "gone"; outside it, or for any other id, a 404 is a real
# error and stays fatal. Do not delete this as defensive noise -- it cost a real
# deck run: ``cards/generations.py::_submit_poll_collect`` polls the instant it
# submits, so a regenerated card killed the whole /collect after two of three
# cards had already passed.
OPENROUTER_SUBMIT_GRACE_SECONDS = 60.0

# How long one HTTP call to the batch API may take before it is abandoned.
# ``urlopen``'s single ``timeout`` covers the connect AND every socket read, so
# one number bounds the whole call. Without it a hung connection blocks FOREVER:
# ``urlopen`` inherits Python's default socket timeout, which is ``None`` --
# and the block happens inside a ``/collect`` that prints nothing while it waits,
# so the operator sees a deck that is simply never collected, with no error, no
# traceback and no log line to act on.
#
# Two minutes, not ten seconds: submit POSTs the whole compiled deck inline and
# a completed poll returns every result inline (see the class docstring), so
# these are genuinely large bodies on a slow link. The number bounds a wedged
# socket, it does not police latency.
OPENROUTER_HTTP_TIMEOUT_SECONDS = 120.0


class _BatchNotVisibleYet(RuntimeError):
    """Internal: a 404 for a just-submitted id, inside the grace period above.

    Never escapes :class:`OpenRouterBatchBackend` -- ``status`` turns it into
    ``"in_progress"`` (the word the polling loop understands as "keep going")
    and ``collect`` into :class:`BatchNotReady` (the "come back later" signal
    ``flows/morph.py`` renders as "still in progress"). It exists so the one
    404-is-not-fatal decision lives in ``_retrieve``, in one place, while the
    two callers each answer it in their own vocabulary.
    """


def _default_openrouter_transport(api_key: Optional[str]) -> Callable[[str, str, Optional[str]], Tuple[int, dict]]:
    """Build the real HTTP transport for one backend instance.

    OpenRouter ships no SDK for the batch API and the endpoints are two plain
    JSON calls, so this uses ``urllib.request`` from the stdlib rather than
    adding a dependency (the development plan's "no new heavy dependencies"
    rule). The returned callable is the whole network surface of
    :class:`OpenRouterBatchBackend`, which is what makes the backend testable
    offline: a fake with the same signature replaces it wholesale.

    A 4xx/5xx is returned as ``(status, body)`` rather than raised -- the
    backend turns it into an error message naming the status code, and an error
    body is exactly the part an operator needs to read.

    A TIMEOUT is the one transport-level failure translated here (see
    :data:`OPENROUTER_HTTP_TIMEOUT_SECONDS`). It arrives in two shapes --
    ``socket.timeout`` raised out of a read, or wrapped in a
    ``urllib.error.URLError`` when the connect is what expired -- and both are
    re-raised as a ``RuntimeError`` naming the seconds, the method and the URL,
    because "the batch API did not answer in 120s" is something an operator can
    act on and a bare socket traceback is not.
    """

    def transport(method: str, url: str, payload: Optional[str]) -> Tuple[int, dict]:
        data = payload.encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Authorization", f"Bearer {api_key}")
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(
                    request, timeout=OPENROUTER_HTTP_TIMEOUT_SECONDS) as response:
                return response.status, _parse_json(response.read())
        except urllib.error.HTTPError as error:
            # Checked before URLError: HTTPError subclasses it, and an HTTP
            # error status is an ANSWER, not a transport failure.
            return error.code, _parse_json(error.read())
        except (socket.timeout, urllib.error.URLError) as error:
            if isinstance(error, socket.timeout) or isinstance(
                    getattr(error, "reason", None), socket.timeout):
                raise RuntimeError(
                    f"OpenRouter batch request timed out after "
                    f"{OPENROUTER_HTTP_TIMEOUT_SECONDS:.0f}s ({method} {url}); "
                    f"the batch may still exist -- re-run /collect before "
                    f"re-submitting") from error
            raise

    return transport


def _parse_json(raw: bytes) -> dict:
    """Decode a response body, degrading to ``{"raw": ...}`` when it is not JSON.

    An error page (a gateway timeout, an HTML 502) must not crash the caller
    before it can report the status code it came with.
    """
    text = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except ValueError:
        return {"raw": text}
    if not isinstance(parsed, dict):
        return {"raw": parsed}
    return parsed


class OpenRouterBatchBackend(BatchBackend):
    """Execute a deck through the OpenRouter Batch API (half price, 24h window).

    OpenRouter's batch API is its own shape, not an OpenAI-compatible one, so
    none of :class:`OpenAIBatchBackend` is reusable here:

    * **Submit is inline.** There is no file upload and no JSONL -- the whole
      deck is POSTed as one JSON document to ``<base>/batches`` and OpenRouter
      persists it internally. :func:`cards.compiler.serialize_openrouter` builds
      that body, including the ``endpoint`` -> ``model`` -> ``requests`` key
      order the stream-parsing service requires.
    * **One model per batch.** ``model`` is a batch-level field; the serializer
      rejects a deck that pins a different one per card.
    * **Results are inline too.** A completed batch carries its ``results``
      array in the very same poll response, so ``collect`` makes exactly one
      request and never downloads an output file.

    A batch the provider rejected carries the explanation on the batch object
    itself -- ``{"error": {"message": "HTTP 400: invalid batch inference job:
    ..."}}`` -- which ``status`` would reduce to a bare ``"failed"``;
    :meth:`failure_reason` surfaces that message so an operator whose deck
    reports every card failed can read WHY without curling the batch API by
    hand.

    ``transport`` is the test seam, mirroring the other backends'
    ``client_factory``: a callable ``(method, url, payload) -> (status_code,
    parsed_json)``. The default one does real HTTP over ``urllib.request``.
    ``now`` is the second seam: a zero-arg clock, so the read-after-write grace
    period (see :data:`OPENROUTER_SUBMIT_GRACE_SECONDS`) can be tested without
    sleeping.
    """

    def __init__(self, model: str, api_key: str = None, base_url: str = None,
                 transport: Callable[[str, str, Optional[str]], Tuple[int, dict]] = None,
                 now: Callable[[], float] = None):
        self.model = model
        self.base_url = (base_url or OPENROUTER_BATCH_BASE_URL).rstrip("/")
        self._transport = transport or _default_openrouter_transport(api_key)
        # ``monotonic``, not ``time()``: the grace period measures an elapsed
        # few seconds and must not be moved by an NTP step or a DST change.
        self._now = now or time.monotonic
        # The id this instance submitted most recently, and when -- the only
        # id whose 404 we are willing to read as "not propagated yet".
        self._submitted_id: Optional[str] = None
        self._submitted_at: Optional[float] = None

    def submit(self, requests: List[dict]) -> str:
        payload = serialize_openrouter(requests, default_model=self.model)
        status_code, body = self._transport("POST", f"{self.base_url}/batches", payload)
        # The service answers a successful submission with 202 Accepted (the
        # deck is queued, not run); 200 is accepted too, so a future change of
        # heart about the code does not wedge every deck.
        if status_code not in (200, 202):
            raise RuntimeError(
                f"OpenRouter batch submission failed with HTTP {status_code}: "
                f"{self._trim(body)}")
        batch_id = body["id"]
        self._submitted_id = batch_id
        self._submitted_at = self._now()
        return batch_id

    def _retrieve(self, batch_id: str) -> dict:
        """GET one batch object, raising on anything but a 200.

        The single exception is a 404 within the read-after-write window of our
        own submission, which raises :class:`_BatchNotVisibleYet` instead: the
        batch exists, the service just cannot see its own write yet. Every other
        status code -- a 401, a 429, a 500, and a 404 for an id we did not
        submit or one whose grace period has run out -- stays the RuntimeError
        it has always been, body included, because an operator must read it.
        """
        status_code, body = self._transport(
            "GET", f"{self.base_url}/batches/{batch_id}", None)
        if status_code == 404 and self._inside_submit_grace(batch_id):
            raise _BatchNotVisibleYet(
                f"OpenRouter batch {batch_id!r} is not readable yet: HTTP 404 "
                f"within {OPENROUTER_SUBMIT_GRACE_SECONDS:g}s of its submission")
        if status_code != 200:
            raise RuntimeError(
                f"OpenRouter batch {batch_id!r} could not be read: HTTP "
                f"{status_code}: {self._trim(body)}")
        return body

    @staticmethod
    def _trim(body: dict) -> str:
        """A response body, shortened to something an operator can read."""
        text = json.dumps(body, ensure_ascii=False)
        if len(text) > _ERROR_BODY_LIMIT:
            return f"{text[:_ERROR_BODY_LIMIT]}..."
        return text

    def _inside_submit_grace(self, batch_id: str) -> bool:
        """Did THIS instance submit ``batch_id`` within the grace period?

        Deliberately narrow: an id we never submitted (a typo, a batch from a
        previous process, a deleted one) is never covered, so a genuine 404
        still fails loudly instead of polling forever.
        """
        if batch_id != self._submitted_id or self._submitted_at is None:
            return False
        return (self._now() - self._submitted_at) < OPENROUTER_SUBMIT_GRACE_SECONDS

    def status(self, batch_id: str) -> str:
        try:
            batch = self._retrieve(batch_id)
        except _BatchNotVisibleYet:
            # "Keep polling" -- the next poll, seconds later, reads the real one.
            return "in_progress"
        return self._normalize(batch.get("status"))

    @staticmethod
    def _normalize(status: str) -> str:
        # OpenRouter batch states: validating, in_progress, finalizing,
        # completed, failed, expired, cancelling, cancelled -- the same
        # vocabulary OpenAI uses, mapped onto the same three words
        # ``cards/generations.py`` and ``cards/store.py`` depend on.
        if status == "completed":
            return "completed"
        if status in ("failed", "expired", "cancelled", "canceled", "cancelling"):
            return "failed"
        return "in_progress"

    def collect(self, batch_id: str) -> Dict[str, Optional[str]]:
        # ONE request: the batch object carries both the status and, once it is
        # completed, the full results array. Polling again to fetch results
        # would be a second round trip for data already in hand.
        try:
            batch = self._retrieve(batch_id)
        except _BatchNotVisibleYet as error:
            # Same window, the caller's vocabulary: collect's "come back later".
            raise BatchNotReady(str(error)) from error
        current = self._normalize(batch.get("status"))
        if current != "completed":
            raise BatchNotReady(
                f"OpenRouter batch {batch_id!r} is {current!r}, not yet completed")

        results: Dict[str, Optional[str]] = {}
        for entry in batch.get("results") or []:
            results[entry.get("custom_id")] = self._parse_result(entry)
        return results

    def failure_reason(self, batch_id: str) -> Optional[str]:
        """The provider's explanation for a rejected batch, or ``None``.

        ``status`` reads only the ``status`` string, so a batch the provider
        rejected -- GET answers with the batch object carrying e.g.
        ``{"error": {"message": "HTTP 400: invalid batch inference job:
        job-submission-count for account alex-79b5d6, in use: 16, quota: 16"}}``
        -- normalizes to plain ``"failed"`` and the reason is dropped: every
        card in the deck reports as failed with no why, and the operator has to
        curl the batch API by hand to learn the deck was over a submission
        quota. This returns the message the batch object carries, so the caller
        can print it next to the failure; ``None`` means the object names no
        error (a healthy or merely unfinished batch).

        One ``_retrieve``, so the read-after-write grace window of
        :data:`OPENROUTER_SUBMIT_GRACE_SECONDS` applies unchanged. Within it, a
        404 for the id this instance just submitted returns ``None`` -- "not
        visible yet" is not a failure and must never raise here, the batch may
        still be on its way -- while a 404 outside the window, or any other
        status code, still raises the same RuntimeError ``status`` would, body
        and all, because an operator must read it.
        """
        try:
            batch = self._retrieve(batch_id)
        except _BatchNotVisibleYet:
            return None
        return self._error_message(batch)

    @staticmethod
    def _error_message(batch: dict) -> Optional[str]:
        """A batch object -> the error message it carries, or ``None``.

        The documented shape is ``{"error": {"message": ...}}``; anything else
        (the key absent, ``null``, an empty message) is "no reason available"
        and yields ``None`` rather than a stringified non-answer.
        """
        error = batch.get("error")
        if not error:
            return None
        if isinstance(error, dict):
            message = error.get("message")
            return message or None
        # Not the documented shape, but a non-empty error is still a reason:
        # report it as text instead of discarding it.
        return str(error)

    @staticmethod
    def _parse_result(entry: dict) -> Optional[str]:
        """One inline result item -> response text, or ``None`` if it failed.

        Exactly one of ``response``/``error`` is populated per item, so an
        errored item -- and a response that did not come back 200, or came back
        without a choice -- degrades to ``None``, the same way the other
        backends report a per-request failure.
        """
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
