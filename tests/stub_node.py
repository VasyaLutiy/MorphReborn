"""A stub inference node: the fast circuit of Morph's test and demo rig.

Every test above the unit level needs something that answers like a llama.cpp
node. The real ones are a cloud batch queue (~20 minutes per generation) or a
GPU box on the LAN -- neither can be part of a test run, and neither can be
driven live in front of an audience. This server is the third option: it speaks
exactly the slice of the OpenAI API that
:class:`processors.llama_cpp_processor.LlamaCppProcessor` uses, and answers in
milliseconds with text a scenario prepared in advance.

**The answer is chosen by the PROMPT, not by the server.** A card's instruction
carries a marker ``[[STUB:<key>]]`` anywhere in its text; the key names an entry
in a JSON answers file given on the command line. So a demo or a test writes its
own morphs -- a two-file changeset, a deliberately failing one -- without
touching this file. An answers key may be written ``<key>@<n>``, which answers
the n-th call for that key: that is how the regeneration path (first attempt
fails acceptance, second passes) and best-of-N variants become reproducible.

Nothing here may fail silently. A key with no answer, and a prompt with no
marker at all, both return a loudly marked stub answer instead of raising: the
run continues and fails at ACCEPTANCE, where the operator sees which card was
misconfigured, rather than dying inside the transport with a stack trace.

Standard library only, and no import of the project: the node is started as a
plain subprocess (``python3 tests/stub_node.py --port 8080 --answers ...``) and
must run in an environment where Morph's own dependencies are absent.

Usage as a Morph processor -- one node per registry id::

    MRPH_PROCESSORS=stub-a,stub-b
    MRPH_PROCESSOR_stub-a_TYPE=llama_cpp
    MRPH_PROCESSOR_stub-a_ENDPOINT_URI=http://127.0.0.1:8080/v1
    MRPH_PROCESSOR_stub-a_MODEL=stub-a
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The marker a prompt uses to name its answer. Matched anywhere in any message:
# the compiler wraps the card's instruction in context and boilerplate, so the
# marker's position in the final prompt is not ours to predict.
STUB_MARKER = re.compile(r"\[\[STUB:([^\]\s]+)\]\]")

# A key may name the attempt it answers: ``cs-retry@2`` is the second call made
# with the key ``cs-retry``.
ATTEMPT_SEPARATOR = "@"

# The two ways a scenario can be misconfigured, written INTO the answer so the
# failure is visible in the acceptance output and in the morphed file itself,
# not just in a server log nobody tails during a demo.
UNKNOWN_KEY_ANSWER = (
    "STUB NODE ERROR: the prompt asked for [[STUB:{key}]] (call {attempt}), and "
    "the answers file has no entry for that key. Add \"{key}\" -- or "
    "\"{key}@{attempt}\" -- to the answers file, or fix the marker in the card's "
    "instruction. This text is not a morph and will not pass any acceptance."
)

NO_MARKER_ANSWER = (
    "STUB NODE ERROR: no [[STUB:<key>]] marker was found anywhere in this "
    "prompt, so the stub node has no way to know which answer was wanted. Put "
    "the marker in the card's instruction. This text is not a morph and will "
    "not pass any acceptance."
)

# How much answer text goes into one SSE chunk. Small enough that every answer
# arrives in several chunks, so a client that mishandles reassembly fails here
# rather than against a real node.
CHUNK_SIZE = 64


class AnswerBook:
    """The prepared answers, and how many times each key has been asked for.

    Call counting lives here (not in the handler) because it is the only mutable
    state in the node and it is touched from several request threads at once.
    """

    def __init__(self, answers):
        self._answers = dict(answers)
        self._calls = {}
        self._lock = threading.Lock()

    def __len__(self):
        return len(self._answers)

    def keys(self):
        return sorted(self._answers)

    def resolve(self, prompt):
        """Answer the prompt: ``(text, key or None, attempt number)``.

        ``key`` is ``None`` when the prompt carries no marker. The attempt is
        1-based and counted per key across the life of the node, so a card that
        is regenerated twice gets attempts 1, 2, 3 -- the same numbering a
        scenario writes into its ``<key>@<n>`` entries.

        When no entry matches the exact attempt, the plain ``<key>`` entry
        answers; failing that, the highest ``<key>@<m>`` with ``m < n`` does, so
        a pair of ``@1``/``@2`` answers keeps behaving sensibly on a third call
        instead of collapsing into the misconfiguration answer.
        """
        match = STUB_MARKER.search(prompt)
        if match is None:
            return NO_MARKER_ANSWER, None, 0

        key = match.group(1)
        with self._lock:
            attempt = self._calls.get(key, 0) + 1
            self._calls[key] = attempt

        text = self._lookup(key, attempt)
        if text is None:
            return UNKNOWN_KEY_ANSWER.format(key=key, attempt=attempt), key, attempt
        return text, key, attempt

    def _lookup(self, key, attempt):
        """The best entry for ``key`` at ``attempt``, or ``None`` if there is none."""
        exact = self._answers.get("%s%s%d" % (key, ATTEMPT_SEPARATOR, attempt))
        if exact is not None:
            return exact
        plain = self._answers.get(key)
        if plain is not None:
            return plain

        prefix = key + ATTEMPT_SEPARATOR
        earlier = []
        for candidate in self._answers:
            if not candidate.startswith(prefix):
                continue
            suffix = candidate[len(prefix):]
            if suffix.isdigit() and int(suffix) < attempt:
                earlier.append((int(suffix), candidate))
        if not earlier:
            return None
        return self._answers[max(earlier)[1]]


def load_answers(path):
    """Read an answers file: ``{key: answer text}``, JSON, UTF-8.

    Keys beginning with ``__`` are dropped: JSON has no comments, so that is
    where an answers file keeps its own documentation.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("%s: expected a JSON object of key -> answer text" % path)

    answers = {}
    for key, value in raw.items():
        if key.startswith("__"):
            continue
        if not isinstance(value, str):
            raise ValueError("%s: answer for %r is not a string" % (path, key))
        answers[key] = value
    return answers


def prompt_text(messages):
    """Every message flattened into one string, for marker matching.

    A marker put in the system message, the context slice or the instruction all
    count the same: the scenario, not the compiler's message layout, decides
    where it lands.
    """
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # The OpenAI content-parts shape; a real client may send it.
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
    return "\n".join(parts)


def _chunks(text, size=CHUNK_SIZE):
    """``text`` cut into SSE-sized pieces (at least one, even for an empty answer)."""
    if not text:
        return [""]
    return [text[at:at + size] for at in range(0, len(text), size)]


class StubNodeHandler(BaseHTTPRequestHandler):
    """The OpenAI-compatible surface: chat completions (streaming or not) and models.

    The class attributes are filled in by :func:`make_server`; a handler is
    instantiated per request, so there is nowhere else to hang them.
    """

    protocol_version = "HTTP/1.1"

    book = None
    node_name = "stub"
    log_dir = None
    _log_lock = threading.Lock()
    _log_sequence = 0

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": self.node_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "morph-stub",
                }],
            })
            return
        self._send_error(404, "no such path: %s" % self.path)

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._send_error(404, "no such path: %s" % self.path)
            return

        try:
            payload = self._read_json_body()
        except ValueError as error:
            self._send_error(400, str(error))
            return

        messages = payload.get("messages") or []
        model = payload.get("model") or self.node_name
        text, key, attempt = self.book.resolve(prompt_text(messages))
        self._log_request(messages, model, key, attempt, bool(payload.get("stream")))

        if payload.get("stream"):
            self._send_stream(text, model)
        else:
            self._send_json(200, self._completion(text, model))

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("request body is not JSON: %s" % error)
        if not isinstance(payload, dict):
            raise ValueError("request body is not a JSON object")
        return payload

    def _completion(self, text, model):
        """The non-streaming response body."""
        return {
            "id": self._completion_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def _send_stream(self, text, model):
        """The answer as SSE: a role chunk, content chunks, a stop chunk, ``[DONE]``.

        The length is unknown up front, so the body ends with the connection
        (``Connection: close``) rather than a Content-Length -- the same framing
        llama-server uses for a stream.
        """
        completion_id = self._completion_id()
        created = int(time.time())

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def frame(delta, finish_reason=None):
            return {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }],
            }

        try:
            self._send_event(frame({"role": "assistant", "content": ""}))
            for piece in _chunks(text):
                self._send_event(frame({"content": piece}))
            self._send_event(frame({}, finish_reason="stop"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The client hung up mid-stream (a cancelled card, a killed CLI).
            # That is its business; the node stays up for the next request.
            pass

    def _send_event(self, frame):
        self.wfile.write(("data: %s\n\n" % json.dumps(frame)).encode("utf-8"))
        self.wfile.flush()

    def _send_json(self, status, body):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, status, message):
        """An OpenAI-shaped error, so a real client reports it as one."""
        self._send_json(status, {"error": {
            "message": message,
            "type": "stub_node_error",
            "code": status,
        }})

    def _completion_id(self):
        return "chatcmpl-stub-%s-%d" % (self.node_name, int(time.time() * 1000))

    def _log_request(self, messages, model, key, attempt, stream):
        """Drop the whole request into ``--log-dir`` as one JSON file.

        This is how a test or an operator inspects what the COMPILER sent --
        which context slice, which instruction, which error context on a
        regeneration -- without instrumenting the CLI.
        """
        if not self.log_dir:
            return

        with StubNodeHandler._log_lock:
            StubNodeHandler._log_sequence += 1
            sequence = StubNodeHandler._log_sequence

        safe_key = re.sub(r"[^A-Za-z0-9_.@-]", "_", key or "no-marker")
        path = os.path.join(self.log_dir, "%04d-%s.json" % (sequence, safe_key))
        record = {
            "node": self.node_name,
            "received": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "key": key,
            "attempt": attempt,
            "model": model,
            "stream": stream,
            "messages": messages,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)

    def log_message(self, message_format, *args):
        """One line per request, on STDERR.

        Stdout stays empty: a supervisor that captures this node's stdout must
        see the protocol and nothing else.
        """
        sys.stderr.write("[%s] %s\n" % (self.node_name, message_format % args))


def make_server(host, port, book, node_name="stub", log_dir=None):
    """A bound, not-yet-serving node. ``port`` 0 binds an ephemeral one.

    Returned rather than served so a test can run it on a thread and read the
    real port off ``server.server_address``.
    """
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    handler = type("BoundStubNodeHandler", (StubNodeHandler,), {
        "book": book,
        "node_name": node_name,
        "log_dir": log_dir,
    })
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def build_parser():
    parser = argparse.ArgumentParser(
        prog="stub_node.py",
        description="A stub OpenAI-compatible inference node for Morph's fast "
                    "test and demo circuit. Answers are chosen by a "
                    "[[STUB:<key>]] marker in the prompt.",
        epilog="example: python3 tests/stub_node.py --port 8080 "
               "--answers tests/stub_answers.json --name stub-a",
    )
    parser.add_argument("--port", type=int, default=8080,
                        help="port to listen on (default: 8080; 0 picks a free one)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to bind (default: 127.0.0.1)")
    parser.add_argument("--answers", required=True,
                        help="JSON file mapping a marker key to its answer text; "
                             "a key may be written <key>@<n> to answer the n-th "
                             "call for that key")
    parser.add_argument("--log-dir", default=None,
                        help="write one JSON file per request here, holding the "
                             "full prompt (created if missing)")
    parser.add_argument("--name", default="stub",
                        help="node name, reported by /v1/models and used in the "
                             "request log (default: stub)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    book = AnswerBook(load_answers(args.answers))
    server = make_server(args.host, args.port, book,
                         node_name=args.name, log_dir=args.log_dir)

    host, port = server.server_address[:2]
    sys.stderr.write(
        "stub node \"%s\" on http://%s:%d/v1 -- %d answer(s) from %s%s\n" % (
            args.name, host, port, len(book), args.answers,
            ", logging prompts to %s" % args.log_dir if args.log_dir else ""))
    sys.stderr.flush()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
