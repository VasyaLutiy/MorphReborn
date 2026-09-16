"""Executable conformance spec for the `morph_mcp` MCP stdio server.

This file is the specification in its testable form.  It treats the server
strictly as a black box: it spawns `python3 -m morph_mcp.server` with pipes
on stdin/stdout (text mode) and speaks newline-delimited, compact JSON-RPC
2.0 at it, asserting only on the bytes that come back.  Nothing here imports
the server, so the suite pins down the *contract*, not an implementation:

  -> {"jsonrpc":"2.0","id":1,"method":"initialize",...}
  <- {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05",
         "capabilities":{"tools":{}},"serverInfo":{"name":"morph",...}}}
  -> {"jsonrpc":"2.0","method":"notifications/initialized"}   (never answered)
  -> {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
  <- ... exactly deck_status, card_add, deck_submit, deck_collect ...
  -> {"jsonrpc":"2.0","id":3,"method":"tools/call",
        "params":{"name":"deck_status","arguments":{}}}
  <- {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text",...}],
         "isError":false}}
  unknown method -> {"jsonrpc":"2.0","id":N,"error":{"code":-32601,...}}

Robustness rules the harness lives by:

  * every read from the server is bounded by READ_TIMEOUT, so a wedged or
    mute server shows up as a failed assertion, never as a hung test;
  * the subprocess is always terminated via addCleanup, pass or fail;
  * the server runs with a throwaway temporary directory as its cwd (the
    project root is put on PYTHONPATH so `-m morph_mcp.server` still
    resolves, and PYTHONDONTWRITEBYTECODE keeps the checkout clean), so a
    test run can never scribble into the real project backlog.
"""

import collections
import json
import os
import queue
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path


READ_TIMEOUT = 10.0   # seconds to wait for any single message from the server
QUIET_WINDOW = 1.0    # silence window proving a notification drew no reply
STOP_TIMEOUT = 5.0    # grace period when shutting the server down

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The contract: the server is launched exactly this way, stdio piped, text mode.
SERVER_ARGV = ["python3", "-m", "morph_mcp.server"]

EXPECTED_TOOLS = ["deck_status", "card_add", "deck_submit", "deck_collect"]


class MorphServerClient:
    """Black-box driver for one server process.

    Speaks one compact JSON object per line over the child's stdio.  All
    reads go through a bounded queue fed by a background thread, so a server
    that never answers (or writes garbage) fails the test with a diagnostic
    instead of hanging it.
    """

    def __init__(self, project_root, workspace):
        env = dict(os.environ)
        parts = [str(project_root)]
        if env.get("PYTHONPATH"):
            parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(parts)
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        self._proc = subprocess.Popen(
            SERVER_ARGV,
            cwd=workspace,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._messages = queue.Queue()
        self._stderr_lines = collections.deque(maxlen=40)
        self._stderr_lock = threading.Lock()
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    # ---- plumbing ---------------------------------------------------------

    def _pump_stdout(self):
        try:
            for raw in self._proc.stdout:
                line = raw.strip()
                if line:
                    self._messages.put(line)
        except (ValueError, OSError):
            pass  # streams torn down while the server was being stopped
        finally:
            self._messages.put(None)  # EOF sentinel

    def _pump_stderr(self):
        try:
            for raw in self._proc.stderr:
                with self._stderr_lock:
                    self._stderr_lines.append(raw)
        except (ValueError, OSError):
            pass

    def stderr_tail(self):
        with self._stderr_lock:
            return "".join(self._stderr_lines).strip() or "<no stderr output>"

    def send(self, message):
        line = json.dumps(message, separators=(",", ":"))
        if self._proc.poll() is not None:
            raise AssertionError(
                f"cannot send {line!r}: server already exited "
                f"(code {self._proc.returncode}); stderr: {self.stderr_tail()}"
            )
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise AssertionError(
                f"failed writing to the server's stdin ({exc}); "
                f"stderr: {self.stderr_tail()}"
            )

    def recv(self):
        try:
            item = self._messages.get(timeout=READ_TIMEOUT)
        except queue.Empty:
            raise AssertionError(
                f"timed out after {READ_TIMEOUT:.1f}s waiting for a message "
                f"from the server; stderr: {self.stderr_tail()}"
            )
        if item is None:
            raise AssertionError(
                "server closed stdout before the expected message "
                f"(exit code {self._proc.poll()}); stderr: {self.stderr_tail()}"
            )
        try:
            return json.loads(item)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"server wrote a line that is not valid JSON: {item!r} ({exc})"
            )

    def assert_quiet(self):
        """Fail if any stray message arrives within the quiet window."""
        try:
            item = self._messages.get(timeout=QUIET_WINDOW)
        except queue.Empty:
            return
        if item is None:
            raise AssertionError("server closed stdout during the quiet-period check")
        raise AssertionError(f"server sent an unexpected message: {item!r}")

    # ---- protocol helpers ---------------------------------------------------

    def request(self, id, method, params=None):
        message = {"jsonrpc": "2.0", "id": id, "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)
        response = self.recv()
        if response.get("jsonrpc") != "2.0":
            raise AssertionError(f"response is not JSON-RPC 2.0: {response!r}")
        if response.get("id") != id:
            raise AssertionError(
                f"expected a response with id={id!r}, got: {response!r}"
            )
        return response

    def notify(self, method, params=None):
        """Send a notification: no id, and by contract NO response at all."""
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)

    def initialize(self):
        response = self.request(
            1,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "c", "version": "1"},
            },
        )
        if "error" in response:
            raise AssertionError(f"initialize failed: {response['error']!r}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise AssertionError(f"initialize returned no result object: {response!r}")
        return result

    def stop(self):
        proc = self._proc
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=STOP_TIMEOUT)
                except subprocess.TimeoutExpired:
                    pass
        else:
            try:
                proc.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                pass


class MorphMcpConformanceCase(unittest.TestCase):
    """Harness shared by all conformance tests.

    Every test gets a fresh server process and a throwaway working directory;
    addCleanup guarantees the subprocess is terminated and the directory is
    removed even when assertions fail.
    """

    def setUp(self):
        workspace = tempfile.TemporaryDirectory(prefix="morph-mcp-conformance-")
        self.addCleanup(workspace.cleanup)
        self.workspace = workspace.name
        self.server = MorphServerClient(PROJECT_ROOT, self.workspace)
        self.addCleanup(self.server.stop)

    def handshake(self):
        """initialize + notifications/initialized, per the protocol contract."""
        result = self.server.initialize()
        self.server.notify("notifications/initialized")
        return result


class InitializeHandshakeTests(MorphMcpConformanceCase):

    def test_initialize_returns_protocol_version_server_info_and_tools_capability(self):
        result = self.server.initialize()

        self.assertEqual(
            result.get("protocolVersion"),
            "2024-11-05",
            "the server must speak the 2024-11-05 protocol version",
        )

        server_info = result.get("serverInfo")
        self.assertIsInstance(
            server_info, dict, "initialize result must carry a serverInfo object"
        )
        self.assertEqual(server_info.get("name"), "morph")
        self.assertIsInstance(server_info.get("version"), str)

        capabilities = result.get("capabilities")
        self.assertIsInstance(
            capabilities, dict, "initialize result must carry a capabilities object"
        )
        self.assertIn("tools", capabilities, "server must advertise a tools capability")
        self.assertIsInstance(capabilities["tools"], dict)


class ToolsListTests(MorphMcpConformanceCase):

    def list_tools(self):
        response = self.server.request(2, "tools/list", {})
        result = response.get("result")
        self.assertIsInstance(
            result, dict, f"tools/list returned no result object: {response!r}"
        )
        tools = result.get("tools")
        self.assertIsInstance(
            tools, list, f"tools/list result must contain a 'tools' list: {result!r}"
        )
        return tools

    def test_tools_list_returns_exactly_the_four_deck_tools(self):
        self.handshake()
        names = [tool.get("name") for tool in self.list_tools()]
        self.assertEqual(
            sorted(names),
            sorted(EXPECTED_TOOLS),
            "tools/list must advertise exactly deck_status, card_add, "
            "deck_submit and deck_collect",
        )

    def test_every_listed_tool_has_a_description_and_an_object_input_schema(self):
        self.handshake()
        for tool in self.list_tools():
            name = tool.get("name", "<unnamed>")
            self.assertIsInstance(tool, dict)
            description = tool.get("description")
            self.assertIsInstance(
                description, str, f"tool {name!r} must declare a description"
            )
            self.assertTrue(description.strip(), f"tool {name!r} has an empty description")
            schema = tool.get("inputSchema")
            self.assertIsInstance(
                schema, dict, f"tool {name!r} must declare an inputSchema object"
            )
            self.assertEqual(
                schema.get("type"),
                "object",
                f"tool {name!r} inputSchema.type must be 'object'",
            )


class DeckStatusToolCallTests(MorphMcpConformanceCase):

    def test_deck_status_call_returns_text_content_and_no_error_flag(self):
        self.handshake()
        response = self.server.request(
            3, "tools/call", {"name": "deck_status", "arguments": {}}
        )
        result = response.get("result")
        self.assertIsInstance(
            result, dict, f"tools/call returned no result object: {response!r}"
        )
        self.assertFalse(
            result.get("isError", False),
            f"deck_status reported a tool error: {result!r}",
        )
        content = result.get("content")
        self.assertIsInstance(
            content, list, f"tools/call result must carry a content list: {result!r}"
        )
        self.assertTrue(content, "tools/call result content must not be empty")
        text_items = [
            item for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        self.assertTrue(
            text_items,
            f"expected at least one {{'type': 'text'}} item in content: {content!r}",
        )
        self.assertIsInstance(text_items[0].get("text"), str)


class UnknownMethodTests(MorphMcpConformanceCase):

    def test_unknown_method_returns_a_32601_method_not_found_error(self):
        self.handshake()
        response = self.server.request(4, "morph/no_such_method", {})
        self.assertNotIn(
            "result", response, "an error response must not carry a result"
        )
        error = response.get("error")
        self.assertIsInstance(
            error, dict, f"unknown method did not produce an error object: {response!r}"
        )
        self.assertEqual(error.get("code"), -32601)
        self.assertIsInstance(error.get("message"), str)
        self.assertTrue(error.get("message"))


class NotificationTests(MorphMcpConformanceCase):

    def test_initialized_notification_draws_no_response_at_all(self):
        self.server.initialize()

        # The contract notification: no id, and it must produce NO message.
        self.server.notify("notifications/initialized")

        # Any reply to the notification would have to arrive before the reply
        # to this probe, so the probe's own reply being next proves silence.
        self.server.request(99, "tools/list", {})

        # And nothing else may be pending afterwards either.
        self.server.assert_quiet()

    def test_unknown_notification_also_draws_no_response(self):
        self.handshake()
        self.server.notify("notifications/definitely_not_a_real_notification")
        self.server.request(98, "tools/list", {})
        self.server.assert_quiet()


if __name__ == "__main__":
    unittest.main()
