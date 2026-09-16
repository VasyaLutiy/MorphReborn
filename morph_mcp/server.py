"""The morph MCP stdio server: framing and tools, wired into one event loop.

This is the process an MCP client launches (``python3 -m morph_mcp.server``).
It owns the loop and nothing else: :mod:`morph_mcp.jsonrpc` turns the text
stream into frames and back, :mod:`morph_mcp.tools` answers tool calls, and
this module decides which method gets which treatment::

    initialize                 -> protocolVersion, capabilities, serverInfo
    notifications/initialized  -> silence (like every notification)
    tools/list                 -> the descriptors from morph_mcp.tools.TOOLS
    tools/call                 -> one tool run, wrapped as text content; a
                                  tool that raises comes back as a result
                                  with isError:true carrying the message,
                                  never as a JSON-RPC error
    any other request          -> {"code":-32601} method not found
    any other notification     -> silence

Protocol contract (stdio MCP, JSON-RPC 2.0): one compact JSON object per
line is read from stdin until EOF; a frame that carries an ``id`` is a
request and draws exactly one reply, a frame without one is a notification
and draws none.  Nothing but protocol frames is ever written to stdout --
diagnostics belong on stderr.

The deck root handed to the tools is the server's working directory,
overridable with the MORPH_ROOT environment variable.  It is resolved per
``tools/call``, so the environment alone decides where the ``.morph/``
directory lives and what the cards are compiled against.

Standard library plus this project only; compatible with Python 3.9.
"""

import os
import sys
from typing import Any, Dict, Optional, TextIO

from .jsonrpc import error, read_message, success, write_message
from .tools import TOOLS, call_tool

__all__ = ["main", "serve"]


# -- what this server is -------------------------------------------------------


_PROTOCOL_VERSION = "2024-11-05"  # the MCP dialect the conformance suite pins
_SERVER_NAME = "morph"
_SERVER_VERSION = "0.1.0"

# JSON-RPC 2.0 error codes this server emits.
_PARSE_ERROR = -32700        # a frame that would not parse
_METHOD_NOT_FOUND = -32601   # a request for a method we do not serve


# -- diagnostics -----------------------------------------------------------------


def _log(message: str) -> None:
    """One diagnostic line on stderr; stdout carries protocol frames only.

    Best-effort by design: if stderr itself is gone -- a client shutting the
    server down may have closed it -- the line is dropped, never raised, so
    diagnostics can never kill the loop they are trying to describe.
    """
    try:
        print(f"morph-mcp: {message}", file=sys.stderr, flush=True)
    except (BrokenPipeError, ValueError, OSError):
        pass


# -- responses -------------------------------------------------------------------


def _initialize_result() -> Dict[str, Any]:
    """The ``initialize`` result: what this server speaks and what it offers."""
    return {
        "protocolVersion": _PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": _SERVER_NAME, "version": _SERVER_VERSION},
    }


def _tool_result(text: str, is_error: bool = False) -> Dict[str, Any]:
    """Wrap a tool's text report as MCP content: one text item, an isError flag."""
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _deck_root() -> str:
    """The deck root: ``$MORPH_ROOT`` when set, else the working directory."""
    return os.environ.get("MORPH_ROOT") or os.getcwd()


def _run_tool(params: Any) -> Dict[str, Any]:
    """Run one ``tools/call`` and wrap its text report.

    A tool that raises -- an unknown name, a malformed card, a refused run
    transition, anything at all -- comes back as a *result* with ``isError``
    true and the message as the text, never as a JSON-RPC error: a tool
    failure is data for the caller, while a JSON-RPC error would mean the
    transport itself failed.  The deck root is resolved here, per call, from
    ``$MORPH_ROOT`` or the working directory.
    """
    if not isinstance(params, dict):
        params = {}
    name = params.get("name")
    arguments = params.get("arguments")
    try:
        text = call_tool(name, arguments, _deck_root())
    except Exception as exc:  # the tool boundary swallows everything
        message = str(exc) or type(exc).__name__
        _log(f"tool {name!r} failed: {message}")
        return _tool_result(message, is_error=True)
    return _tool_result(text)


# -- dispatch --------------------------------------------------------------------


def _handle_frame(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Dispatch one decoded frame; return the reply, or None for silence.

    The ``id`` member is the whole distinction: a frame that carries one is a
    request and gets exactly one reply, a frame that does not is a
    notification and gets none -- whatever its method, known or not.
    """
    method = message.get("method")
    if not isinstance(method, str):
        # No method member: neither a request nor a notification, so the
        # protocol offers nothing to answer it with.  Ignore it.
        keys = ", ".join(sorted(message))
        _log(f"ignoring a frame with no method member (keys: {keys})")
        return None

    is_notification = "id" not in message
    request_id = message.get("id")

    if method == "initialize" and not is_notification:
        return success(request_id, _initialize_result())

    if method == "tools/list" and not is_notification:
        return success(request_id, {"tools": TOOLS})

    if method == "tools/call" and not is_notification:
        return success(request_id, _run_tool(message.get("params")))

    if is_notification:
        # notifications/initialized lands here, and so does anything else
        # without an id: the contract is NO response at all.
        _log(f"notification {method!r}: acknowledged with silence")
        return None

    _log(f"method not found: {method!r}")
    return error(request_id, _METHOD_NOT_FOUND, f"method not found: {method}")


# -- the event loop ----------------------------------------------------------------


def serve(stdin: TextIO, stdout: TextIO) -> int:
    """Serve JSON-RPC over a pair of text streams until stdin hits EOF.

    One frame at a time: :func:`morph_mcp.jsonrpc.read_message` parses it,
    :func:`_handle_frame` decides the reply, and
    :func:`morph_mcp.jsonrpc.write_message` writes and flushes it, so a peer
    reading line by line sees every frame immediately.  A frame that will not
    parse is answered with a ``-32700`` parse error against a null id (the id
    is unknowable) and the loop carries on.  A stream that dies underneath us
    ends the server quietly: the peer is gone, there is nobody left to answer.

    Returns a process exit code; 0 on every path that is not a crash.
    """
    _log(f"morph MCP server listening on stdio (protocol {_PROTOCOL_VERSION})")
    try:
        while True:
            if stdin.closed:
                _log("stdin is closed; shutting down")
                return 0
            try:
                message = read_message(stdin)
            except ValueError as exc:
                # Malformed JSON, or a parsed frame that is not an object --
                # exactly what the framing layer promises to raise so this
                # loop can answer -32700 instead of dying.
                _log(f"parse error on an incoming frame: {exc}")
                write_message(
                    stdout, error(None, _PARSE_ERROR, f"parse error: {exc}")
                )
                continue
            if message is None:  # EOF: the client is done with us.
                _log("stdin reached EOF; shutting down")
                return 0
            response = _handle_frame(message)
            if response is not None:
                write_message(stdout, response)
    except (BrokenPipeError, ValueError, OSError) as exc:
        # A stream failed underneath us (the client closed its ends while
        # shutting the server down).  Nobody is left to answer: stop quietly.
        _log(f"stdio stream failed; shutting down ({exc})")
        return 0


# -- entry point ---------------------------------------------------------------------


def main() -> int:
    """Run the server on the process's own stdio (no CLI arguments)."""
    return serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    sys.exit(main())
