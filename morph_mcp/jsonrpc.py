"""JSON-RPC 2.0 framing for the morph MCP stdio server.

This is the framing layer only: it turns a text stream into parsed JSON-RPC
message objects and back. Method dispatch, tools, and notification handling
live upstream; this module knows nothing about them.

Protocol contract (stdio MCP, JSON-RPC 2.0):

    * Exactly one *compact* JSON object per line, newline-delimited.
    * Requests carry an ``id`` and get exactly one response; notifications
      carry no ``id`` and get NO response at all.
    * A response carries ``result`` or ``error`` -- never both::

          -> {"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}
          <- {"jsonrpc":"2.0","id":1,"result":{...}}
          -> {"jsonrpc":"2.0","method":"notifications/initialized"}
             (notification: no response)
          <- {"jsonrpc":"2.0","id":2,"error":{"code":-32601,"message":"..."}}

    * stdin is read until EOF; nothing but protocol frames is ever written
      to stdout (diagnostics, if any, belong on stderr).

Standard library only; compatible with Python 3.9.
"""

import json
from typing import Any, Dict, Optional, TextIO

__all__ = ["read_message", "write_message", "success", "error"]

_JSONRPC_VERSION = "2.0"


def read_message(stream: TextIO) -> Optional[Dict[str, Any]]:
    """Read ONE newline-delimited JSON object from ``stream``.

    Reads lines until a non-blank one appears; blank (whitespace-only) lines
    are skipped. Returns ``None`` at EOF. A final line without a trailing
    newline is still processed.

    Malformed JSON raises :class:`json.JSONDecodeError`; a frame that parses
    but is not a JSON object raises :class:`ValueError`. Both surface as
    ``ValueError`` (``JSONDecodeError`` subclasses it), so the server loop
    can catch it and answer with a ``-32700`` parse error instead of dying.
    """
    while True:
        line = stream.readline()
        if line == "":  # EOF: readline() returns '' and nothing else.
            return None
        text = line.strip()
        if not text:
            continue  # Blank line between frames -- skip it.
        message = json.loads(text)
        if not isinstance(message, dict):
            raise ValueError(
                "JSON-RPC frame must be a JSON object, got %s"
                % type(message).__name__
            )
        return message


def write_message(stream: TextIO, message: Dict[str, Any]) -> None:
    """Write ONE compact JSON object followed by a single newline, then flush.

    "Compact" means ``separators=(",", ":")`` -- no padding spaces -- so a
    frame is exactly one line. Non-ASCII characters are escaped
    (``ensure_ascii=True``), keeping the frame pure ASCII and safe on any
    text-mode stdout. The flush is not optional: a peer reading line by line
    sees nothing until the buffer drains.
    """
    if not isinstance(message, dict):
        raise TypeError(
            "JSON-RPC frame must be a dict, got %s" % type(message).__name__
        )
    stream.write(json.dumps(message, separators=(",", ":")) + "\n")
    stream.flush()


def success(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    """Build a JSON-RPC success response: ``result`` set, ``error`` absent."""
    return {"jsonrpc": _JSONRPC_VERSION, "id": request_id, "result": result}


def error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    """Build a JSON-RPC error response: ``error`` set, ``result`` absent."""
    return {
        "jsonrpc": _JSONRPC_VERSION,
        "id": request_id,
        "error": {"code": code, "message": message},
    }
