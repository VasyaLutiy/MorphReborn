"""
The output discipline of a headless ``mrph`` CLI: one JSON document, always.

A headless run is consumed by a script, not read by an engineer, so stdout has
to be exactly one JSON document on EVERY outcome -- including every failure.
That contract is easy to state and easier to break: a traceback on stdout, a
stray ``print`` in a handler, or an exception escaping before anything is
written all leave the caller parsing half a Python backtrace. This module is
where the guarantee lives, so that no handler has to remember it: a handler
returns a payload -- or raises anything at all -- and :func:`run_cli` turns
that into one document on the stream it is handed plus one exit code to the
shell. Nothing else reaches the stream, and nothing escapes the function but
the code.

The exit codes are the whole vocabulary a headless caller gets, deliberately
coarse: a script branches on "did it finish", "did it refuse before costing
money", "did the transport fall over", "did I ask for something impossible" --
not on fifteen exception classes, and a code that means one thing forever is
worth more than a precise one that drifts.

* ``EXIT_OK`` (0) -- done. For a whole run: every card accepted.
* ``EXIT_INCOMPLETE`` (1) -- the run finished, but some cards failed or were
  skipped; the payload says which.
* ``EXIT_REFUSED`` (2) -- refused BEFORE spending money: a file-ownership
  conflict, a dirty working tree, a :class:`cards.store.StoreError`. The deck
  or the tree is wrong; nothing was submitted, and the error document says why.
* ``EXIT_TRANSPORT`` (3) -- transport/provider trouble: a batch the provider
  rejected, a wait that exhausted its timeout. The request was reasonable;
  retrying is meaningful.
* ``EXIT_USAGE`` (4) -- usage: no such file, malformed JSON, a malformed card,
  an unknown processor. The caller asked for something that cannot work.

The pieces: :func:`error_document` is the failure shape -- exactly one shape,
so a script can read ``["error"]["message"]`` without checking; :func:`emit`
writes one document and flushes it, degrading anything unexpected to a string
rather than raising; :func:`classify` maps any exception onto the table above;
:func:`run_cli` composes them and carries the guarantee.

Pure logic, stdlib only: no argument parsing, no store calls, no I/O beyond
writing to the stream the caller hands in. Like the rest of ``cards`` it
imports nothing from ``flows`` -- and its ``cards.*`` imports sit INSIDE
:func:`classify` (a local import), so importing this module stays cheap and
free of import cycles with the modules it names.
"""

import json
from typing import Callable, TextIO, Tuple, Type, Union

# The exit codes, in the order a run meets them: a submission is either
# refused (2) or it spends, and a spent run ends done (0), incomplete (1), or
# dead on the transport (3) -- with usage (4) the answer to a caller that
# never got as far as a deck. Documented in the module docstring; defined
# here as the single vocabulary every handler and the classify table share.
EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_REFUSED = 2
EXIT_TRANSPORT = 3
EXIT_USAGE = 4


class CliError(Exception):
    """A failure already in the CLI's own vocabulary: code, kind, message.

    :func:`classify` produces one from any exception; :func:`run_cli` turns
    one into the error document it emits. ``kind`` is a short class name for
    the failure -- ``"StoreError"``, ``"HazardError"``, ``"WaitTimeout"`` --
    so a script can branch on the family without parsing the message. A
    :class:`CliError` raised BY a handler is honoured as raised:
    :func:`classify` passes it through unchanged.
    """

    def __init__(self, code: int, kind: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.kind = kind
        self.message = message


def error_document(code: int, kind: str, message: str) -> dict:
    """The failure document, and exactly it: ``{"error": {code, kind, message}}``.

    Nothing else -- no traceback, no payload fragments, no extra keys. A
    script that reads ``["error"]["message"]`` must be able to trust that the
    shape never grows sideways, the same way the exit codes never grow new
    meanings: the payload of a SUCCESSFUL command is each handler's business,
    but the shape of a failure is this module's, and one shape it stays.
    """
    return {"error": {"code": code, "kind": kind, "message": message}}


# -- the stream --------------------------------------------------------------


def emit(payload: dict, stream: TextIO, pretty: bool = False) -> None:
    """Write ``payload`` to ``stream`` as ONE JSON document plus ONE newline.

    Every choice here serves "the caller can parse this, always":

    * compact by default (``separators=(",", ":")``) -- a single line, the
      form a script wants; ``pretty=True`` gives ``indent=2`` for a human
      reading the same document in a terminal. Both end in exactly one
      newline, so a line-oriented caller and a whole-document caller both
      work.
    * ``default=str`` -- an unexpected non-serialisable value degrades to its
      string form instead of raising and destroying the contract
      mid-document.
    * ``ensure_ascii`` stays at its ``True`` default: a non-ASCII character
      is escaped to ``\\uXXXX`` -- which ``json.loads`` reads back as the
      same string -- rather than risking a ``UnicodeEncodeError`` on a stream
      whose encoding cannot represent it. The document must get OUT; how it
      is spelled is negotiable.
    * the stream is flushed before returning, so the document is complete on
      the wire the moment this call ends -- a caller that kills the process
      immediately afterwards still reads all of it.
    """
    if pretty:
        text = json.dumps(payload, indent=2, default=str)
    else:
        text = json.dumps(payload, separators=(",", ":"), default=str)
    stream.write(text + "\n")
    stream.flush()


# -- classification ----------------------------------------------------------


# One row of the classification table: (exception class, exit code, kind).
# The table itself is built inside :func:`classify`, because its first rows
# name ``cards`` classes that are imported lazily there (see that function).
_ClassRule = Tuple[Type[BaseException], int, str]


def classify(exc: BaseException) -> CliError:
    """Map any exception onto the exit-code table, as a :class:`CliError`.

    The mapping is a TABLE walked in order, not a chain of ``isinstance``
    branches written as they occurred -- because order IS part of the mapping
    here, and a chain hides it. Two pairs in this table are parent and child:
    :class:`cards.hazards.HazardError` subclasses
    :class:`cards.deck.DeckError`, and :class:`json.JSONDecodeError`
    subclasses ``ValueError``. Tested in the wrong order, both still land on
    the right CODE today (a hazard is as fatal as a broken deck, malformed
    JSON as much a usage error as a bad value) but under the wrong KIND -- and
    kind is what a script reads when it wants to know WHICH refusal happened.
    The table keeps every subclass ahead of its parent, visibly.

    Three codes cover everything: refusal (2) for the deck/store family -- the
    run was refused BEFORE money was spent; usage (4) for malformed input from
    the caller -- a bad card, unreadable JSON, a missing file, a wrong type;
    transport (3) for the rest, :class:`OSError` included. The catch-all is
    transport on purpose: an exception nothing in the table names is a failure
    of the machine or the provider, not proof the caller misspelled anything,
    and "retrying is meaningful" is the honest default for what we cannot
    classify. Its kind is simply the exception's class name.

    The message is ``str(exc)``, falling back to the class name when that is
    empty -- a document whose message is a bare empty string tells the caller
    nothing at all.

    The ``cards`` classes are imported HERE, inside the function, so that
    importing :mod:`cards.cli_json` stays cheap and cannot close an import
    cycle with the modules it names. A :class:`CliError` passed in is returned
    unchanged: classification is idempotent, and a caller that already knows
    the code must not have it second-guessed.
    """
    if isinstance(exc, CliError):
        return exc

    from cards.deck import DeckError
    from cards.hazards import HazardError
    from cards.schema import CardError
    from cards.store import StoreError

    # (exception class, exit code, kind) -- subclass before parent, everywhere
    # it matters (HazardError before DeckError; JSONDecodeError before
    # ValueError; the three OSError species before OSError). The kind is every
    # current entry's own class name; the third column exists so a future row
    # can file a class under a kind that is not its name -- a ``WaitTimeout``
    # exception under "WaitTimeout" -- without a second mechanism.
    rules: Tuple[_ClassRule, ...] = (
        (StoreError, EXIT_REFUSED, "StoreError"),
        (HazardError, EXIT_REFUSED, "HazardError"),
        (DeckError, EXIT_REFUSED, "DeckError"),
        (CardError, EXIT_USAGE, "CardError"),
        (json.JSONDecodeError, EXIT_USAGE, "JSONDecodeError"),
        (FileNotFoundError, EXIT_USAGE, "FileNotFoundError"),
        (IsADirectoryError, EXIT_USAGE, "IsADirectoryError"),
        (NotADirectoryError, EXIT_USAGE, "NotADirectoryError"),
        (ValueError, EXIT_USAGE, "ValueError"),
        (TypeError, EXIT_USAGE, "TypeError"),
        (KeyError, EXIT_USAGE, "KeyError"),
        (OSError, EXIT_TRANSPORT, "OSError"),
    )
    for cls, code, kind in rules:
        if isinstance(exc, cls):
            return CliError(code, kind, str(exc) or type(exc).__name__)
    return CliError(EXIT_TRANSPORT, type(exc).__name__,
                    str(exc) or type(exc).__name__)


# -- the wrapper -------------------------------------------------------------


def run_cli(
    handler: Callable[[], Union[dict, Tuple[dict, int]]],
    stream: TextIO,
    pretty: bool = False,
) -> int:
    """Run ``handler`` and carry the one-document guarantee on its behalf.

    ``handler`` returns either a payload ``dict`` -- which exits 0 -- or a
    ``(payload, exit_code)`` pair; it may raise anything at all. Exactly one
    document is written to ``stream`` and one code returned: a payload becomes
    that document, a raised exception is classified (:func:`classify`) and
    becomes the :func:`error_document` shape with its code. This is the whole
    contract a headless caller has, which is why the function catches
    ``BaseException`` and not merely ``Exception``: a ``GeneratorExit`` or a
    ``MemoryError`` escaping with nothing on the stream breaks the contract
    just as dead as a ``ValueError`` would.

    Two exceptions are NOT ours: ``KeyboardInterrupt`` and ``SystemExit`` are
    re-raised untouched. They are not failures of the command but of the
    process -- the engineer hit Ctrl-C, or something is mid-``sys.exit`` -- and
    the shell must be able to tell an interrupted run from a completed one
    with a nonzero code. Nothing is written for them either: half a document
    is worse than none.

    The net covers the emission as well as the call. :func:`emit` is built not
    to raise (``default=str``, ASCII-safe output), but a payload that defeats
    even that -- a circular reference is the one shape that does -- would
    otherwise escape with NOTHING on the stream, the one outcome this function
    exists to make impossible; so that failure is classified and reported like
    any other.

    ``pretty`` passes through to :func:`emit`. Returns the exit code.
    """
    try:
        result = handler()
        if isinstance(result, tuple) and len(result) == 2:
            payload, code = result
        else:
            payload, code = result, EXIT_OK
        emit(payload, stream, pretty)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        cli_error = classify(exc)
        emit(error_document(cli_error.code, cli_error.kind, cli_error.message),
             stream, pretty)
        return cli_error.code
    return code
