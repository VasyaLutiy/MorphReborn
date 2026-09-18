"""
The headless front door of ``mrph``: argv in, one JSON document out.

WHY this surface exists: the interactive REPL is a console for a person -- a
person can answer a question, watch a queue for twenty minutes, and read a
progress line wherever it happens to land. A script -- and most of all an
agent driving this system -- can do none of that: what it needs from a
process is DATA. So every headless invocation is one decision in and one
answer out: ``run`` is "run the deck", ``submit`` is "send the current
generation", ``report`` is "show the last run" -- and each answer arrives as
a single JSON document a caller parses without knowing anything about
Python, the REPL or the store, plus one exit code it can branch on.

WHY stdout is sacred: the one-document contract of :mod:`cards.cli_json` is
only as strong as the weakest writer on the stream. A stray ``print``, a
traceback, an argparse usage block -- any of them lands between the braces a
caller is parsing, and for that run the contract is dead. This module is
where every deviation from the contract is prevented or confined:

* a command's MEANING is a handler in :mod:`cards.cli_deck`,
  :mod:`cards.cli_run` or :mod:`cards.cli_cycle`, and a handler RETURNS its
  payload instead of printing; this module is only the wiring that turns
  argv into one of those functions called with its arguments;
* the ONE document -- on success and on every failure alike -- is written by
  :func:`cards.cli_json.run_cli`, which also maps whatever a handler raises
  onto the coarse exit-code table; no code is invented here;
* the human log is a one-argument callable writing to ``sys.stderr``, passed
  as ``log=`` into every handler that accepts one, so the store's progress
  lines stay off stdout by construction rather than by vigilance;
* argparse is defused at the source: :class:`_Parser.error` keeps argparse's
  own usage and diagnosis on stderr, where its text belongs, but raises with
  the message attached instead of ``SystemExit(2)`` -- so a bad command
  line, an unknown subcommand, or NO subcommand at all is answered with the
  same one error document and exit code 4 as any other usage failure.

Nothing here reads stdin, asks a question, or imports from ``flows`` -- not
transitively either: a scripted run must complete without the console bot
ever loading, which is why ``bin/mrph`` tests ``sys.argv`` before that
import happens.
"""

import argparse
import sys
from typing import Callable, List, NoReturn, Optional, Tuple, Union

from cards import cli_cycle, cli_deck, cli_run
from cards.cli_json import EXIT_OK, EXIT_USAGE, emit, error_document, run_cli

__all__ = ["main"]

# The wait budget the two long-running commands default to, in seconds: six
# hours of wall clock, the same default the step handlers and the cycle
# runner already use.
DEFAULT_TIMEOUT = 6 * 3600.0


class _UsageError(Exception):
    """argparse's diagnosis, carried out of the parse for the JSON contract.

    argparse signals a bad command line by writing to stderr and raising
    ``SystemExit(2)`` -- which would leave stdout empty and hand the shell a
    code this CLI's table reads as something else. :meth:`_Parser.error`
    raises this instead, with the same message, so :func:`main` can emit the
    one error document and exit 4.
    """


class _Parser(argparse.ArgumentParser):
    """An :class:`argparse.ArgumentParser` that fails into the JSON contract.

    The override keeps argparse's behaviour on stderr -- the usage block and
    the ``prog: error: ...`` line, exactly as argparse writes them -- and
    replaces only the ``SystemExit`` with :class:`_UsageError`, so the
    message survives into the error document. Subparsers inherit the class
    (``add_subparsers`` defaults to the parent's type), so a mistyped nested
    command fails the same way as a mistyped top-level one.
    """

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        sys.stderr.write(f"{self.prog}: error: {message}\n")
        raise _UsageError(message)


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """Add the two options every leaf command takes: ``--pretty``, ``--root``.

    They live on the leaves, not on the top-level parser, because the leaf is
    where a caller writes them (``mrph deck status --root TMP``) -- and
    because an option defined at two levels would let the inner default
    silently overwrite the outer parse.
    """
    parser.add_argument(
        "--pretty", action="store_true",
        help="print the JSON document indented (default: one compact line)")
    parser.add_argument(
        "--root", default=".",
        help="project root holding .morph/ (default: the current directory)")


def _build_parser() -> argparse.ArgumentParser:
    """The whole headless grammar: five commands, seven leaves, nothing else.

    The grammar is the contract's front half: whatever it accepts, a handler
    must be able to answer; whatever it refuses, :func:`main` answers with
    the usage error document. Everything a handler needs is parsed HERE --
    no handler ever sees argv.
    """
    parser = _Parser(
        prog="mrph",
        description="the headless morph orchestrator: one JSON document on "
                    "stdout, the human log on stderr")
    commands = parser.add_subparsers(
        dest="command", required=True, title="commands")

    deck = commands.add_parser(
        "deck", help="inspect and edit the backlog (.morph/deck.json)")
    deck_commands = deck.add_subparsers(
        dest="deck_command", required=True, title="deck commands")

    add = deck_commands.add_parser(
        "add", help="add one card, or a JSON array of cards, to the backlog")
    _add_common_options(add)
    add.add_argument(
        "--file", required=True,
        help="path to a JSON card or a JSON array of cards")

    check = deck_commands.add_parser(
        "check", help="judge the backlog's file ownership")
    _add_common_options(check)

    status = deck_commands.add_parser(
        "status", help="show the backlog and the run state")
    _add_common_options(status)

    submit = commands.add_parser(
        "submit", help="compile and submit the deck's current generation")
    _add_common_options(submit)
    submit.add_argument(
        "--processor", default=None,
        help="processor id to run under (default: the registry's default)")
    submit.add_argument(
        "--nogit", action="store_true",
        help="skip the run branch; the run itself is not skipped")

    collect = commands.add_parser(
        "collect", help="poll the in-flight batch once (or --wait)")
    _add_common_options(collect)
    collect.add_argument(
        "--processor", default=None,
        help="processor id to poll under (default: the registry's default)")
    collect.add_argument(
        "--wait", action="store_true",
        help="poll until the generation settles, not just once")
    collect.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help="seconds the WHOLE wait may take (default: 21600, six hours)")

    run = commands.add_parser(
        "run", help="run the whole deck once, blocking, to the archive")
    _add_common_options(run)
    run.add_argument(
        "--processor", default=None,
        help="processor id to run under (default: the registry's default)")
    run.add_argument(
        "--nogit", action="store_true",
        help="skip the run branch; the run itself is not skipped")
    run.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help="seconds the WHOLE wait may take (default: 21600, six hours)")

    report = commands.add_parser(
        "report", help="show one archived run report")
    _add_common_options(report)
    report.add_argument(
        "deck_id", nargs="?", default=None,
        help="the archived run's deck id (default: the latest)")

    return parser


def _stderr_log(line: str) -> None:
    """The human log: one line to ``sys.stderr``, resolved at call time.

    ``sys.stderr`` is looked up on every call, not captured once, so a caller
    that swaps the stream around the invocation (the tests do exactly that)
    still receives the lines. The trailing newline is normalised: one line
    in, one line out, never two.
    """
    sys.stderr.write(line.rstrip("\n") + "\n")


def _handler(
    args: argparse.Namespace,
) -> Callable[[], Union[dict, Tuple[dict, int]]]:
    """Turn a parsed command line into the zero-argument handler ``run_cli``
    will call.

    Pure wiring -- each branch names its handler and hands over exactly the
    parsed values, and nothing else. The stderr logger goes to precisely the
    three handlers that accept one (the two provider-facing steps and the
    whole cycle); the read-only commands have nothing to log and take
    nothing else.
    """
    if args.command == "deck":
        if args.deck_command == "add":
            return lambda: cli_deck.deck_add(args.root, args.file)
        if args.deck_command == "check":
            return lambda: cli_deck.deck_check(args.root)
        return lambda: cli_deck.deck_status(args.root)
    if args.command == "submit":
        return lambda: cli_run.submit(
            args.root, processor=args.processor, nogit=args.nogit,
            log=_stderr_log)
    if args.command == "collect":
        return lambda: cli_run.collect(
            args.root, processor=args.processor, wait=args.wait,
            timeout=args.timeout, log=_stderr_log)
    if args.command == "run":
        return lambda: cli_cycle.run(
            args.root, processor=args.processor, nogit=args.nogit,
            timeout=args.timeout, log=_stderr_log)
    return lambda: cli_deck.report(args.root, args.deck_id)


def _usage_failure(message: str) -> int:
    """Emit the usage error document on stdout and return exit code 4.

    The parse failed before any handler ran, so ``--pretty`` was never
    parsed either: the document is the compact one-line form regardless.
    """
    emit(error_document(EXIT_USAGE, "UsageError", message), sys.stdout)
    return EXIT_USAGE


def main(argv: Optional[List[str]] = None) -> int:
    """Parse ``argv`` (default ``sys.argv[1:]``), dispatch, return the code.

    The whole headless entry point, and the only place the shell touches:
    the parsed command names a handler, :func:`cards.cli_json.run_cli` calls
    it and writes its ONE document to ``sys.stdout`` -- success or failure --
    and the returned code is the process exit code. The code is the
    handler's, never this module's: a bare dict means 0, a ``(dict, code)``
    pair means that code, and a raised exception is classified by
    :func:`cards.cli_json.classify`. Nothing prints to stdout but
    ``run_cli``; the human log goes to ``sys.stderr`` (:func:`_stderr_log`);
    stdin is never read and no question is ever asked.

    A parse failure -- including no subcommand at all, which the required
    subcommand choice refuses like any other -- is answered by
    :func:`_usage_failure`: the error document, exit code 4, with argparse's
    own usage and diagnosis still on stderr. The one argparse exit that is
    NOT a failure is ``-h``/``--help``: argparse has already written the
    help to stdout for a human, and ``main`` returns 0 around it.
    """
    if argv is None:
        argv = sys.argv[1:]
    try:
        args = _build_parser().parse_args(argv)
    except _UsageError as exc:
        return _usage_failure(str(exc) or "invalid arguments")
    except SystemExit as exc:
        # With error() overridden, argparse's own exits are the help path
        # (code 0). Anything non-zero cannot arise from this parser, but if
        # it ever does it is a usage failure like any other.
        if exc.code in (0, None):
            return EXIT_OK
        return _usage_failure("invalid arguments (the usage is on stderr)")
    return run_cli(_handler(args), sys.stdout, pretty=args.pretty)


if __name__ == "__main__":
    sys.exit(main())
