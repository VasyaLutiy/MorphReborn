"""
The independent judge of the headless ``mrph`` CLI's output contract.

This file adds no production code and modifies nothing: it holds the headless
CLI -- ``cards.cli.main`` -- to the contract a scripted caller (and most of all
an agent) builds on. The contract is stated here as RULES, so the tests judge
the promise and not what the code happens to do today:

* **B1.** stdout carries exactly one JSON document, on EVERY outcome of EVERY
  subcommand, success and failure alike. ``json.loads`` over the WHOLE captured
  stdout must succeed every single time -- which also forbids anything else on
  the stream, because a log line before the document or trailing junk after it
  is exactly what ``json.loads`` refuses ("Extra data"). The sweep runs every
  command without ``--pretty``, so the one line is also asserted outright.
* **B2.** Without ``--pretty`` that document is one line; with ``--pretty`` it
  is indented over several lines. Both parse, and both carry the SAME document.
* **B3.** The human log goes to stderr. Whatever appears there must never
  appear on stdout.
* **B4.** Nothing reads stdin and nothing asks a question. Any subcommand, run
  with stdin empty -- or closed outright -- must answer without hanging or
  raising. Every invocation in this file already runs with an EMPTY
  ``io.StringIO`` as stdin; the dedicated B4 test closes the stream entirely.
* **B5.** The exit codes are: 0 done; 1 finished with failed or skipped cards;
  2 refused before spending anything (ownership conflict, dirty tree,
  ``StoreError``); 3 transport or an exhausted wait; 4 usage (missing file,
  malformed JSON, unknown processor, bad arguments). Where the table rules
  unambiguously the code is PINNED; where an empty root makes two readings
  honest (an empty deck submitted, an empty archive reported) the test says so
  and accepts exactly the defensible set -- never the whole vocabulary as a
  shrug.
* **B6.** Every error document is exactly ``{"error": {"code": <int>,
  "kind": <str>, "message": <str>}}`` -- those three keys, that nesting, a
  non-empty kind and a non-empty message -- and the document's code agrees
  with the code the process exits with.

The method: everything is driven IN PROCESS through ``cards.cli.main([...])``
-- never a subprocess -- with ``sys.stdout``/``sys.stderr`` captured into
``io.StringIO`` and ``sys.stdin`` replaced. Every invocation gets a fresh
``tempfile.TemporaryDirectory`` passed as ``--root``, so no test touches this
project's ``.morph/`` and no test leaves state behind for the next. The one
invocation helper fails with a NAMED message carrying the raw stdout (and the
stderr for context) when the document does not parse -- a judge that says only
``JSONDecodeError`` burns the attempt blind.

Every failure is forced locally, so no test reaches the network, sleeps for
real, or takes more than a moment: bad argv, a missing file, malformed JSON,
cards that violate ``cards.schema`` (each fixture card is one
``MorphCard.from_dict`` would ACCEPT -- ``intent`` is exactly one of
``generate``/``patch``/``todo``, never prose), an unknown processor id
(resolved against the local registry before any provider is contacted),
nothing in flight, and a deck whose two cards claim one file in one
generation.
"""

import io
import json
import sys
import tempfile
from pathlib import Path

import pytest

from cards import cli

# An unknown processor id: resolved against the local registry before anything
# is submitted, so naming this forces a clean LOCAL failure -- the contract's
# usage case -- with no provider ever contacted.
NO_SUCH_PROCESSOR = "no-such-processor"

# B5's whole vocabulary. Every invocation must land inside it; the pinned
# tests narrow from here, never outside it.
EXIT_CODES = frozenset({0, 1, 2, 3, 4})


# ---------------------------------------------------------------------------
# fixtures, fixtures-builders, and the one invocation helper
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_root() -> Path:
    """A brand-new empty project root for exactly one test.

    A fresh ``tempfile.TemporaryDirectory`` per run: nothing this suite does
    can touch this project's own ``.morph/``, and nothing it creates in the
    root survives the test -- the temporary directory is deleted afterwards.
    """
    with tempfile.TemporaryDirectory(prefix="mrph-cli-contract-") as tmp:
        yield Path(tmp)


def _invoke(argv, root, stdin_stream=None):
    """One headless invocation, in process: argv in, (code, stdout, stderr) out.

    Never a subprocess: ``cards.cli.main`` is called directly, with the three
    standard streams swapped for capturable ones around the call and restored
    afterwards -- in a ``finally``, so a contract breach surfaces as a test
    failure without poisoning the streams for the next test. ``--root`` is
    appended here so every invocation without exception points at the fresh
    temporary root. ``stdin`` defaults to an EMPTY ``io.StringIO`` -- B4's
    "empty" half therefore holds across the whole sweep, not only in the
    dedicated test.
    """
    argv = list(argv) + ["--root", str(root)]
    out, err = io.StringIO(), io.StringIO()
    if stdin_stream is None:
        stdin_stream = io.StringIO()
    saved = (sys.stdin, sys.stdout, sys.stderr)
    sys.stdin, sys.stdout, sys.stderr = stdin_stream, out, err
    try:
        code = cli.main(argv)
    finally:
        sys.stdin, sys.stdout, sys.stderr = saved
    return code, out.getvalue(), err.getvalue()


def _parse_stdout(stdout_text, stderr_text, label):
    """``json.loads`` the WHOLE captured stdout, or fail naming the run.

    The raw stdout goes into the message, with the stderr for context: a
    judge that reports only ``JSONDecodeError`` burns the attempt blind.
    Parsing the whole text (not a stripped prefix) is deliberate -- trailing
    junk after the document is as dead a contract as a leading log line.
    """
    try:
        return json.loads(stdout_text)
    except ValueError as exc:
        raise AssertionError(
            f"B1 violated [{label}]: stdout must be exactly one JSON document "
            f"on every outcome, but json.loads failed: {exc}; raw stdout: "
            f"{stdout_text!r}; stderr for context: {stderr_text!r}"
        ) from exc


def _assert_stderr_off_stdout(stdout_text, stderr_text, label):
    """B3: no line the human log put on stderr may appear on stdout."""
    for line in stderr_text.splitlines():
        if line.strip():
            assert line not in stdout_text, (
                f"B3 violated [{label}]: a human-log line appeared on stdout: "
                f"{line!r}; raw stdout: {stdout_text!r}"
            )


def run_invocation(argv, root, label, stdin_stream=None):
    """Run one invocation and JUDGE its output contract; return the findings.

    Returns ``(exit_code, parsed_stdout, stderr_text)``. On the way it checks,
    for every invocation in the suite: B1 -- stdout parses as one JSON
    document and is exactly one compact line (the sweep never passes
    ``--pretty``), with nothing before or after it; B3 -- no stderr line
    leaked onto stdout; and the baseline of B5 -- the exit code comes from the
    documented vocabulary at all. Failing any of these names the subcommand,
    the situation, and what was expected, and carries the raw streams.
    """
    code, stdout_text, stderr_text = _invoke(argv, root, stdin_stream=stdin_stream)
    doc = _parse_stdout(stdout_text, stderr_text, label)
    lines = stdout_text.splitlines()
    assert len(lines) == 1, (
        f"B1 violated [{label}]: stdout must carry exactly one compact JSON "
        f"document -- one line, no blank noise -- got {len(lines)} lines; raw "
        f"stdout: {stdout_text!r}; stderr for context: {stderr_text!r}"
    )
    _assert_stderr_off_stdout(stdout_text, stderr_text, label)
    assert code in EXIT_CODES, (
        f"B5 violated [{label}]: the exit code must come from the documented "
        f"vocabulary {sorted(EXIT_CODES)}, got {code}"
    )
    return code, doc, stderr_text


def assert_exit(code, expected, label, doc=None):
    """B5: ``expected`` is a pinned int, or the exact set of defensible codes."""
    if isinstance(expected, int):
        expected = frozenset({expected})
    assert code in expected, (
        f"B5 violated [{label}]: expected exit "
        f"{' or '.join(str(c) for c in sorted(expected))}, got {code}"
        + (f"; document: {doc!r}" if doc is not None else "")
    )


def assert_error_document(doc, label, code=None):
    """B6: the failure shape, and exactly it: error -> code/kind/message."""
    assert isinstance(doc, dict), (
        f"B6 violated [{label}]: the error document must be a JSON object, "
        f"got {type(doc).__name__}; document: {doc!r}"
    )
    assert set(doc.keys()) == {"error"}, (
        f"B6 violated [{label}]: the error document must have exactly one key "
        f"'error', got {sorted(doc.keys())}; document: {doc!r}"
    )
    inner = doc["error"]
    assert isinstance(inner, dict), (
        f"B6 violated [{label}]: the 'error' value must be a JSON object, got "
        f"{type(inner).__name__}; document: {doc!r}"
    )
    assert set(inner.keys()) == {"code", "kind", "message"}, (
        f"B6 violated [{label}]: the error object must have exactly the keys "
        f"'code', 'kind' and 'message', got {sorted(inner.keys())}; document: "
        f"{doc!r}"
    )
    assert isinstance(inner["code"], int) and not isinstance(inner["code"], bool), (
        f"B6 violated [{label}]: the error code must be an int, got "
        f"{inner['code']!r}; document: {doc!r}"
    )
    assert isinstance(inner["kind"], str) and inner["kind"], (
        f"B6 violated [{label}]: the error kind must be a non-empty string, "
        f"got {inner['kind']!r}; document: {doc!r}"
    )
    assert isinstance(inner["message"], str) and inner["message"], (
        f"B6 violated [{label}]: the error message must be a non-empty "
        f"string, got {inner['message']!r}; document: {doc!r}"
    )
    if code is not None:
        assert inner["code"] == code, (
            f"B5/B6 violated [{label}]: the document's error code "
            f"{inner['code']!r} must be the same code the process exits with "
            f"({code}); document: {doc!r}"
        )


def _card(custom_id, target, instruction="Write the file named by the target."):
    """A morph card ``cards.schema.MorphCard.from_dict`` ACCEPTS.

    ``intent`` is EXACTLY one of the three flow names -- it is not a prose
    description of what the card does; the prose belongs in ``instruction``.
    ``custom_id`` matches the safe character set, and the card names exactly
    one ``target``.
    """
    return {
        "custom_id": custom_id,
        "intent": "generate",
        "target": target,
        "instruction": instruction,
    }


def _write_card_file(root, name, payload):
    """Write one card (or a future array) as JSON inside the fresh root."""
    path = root / name
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return str(path)


def _prepared_happy_argv(argv, root):
    """Replace the ``@CARD`` placeholder with a valid card file in ``root``."""
    card_path = _write_card_file(root, "card.json", _card("probe.hello", "src/hello.py"))
    return [card_path if part == "@CARD" else part for part in argv]


# ---------------------------------------------------------------------------
# the B1 sweep: seven subcommands x the nine situations
# ---------------------------------------------------------------------------

# Situation 1, one case per subcommand. Pinned where the table rules
# unambiguously:
#   deck status / deck check / deck add -- an empty backlog shown, judged (no
#     claims, so no conflict), or extended by one valid card is DONE: 0.
#   run -- running zero cards completes with nothing failed or skipped: done,
#     0 (its answer is the "empty" document).
#   collect -- nothing in flight is a state refusal before anything is spent:
#     2 (situation 8 pins this same invocation on its own).
# Left as the defensible SET where an empty root makes two readings honest:
#   submit -- an empty deck may be answered as done (nothing to submit, the
#     "empty" document) or refused (nothing to submit is a state refusal):
#     0 or 2.
#   report -- an empty archive may be answered as done (no runs archived yet),
#     as the store's refusal to show a latest run, or as the usage-flavoured
#     absence of the artefact asked for: 0, 2 or 4.
HAPPY_CASES = [
    (["deck", "status"], "deck status", 0),
    (["deck", "check"], "deck check", 0),
    (["deck", "add", "--file", "@CARD"], "deck add", 0),
    (["submit", "--nogit"], "submit", frozenset({0, 2})),
    (["collect"], "collect", 2),
    (["run", "--nogit"], "run", 0),
    (["report"], "report", frozenset({0, 2, 4})),
]

HAPPY_IDS = [
    "deck-status", "deck-check", "deck-add", "submit", "collect", "run", "report",
]


@pytest.mark.parametrize(("argv", "name", "expected"), HAPPY_CASES, ids=HAPPY_IDS)
def test_situation1_empty_root(argv, name, expected, fresh_root):
    """Situation 1: the happy-ish path on an empty temporary root.

    B1 and B3 hold for every outcome; B5 is pinned exactly where the table is
    unambiguous and stated as the defensible set where it is not (see the
    case table above).
    """
    label = f"{name} / situation 1 (empty temporary root)"
    argv = _prepared_happy_argv(argv, fresh_root)
    code, doc, _ = run_invocation(argv, fresh_root, label)
    assert_exit(code, expected, label, doc)


@pytest.mark.parametrize(
    ("argv", "name"),
    [
        (["deck", "add"], "deck add (missing --file)"),
        (["deck"], "deck (missing deck subcommand)"),
    ],
    ids=["deck-add-no-file", "deck-no-subcommand"],
)
def test_situation2_missing_required_argument(argv, name, fresh_root):
    """Situation 2: a missing required argument is usage -- exit 4.

    Argparse's own usage block and diagnosis belong on stderr (B3), while
    stdout still carries the one error document (B1, B6).
    """
    label = f"{name} / situation 2 (a missing required argument)"
    code, doc, stderr_text = run_invocation(argv, fresh_root, label)
    assert_exit(code, 4, label, doc)
    assert_error_document(doc, label, code=4)
    assert stderr_text.strip(), (
        f"B3 [{label}]: argparse's usage and diagnosis belong on stderr, but "
        f"stderr is empty"
    )


def test_situation3_nonexistent_file(fresh_root):
    """Situation 3: a nonexistent --file path is usage -- exit 4 (B1/B5/B6)."""
    label = "deck add / situation 3 (a nonexistent --file path)"
    missing = fresh_root / "no-such-cards.json"
    code, doc, _ = run_invocation(
        ["deck", "add", "--file", str(missing)], fresh_root, label)
    assert_exit(code, 4, label, doc)
    assert_error_document(doc, label, code=4)


def test_situation4_malformed_json(fresh_root):
    """Situation 4: a --file holding malformed JSON is usage -- exit 4."""
    label = "deck add / situation 4 (a --file holding malformed JSON)"
    broken = fresh_root / "broken.json"
    broken.write_text('{"custom_id": "broken", "intent": "generat', encoding="utf-8")
    code, doc, _ = run_invocation(
        ["deck", "add", "--file", str(broken)], fresh_root, label)
    assert_exit(code, 4, label, doc)
    assert_error_document(doc, label, code=4)


# Situation 5: three cards that violate cards.schema -- each is rejected at
# construction, so deck add must answer with the usage error. The first is
# the classic mistake: a PROSE intent instead of one of the three flow names.
SCHEMA_VIOLATIONS = [
    (
        {
            "custom_id": "prose-intent",
            "intent": "write a friendly hello module",
            "target": "src/hello.py",
            "instruction": "Write a friendly hello module.",
        },
        "prose-intent",
    ),
    (
        {
            "custom_id": "no-target",
            "intent": "generate",
            "instruction": "Write something, somewhere.",
        },
        "no-target",
    ),
    (
        {
            "custom_id": "both-forms",
            "intent": "generate",
            "target": "src/a.py",
            "targets": ["src/b.py"],
            "instruction": "Write both files.",
        },
        "both-target-forms",
    ),
]


@pytest.mark.parametrize(
    ("card", "case"), SCHEMA_VIOLATIONS, ids=[case for _, case in SCHEMA_VIOLATIONS]
)
def test_situation5_schema_violation(card, case, fresh_root):
    """Situation 5: a --file holding a schema-violating card is usage -- 4."""
    label = f"deck add / situation 5 (a card violating the schema: {case})"
    path = _write_card_file(fresh_root, "violating.json", card)
    code, doc, _ = run_invocation(["deck", "add", "--file", path], fresh_root, label)
    assert_exit(code, 4, label, doc)
    assert_error_document(doc, label, code=4)


@pytest.mark.parametrize(
    ("argv", "name", "needs_deck"),
    [
        (["submit", "--nogit", "--processor", NO_SUCH_PROCESSOR], "submit", False),
        (["collect", "--processor", NO_SUCH_PROCESSOR], "collect", False),
        (["run", "--nogit", "--processor", NO_SUCH_PROCESSOR], "run", True),
    ],
    ids=["submit", "collect", "run"],
)
def test_situation6_unknown_processor(argv, name, needs_deck, fresh_root):
    """B1/B5/B6: an unknown processor is usage -- exit 4, the error shape.

    The refusal happens locally at processor resolution, before anything is
    submitted, so no network is touched.

    ``run`` is given a deck first: on an EMPTY deck ``run`` answers with its
    "empty" document and exit 0 without ever consulting the processor -- the
    processor only matters once there is a card to run under it, so there the
    unknown name would force nothing and no ruling applies. One valid card
    (added through ``deck add``) makes the processor genuinely needed, and
    then the table's usage ruling does: exit 4.
    """
    label = f"{name} / situation 6 (an unknown --processor)"
    if needs_deck:
        card_path = _write_card_file(
            fresh_root, "probe.json", _card("processor-probe", "src/probe.py"))
        add_label = f"deck add / situation 6 (the probe deck for {name})"
        add_code, add_doc, _ = run_invocation(
            ["deck", "add", "--file", card_path], fresh_root, add_label)
        assert_exit(add_code, 0, add_label, add_doc)
    code, doc, _ = run_invocation(argv, fresh_root, label)
    assert_exit(code, 4, label, doc)
    assert_error_document(doc, label, code=4)


@pytest.mark.parametrize(
    ("argv", "name"),
    [
        (["frobnicate"], "an unknown subcommand"),
        ([], "no subcommand at all"),
        (["deck", "frobnicate"], "an unknown deck subcommand"),
    ],
    ids=["unknown-top", "none-at-all", "unknown-nested"],
)
def test_situation7_bad_command(argv, name, fresh_root):
    """Situation 7: bad argv is usage -- exit 4, on stdout as the error shape.

    argparse's own usage and diagnosis stay on stderr (B3); the missing or
    mistyped subcommand is answered by the same one error document as any
    other usage failure (B1, B6).
    """
    label = f"{name} / situation 7 (bad command)"
    code, doc, stderr_text = run_invocation(argv, fresh_root, label)
    assert_exit(code, 4, label, doc)
    assert_error_document(doc, label, code=4)
    assert stderr_text.strip(), (
        f"B3 [{label}]: argparse's usage and diagnosis belong on stderr, but "
        f"stderr is empty"
    )


def test_situation8_collect_nothing_in_flight(fresh_root):
    """Situation 8: a state refusal -- collect with nothing in flight is 2.

    The request was reasonable; there is simply nothing to poll, and the
    refusal happens before anything could be spent.
    """
    label = "collect / situation 8 (nothing in flight)"
    code, doc, _ = run_invocation(["collect"], fresh_root, label)
    assert_exit(code, 2, label, doc)
    assert_error_document(doc, label, code=2)


def test_situation9_same_file_in_one_generation(fresh_root):
    """Situation 9: two cards claiming one file in one generation -- exit 2.

    The two cards are each valid on their own (cards.schema accepts both);
    the conflict is BETWEEN them, in one generation, and it is refused before
    anything is submitted or spent.
    """
    label = "run --nogit / situation 9 (two cards writing the same file)"
    first = _write_card_file(fresh_root, "first.json", _card("dup-first", "src/shared.py"))
    second = _write_card_file(fresh_root, "second.json", _card("dup-second", "src/shared.py"))
    for path, part in ((first, "the first add"), (second, "the second add")):
        add_label = f"deck add / situation 9 ({part})"
        add_code, add_doc, _ = run_invocation(
            ["deck", "add", "--file", path], fresh_root, add_label)
        assert_exit(add_code, 0, add_label, add_doc)
    code, doc, _ = run_invocation(["run", "--nogit"], fresh_root, label)
    assert_exit(code, 2, label, doc)
    assert_error_document(doc, label, code=2)


# ---------------------------------------------------------------------------
# the two explicit contract points beyond the sweep
# ---------------------------------------------------------------------------


def test_b2_compact_one_line_and_pretty_indented(fresh_root):
    """B2: the same deck status document, spelled both ways.

    Without ``--pretty`` it is one compact line; with ``--pretty`` it is
    indented over several lines; both parse, and both carry the SAME
    document -- only the spelling differs.
    """
    compact_label = "deck status / B2 (without --pretty)"
    code_c, doc_c, _ = run_invocation(["deck", "status"], fresh_root, compact_label)
    assert_exit(code_c, 0, compact_label, doc_c)

    pretty_label = "deck status / B2 (with --pretty)"
    code_p, out_p, err_p = _invoke(["deck", "status", "--pretty"], fresh_root)
    doc_p = _parse_stdout(out_p, err_p, pretty_label)
    _assert_stderr_off_stdout(out_p, err_p, pretty_label)
    assert_exit(code_p, 0, pretty_label, doc_p)
    pretty_lines = out_p.splitlines()
    assert len(pretty_lines) > 1, (
        f"B2 violated [{pretty_label}]: with --pretty the document is "
        f"indented over several lines; raw stdout: {out_p!r}"
    )
    assert any(line.startswith("  ") for line in pretty_lines), (
        f"B2 violated [{pretty_label}]: --pretty must indent the document; "
        f"raw stdout: {out_p!r}"
    )
    assert doc_p == doc_c, (
        f"B2 violated: --pretty must carry the SAME document, only spelled "
        f"differently; compact: {doc_c!r}; pretty: {doc_p!r}"
    )


@pytest.mark.parametrize(("argv", "name", "expected"), HAPPY_CASES, ids=HAPPY_IDS)
def test_b4_closed_stdin(argv, name, expected, fresh_root):
    """B4, the closed half: no subcommand reads stdin or asks a question.

    Every other test in this file already runs with an EMPTY ``io.StringIO``
    as stdin; here the stream is CLOSED outright, so much as one read would
    raise and be judged. Each subcommand must still answer normally: stdout
    parses, stderr stays off stdout, and the exit code is the same one the
    empty-root sweep demands.
    """
    label = f"{name} / B4 (stdin closed)"
    closed_stdin = io.StringIO()
    closed_stdin.close()
    argv = _prepared_happy_argv(argv, fresh_root)
    code, doc, _ = run_invocation(argv, fresh_root, label, stdin_stream=closed_stdin)
    assert_exit(code, expected, label, doc)
