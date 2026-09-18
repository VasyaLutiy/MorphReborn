"""
The headless handlers that never talk to a provider: ``deck add``, ``deck
check``, ``deck status`` and ``report``.

This module is the handler layer of the headless ``mrph`` CLI. Each function
below is one command's MEANING -- read this fragment into the backlog, judge
the deck's file ownership, say where the run stands, show one archived run --
and nothing else. Everything around the meaning belongs to another layer, on
purpose: no function here parses arguments (the dispatcher's whole job is
turning argv into one of these functions called with its arguments), none
prints or touches ``sys`` (that is :func:`cards.cli_json.run_cli`'s job: one
JSON document on the stream handed to it, one exit code to the shell), and
none performs provider traffic -- these four read and write the project's
``.morph/`` and stop there. A handler returns a JSON-ready ``dict``, or a
``(dict, exit_code)`` pair when the shell must hear a verdict, as
``deck_check`` does, or raises anything at all;
:func:`cards.cli_json.run_cli` turns each of those outcomes into the one
document and the one code.

WHY the handlers are a layer of their own rather than functions that print as
they go: a function that RETURNS its result is testable without capturing
stdout. Exercising the command as a process couples the test to the
dispatcher, the emitter and the handler at once, and when such a test fails
there is no saying which of the three broke. A handler that returns the
payload it would have printed is asserted on directly -- ``payload, code =
deck_check(root)`` -- which is exactly how ``tests/test_cli_deck.py`` tests
every function in this module, each against a temporary project root. The
end-to-end contract (one document, one code) is enforced once, by ``run_cli``
wrapping whatever handler the dispatcher names; the per-command tests do not
re-prove it, because they do not have to.

Two disciplines hold across the layer. The SHAPES come from
:mod:`cards.cli_views` and nowhere else: a dict built by hand here would be a
second serialiser waiting to drift away from the printed one, so every
entity-shaped value -- the hazards, the deck view, the run report -- passes
through a ``cli_views`` function on its way out. And the REFUSALS speak the
vocabulary of :mod:`cards.cli_json`: a handler that must stop the caller
raises :class:`cards.cli_json.CliError` carrying one of that module's
exit-code constants (``classify`` honours a ``CliError`` as raised), while
every other failure simply propagates and is classified by the table there.
What is deliberately NOT refused anywhere in this module is a file-ownership
hazard: ``deck_add`` and ``deck_check`` put the hazards in their payloads --
and ``deck_check`` turns errors into its exit code -- but neither refuses to
let an operator see a deck; refusing is the run preflight's job
(:func:`cards.store.preflight_deck`), which runs when money is about to be
spent.

Like the rest of ``cards``, this module imports nothing from ``flows``.
"""

import json
from typing import Dict, List, Optional, Tuple

from cards.cli_json import EXIT_OK, EXIT_REFUSED, EXIT_USAGE, CliError
from cards.cli_views import (
    deck_status_to_dict,
    hazards_to_list,
    report_to_dict,
)
from cards.hazards import (
    errors as hazard_errors,
    find_hazards,
    warnings as hazard_warnings,
)
from cards.store import DeckStore, build_deck_status, list_runs


def deck_add(root: str, path: str) -> Dict[str, object]:
    """Add one card -- or a JSON array of cards -- from ``path`` to the backlog.

    The file at ``path`` holds either a JSON array of card dicts or one card
    dict; a single object is wrapped in a list, so adding one card by hand
    does not require remembering the array. Anything else -- a bare string, a
    number -- is refused as a :class:`cards.cli_json.CliError` with
    ``EXIT_USAGE``: no reading of a scalar is a deck fragment.

    The addition itself is :meth:`cards.store.DeckStore.add_cards`, which is
    already atomic: the whole fragment is validated against the existing
    backlog and either all of it is saved or none of it is. A malformed card
    raises :class:`cards.schema.CardError` (exit 4) and a fragment that would
    break the deck -- a duplicate ``custom_id``, a dangling or cyclic
    dependency -- raises :class:`cards.deck.DeckError` (exit 2); both simply
    propagate to :func:`cards.cli_json.classify`. A missing file raises
    :class:`FileNotFoundError` and malformed JSON a
    :class:`json.JSONDecodeError`; those propagate too, because the error
    table maps each to exit 4 -- the caller asked for something that cannot
    work, and nothing has been written.

    File-ownership hazards are REPORTED here, never refused. This command is
    ``/card`` in headless form, and ``/card`` reports a contested file and
    lets the operator decide; so the payload carries the hazards and the call
    succeeds even when a hazard's severity is ``error``. Refusing is the run
    preflight's job (:func:`cards.store.preflight_deck`), which runs later,
    when a batch is about to cost money.

    Returns ``{"added": [...], "hazards": [...]}``: the added custom_ids in
    the order given, and the hazards of the WHOLE backlog -- the deck the
    fragment landed in, pre-existing cards included, so a conflict with a card
    added yesterday is reported exactly like one inside the fragment --
    through :func:`cards.cli_views.hazards_to_list`.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if isinstance(raw, list):
        card_dicts: List[dict] = raw
    elif isinstance(raw, dict):
        card_dicts = [raw]
    else:
        raise CliError(
            EXIT_USAGE, "UsageError",
            f"{path}: a deck fragment must be a JSON array of cards or a "
            f"single card object, got {type(raw).__name__}")

    store = DeckStore(root)
    added = store.add_cards(card_dicts)
    hazards = find_hazards(store.load_cards())
    return {
        "added": [card.custom_id for card in added],
        "hazards": hazards_to_list(hazards),
    }


def deck_check(root: str) -> Tuple[Dict[str, object], int]:
    """Check the backlog's file ownership; return ``(payload, exit_code)``.

    The payload counts what the deck holds -- ``cards`` is the backlog size,
    ``errors`` and ``warnings`` the lengths of :func:`cards.hazards.errors`
    and :func:`cards.hazards.warnings` over
    :func:`cards.hazards.find_hazards` -- and lists EVERY hazard, both
    severities, through :func:`cards.cli_views.hazards_to_list`, so the caller
    of a failed check can print exactly what a human needs to read.

    The exit code is the command's whole verdict, which is why it is coarse
    exactly here: ``EXIT_REFUSED`` (2) when at least one error hazard exists,
    ``EXIT_OK`` (0) otherwise. A check that cannot fail a script is not a
    check -- a headless caller runs ``deck check`` precisely to learn whether
    the deck as it stands would stop a run. Warnings alone do NOT move the
    code: they are real, and they are in the payload to be read, but an
    operator who has read them and chosen to proceed must not be wedged by
    them either -- the same bargain ``/deck check`` makes interactively, and
    the same line the run preflight draws (errors refuse, warnings report).
    """
    store = DeckStore(root)
    cards = store.load_cards()
    hazards = find_hazards(cards)
    error_count = len(hazard_errors(hazards))
    payload = {
        "cards": len(cards),
        "errors": error_count,
        "warnings": len(hazard_warnings(hazards)),
        "hazards": hazards_to_list(hazards),
    }
    exit_code = EXIT_REFUSED if error_count else EXIT_OK
    return payload, exit_code


def deck_status(root: str) -> Dict[str, object]:
    """The backlog and run state, as ``/deck`` prints them, as one dict.

    A deliberate pass-through, and the shortest handler here for that reason:
    :func:`cards.store.build_deck_status` derives the snapshot from the
    backlog and ``state.json`` (per-card display status, the composition, the
    current generation), and :func:`cards.cli_views.deck_status_to_dict` is
    that view's one JSON shape. Named anyway so the dispatcher has one handler
    per command and the tests can exercise the command without composing its
    two parts -- the layer's promise is that a command is a callable, not a
    recipe.
    """
    return deck_status_to_dict(build_deck_status(DeckStore(root)))


def report(root: str, deck_id: Optional[str] = None) -> Dict[str, object]:
    """One archived run report, through :func:`cards.cli_views.report_to_dict`.

    With ``deck_id``, the run archived under that id; without it, the LATEST
    one. The latest is ``runs[0]``: :func:`cards.store.list_runs` returns the
    archive NEWEST FIRST -- its final statement sorts by ``(completed_at,
    deck_id)`` with ``reverse=True``, which is read out of ``cards/store.py``
    itself, not assumed -- so the newest run is at the HEAD of the list, not
    the tail. ``completed_at`` is recorded to millisecond precision precisely
    so two runs started within the same second still order by when they
    FINISHED, and the ``deck_id`` in the sort key only breaks ties between
    runs finishing in the same millisecond. No re-sort happens here:
    re-ordering what the store already ordered would be a second opinion on
    "latest", and the two would eventually disagree.

    Raises :class:`cards.cli_json.CliError` with ``EXIT_USAGE`` in the two
    cases where the request names nothing that exists: when no run is
    archived at all -- a project that has never finished a run has nothing to
    report -- and when the requested ``deck_id`` is not among the archived
    ones. In the second case the message NAMES the ids that ARE archived, so
    the operator gets a diagnosis and the id they meant rather than a dead
    end.
    """
    store = DeckStore(root)
    runs = list_runs(store)
    if not runs:
        raise CliError(
            EXIT_USAGE, "UsageError",
            f"no runs are archived under {store.runs_dir} yet -- a run is "
            f"archived when it finishes (/nightly, or the /collect that "
            f"completes the deck)")
    if deck_id is None:
        chosen = runs[0]
    else:
        chosen = next((run for run in runs if run.deck_id == deck_id), None)
        if chosen is None:
            raise CliError(
                EXIT_USAGE, "UsageError",
                f"no archived run named {deck_id!r}; the archive under "
                f"{store.runs_dir} holds: "
                f"{', '.join(run.deck_id for run in runs)}")
    return report_to_dict(chosen)
