"""
The MCP tool layer: four tools over the Morph batch orchestrator.

Where the ``mrph`` CLI drives the backlog and the submit / collect cycle from
a terminal, this module exposes the same operations as MCP tools so an agent
can drive a deck from a conversation. It is a deliberately thin layer: every
capability is delegated to :mod:`cards.store` -- :func:`build_deck_status`,
:meth:`DeckStore.add_card`, :func:`submit_generation` and
:func:`collect_generation` -- and this module only adapts arguments in and
renders the results as text out. It keeps no state of its own (the backlog and
the run state live entirely in ``<root>/.morph/``) and, like ``cards`` itself,
it never imports from ``flows``.

Two public names, the whole contract of the server (see ``__all__``):

* :data:`TOOLS` -- the tool descriptors served over ``tools/list``: name,
  description and a JSON-schema ``inputSchema`` for each of ``deck_status``,
  ``card_add``, ``deck_submit`` and ``deck_collect``.
* :func:`call_tool` -- the dispatcher behind ``tools/call``. It returns TEXT
  (a human-readable report, never a dict) and honours ``root``, so one server
  process can drive decks in any project directory: ``root`` is where the
  ``.morph/`` directory lives *and* the project root cards are compiled
  against.

The batch backend for ``deck_submit`` / ``deck_collect`` is built *inside*
those calls, from the processor registry
(``processors.registry.ProcessorRegistry.from_env()``), so merely importing
this module requires no configuration. When no processor is configured, those
two tools answer with a clear message instead of raising. Every other failure
propagates: a malformed card raises :class:`cards.schema.CardError` (a
``ValueError``), a refused run transition (submitting twice, collecting with
nothing in flight) a :class:`cards.store.StoreError`; surfacing those is the
MCP server layer's job, exactly as it surfaces the ``ValueError`` an unknown
tool name gets.
"""

from typing import Any, Dict, List, Optional, Tuple

from cards.store import (
    CardOutcome,
    DeckStatusView,
    DeckStore,
    PHASE_DONE,
    build_deck_status,
    collect_generation,
    submit_generation,
)
from processors.registry import ProcessorRegistry

__all__ = ["TOOLS", "call_tool"]

# Progress lines logged by the store / generation runner carry the CLI prompt;
# it is stripped when they are folded into a tool report.
_LOG_PREFIX = "mrph> "


# -- tool descriptors (tools/list) --------------------------------------------


TOOLS: List[Dict[str, Any]] = [
    {
        "name": "deck_status",
        "description": (
            "Render the Morph backlog: every card with its status (pending, "
            "in_flight, written, failed, skipped) and the generation "
            "composition of the current run. Takes no arguments."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "card_add",
        "description": (
            "Add one morph card to the backlog. The card is validated on its "
            "own and against the existing deck (duplicate custom_id, dangling "
            "or cyclic dependency) before anything is written."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "card": {
                    "type": "object",
                    "description": (
                        "One morph card, in the nested form (custom_id and "
                        "instruction at the top level, meta fields under "
                        "'meta') or the equivalent flat form."
                    ),
                },
            },
            "required": ["card"],
        },
    },
    {
        "name": "deck_submit",
        "description": (
            "Compile and submit the current generation's runnable cards as "
            "one batch. The batch backend is built from the processor "
            "registry inside the call; the run state is persisted so "
            "deck_collect can pick the batch up later. Cards whose "
            "dependencies already failed or were skipped are reported as "
            "skipped while advancing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "processor": {
                    "type": "string",
                    "description": (
                        "Optional processor label used to select the batch "
                        "backend and recorded with the in-flight batch."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "deck_collect",
        "description": (
            "Poll the in-flight batch once. While it is still running, report "
            "that and change nothing (call again later); once it has ended, "
            "verify and integrate every morph, advance the run and report "
            "each card's outcome. Takes no arguments."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
]


# -- rendering helpers ---------------------------------------------------------


def _strip_log_prefix(line: str) -> str:
    """Drop the CLI prompt a store progress line may carry."""
    if line.startswith(_LOG_PREFIX):
        return line[len(_LOG_PREFIX):]
    return line


def _render_deck_status(view: DeckStatusView) -> str:
    """Human-readable rendering of a :class:`DeckStatusView`."""
    if view.empty:
        return "backlog is empty; add cards with card_add"

    total = len(view.generations)
    header = f"backlog: {len(view.card_status)} card(s); phase {view.phase}"
    if total:
        header += f"; generation {min(view.current_generation + 1, total)} of {total}"
    lines = [header]

    if view.generations:
        lines.append("generations:")
        for number, generation in enumerate(view.generations, start=1):
            ids = ", ".join(generation) if generation else "(no cards)"
            marker = "   [current]" if number - 1 == view.current_generation else ""
            lines.append(f"  {number}: {ids}{marker}")

    lines.append("cards:")
    for custom_id, status in view.card_status:
        lines.append(f"  {custom_id}: {status}")
    return "\n".join(lines)


def _render_outcome(outcome: CardOutcome) -> str:
    """One card's outcome as a (possibly multi-line) indented report line."""
    line = f"  {outcome.custom_id}: {outcome.status}"
    bits: List[str] = []
    if outcome.paths:
        bits.append("paths: " + ", ".join(outcome.paths))
    if outcome.attempts and outcome.attempts != 1:
        bits.append(f"{outcome.attempts} attempts")
    if outcome.winning_variant is not None:
        bits.append(f"variant {outcome.winning_variant}")
    if bits:
        line += f" ({'; '.join(bits)})"
    if outcome.reason:
        line += f"\n      reason: {outcome.reason}"
    acceptance = (outcome.acceptance_output or "").strip()
    if acceptance:
        if len(acceptance) > 200:
            acceptance = acceptance[:200] + "..."
        line += f"\n      acceptance: {acceptance}"
    return line


# -- the per-call batch backend -------------------------------------------------


def _backend_from(registry: Any, label: Optional[str]) -> Optional[Any]:
    """Duck-type a batch backend out of a processor registry.

    The backend this layer needs is the one ``cards.store`` already talks to:
    ``submit`` / ``status`` / ``collect``. The registry may expose that
    protocol itself, hand a backend out through an accessor, or be a plain
    ``{label: backend}`` mapping; all three shapes are accepted so this thin
    layer does not pin the registry's API any harder than it has. Returns
    ``None`` when nothing backend-shaped can be obtained.
    """

    def _is_backend(obj: Any) -> bool:
        return obj is not None and callable(getattr(obj, "submit", None))

    # The registry speaks the backend protocol itself.
    if _is_backend(registry):
        return registry

    # ... or hands one out through an accessor (labelled first when we have a
    # label; a TypeError from an accessor that takes no label is skipped).
    for name in ("backend", "get", "build", "resolve"):
        accessor = getattr(registry, name, None)
        if not callable(accessor):
            continue
        arg_sets: List[Tuple[Any, ...]] = [(label,)] if label is not None else []
        arg_sets.append(())
        for args in arg_sets:
            try:
                candidate = accessor(*args)
            except Exception:
                continue
            if _is_backend(candidate):
                return candidate

    # ... or is a plain mapping: use the named entry, or the only one when no
    # label was given (an ambiguous mapping without a label is refused).
    try:
        entries = list(registry.values())
    except Exception:
        return None
    if label is None and len(entries) == 1 and _is_backend(entries[0]):
        return entries[0]
    return None


def _build_backend(label: Optional[str] = None) -> Tuple[Optional[Any], Optional[str]]:
    """Build the batch backend for one tool call; never raises.

    The registry is consulted *inside the call*, so importing this module (and
    serving the configuration-free tools) requires no provider setup. Returns
    ``(backend, None)`` on success, or ``(None, reason)`` when no processor is
    configured -- the caller turns that into a clear message.
    """
    try:
        registry = ProcessorRegistry.from_env()
    except Exception as exc:  # nothing configured / unusable configuration
        return None, f"no batch processor is configured ({exc})"
    if registry is None:
        return None, "no batch processor is configured"
    backend = _backend_from(registry, label)
    if backend is None:
        reason = "no batch processor could be built from the processor registry"
        if label is not None:
            reason += f" for processor {label!r}"
        return None, reason
    return backend, None


# -- the four tools --------------------------------------------------------------


def _tool_deck_status(arguments: Dict[str, Any], root: str) -> str:
    """Render the backlog: per-card statuses and generation composition."""
    return _render_deck_status(build_deck_status(DeckStore(root)))


def _tool_card_add(arguments: Dict[str, Any], root: str) -> str:
    """Validate and append one card; confirm with its custom_id and target."""
    card = arguments.get("card")
    if not isinstance(card, dict):
        raise ValueError(
            "card_add requires a 'card' object argument: one morph card in "
            "the flat or nested form"
        )
    store = DeckStore(root)
    added = store.add_card(card)
    total = len(store.load_cards())
    return (
        f"added card {added.custom_id!r} (target {added.target!r}, intent "
        f"{added.intent!r}); the backlog now holds {total} card(s)"
    )


def _tool_deck_submit(arguments: Dict[str, Any], root: str) -> str:
    """Compile and submit the current generation; report what was sent."""
    label = arguments.get("processor")
    if label is not None and not isinstance(label, str):
        raise ValueError("deck_submit: 'processor' must be a string")
    label = label or None

    backend, problem = _build_backend(label)
    if backend is None:
        return f"deck_submit: {problem}; nothing was submitted"

    store = DeckStore(root)
    progress: List[str] = []
    result = submit_generation(
        store, backend, root=root, backend_label=label, log=progress.append
    )

    lines: List[str] = []
    if result.submitted:
        lines.append(
            f"submitted generation {result.generation_number} of "
            f"{result.total_generations} as batch {result.batch_id!r} "
            f"({len(result.card_ids)} card(s))"
        )
        if result.card_ids:
            lines.append(f"  cards: {', '.join(result.card_ids)}")
    elif result.total_generations == 0:
        lines.append("nothing to submit: the backlog is empty")
    else:
        lines.append(
            "nothing to submit: no runnable cards remain; the deck run is "
            f"complete ({result.total_generations} generation(s))"
        )
    for custom_id, reason in result.skipped:
        lines.append(f"  skipped {custom_id!r}: {reason}")
    lines.extend(f"  {_strip_log_prefix(entry)}" for entry in progress)
    if result.submitted:
        lines.append("call deck_collect to pick up the results")
    return "\n".join(lines)


def _tool_deck_collect(arguments: Dict[str, Any], root: str) -> str:
    """Collect the in-flight generation; report each card's outcome."""
    backend, problem = _build_backend()
    if backend is None:
        return f"deck_collect: {problem}; nothing to collect"

    store = DeckStore(root)
    progress: List[str] = []
    result = collect_generation(store, backend, root=root, log=progress.append)

    if result.in_progress:
        return (
            f"generation {result.generation_number} of "
            f"{result.total_generations} is still running; call deck_collect "
            "again later"
        )

    lines = [
        f"generation {result.generation_number} of {result.total_generations} "
        "collected:"
    ]
    if not result.outcomes:
        lines.append("  (no cards in the collected generation)")
    for outcome in result.outcomes.values():
        lines.append(_render_outcome(outcome))
    if result.phase == PHASE_DONE:
        lines.append("the deck run is complete")
    else:
        lines.append(
            f"next: deck_submit for generation {result.generation_number + 1} "
            f"of {result.total_generations}"
        )
    lines.extend(f"  {_strip_log_prefix(entry)}" for entry in progress)
    return "\n".join(lines)


_HANDLERS = {
    "deck_status": _tool_deck_status,
    "card_add": _tool_card_add,
    "deck_submit": _tool_deck_submit,
    "deck_collect": _tool_deck_collect,
}


# -- dispatch (tools/call) --------------------------------------------------------


def call_tool(name: str, arguments: Dict[str, Any], root: str = ".") -> str:
    """Run one tool by name against a :class:`DeckStore` rooted at ``root``.

    ``root`` is honoured end to end: it locates the ``.morph/`` directory
    (backlog and run state) and is the project root the cards are compiled
    against, so one server process can drive decks in any directory.

    Returns a human-readable text report -- never a dict. An unknown tool name
    raises ``ValueError``; a malformed card raises
    :class:`cards.schema.CardError`; a refused run transition (submitting
    twice, collecting with nothing in flight) raises
    :class:`cards.store.StoreError`. Only a missing processor is answered with
    a message instead of an exception.
    """
    if not isinstance(name, str) or name not in _HANDLERS:
        known = ", ".join(sorted(_HANDLERS))
        raise ValueError(f"unknown tool {name!r}; known tools: {known}")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ValueError(
            f"tool {name!r} arguments must be an object, "
            f"got {type(arguments).__name__}"
        )
    return _HANDLERS[name](arguments, root)
