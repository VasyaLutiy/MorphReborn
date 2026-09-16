"""
The deck compiler: morph card + context slice -> a provider-specific batch payload.

Phase 1 of ``documentation/DEVELOPMENT_PLAN.md``. A morph card
(:class:`cards.schema.MorphCard`) is compiled into one *self-contained*
conversation -- exactly the shape Morph 1.0 builds interactively in
``flows/morph.py`` -- and then serialized into the on-the-wire batch format of
a provider (Anthropic Message Batches, OpenAI Batch, OpenRouter Batch). See
``documentation/batch-orchestrator.md`` ("What batch APIs offer") for why a
Morph request is already a batch request: a fresh, complete conversation from
disk, one instruction, one response, no session state.

Two guarantees this module exists to keep:

* **Fidelity to Morph 1.0.** The conversation for each intent mirrors
  ``flows/morph.py`` message-for-message: ``generate`` is the project context
  followed by the user instruction; ``patch`` prepends the two assistant
  framing messages carrying the original file; ``todo`` is the fixed
  criticism prompt with no project context (matching ``build_todo_transition``).
* **Determinism.** Identical inputs must produce byte-identical output. The
  transport-only ``time`` field is stripped from every message (mirroring
  ``MorphBot.clean_conversation``), whole-project walks are sorted before
  emission, and JSON is dumped with a stable key-construction order. This is
  what makes the golden-file tests meaningful.

Constraint (see the development plan): ``cards`` must not import from
``flows`` -- ``flows`` imports the world and has side effects. The small source
filter is therefore duplicated below rather than imported; it mirrors
``flows.morph.filter_source_code_file_names`` and must be kept in step with it.
"""

import copy
import json
import os
from typing import List

from cards.schema import MorphCard
from context_folder_dialog import ContextFolderDialog
from llm_dialog import LLMDialog


# Fixed instruction for the ``todo`` intent, copied verbatim from
# ``flows.morph.MorphBot.build_todo_transition`` so a compiled todo card and an
# interactive /todo produce the identical user message.
TODO_INSTRUCTION = (
    "Please, criticize this file contents and add \"TODO:\" comments, saying, "
    "what can be improved."
)


def _filter_source_code_file_names(file_path: str) -> bool:
    """Whether a walked file belongs in the whole-project context.

    Mirrors ``flows.morph.filter_source_code_file_names`` exactly. Duplicated
    (not imported) to keep ``cards`` free of any dependency on ``flows``; keep
    the two in step when either changes.
    """

    if 'node_modules' in file_path:
        return False

    if 'typechain-types' in file_path:
        return False

    if 'venv/' in file_path:
        return False

    if 'venvy/' in file_path:
        return False

    return (
            file_path.endswith('Dockerfile') or
            file_path.endswith('package.json') or
            file_path.endswith('requirements.txt') or
            file_path.endswith('.md') or
            file_path.endswith('.dot') or
            file_path.endswith('.py') or
            file_path.endswith('.sol') or
            file_path.endswith('.sh') or
            file_path.endswith('.rs') or
            file_path.endswith('.js') or
            file_path.endswith('.jsx') or
            file_path.endswith('.go') or
            file_path.endswith('.fc') or
            file_path.endswith('.yaml') or
            file_path.endswith('.yml') or
            file_path.endswith('.sql') or
            file_path.endswith('.ino') or
            file_path.endswith('.proto') or
            file_path.endswith('.txt') or
            file_path.endswith('.ts') or
            file_path.endswith('.tsx')
    )


def _resolve_slice(card: MorphCard, root: str) -> List[str]:
    """The list of context files to embed, relative to ``root``, SORTED.

    A non-empty ``context_slice`` is used verbatim (sorted for determinism); an
    empty one falls back to Morph 1.0's whole-project behaviour -- walk ``root``,
    keep what the source filter accepts, and sort. Sorting both paths means the
    emitted context order never depends on filesystem walk order.
    """
    if card.context_slice:
        return sorted(card.context_slice)

    matched = []
    for walk_root, _dirs, files in os.walk(root):
        for file_name in files:
            file_path = os.path.join(walk_root, file_name)
            if _filter_source_code_file_names(file_path):
                matched.append(os.path.relpath(file_path, root))
    return sorted(matched)


def _context_messages(card: MorphCard, root: str) -> List[dict]:
    """Build the project-context messages for a card (cleaned of ``time``).

    Whole-project mode is expressed as a sorted file list fed through
    :class:`ContextFolderDialog`'s file-list mode, so both slice and
    whole-project paths share one deterministic emission order and one message
    template.
    """
    file_list = _resolve_slice(card, root)
    context = ContextFolderDialog(root, file_list=file_list)
    context.process([])
    return _clean(context.conversation)


def _clean(conversation: List[dict]) -> List[dict]:
    """Deep-copy a conversation and strip the transport-only ``time`` field.

    Mirrors ``flows.morph.MorphBot.clean_conversation`` -- the ``time`` values
    are wall-clock timestamps and would destroy byte-for-byte reproducibility.
    """
    conversation = copy.deepcopy(conversation)
    for message in conversation:
        message.pop("time", None)
    return conversation


def _read_target(card: MorphCard, root: str) -> str:
    """Read the ``patch``/``todo`` target file, relative to ``root``."""
    target_path = os.path.join(root, card.target)
    with open(target_path, "r", encoding="utf-8") as handle:
        return handle.read()


def _build_conversation(card: MorphCard, root: str) -> List[dict]:
    """Assemble the self-contained conversation for one card.

    The three intents mirror ``flows/morph.py`` message-for-message; see the
    module docstring. The returned messages carry only ``role``/``content``
    (no ``time``), so they are ready to serialize.
    """
    if card.intent == "generate":
        messages = _context_messages(card, root)
        messages.append({"role": "user", "content": card.instruction})
        return messages

    if card.intent == "patch":
        dialog = LLMDialog()
        dialog.assign("assistant", f"Let's update the {card.target} file provided.")
        file_contents = _read_target(card, root)
        dialog.assign("assistant", f"Original file:\n\n---\n{file_contents}\n---\n")
        messages = _clean(dialog.conversation)
        messages.extend(_context_messages(card, root))
        messages.append({"role": "user", "content": card.instruction})
        return messages

    if card.intent == "todo":
        dialog = LLMDialog()
        dialog.assign("assistant", f"Let's update the {card.target} file provided.")
        file_contents = _read_target(card, root)
        dialog.assign("assistant", f"Original file:\n\n---\n{file_contents}\n---\n")
        dialog.assign("user", TODO_INSTRUCTION)
        return _clean(dialog.conversation)

    # MorphCard.validate() guarantees intent is one of the above; this guards
    # against a future intent being added to the schema without a compiler arm.
    raise ValueError(f"card {card.custom_id!r}: unsupported intent {card.intent!r}")


def compile_card(card: MorphCard, root: str = ".") -> List[dict]:
    """Compile one morph card into one request dict per variant.

    Returns ``card.variants`` request dicts. With ``variants == 1`` the single
    request keeps the card's ``custom_id``; with ``N > 1`` each request is
    suffixed ``.v1 .. .vN`` (best-of-N, per ``batch-orchestrator.md``). The
    conversation is built once and shared across variants -- the fan-out is over
    sampling, not over inputs. ``model`` is the card's hint (may be ``None``);
    the serializers resolve ``None`` against their ``default_model``.
    """
    messages = _build_conversation(card, root)

    requests = []
    if card.variants == 1:
        custom_ids = [card.custom_id]
    else:
        custom_ids = [f"{card.custom_id}.v{n}" for n in range(1, card.variants + 1)]

    for custom_id in custom_ids:
        requests.append({
            "custom_id": custom_id,
            "model": card.model,
            "messages": messages,
        })
    return requests


def compile_deck(cards: List[MorphCard], root: str = ".") -> List[dict]:
    """Compile a list of cards into a flat list of request dicts, in deck order."""
    requests = []
    for card in cards:
        requests.extend(compile_card(card, root))
    return requests


# -- serializers -------------------------------------------------------------


def _dumps(obj: dict) -> str:
    """Compact, stable JSON for one JSONL line.

    ``ensure_ascii=False`` keeps source bytes intact; ``separators`` drops the
    default whitespace; key order is the construction order of the dicts we
    build (never re-sorted), which is what makes the output byte-stable.
    """
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def serialize_anthropic(requests: List[dict], default_model: str, max_tokens: int = 8192) -> str:
    """Serialize requests as Anthropic Message Batches JSONL.

    One line per request: ``{"custom_id", "params": {"model", "max_tokens",
    "messages"}}``. Anthropic messages use only the ``user``/``assistant``
    roles, which our dialogs already satisfy. The returned string ends with a
    single trailing newline.
    """
    lines = []
    for request in requests:
        model = request["model"] or default_model
        line = _dumps({
            "custom_id": request["custom_id"],
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "messages": request["messages"],
            },
        })
        lines.append(line)
    return "".join(f"{line}\n" for line in lines)


def serialize_openai(requests: List[dict], default_model: str) -> str:
    """Serialize requests as OpenAI Batch JSONL.

    One line per request: ``{"custom_id", "method": "POST", "url":
    "/v1/chat/completions", "body": {"model", "messages"}}``. The returned
    string ends with a single trailing newline.
    """
    lines = []
    for request in requests:
        model = request["model"] or default_model
        line = _dumps({
            "custom_id": request["custom_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": request["messages"],
            },
        })
        lines.append(line)
    return "".join(f"{line}\n" for line in lines)


def serialize_openrouter(requests: List[dict], default_model: str) -> str:
    """Serialize requests as an OpenRouter Batch API submit payload.

    OpenRouter's batch API is *not* the OpenAI shape: there is no file upload
    and no JSONL. The whole deck goes out inline as one JSON document with
    exactly three top-level fields::

        {"endpoint": "/v1/chat/completions", "model": ..., "requests": [...]}

    Two consequences shape this function:

    * **Key order is load-bearing.** The service stream-parses the body and
      rejects it with a 400 if ``requests`` arrives before ``endpoint`` and
      ``model``. Returning a *string* (rather than a dict for the caller to
      dump) is what makes that ordering a guarantee of this module: ``_dumps``
      never re-sorts keys, so the construction order below is the wire order.
    * **One model per batch.** ``model`` applies to the entire batch, so each
      request's ``body`` deliberately omits it and inherits the batch-level
      one. A request body that names a *different* model is rejected by the
      service, so a card pinned to another model is caught here instead --
      such a deck must be split into one batch per model.

    Unlike the JSONL serializers this returns a single JSON document with no
    trailing newline: it is a request body, not a file.
    """
    items = []
    for request in requests:
        model = request["model"]
        if model is not None and model != default_model:
            raise ValueError(
                f"request {request['custom_id']!r} pins model {model!r}, but the "
                f"OpenRouter batch runs on {default_model!r}: OpenRouter applies "
                f"one model to the whole batch, so a deck mixing models must be "
                f"split into one batch per model")
        items.append({
            "custom_id": request["custom_id"],
            # No "model" key: every request inherits the batch-level model.
            "body": {
                "messages": request["messages"],
            },
        })

    return _dumps({
        "endpoint": "/v1/chat/completions",
        "model": default_model,
        "requests": items,
    })
