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
  One deliberate departure: a ``patch`` conversation never repeats its target
  in the project context, because it already ships it as "Original file" --
  see :func:`_context_messages` for the measurement that motivated it.
  A card that writes a SET of files (``targets``) extends -- never rewrites --
  that shape: one framing pair per existing target, and the output directive
  of :func:`_output_directive` appended to the closing user message. A
  single-target card compiles byte for byte as it always did, which the golden
  files in ``tests/golden`` hold to.
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
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

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


# What a multi-target card appends to its closing user message. WHY it is
# generated rather than left to the card author: a changeset card is worthless
# unless the answer can be split back into files, so the ONE format the parser
# understands (:func:`cards.generations.response_to_files`) has to be stated in
# every such prompt -- and a format restated by hand in every card is a format
# that drifts card by card until the parser rejects the answer.
MULTI_TARGET_DIRECTIVE = (
    "\n\n---\n"
    "This card writes SEVERAL files. Return each of them as a path line "
    "followed by one fenced block holding the COMPLETE file body:\n"
    "\n"
    "FILE: <path>\n"
    "```python\n"
    "<complete file body>\n"
    "```\n"
    "\n"
    "(the fence tag may name the file's own language). One such pair per file, "
    "in this exact order:\n"
    "{files}\n"
    "Use exactly these paths, give every one of them in full -- no diffs, no "
    "elisions -- and return no file this list does not name: an answer that "
    "misses a file or invents one is discarded whole and the card is retried."
)


def _output_directive(card: MorphCard) -> str:
    """The multi-file output convention for ``card``, or ``""`` for one target.

    Appended to the closing user message, so the executor is told how to return
    a set of files in the same breath as what to put in them. Empty for a
    single-target card -- its prompt must stay byte-identical to Morph 1.0's.
    """
    if len(card.targets) < 2:
        return ""
    files = "\n".join(f"- {target}" for target in card.targets)
    return MULTI_TARGET_DIRECTIVE.format(files=files)


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


def _path_key(root: str, relative_path: str) -> str:
    """A canonical identity for one context path, for comparison only.

    Slice entries and ``card.target`` are both written relative to ``root``, but
    the same file may legitimately be spelled ``flows/morph.py`` or
    ``./flows/morph.py``. Comparing the raw strings would miss the match, so
    both sides are joined to ``root`` and normalised to an absolute path first.
    The key is never emitted -- the original relative spelling is what reaches
    the prompt.
    """
    return os.path.abspath(os.path.join(root, relative_path))


def _resolve_slice(card: MorphCard, root: str, skip_target: bool = False) -> List[str]:
    """The list of context files to embed, relative to ``root``, SORTED.

    A non-empty ``context_slice`` is used verbatim (sorted for determinism); an
    empty one falls back to Morph 1.0's whole-project behaviour -- walk ``root``,
    keep what the source filter accepts, and sort. Sorting both paths means the
    emitted context order never depends on filesystem walk order.

    ``skip_target`` drops EVERY path the card writes (``card.targets``, which
    is the single ``target`` for a one-file card) from the result. Both paths
    need it: an author naturally lists the target in the slice (it *is* relevant
    context), and the whole-project walk yields the target too -- so for an
    intent that already ships the targets as "Original file" either path would
    otherwise send the same file twice. See :func:`_context_messages`.
    """
    if card.context_slice:
        paths = sorted(card.context_slice)
    else:
        matched = []
        for walk_root, _dirs, files in os.walk(root):
            for file_name in files:
                file_path = os.path.join(walk_root, file_name)
                if _filter_source_code_file_names(file_path):
                    matched.append(os.path.relpath(file_path, root))
        paths = sorted(matched)

    if skip_target:
        target_keys = {_path_key(root, target) for target in card.targets}
        paths = [path for path in paths if _path_key(root, path) not in target_keys]
    return paths


# -- what one compilation actually read ---------------------------------------
#
# WHY a card's inputs are captured and re-checked, and not merely compiled.
# A generation's cards are compiled TOGETHER, from the tree as it stands before
# the batch goes out -- and then verified ONE BY ONE, into that same tree
# (:func:`cards.acceptance.verify_card` writes at the real target paths, because
# acceptance must test the files where they will live). So by the time card N is
# judged, cards 1..N-1 have already written theirs. If one of those files is a
# file card N was compiled from, card N's answer was written against a version
# of the project that no longer exists, and accepting it overwrites a sibling's
# accepted work with no trace: both cards report ``written`` and the run is
# green. :mod:`cards.hazards` refuses such a deck up front; this is the runtime
# half of the same guarantee, and it also covers what no deck check can -- a
# human editing the tree while a batch is in a provider's queue for an hour.
#
# WHAT is watched, and why it is not everything the executor saw. The inputs
# checked are the card's DECLARED ones: its targets, plus the paths its
# ``context_slice`` names. The whole-project walk an EMPTY slice resolves to is
# deliberately not watched, although the executor did read it. Watching it would
# declare every card of a generation stale the moment the first one is accepted
# -- an empty slice contains every sibling's target -- and so buy one extra
# provider queue (20-40 minutes, ``Head_Pains.md`` 3.3) per generation to
# regenerate answers that are almost always fine. An empty slice IS a
# declaration that the project is read as a snapshot; :mod:`cards.hazards`
# reports what that snapshot will miss (``implicit-read``) at the only time
# anything can be done about it, while the deck is being written.
#
# What the watched set does cover is the whole of the destructive case: a card
# can only destroy another card's accepted work through a file it WRITES, and
# every target is watched. A stale read of a named slice file is watched too,
# because naming a file is a statement that its contents matter to this card.
#
# The paths are captured rather than re-derived, so the check re-reads exactly
# the files the record was taken from -- a file the tree merely GREW in the
# meantime (``.pytest_cache/README.md``, dropped by the first acceptance run of
# the very generation being judged) is not a change to anything this card was
# compiled from, and must not read as one.

# Per-path content digest width. Full SHA-256 is 64 hex characters and this map
# is persisted per card in ``.morph/state.json``; 16 hex characters are 64 bits,
# which no accidental edit collides with, and keeps a whole-project card's
# record to a couple of kilobytes.
_DIGEST_WIDTH = 16

# What :func:`_digest_file` records for a path that is not there. Distinct from
# any hash, so a file appearing or disappearing reads as a change like any other.
_ABSENT = "absent"


def _digest_file(path: str) -> str:
    """The content digest of one file, or :data:`_ABSENT` when it is missing.

    Read in binary and hashed whole: a card is stale when the BYTES it was
    compiled from differ, and nothing cheaper is trustworthy here -- the
    acceptance layer deliberately rewrites mtimes (see
    :func:`cards.acceptance._stamp_distinct_mtime`), so any stat-based shortcut
    would be reading the one field another guard is already moving on purpose.
    """
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return _ABSENT
    return hashlib.sha256(data).hexdigest()[:_DIGEST_WIDTH]


def card_inputs(card: MorphCard, root: str = ".") -> List[str]:
    """The files a card DECLARES it is compiled from, relative to ``root``, sorted.

    Its ``context_slice`` as written -- an empty one resolves to the whole
    project for the prompt, but declares nothing here, for the reason given
    above -- plus its targets, which ``patch`` and ``todo`` ship as "Original
    file" and which a ``generate`` card may legitimately overwrite.
    Deduplicated by canonical path, so a file named twice under two spellings is
    one input.
    """
    paths = sorted(card.context_slice)
    seen = {_path_key(root, path) for path in paths}
    for target in card.targets:
        key = _path_key(root, target)
        if key not in seen:
            seen.add(key)
            paths.append(target)
    return sorted(paths)


@dataclass(frozen=True)
class CompiledInputs:
    """The files one compilation read, and what they held at that moment.

    Captured next to :func:`compile_card` and carried until the card's answer is
    judged -- through ``.morph/state.json`` when the two happen in different CLI
    sessions. :meth:`changed` is the question it exists to answer, and it names
    the files rather than merely reporting a boolean, because "this card was
    compiled from a foo.py that no longer exists" is only actionable if the log
    says ``foo.py``.
    """

    digests: Dict[str, str] = field(default_factory=dict)

    @property
    def paths(self) -> List[str]:
        """The files read, sorted -- what the executor actually saw."""
        return sorted(self.digests)

    @classmethod
    def capture(cls, card: MorphCard, root: str = ".") -> "CompiledInputs":
        """Record what a compilation of ``card`` declares it reads, right now."""
        return cls({path: _digest_file(os.path.join(root, path))
                    for path in card_inputs(card, root)})

    def changed(self, root: str = ".") -> List[str]:
        """The captured paths whose bytes differ now, sorted; empty if none do.

        Only the captured paths are re-read: a file the tree has GROWN since,
        or one an empty slice swept into the prompt without declaring, is
        deliberately invisible here -- see the note above :func:`_digest_file`.
        """
        return sorted(path for path, digest in self.digests.items()
                      if _digest_file(os.path.join(root, path)) != digest)

    def to_dict(self) -> dict:
        """The persisted shape (``.morph/state.json``)."""
        return {"digests": dict(self.digests)}

    @classmethod
    def from_dict(cls, data: dict) -> "CompiledInputs":
        """Read back :meth:`to_dict`; a missing/!malformed record captures nothing.

        A record that cannot be read comes back EMPTY, which means "no captured
        inputs" and disables the staleness check for that card rather than
        failing the run: this guard exists to prevent silent loss, and turning a
        state file written by an older version into a crash would be a louder
        failure than the one it prevents.
        """
        if not isinstance(data, dict):
            return cls({})
        digests = data.get("digests")
        if not isinstance(digests, dict):
            return cls({})
        return cls({str(path): str(digest) for path, digest in digests.items()})


def _context_messages(card: MorphCard, root: str, skip_target: bool = False) -> List[dict]:
    """Build the project-context messages for a card (cleaned of ``time``).

    Whole-project mode is expressed as a sorted file list fed through
    :class:`ContextFolderDialog`'s file-list mode, so both slice and
    whole-project paths share one deterministic emission order and one message
    template.

    ``skip_target`` is set by the intents that already carry the target files in
    an "Original file" assistant message (``patch``). Sending such a file a second
    time as project context tells the model nothing it does not already have and
    costs its tokens twice -- measured on a real card, one 58.8 KB target sent
    twice was 92% of a 127 KB prompt, and on a larger target the duplicate is a
    way to walk into the context window for no benefit at all. The remaining
    slice entries keep their order and rendering.
    """
    file_list = _resolve_slice(card, root, skip_target=skip_target)
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


def _read_file(root: str, relative_path: str) -> str:
    """Read one file relative to ``root``."""
    with open(os.path.join(root, relative_path), "r", encoding="utf-8") as handle:
        return handle.read()


def _original_file_messages(card: MorphCard, root: str) -> List[dict]:
    """The assistant framing pair for every target the card ALREADY has on disk.

    One pair per target, in the card's order: a message naming the path ("Let's
    update the X file provided.") and the message carrying its bytes. For a
    single-target card this is exactly the two messages ``flows/morph.py``
    builds, byte for byte.

    A target that does not exist yet is simply not sent: a changeset card
    routinely patches one module while CREATING its test, and there is no
    original to show for the second. (For a single-target ``patch`` card this
    replaces a bare ``FileNotFoundError`` with a prompt that asks for the file
    from scratch -- which is what such a card meant.)
    """
    dialog = LLMDialog()
    for target in card.targets:
        if not os.path.exists(os.path.join(root, target)):
            continue
        dialog.assign("assistant", f"Let's update the {target} file provided.")
        file_contents = _read_file(root, target)
        dialog.assign("assistant", f"Original file:\n\n---\n{file_contents}\n---\n")
    return _clean(dialog.conversation)


def _build_conversation(card: MorphCard, root: str) -> List[dict]:
    """Assemble the self-contained conversation for one card.

    The three intents mirror ``flows/morph.py`` message-for-message; see the
    module docstring. The returned messages carry only ``role``/``content``
    (no ``time``), so they are ready to serialize.

    The closing user message carries the card's instruction plus -- for a card
    that writes a SET of files -- the output directive that tells the executor
    how to return them (:func:`_output_directive`, empty for one target).
    """
    directive = _output_directive(card)

    if card.intent == "generate":
        messages = _context_messages(card, root)
        messages.append({"role": "user", "content": card.instruction + directive})
        return messages

    if card.intent == "patch":
        messages = _original_file_messages(card, root)
        # The targets have just gone out as "Original file"; skip_target keeps
        # the context slice from shipping them again (see _context_messages).
        messages.extend(_context_messages(card, root, skip_target=True))
        messages.append({"role": "user", "content": card.instruction + directive})
        return messages

    if card.intent == "todo":
        messages = _original_file_messages(card, root)
        messages.append({"role": "user", "content": TODO_INSTRUCTION + directive})
        return messages

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
