"""
The morph card: the JCL of an LLM batch job.

A *morph card* is the unit of specification the orchestrator compiles and the
executor consumes -- see ``documentation/batch-orchestrator.md`` ("The morph
card: JCL for LLM jobs"). Each card names a single output ``target``, the
``instruction`` that produces it, the ``context_slice`` of files its executor
must see, and the machine-checkable ``acceptance`` criterion that verifies it.

This module is the format frozen in code, deliberately ahead of any consumer
(Phase 0 of ``documentation/DEVELOPMENT_PLAN.md``). It is pure, stdlib-only
data + validation: no I/O, no registry lookup, no imports from ``processors/``
or ``flows/``. ``model`` is stored verbatim as a hint here; resolving it
against the processor registry happens in a later phase. ``MorphCard.from_dict``
accepts both the nested JSON shape documented in ``batch-orchestrator.md``
(meta fields under a ``"meta"`` key) and an equivalent flat shape.

Every validation failure raises :class:`CardError`, whose message names the
offending ``custom_id`` (when known) and field, so a broken deck reports *what*
is wrong rather than surfacing a bare ``KeyError``/``TypeError``.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


# custom_id becomes part of batch custom_ids and output file names
# (``<name>.<custom_id>.<ext>``), so it is restricted to filesystem- and
# provider-safe characters.
CUSTOM_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

VALID_INTENTS = ("generate", "patch", "todo")

# The meta fields recognised in both the nested and the flat form. Kept here so
# ``from_dict`` can reject anything else by name instead of silently ignoring
# typos in a hand-written card.
_META_FIELDS = (
    "intent",
    "target",
    "context_slice",
    "acceptance",
    "model",
    "variants",
    "generation",
    "depends_on",
)


class CardError(ValueError):
    """A single morph card is malformed.

    Distinct from :class:`cards.deck.DeckError`, which concerns relationships
    *between* cards. Messages are prefixed with the card's ``custom_id`` when it
    is known so failures point at the offending card.
    """


def _err(custom_id: Optional[str], message: str) -> "CardError":
    """Build a :class:`CardError` that names the card when we know its id."""
    if custom_id:
        return CardError(f"card {custom_id!r}: {message}")
    return CardError(message)


@dataclass
class MorphCard:
    """One self-contained LLM batch job.

    See the module docstring and ``documentation/batch-orchestrator.md`` for the
    meaning of each field. Instances are normally built via :meth:`from_dict`;
    constructing one directly still goes through :meth:`validate`.
    """

    custom_id: str
    intent: str
    target: str
    instruction: str = ""
    context_slice: List[str] = field(default_factory=list)
    acceptance: Optional[str] = None
    model: Optional[str] = None
    variants: int = 1
    generation: int = 0
    depends_on: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    # -- validation -----------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`CardError` if any field is malformed."""
        cid = self.custom_id

        # custom_id: required, non-empty, safe character set.
        if not isinstance(self.custom_id, str) or not self.custom_id:
            raise _err(None, "custom_id is required and must be a non-empty string")
        if not CUSTOM_ID_RE.match(self.custom_id):
            raise _err(
                cid,
                "custom_id must match ^[A-Za-z0-9._-]+$ "
                "(no spaces or path separators)",
            )

        # intent: one of the known flows.
        if self.intent not in VALID_INTENTS:
            raise _err(
                cid,
                "intent must be one of "
                f"{{{', '.join(VALID_INTENTS)}}}, got {self.intent!r}",
            )

        # target: required, non-empty.
        if not isinstance(self.target, str) or not self.target:
            raise _err(cid, "target is required and must be a non-empty string")

        # instruction: required for generate/patch; may be empty for todo
        # (the todo instruction is fixed elsewhere).
        if not isinstance(self.instruction, str):
            raise _err(cid, "instruction must be a string")
        if self.intent != "todo" and not self.instruction:
            raise _err(
                cid,
                f"instruction is required and must be non-empty for intent "
                f"{self.intent!r}",
            )

        # context_slice: list of strings (empty => whole project).
        _require_str_list(cid, "context_slice", self.context_slice)

        # acceptance: optional string.
        if self.acceptance is not None and not isinstance(self.acceptance, str):
            raise _err(cid, "acceptance must be a string or null")

        # model: optional string hint (not resolved here).
        if self.model is not None and not isinstance(self.model, str):
            raise _err(cid, "model must be a string or null")

        # variants: integer >= 1.
        if isinstance(self.variants, bool) or not isinstance(self.variants, int):
            raise _err(cid, "variants must be an integer")
        if self.variants < 1:
            raise _err(cid, f"variants must be >= 1, got {self.variants}")

        # generation: non-negative integer.
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise _err(cid, "generation must be an integer")
        if self.generation < 0:
            raise _err(cid, f"generation must be >= 0, got {self.generation}")

        # depends_on: list of custom_ids (cross-card resolution is the deck's job).
        _require_str_list(cid, "depends_on", self.depends_on)

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "MorphCard":
        """Build a :class:`MorphCard` from its JSON dict shape.

        Accepts both the nested form documented in ``batch-orchestrator.md``::

            {"custom_id": ..., "meta": {"intent": ..., "target": ...},
             "instruction": ...}

        and a flat form where the meta fields sit at the top level::

            {"custom_id": ..., "intent": ..., "target": ..., "instruction": ...}

        Unknown fields raise :class:`CardError` naming the field.
        """
        if not isinstance(data, dict):
            raise CardError(f"card must be a JSON object, got {type(data).__name__}")

        custom_id = data.get("custom_id")
        # Validate the id shape early so every later message can name the card.
        if not isinstance(custom_id, str) or not custom_id:
            raise _err(None, "custom_id is required and must be a non-empty string")

        instruction = data.get("instruction", "")

        # Merge meta from either the nested "meta" object or the top level, but
        # never both -- a card that mixes the two forms is almost certainly a
        # mistake and we would not know which value wins.
        has_nested = "meta" in data
        meta = data.get("meta", {})
        if has_nested and not isinstance(meta, dict):
            raise _err(custom_id, "'meta' must be a JSON object")

        top_level_meta = [key for key in _META_FIELDS if key in data]
        if has_nested and top_level_meta:
            raise _err(
                custom_id,
                "meta fields must live under 'meta' or at the top level, "
                f"not both (found {sorted(top_level_meta)} alongside 'meta')",
            )

        source = meta if has_nested else data

        # Reject unknown keys by name. The card-level keys are custom_id,
        # instruction and meta; everything else must be a known meta field.
        allowed = set(_META_FIELDS) | {"custom_id", "instruction", "meta"}
        for key in data:
            if key not in allowed:
                raise _err(custom_id, f"unknown field {key!r}")
        if has_nested:
            for key in meta:
                if key not in _META_FIELDS:
                    raise _err(custom_id, f"unknown field 'meta.{key}'")

        kwargs = {
            "custom_id": custom_id,
            "intent": source.get("intent"),
            "target": source.get("target"),
            "instruction": instruction,
        }
        # Only pass optional fields when present so dataclass defaults apply.
        for key in (
            "context_slice",
            "acceptance",
            "model",
            "variants",
            "generation",
            "depends_on",
        ):
            if key in source:
                kwargs[key] = source[key]

        if kwargs["intent"] is None:
            raise _err(custom_id, "intent is required")
        if kwargs["target"] is None:
            raise _err(custom_id, "target is required")

        return cls(**kwargs)


def _require_str_list(custom_id: Optional[str], name: str, value) -> None:
    """Validate that ``value`` is a list of strings, else raise CardError."""
    if not isinstance(value, list):
        raise _err(custom_id, f"{name} must be a list of strings")
    for item in value:
        if not isinstance(item, str):
            raise _err(custom_id, f"{name} must contain only strings")
