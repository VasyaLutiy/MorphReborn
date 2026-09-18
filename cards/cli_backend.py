"""
Turning a ``--processor`` label into a batch backend, for headless ``mrph``.

Where the REPL resolves its processor inside ``flows/morph.py``, the headless
CLI cannot follow: that module *is* the console bot, and importing it from
here would drag the bot and its side effects in behind a mere flag parse. So
this module re-derives the same answer from ``processors.registry`` alone and
never imports from ``flows``.

The shape is the one ``morph_mcp/tools.py`` already worked out for the MCP
layer (``_backend_from`` / ``_build_backend``): the registry is built
*inside* the call, never at import time, so importing this module -- and even
printing the CLI's help -- requires no processor configuration at all. The
one deliberate difference is the error style: the MCP tools answer a missing
processor with a reason string, while this module RAISES, because the CLI
turns exceptions into its error document. :class:`BackendError` is a
``ValueError`` on purpose: the CLI's error table already maps ``ValueError``
to exit code 4 ("usage: unknown processor"), so a bad ``--processor`` needs
no new wiring.

Three public names, the whole contract (see ``__all__``):

* :class:`BackendError` -- nothing is configured, or the named processor
  cannot be turned into a batch backend.
* :func:`resolve_backend` -- ``(backend, resolved_label)`` for a
  ``--processor`` label. ``None`` means the registry's default; a leading
  ``@`` is accepted and stripped, because an operator types ``@glm`` out of
  REPL habit. The label returned is the stripped id -- the one the caller
  records as the run's ``backend_label``.
* :func:`available_labels` -- the configured ids, for error messages that
  list the alternatives. Never raises.

Deliberately absent: ``@all`` and any pool fan-out. One model per deck is a
rule of this system -- a batch leaves under a single model (see
``processors/batch.py``) -- so a mixed deck could not leave as one batch, and
a label that names no single processor (``@all`` included) fails as the
unknown id it is, with the configured list in the message.
"""

from typing import List, Optional, Tuple

from processors.batch import BatchBackend
from processors.registry import ProcessorRegistry

__all__ = ["BackendError", "resolve_backend", "available_labels"]

# The diagnosis for the one failure that names no id: the registry came back
# with nothing in it, so there is nothing to list but the fix.
_NO_PROCESSOR_MESSAGE = (
    "no processor is configured; define one with MRPH_PROCESSOR_<ID>_TYPE "
    "(plus MODEL / ENDPOINT_URI / API_KEY), or one of the legacy "
    "LLAMA_CPP_* / OLLAMA_* / OPENAI_* variables")


class BackendError(ValueError):
    """No processor is configured, or the named one cannot be built.

    A ``ValueError`` on purpose, not a new species for the CLI to learn: the
    CLI's error table already maps ``ValueError`` to exit code 4 ("usage:
    unknown processor"), so this type rides wiring that exists. The message
    is a diagnosis rather than a refusal -- it names the id that failed and
    lists the configured ones -- because the operator reading it is about to
    retype the flag.
    """


def _strip_at(label: str) -> str:
    """Drop the leading ``@`` the REPL habit puts before a processor name."""
    if label.startswith("@"):
        return label[1:]
    return label


def resolve_backend(label: Optional[str] = None) -> Tuple[BatchBackend, str]:
    """Turn a ``--processor`` label into ``(backend, resolved_label)``.

    The registry is built inside the call
    (``processors.registry.ProcessorRegistry.from_env()``), never at import
    time, so importing this module requires no configuration; the cost is
    paid only when a deck is actually submitted.

    ``label`` may be ``None`` -- the registry's ``default_id()`` is used then
    -- or an id, optionally with the leading ``@`` the REPL habit adds:
    ``@glm`` and ``glm`` resolve identically, and the label returned is the
    stripped id, the one the caller records as the run's ``backend_label``.
    An empty label, or a bare ``@``, names nothing and falls back to the
    default too.

    Every failure raises :class:`BackendError`: an environment that cannot be
    read, a registry with no processors at all, an id that is not among
    ``registry.ids`` -- the message names it and lists the configured ones, a
    diagnosis and not a refusal -- and a configured processor whose batch
    backend cannot be built. There is no ``@all``: one model per deck is a
    rule of this system, so ``@all`` fails as the unknown id it is.
    """
    if label is not None and not isinstance(label, str):
        raise BackendError(
            f"the processor label must be a string, got "
            f"{type(label).__name__}")

    try:
        registry = ProcessorRegistry.from_env()
    except Exception as exc:
        raise BackendError(
            "the processor registry could not be built from the environment "
            f"({exc})") from exc
    if registry is None:
        # Shared paranoia with morph_mcp/tools.py: from_env builds a registry
        # or raises, it does not return None -- but if one ever did, it must
        # read as "nothing configured" here, not crash two lines down.
        raise BackendError(_NO_PROCESSOR_MESSAGE)

    if not label:
        # None or "": no id named at all -- the registry's default decides.
        resolved = registry.default_id()
    else:
        stripped = _strip_at(label)
        # A bare "@" strips to nothing and names nothing: default again.
        resolved = stripped if stripped else registry.default_id()

    configured = list(registry.ids)
    if resolved is None:
        raise BackendError(_NO_PROCESSOR_MESSAGE)
    if resolved not in configured:
        listed = ", ".join(configured) if configured else "(none)"
        raise BackendError(
            f"unknown processor {resolved!r}; configured processors: {listed}")

    try:
        backend = registry.batch(resolved)
    except Exception as exc:
        raise BackendError(
            f"processor {resolved!r} could not be turned into a batch "
            f"backend ({exc})") from exc
    return backend, resolved


def available_labels() -> List[str]:
    """The configured processor ids, in registry (priority) order.

    ``[]`` when nothing is configured, when the registry cannot be built from
    the environment, or when anything else goes wrong on the way to the id
    list: this function exists so an error message can list the
    alternatives, and a listing helper that raised would defeat that purpose.
    Like :func:`resolve_backend` it builds the registry inside the call, so
    importing this module still requires no configuration.
    """
    try:
        registry = ProcessorRegistry.from_env()
        return list(registry.ids)
    except Exception:
        return []
