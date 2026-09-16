"""
Named processor registry for GPT Morph.

Historically the bot read one fixed set of environment variables per backend
(``LLAMA_CPP_*``, ``OLLAMA_*``, ``OPENAI_*``) and picked a single processor by a
hard-coded priority. This module generalises that into a registry of *named*
processor instances, so many instances of each backend can be configured at once
and addressed by identifier. This is what turns GPT Morph into a multi-agent
system: several local nodes on the network can morph the same file in parallel.

Two configuration schemes are supported:

1. Namespaced scheme (preferred, arbitrary number of instances)::

       MRPH_PROCESSORS=k80-a,k80-b,gpt4          # optional, fixes id order
       MRPH_PROCESSOR_k80-a_TYPE=llama_cpp
       MRPH_PROCESSOR_k80-a_ENDPOINT_URI=http://192.168.0.14:8080/v1
       MRPH_PROCESSOR_k80-a_MODEL=k80-model

       MRPH_PROCESSOR_k80-b_TYPE=llama_cpp
       MRPH_PROCESSOR_k80-b_ENDPOINT_URI=http://192.168.0.15:8080/v1
       MRPH_PROCESSOR_k80-b_MODEL=k80-model

       MRPH_PROCESSOR_gpt4_TYPE=openai
       MRPH_PROCESSOR_gpt4_API_KEY=sk-...
       MRPH_PROCESSOR_gpt4_MODEL=gpt-4o

   Recognised per-instance keys (all prefixed ``MRPH_PROCESSOR_<ID>_``):
   ``TYPE`` (``llama_cpp`` | ``ollama`` | ``openai`` | ``anthropic`` |
   ``openrouter``), ``ENDPOINT_URI``, ``MODEL``, ``API_KEY``, ``BASE_URL``.

2. Legacy scheme (kept for backward compatibility). If the namespaced scheme
   defines nothing, the classic single-instance variables are mapped to the
   default identifiers ``llama_cpp``, ``ollama`` and ``openai`` in that priority
   order, exactly reproducing the previous selection behaviour.
"""

import os
from typing import Dict, List, Optional

from processors.batch import (
    BatchBackend,
    OpenAIBatchBackend,
    AnthropicBatchBackend,
    OpenRouterBatchBackend,
    LocalBatchBackend,
)

NAMESPACE_PREFIX = "MRPH_PROCESSOR_"
KNOWN_TYPES = ("llama_cpp", "ollama", "openai", "anthropic", "openrouter")

# OpenRouter's *synchronous* API is OpenAI-compatible, so one openrouter
# processor id serves both worlds: interactive /generate and /patch go through
# an OpenAIProcessor pointed here, while decks go through the batch API on the
# separate /api/beta path (``processors.batch.OPENROUTER_BATCH_BASE_URL``).
OPENROUTER_SYNC_BASE_URL = "https://openrouter.ai/api/v1"


class ProcessorConfig:
    """Backend-agnostic description of a single named processor instance."""

    def __init__(self, identifier: str, kind: str, params: Dict[str, str]):
        self.identifier = identifier
        self.kind = kind
        self.params = params

    def describe(self) -> str:
        if self.kind == "llama_cpp":
            return f"[{self.identifier}] llama.cpp @ {self.params.get('endpoint_uri')} " \
                   f"(model \"{self.params.get('model')}\")"
        if self.kind == "ollama":
            return f"[{self.identifier}] Ollama @ {self.params.get('endpoint_uri')} " \
                   f"(model \"{self.params.get('model')}\")"
        if self.kind == "openai":
            target = self.params.get("base_url") or "OpenAI / Azure OpenAI"
            return f"[{self.identifier}] OpenAI @ {target} " \
                   f"(model \"{self.params.get('model') or os.environ.get('OPENAI_MODEL_NAME')}\")"
        if self.kind == "anthropic":
            model = self.params.get("model") or os.environ.get("ANTHROPIC_MODEL_NAME") \
                or "claude-sonnet-5"
            return f"[{self.identifier}] Anthropic (model \"{model}\")"
        if self.kind == "openrouter":
            return f"[{self.identifier}] OpenRouter (model \"{self.params.get('model')}\")"
        return f"[{self.identifier}] {self.kind}"


class ProcessorRegistry:
    """An ordered collection of named :class:`ProcessorConfig` instances."""

    def __init__(self, configs: Dict[str, ProcessorConfig]):
        # ``dict`` preserves insertion order, which we use as the selection priority.
        self._configs = configs

    def __len__(self) -> int:
        return len(self._configs)

    @property
    def ids(self) -> List[str]:
        return list(self._configs.keys())

    def get(self, identifier: str) -> Optional[ProcessorConfig]:
        return self._configs.get(identifier)

    def default_id(self) -> Optional[str]:
        return next(iter(self._configs), None)

    def describe_all(self) -> List[str]:
        return [config.describe() for config in self._configs.values()]

    def create(self, identifier: str):
        """Instantiate the concrete processor object for ``identifier``.

        The concrete processor classes are imported lazily here, not at module
        import time, so each backend's SDK (``openai``, ``ollama``,
        ``anthropic``) is required only when an instance of that backend is
        actually created -- constructing the registry, listing/describing
        configs and choosing a batch backend all work with no SDK installed.
        """
        config = self._configs[identifier]
        if config.kind == "llama_cpp":
            from processors.llama_cpp_processor import LlamaCppProcessor
            return LlamaCppProcessor(config.params["endpoint_uri"], config.params["model"])
        if config.kind == "ollama":
            from processors.ollama_processor import OllamaProcessor
            return OllamaProcessor(config.params["endpoint_uri"], config.params["model"])
        if config.kind == "openai":
            from processors.openai_processor import OpenAIProcessor
            return OpenAIProcessor(
                model=config.params.get("model"),
                api_key=config.params.get("api_key"),
                base_url=config.params.get("base_url"),
            )
        if config.kind == "anthropic":
            from processors.anthropic_processor import AnthropicProcessor
            return AnthropicProcessor(
                model=config.params.get("model"),
                api_key=config.params.get("api_key"),
            )
        if config.kind == "openrouter":
            # The synchronous side of OpenRouter is OpenAI-compatible, so the
            # existing processor drives it verbatim -- only the base URL differs.
            # ``BASE_URL`` overrides the *batch* path only (see ``batch()``), so
            # the interactive path always uses the documented /api/v1 endpoint.
            from processors.openai_processor import OpenAIProcessor
            return OpenAIProcessor(
                model=config.params.get("model"),
                api_key=config.params.get("api_key"),
                base_url=OPENROUTER_SYNC_BASE_URL,
            )
        raise ValueError(f"Unknown processor type \"{config.kind}\" for id \"{identifier}\".")

    def run(self, identifier: str, messages: List[Dict[str, str]]) -> str:
        """Run one processor synchronously and return its text response."""
        stream = self.create(identifier).process(messages)
        if stream:
            return stream[0]["value"]
        return ''

    def batch(self, identifier: str) -> "BatchBackend":
        """The batch backend for ``identifier`` (submit / status / collect).

        Cloud providers expose real batch endpoints; a local llama.cpp/Ollama
        node has none, so it is wrapped in :class:`~processors.batch.LocalBatchBackend`
        which drains the deck through the pool. See ``documentation/batch-orchestrator.md``.
        """
        config = self._configs[identifier]
        if config.kind == "openai":
            return OpenAIBatchBackend(
                model=config.params.get("model") or os.environ.get("OPENAI_MODEL_NAME"),
                api_key=config.params.get("api_key"),
                base_url=config.params.get("base_url"),
            )
        if config.kind == "anthropic":
            return AnthropicBatchBackend(
                model=config.params.get("model") or os.environ.get("ANTHROPIC_MODEL_NAME")
                or "claude-sonnet-5",
                api_key=config.params.get("api_key"),
            )
        if config.kind == "openrouter":
            return OpenRouterBatchBackend(
                model=config.params.get("model"),
                api_key=config.params.get("api_key"),
                base_url=config.params.get("base_url"),
            )
        if config.kind in ("llama_cpp", "ollama"):
            return LocalBatchBackend(self, [identifier])
        raise ValueError(f"Unknown processor type \"{config.kind}\" for id \"{identifier}\".")

    def batch_pool(self, identifiers: List[str]) -> LocalBatchBackend:
        """A single local batch endpoint spanning several local nodes.

        Drains one deck across all of ``identifiers`` through one shared
        scheduler pool -- the K80 farm running an overnight deck as one machine.
        """
        return LocalBatchBackend(self, list(identifiers))

    # -- construction ------------------------------------------------------

    @classmethod
    def from_env(cls) -> "ProcessorRegistry":
        configs: Dict[str, ProcessorConfig] = {}
        cls._add_namespaced(configs)
        cls._add_legacy(configs)
        return cls(configs)

    @staticmethod
    def _discover_namespaced_ids() -> List[str]:
        """Ids are ordered by ``MRPH_PROCESSORS`` when present, else discovered."""
        listed = os.environ.get("MRPH_PROCESSORS")
        if listed:
            return [token.strip() for token in listed.split(",") if token.strip()]

        discovered = []
        suffix = "_TYPE"
        for key in os.environ:
            if key.startswith(NAMESPACE_PREFIX) and key.endswith(suffix):
                identifier = key[len(NAMESPACE_PREFIX):-len(suffix)]
                if identifier and identifier not in discovered:
                    discovered.append(identifier)
        return discovered

    @classmethod
    def _add_namespaced(cls, configs: Dict[str, ProcessorConfig]) -> None:
        for identifier in cls._discover_namespaced_ids():
            base = f"{NAMESPACE_PREFIX}{identifier}_"
            kind = os.environ.get(f"{base}TYPE")
            if not kind:
                continue
            kind = kind.strip().lower()
            if kind not in KNOWN_TYPES:
                continue
            params = {
                "endpoint_uri": os.environ.get(f"{base}ENDPOINT_URI"),
                "model": os.environ.get(f"{base}MODEL"),
                "api_key": os.environ.get(f"{base}API_KEY"),
                "base_url": os.environ.get(f"{base}BASE_URL"),
            }
            if not cls._is_valid(kind, params):
                continue
            configs[identifier] = ProcessorConfig(identifier, kind, params)

    @staticmethod
    def _is_valid(kind: str, params: Dict[str, str]) -> bool:
        if kind in ("llama_cpp", "ollama"):
            return bool(params.get("endpoint_uri") and params.get("model"))
        if kind == "openai":
            # An OpenAI instance is usable with either an explicit key or the
            # process-wide OpenAI/Azure credentials.
            return bool(
                params.get("api_key")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("AZURE_OPENAI_API_KEY")
            )
        if kind == "anthropic":
            # Usable with an explicit key or the process-wide Anthropic key.
            return bool(
                params.get("api_key")
                or os.environ.get("ANTHROPIC_API_KEY")
            )
        if kind == "openrouter":
            # There is no process-wide OpenRouter fallback and no default model:
            # the model slug chooses both the vendor and the price tier (a
            # ":batch" suffix is the half-price variant), so it must be explicit.
            return bool(params.get("api_key") and params.get("model"))
        return False

    @staticmethod
    def _add_legacy(configs: Dict[str, ProcessorConfig]) -> None:
        """Map the classic single-instance variables onto default ids.

        Ordering (llama_cpp -> ollama -> openai) preserves the historical
        selection priority. Ids already provided by the namespaced scheme win.
        """
        llama_uri = os.environ.get("LLAMA_CPP_ENDPOINT_URI")
        llama_model = os.environ.get("LLAMA_CPP_MODEL")
        if llama_uri and llama_model and "llama_cpp" not in configs:
            configs["llama_cpp"] = ProcessorConfig(
                "llama_cpp", "llama_cpp",
                {"endpoint_uri": llama_uri, "model": llama_model})

        ollama_uri = os.environ.get("OLLAMA_ENDPOINT_URI")
        ollama_model = os.environ.get("OLLAMA_MODEL")
        if ollama_uri and ollama_model and "ollama" not in configs:
            configs["ollama"] = ProcessorConfig(
                "ollama", "ollama",
                {"endpoint_uri": ollama_uri, "model": ollama_model})

        openai_api_key = os.environ.get("OPENAI_API_KEY")
        azure_openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if (openai_api_key or azure_openai_api_key) and "openai" not in configs:
            configs["openai"] = ProcessorConfig(
                "openai", "openai",
                {"model": os.environ.get("OPENAI_MODEL_NAME"),
                 "api_key": openai_api_key})

        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_api_key and "anthropic" not in configs:
            configs["anthropic"] = ProcessorConfig(
                "anthropic", "anthropic",
                {"model": os.environ.get("ANTHROPIC_MODEL_NAME") or "claude-sonnet-5",
                 "api_key": anthropic_api_key})
