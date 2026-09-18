"""
Tests for ``cards/cli_backend.py``: the headless CLI's processor resolver.

Nothing here touches the real environment. Every resolution test patches
``ProcessorRegistry.from_env`` at the name ``cards.cli_backend`` actually
looks the class up by -- ``cards.cli_backend.ProcessorRegistry``, the binding
its import created -- and answers with a fake registry exposing exactly the
surface the module consumes: ``ids``, ``default_id()`` and ``batch(id)``. No
``MRPH_PROCESSOR_*`` variable is read, no provider SDK is imported, no
network is touched.

The one test with no patch is the import-hygiene one: importing
``cards.cli_backend`` must never leave ``flows.morph`` in ``sys.modules``.
The headless path exists precisely so the console bot stays out of it; the
REPL's own resolver in ``flows/morph.py`` is unreachable from there by
design, and this test holds that line.
"""

import importlib
import sys
import unittest
from unittest import mock

import cards.cli_backend as cli_backend


class _FakeRegistry:
    """The slice of ``ProcessorRegistry`` that ``cards.cli_backend`` uses."""

    def __init__(self, ids):
        self.ids = list(ids)
        # Every id handed to batch(), in order: what the module asked for.
        self.batches_asked = []

    def default_id(self):
        return self.ids[0] if self.ids else None

    def batch(self, identifier):
        self.batches_asked.append(identifier)
        if identifier not in self.ids:
            raise ValueError(f"unknown processor {identifier!r}")
        return f"batch-backend-for-{identifier}"


def _from_env_returns(registry):
    """Patch ``from_env`` where the module looks it up to return ``registry``."""

    def fake_from_env():
        return registry

    return mock.patch.object(
        cli_backend.ProcessorRegistry, "from_env", staticmethod(fake_from_env))


def _from_env_raises(error):
    """Patch ``from_env`` where the module looks it up to raise ``error``."""

    def fake_from_env():
        raise error

    return mock.patch.object(
        cli_backend.ProcessorRegistry, "from_env", staticmethod(fake_from_env))


class ResolveBackendTests(unittest.TestCase):
    """``resolve_backend``: label in, (backend, resolved id) out -- or raise."""

    def test_none_label_resolves_to_the_default_id(self):
        """No label at all: the registry's default id decides."""
        registry = _FakeRegistry(["k80-a", "gpt4"])
        with _from_env_returns(registry):
            backend, label = cli_backend.resolve_backend()
        self.assertEqual(
            label, "k80-a",
            "a None label must resolve to the registry's default id")
        self.assertEqual(
            backend, "batch-backend-for-k80-a",
            "the backend returned must be the default processor's backend")
        self.assertEqual(
            registry.batches_asked, ["k80-a"],
            "batch() must be asked for the default id and nothing else")

    def test_explicit_label_resolves_to_itself(self):
        """A bare id goes to the registry verbatim."""
        registry = _FakeRegistry(["k80-a", "gpt4"])
        with _from_env_returns(registry):
            backend, label = cli_backend.resolve_backend("gpt4")
        self.assertEqual(
            label, "gpt4",
            "an explicit label must resolve to itself")
        self.assertEqual(
            backend, "batch-backend-for-gpt4",
            "the backend returned must be the named processor's backend")
        self.assertEqual(
            registry.batches_asked, ["gpt4"],
            "batch() must be asked for exactly the explicit id")

    def test_at_prefixed_label_resolves_like_the_bare_one(self):
        """The REPL's '@glm' habit must not change the resolution."""
        registry = _FakeRegistry(["k80-a", "gpt4"])
        with _from_env_returns(registry):
            backend, label = cli_backend.resolve_backend("@gpt4")
        self.assertEqual(
            label, "gpt4",
            "the leading @ must be stripped from the resolved label")
        self.assertEqual(
            backend, "batch-backend-for-gpt4",
            "@gpt4 must resolve to the same backend as the bare gpt4")
        self.assertEqual(
            registry.batches_asked, ["gpt4"],
            "batch() must be asked for the stripped id, never '@gpt4'")

    def test_unknown_label_raises_a_diagnosing_backend_error(self):
        """An unknown id is named, next to the configured alternatives."""
        registry = _FakeRegistry(["k80-a", "gpt4"])
        with _from_env_returns(registry):
            with self.assertRaises(
                    cli_backend.BackendError,
                    msg="an unknown processor label must raise BackendError"
            ) as caught:
                cli_backend.resolve_backend("nope")
        self.assertIn(
            "nope", str(caught.exception),
            "the error message must name the unknown id")
        self.assertIn(
            "k80-a", str(caught.exception),
            "the error message must list a configured id as the alternative")

    def test_at_prefixed_unknown_label_is_unknown_too(self):
        """'@nope' fails like 'nope', named without the @."""
        registry = _FakeRegistry(["k80-a"])
        with _from_env_returns(registry):
            with self.assertRaises(
                    cli_backend.BackendError,
                    msg="an unknown @-prefixed label must raise BackendError"
            ) as caught:
                cli_backend.resolve_backend("@nope")
        self.assertIn(
            "nope", str(caught.exception),
            "the error message must name the stripped id")

    def test_empty_registry_raises_backend_error(self):
        """Nothing configured: no label and an explicit label both refuse."""
        with _from_env_returns(_FakeRegistry([])):
            with self.assertRaises(
                    cli_backend.BackendError,
                    msg="an empty registry must raise, never return a default"
            ):
                cli_backend.resolve_backend()
            with self.assertRaises(
                    cli_backend.BackendError,
                    msg="an empty registry must refuse an explicit label too"
            ):
                cli_backend.resolve_backend("gpt4")

    def test_failing_from_env_is_reported_as_backend_error(self):
        """An unreadable environment surfaces as BackendError, never raw."""
        with _from_env_raises(RuntimeError("environment is unusable")):
            with self.assertRaises(
                    cli_backend.BackendError,
                    msg="a from_env failure must surface as BackendError, "
                        "not propagate raw"
            ) as caught:
                cli_backend.resolve_backend()
        self.assertIn(
            "environment is unusable", str(caught.exception),
            "the BackendError must carry the underlying diagnosis")


class BackendErrorContractTests(unittest.TestCase):
    """The type contract the CLI's exit-code table relies on."""

    def test_backend_error_is_a_value_error(self):
        """ValueError on purpose: exit code 4 needs no new wiring."""
        self.assertTrue(
            issubclass(cli_backend.BackendError, ValueError),
            "BackendError must subclass ValueError so the CLI's existing "
            "error table maps it to exit code 4")


class AvailableLabelsTests(unittest.TestCase):
    """``available_labels``: the alternatives, or [] -- never an exception."""

    def test_returns_the_configured_ids_in_order(self):
        """The ids come back in the registry's (priority) order."""
        with _from_env_returns(_FakeRegistry(["k80-a", "gpt4"])):
            labels = cli_backend.available_labels()
        self.assertEqual(
            labels, ["k80-a", "gpt4"],
            "available_labels must list the configured ids in registry order")

    def test_returns_empty_list_when_nothing_is_configured(self):
        """An empty registry is an empty list, not an error."""
        with _from_env_returns(_FakeRegistry([])):
            labels = cli_backend.available_labels()
        self.assertEqual(
            labels, [],
            "an empty registry must yield an empty list")

    def test_returns_empty_list_when_from_env_raises(self):
        """A broken environment is [] too: this helper must never raise."""
        with _from_env_raises(RuntimeError("no configuration")):
            labels = cli_backend.available_labels()
        self.assertEqual(
            labels, [],
            "a failing from_env must yield [], never an exception")


class ImportHygieneTests(unittest.TestCase):
    """The headless path must not drag the console bot in behind it."""

    def test_importing_cli_backend_does_not_load_flows_morph(self):
        """Re-importing the module leaves flows.morph out of sys.modules."""
        # Judge this module's own imports, not the process's history: drop any
        # flows.morph an earlier test may have loaded, re-execute
        # cards.cli_backend, and check nothing pulled the REPL bot back in.
        sys.modules.pop("flows.morph", None)
        importlib.reload(cli_backend)
        self.assertNotIn(
            "flows.morph", sys.modules,
            "importing cards.cli_backend must not leave flows.morph in "
            "sys.modules")


if __name__ == "__main__":
    unittest.main()
