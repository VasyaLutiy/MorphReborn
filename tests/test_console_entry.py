"""
Tests for the installed command (:mod:`mrph_console`, ``setup.py``).

Two halves, and the second is the one that matters. The first checks the
dispatch: arguments mean the headless surface, no arguments mean the console
bot. The second INSTALLS the project into a throwaway virtual environment and
asks what actually shipped -- because everything else in this suite imports from
the tree, which is exactly why a broken installation stayed invisible for so
long: ``py_modules`` named 18 modules out of 36, and the first headless call of
a real installation died on ``ModuleNotFoundError: cards.cli`` while every test
here stayed green.

The install runs with ``--no-deps``: the third-party wheels are irrelevant to
the question "did OUR modules ship", and fetching them would turn a two-second
test into a minute.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import mrph_console


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# What an installation must be able to import. Deliberately the modules a
# headless run touches first, plus the MCP server -- the two surfaces the old
# py_modules list dropped whole. ``flows.morph`` is absent on purpose: it needs
# the conversation-flow dependency, which --no-deps does not install.
SHIPPED = ("mrph_console", "settings", "cards.cli", "cards.cli_run",
           "cards.cli_cycle", "cards.store", "cards.budget", "cards.notify",
           "processors.registry", "processors.batch", "morph_mcp.server")


class DispatchTest(unittest.TestCase):

    def test_arguments_go_to_the_headless_surface(self):
        with mock.patch("cards.cli.main", return_value=7) as cli_main, \
                mock.patch("settings.load_settings"):
            self.assertEqual(7, mrph_console.main(["deck", "--pretty"]))
        cli_main.assert_called_once_with(["deck", "--pretty"])

    def test_no_arguments_start_the_console_bot(self):
        # The console bot is handed in through sys.modules rather than patched
        # on the real module, because IMPORTING flows.morph here is not free:
        # it pulls the whole conversation-flow stack into a suite that
        # otherwise drives transitions without a bot, and doing so made a later
        # asynchronous wait test hang. The dispatch is what this test is about,
        # so the stub answers it without the import.
        bot = mock.Mock()
        stub = types.ModuleType("flows.morph")
        stub.MorphBot = mock.Mock(return_value=bot)
        with mock.patch("settings.load_settings"), \
                mock.patch.dict(sys.modules, {"flows.morph": stub}):
            self.assertEqual(0, mrph_console.main([]))
        stub.MorphBot.assert_called_once_with(None)
        bot.run.assert_called_once_with()

    def test_settings_are_loaded_before_anything_else(self):
        # A run that reaches the registry without .env loaded reports "no
        # processor configured" -- a diagnosis that sends the operator hunting
        # in the wrong place entirely.
        with mock.patch("settings.load_settings") as load, \
                mock.patch("cards.cli.main", return_value=0):
            mrph_console.main(["deck"])
        load.assert_called_once_with()


class InstallationTest(unittest.TestCase):
    """What ``pip install .`` actually puts into an environment."""

    @classmethod
    def setUpClass(cls):
        cls.home = tempfile.mkdtemp(prefix="mrph-install-")
        cls.env = os.path.join(cls.home, "venv")
        subprocess.run([sys.executable, "-m", "venv", cls.env],
                       check=True, capture_output=True)
        cls.python = os.path.join(cls.env, "bin", "python")
        cls.command = os.path.join(cls.env, "bin", "mrph")

        # Install from a clean export of HEAD, not from the checkout itself.
        # Two reasons, and the first is what makes this test usable at all: pip
        # copies the whole source directory into a temporary build tree, and a
        # developer's checkout carries a venv/ of a few hundred megabytes, which
        # turns a four-second test into minutes. The second is that an export is
        # exactly what a runner clones -- so this asks "does what is COMMITTED
        # install", which is the question that matters.
        cls.export = os.path.join(cls.home, "export")
        os.makedirs(cls.export)
        archive = subprocess.run(["git", "archive", "HEAD"], cwd=ROOT,
                                 capture_output=True)
        subprocess.run(["tar", "-x", "-C", cls.export], input=archive.stdout,
                       check=True, capture_output=True)

        cls.install = subprocess.run(
            [cls.python, "-m", "pip", "install", "--no-deps", "--quiet",
             cls.export],
            capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.home, ignore_errors=True)

    def test_the_install_succeeds(self):
        self.assertEqual(0, self.install.returncode,
                         self.install.stderr[-2000:])

    def test_the_command_is_installed(self):
        self.assertTrue(os.path.exists(self.command),
                        f"no mrph in {os.path.dirname(self.command)}: "
                        f"{os.listdir(os.path.dirname(self.command))}")
        self.assertTrue(os.access(self.command, os.X_OK))

    def test_the_command_is_a_launcher_and_not_a_copy(self):
        # The distinction this whole change exists for: a COPY of bin/mrph goes
        # stale the moment Morph morphs itself, a generated launcher imports the
        # installed package at run time and cannot.
        with open(self.command, "rb") as handle:
            installed = handle.read()
        with open(os.path.join(self.export, "bin", "mrph"), "rb") as handle:
            source = handle.read()
        self.assertNotEqual(source, installed)
        # A generated launcher resolves the entry point at run time; the marker
        # is the console_scripts group, not our module name, which never appears
        # in the script itself.
        self.assertIn(b"console_scripts", installed)

    def test_every_shipped_module_imports(self):
        # One subprocess, all imports: a missing module names itself in the
        # traceback, which is the diagnosis a regeneration needs.
        completed = subprocess.run(
            [self.python, "-c",
             "import " + ", ".join(SHIPPED) + "; print('ok')"],
            capture_output=True, text=True, cwd=self.home)
        self.assertEqual(0, completed.returncode,
                         completed.stderr[-2000:])

    def test_the_installed_command_runs_our_code(self):
        # Run it OUTSIDE the checkout: inside, an import would be satisfied by
        # the working tree and prove nothing about what shipped. With --no-deps
        # the run cannot get past the provider SDK -- and that is precisely the
        # line this test draws: whatever fails, it must not be OUR module. The
        # old packaging failed here with "No module named 'cards.cli'".
        completed = subprocess.run([self.command, "deck"],
                                   capture_output=True, text=True, cwd=self.home)
        ours = ("mrph_console", "settings", "cards", "processors", "flows",
                "morph_mcp")
        for name in ours:
            self.assertNotIn(f"No module named '{name}", completed.stderr,
                             completed.stderr[-2000:])
        # And it did reach our entry point rather than dying in the launcher.
        self.assertIn("mrph_console.py", completed.stderr,
                      completed.stderr[-2000:] or completed.stdout[:200])


if __name__ == "__main__":
    unittest.main()
