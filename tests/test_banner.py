"""
Tests for the welcome banner (:mod:`flows.banner`).

The banner is art, so there is nothing here about how it LOOKS. What these
tests hold is the pair of properties that break silently and are invisible in a
diff: every rendered line is the same width on screen, and colour is a decision
the environment is allowed to make. The width one matters because the octopus
and the copy beside it are padded BEFORE the escape sequences go on -- get that
order wrong and the right-hand border walks off the frame on a coloured
terminal while the plain render still looks perfect.

``flows.banner`` is imported directly: it pulls in nothing from the console bot,
which is the reason the art was moved out of ``flows/morph.py`` in the first
place.
"""

import os
import re
import unittest

from flows.banner import ART, RAMP, TEXT, render_banner, supports_colour


# Matches an SGR sequence: what a plain render must contain none of, and what a
# coloured render must reduce to the plain one once removed.
ESCAPE = re.compile(r"\033\[[0-9;]*m")

# Environment variables the banner reads. Every test puts them back.
COLOUR_VARS = ("NO_COLOR", "FORCE_COLOR", "COLORTERM", "TERM")


class EnvironmentCase(unittest.TestCase):
    """Base class that restores the colour environment after every test."""

    def setUp(self):
        self._saved = {name: os.environ.get(name) for name in COLOUR_VARS}
        for name in COLOUR_VARS:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class ArtTest(unittest.TestCase):

    def test_every_art_line_is_the_same_width(self):
        widths = {len(line) for line in ART}
        self.assertEqual(1, len(widths), "art lines differ in width: {}".format(sorted(widths)))

    def test_the_ramp_covers_the_art(self):
        # One palette step per line; a short ramp would silently flat-fill the
        # tail of the octopus with the tentacle colour.
        self.assertEqual(len(ART), len(RAMP))

    def test_the_copy_is_no_taller_than_the_art(self):
        # The frame is as tall as the taller column. Copy that outgrows the
        # octopus is allowed by the renderer but is a design decision, not an
        # accident, so it should fail here first.
        self.assertLessEqual(len(TEXT), len(ART))


class PlainRenderTest(EnvironmentCase):

    def test_no_escapes_without_a_terminal(self):
        # No FORCE_COLOR and a stream that is not a tty: the pipe case.
        banner = render_banner(version="9.9.9", stream=_NotATerminal())
        self.assertNotIn("\033", banner)

    def test_no_color_beats_force_color(self):
        os.environ["NO_COLOR"] = ""
        os.environ["FORCE_COLOR"] = "1"
        self.assertFalse(supports_colour(_NotATerminal()))
        self.assertNotIn("\033", render_banner(version="9.9.9"))

    def test_dumb_terminal_gets_no_colour(self):
        os.environ["TERM"] = "dumb"
        self.assertFalse(supports_colour(_Terminal()))

    def test_every_line_is_the_same_width(self):
        lines = render_banner(version="9.9.9", stream=_NotATerminal()).split("\n")
        widths = {len(line) for line in lines}
        self.assertEqual(1, len(widths), "frame is ragged: {}".format(sorted(widths)))

    def test_the_version_reaches_the_copy(self):
        banner = render_banner(version="9.9.9", stream=_NotATerminal())
        self.assertIn("9.9.9", banner)

    def test_a_missing_version_still_renders(self):
        # Running from a checkout that was never pip-installed.
        banner = render_banner(version=None, stream=_NotATerminal())
        self.assertIn("GPT MORPH", banner)


class ColouredRenderTest(EnvironmentCase):

    def test_colour_changes_nothing_but_colour(self):
        plain = render_banner(version="9.9.9", stream=_NotATerminal())
        os.environ["FORCE_COLOR"] = "1"
        os.environ["COLORTERM"] = "truecolor"
        coloured = render_banner(version="9.9.9")
        self.assertIn("\033[38;2;", coloured)
        self.assertEqual(plain, ESCAPE.sub("", coloured))

    def test_256_colour_terminals_get_indexed_escapes(self):
        os.environ["FORCE_COLOR"] = "1"
        os.environ["TERM"] = "xterm-256color"
        coloured = render_banner(version="9.9.9")
        self.assertIn("\033[38;5;", coloured)
        self.assertNotIn("\033[38;2;", coloured)

    def test_a_plain_terminal_falls_back_to_magenta(self):
        os.environ["FORCE_COLOR"] = "1"
        os.environ["TERM"] = "vt100"
        coloured = render_banner(version="9.9.9")
        self.assertNotIn("\033[38;", coloured)
        self.assertIn("\033[1;35m", coloured)


class _NotATerminal:
    """The pipe: what ``python -m mrph > file`` hands us."""

    @staticmethod
    def isatty():
        return False


class _Terminal:

    @staticmethod
    def isatty():
        return True


if __name__ == "__main__":
    unittest.main()
