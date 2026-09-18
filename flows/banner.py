"""The mrph welcome banner: an ASCII octopus, filled in purple.

WHY a module of its own, and not the raw string that used to sit inline in
``flows/morph.py``: the banner now has behaviour -- it picks a colour depth from
the environment, and it reads the version from the installed distribution
instead of carrying a copy that drifts on every release. That is logic, and
logic belongs next to a test, not in the middle of a state-machine builder.

The art is rendered ONCE, when the state machine is built, and printed by
``ConsoleBot.send_message``. Colour is therefore decided at start-up: a session
that pipes stdout somewhere gets the plain art for the whole session, which is
what a pipe wants.
"""

import os
import sys

try:
    import pkg_resources
except ImportError:  # pragma: no cover - setuptools is a hard dependency
    pkg_resources = None


__all__ = ["render_banner", "supports_colour"]


# The octopus, drawn in block elements so the mantle can carry shading rather
# than outline alone. Every line is padded to the same width by build time; the
# check lives in tests/test_banner.py, because a single short line here shifts
# the whole right-hand column of the frame.
ART = (
    "       ▄▄▄▄████████▄▄▄▄       ",
    "    ▄████████████████████▄    ",
    "    ▄█████  ██████  █████▄    ",
    "    ███████  ████  ███████    ",
    "   █████████  ██  █████████   ",
    "   ████  ████████████  ████   ",
    "   █████  ██████████  █████   ",
    "   ▀█████    ████    █████▀   ",
    "     ▀▀████████████████▀▀     ",
    "        ▜██▛▀▐██▌▀▜██▛        ",
    "     ▗▟▛▘ ▟▛ ▐██▌ ▜▙ ▝▜▙▖     ",
    "   ▗▞▘ ▝▚▖▐▛ ▐▌▐▌ ▜▌▗▞▘ ▝▚▖   ",
    "   ▞     ▚▌   ▐▌   ▐▞     ▚   ",
    "   ▟  ▗▄▞▘    ▝▘    ▝▚▄▖  ▙   ",
    "  ▝▚▄▞▘                ▝▚▄▞▘  ",
)

# One palette entry per art line: light at the mantle's crown, deepest at the
# tentacle tips. This vertical ramp IS the fill -- the art itself carries no
# colour markers, so re-drawing it never means re-deriving a mask.
RAMP = (0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5)

# Purple, from crown to tip. Three renderings of the same six steps: 24-bit,
# 256-colour, and the eight-colour fallback where every step collapses to one of
# two magentas.
TRUECOLOUR = (
    (221, 190, 255),
    (198, 149, 253),
    (173, 105, 247),
    (152, 68, 232),
    (129, 44, 204),
    (104, 32, 166),
)
XTERM_256 = (183, 177, 141, 134, 92, 54)
BASIC = ("95", "95", "95", "35", "35", "35")

# The frame and the copy that sits beside the octopus.
FRAME_COLOUR = 4
TEXT = (
    ("", 0),
    ("GPT MORPH", 0),
    ("THE GRANDPA OF CLAUDE CODE", 2),
    ("", 0),
    ("> HOW CAN I HELP YOU, KIDDO?", 1),
    ("", 0),
    ("Grandpa writes clean code.", 3),
    ("No frameworks. No fluff.", 3),
    ("", 0),
    ("Memory: 64K    Wisdom: ∞", 4),
    ("", 0),
    ('"WE DEBUGGED WITH', 5),
    ('PRINT STATEMENTS."', 5),
    ("", 0),
    ("> _", 1),
)

GUTTER = 4
PADDING = 2


def supports_colour(stream=None):
    """Decide whether to emit escape sequences at all.

    ``NO_COLOR`` (any value) wins over everything, then ``FORCE_COLOR``, then
    the plain question of whether we are talking to a terminal. A dumb terminal
    counts as no terminal.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    stream = stream if stream is not None else sys.stdout
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def _depth():
    """``"truecolour"``, ``"256"`` or ``"basic"`` for the current terminal."""
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return "truecolour"
    term = os.environ.get("TERM", "")
    if "256" in term or "direct" in term:
        return "256"
    return "basic"


def _paint(text, step, depth):
    """Wrap ``text`` in the escape sequence for palette ``step``."""
    if not text.strip():
        return text
    if depth == "truecolour":
        red, green, blue = TRUECOLOUR[step]
        opening = "\033[38;2;{};{};{}m".format(red, green, blue)
    elif depth == "256":
        opening = "\033[38;5;{}m".format(XTERM_256[step])
    else:
        opening = "\033[1;{}m".format(BASIC[step])
    return opening + text + "\033[0m"


def _version():
    if pkg_resources is None:
        return None
    try:
        return pkg_resources.get_distribution("mrph").version
    except Exception:
        # Running from a checkout that was never installed: the banner drops the
        # version rather than the whole greeting.
        return None


def _rows(version):
    """Pair every art line with its copy, both padded to a fixed width.

    Padding happens BEFORE colour: an escape sequence has no width on screen but
    plenty of length in Python, so anything that measures a painted string
    measures the wrong thing and the right-hand border walks off.
    """
    text = list(TEXT)
    if version:
        text[1] = ("GPT MORPH  v{}".format(version), 0)

    art_width = max(len(line) for line in ART)
    text_width = max(len(line) for line, _ in text)
    height = max(len(ART), len(text))

    rows = []
    for index in range(height):
        art = ART[index] if index < len(ART) else ""
        copy, step = text[index] if index < len(text) else ("", 0)
        rows.append((art.ljust(art_width), RAMP[index] if index < len(RAMP) else RAMP[-1],
                     copy.ljust(text_width), step))
    return rows, art_width + GUTTER + text_width


def render_banner(version=None, stream=None):
    """Return the framed, coloured octopus as one printable string."""
    rows, body_width = _rows(version if version is not None else _version())
    colour = supports_colour(stream)
    depth = _depth()

    def paint(text, step):
        return _paint(text, step, depth) if colour else text

    inner = body_width + PADDING * 2
    pad = " " * PADDING
    lines = [paint("╭" + "─" * inner + "╮", FRAME_COLOUR)]
    for art, art_step, copy, text_step in rows:
        body = paint(art, art_step) + " " * GUTTER + paint(copy, text_step)
        lines.append(paint("│", FRAME_COLOUR) + pad + body + pad + paint("│", FRAME_COLOUR))
    lines.append(paint("╰" + "─" * inner + "╯", FRAME_COLOUR))
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_banner())
