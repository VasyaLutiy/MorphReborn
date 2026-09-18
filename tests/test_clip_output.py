"""
Tests for :func:`cards.acceptance.clip_output` -- the clipper standing between
a failing acceptance run and the executor that has to fix it: its result is
the error context a regeneration is given, capped at
:data:`cards.acceptance._OUTPUT_TAIL_CAP` characters.

Pure string tests: no network, no subprocess, no sleeping. The contract pinned
here, clause by clause: text at or under the cap comes back unchanged; a
longer text comes back at or under the cap; the result is LINE ALIGNED -- every
line of it is a whole line of the input or an elision marker line, with the
one pinned exception of an input whose single line is itself longer than the
cap, where a partial line is the honest answer; head and tail are both kept; a
line of the elided middle carrying a :data:`cards.acceptance.DIAGNOSIS_MARKERS`
marker is rescued, in input order, before the tail spends the rest of the
budget; and every gap between kept stretches is named by exactly one marker
carrying the EXACT count of characters dropped there -- so the result is
``part_1 + marker_1 + ... + part_n`` and a walk that adds each part's length
and each marker's number ends exactly at ``len(text)``. That walk is
performed, not paraphrased: :meth:`WalkChecksMixin._walk` walks the input,
position by position and line by whole line, on every test that clips a
multi-line log.
"""

import unittest

from cards.acceptance import (
    _ELISION_TEMPLATE,
    _OUTPUT_TAIL_CAP,
    DIAGNOSIS_MARKERS,
    clip_output,
)


# -- building logs -----------------------------------------------------------


_PREFIX = "... ["
_SUFFIX = " characters elided] ..."

# The real cause line the measured failure dropped: a validation error naming
# the rejected value, carried in the middle of a log far longer than the cap.
CAUSE_LINE = "ValidationError: intent must be one of ('generate', 'patch')\n"


def _marker_number(line):
    """The number an elision marker line names, or ``None`` for any other line.

    Recognises exactly the line :data:`cards.acceptance._ELISION_TEMPLATE`
    builds, with or without the newline it ends in. A log line imitating one
    would be taken for a marker by the walk below; no test here writes such
    a line.
    """
    body = line[:-1] if line.endswith("\n") else line
    if not body.startswith(_PREFIX) or not body.endswith(_SUFFIX):
        return None
    digits = body[len(_PREFIX):-len(_SUFFIX)]
    if not digits.isdigit():
        return None
    return int(digits)


def _padded(body, width):
    """``body`` padded out with 'x' to a line of exactly ``width`` characters."""
    return body + " " + "x" * (width - len(body) - 2) + "\n"


def _plain_line(index, width=60):
    """A filler line of exactly ``width`` characters, carrying no marker."""
    return _padded("plain filler line {:04d}".format(index), width)


def _assertion_line(index, width=26):
    """A diagnosis line of exactly ``width`` characters: it carries ``assert``."""
    return _padded("assertion {:04d} failed".format(index), width)


def _tail_line(index, width=200):
    """A heavy plain line of exactly ``width`` characters, carrying no marker."""
    return _padded("tail summary line {:04d}".format(index), width)


# -- the walk ----------------------------------------------------------------


class WalkChecksMixin:
    """The literal walk both clip test classes share.

    The contract's walk clause is a property, and the tests check it by
    PERFORMING it: the result is parsed into an alternating sequence of parts
    and markers, a walk starts at the input's first character, adds each
    part's length (after checking the part really is the input at the position
    the walk reached, line by whole line) and each marker's number (after
    checking the number lands the walk on a line start, stepping over the
    whole lines the gap dropped), and the walk must end exactly at
    ``len(text)`` with every input line accounted for.
    """

    def _clip(self, text):
        """Clip, and enforce the cap while at it: no result may exceed it."""
        result = clip_output(text)
        self.assertLessEqual(len(result), _OUTPUT_TAIL_CAP)
        return result

    def _walk(self, text, result):
        """Perform the walk; return ``(parts, numbers)`` in result order."""
        input_lines = text.splitlines(keepends=True)
        offsets = [0]
        for line in input_lines:
            offsets.append(offsets[-1] + len(line))

        pieces = []
        current = []
        for line in result.splitlines(keepends=True):
            number = _marker_number(line)
            if number is None:
                current.append(line)
            else:
                if current:
                    pieces.append(("part", "".join(current)))
                    current = []
                pieces.append(("marker", number))
        if current:
            pieces.append(("part", "".join(current)))

        parts = [value for kind, value in pieces if kind == "part"]
        numbers = [value for kind, value in pieces if kind == "marker"]

        position = 0
        line_at = 0
        for kind, value in pieces:
            if kind == "marker":
                position += value
                while line_at < len(input_lines) and offsets[line_at + 1] <= position:
                    line_at += 1
                self.assertEqual(
                    offsets[line_at],
                    position,
                    "a marker's number does not land the walk on a line start",
                )
            else:
                self.assertEqual(
                    offsets[line_at],
                    position,
                    "a part does not begin where the walk sits",
                )
                for result_line in value.splitlines(keepends=True):
                    self.assertLess(
                        line_at,
                        len(input_lines),
                        "the result keeps more lines than the input has",
                    )
                    self.assertEqual(
                        input_lines[line_at],
                        result_line,
                        "a cut landed inside a line",
                    )
                    line_at += 1
                position += len(value)
        self.assertEqual(
            position, len(text), "the walk did not end exactly at len(text)"
        )
        self.assertEqual(
            line_at,
            len(input_lines),
            "the walk did not consume every input line",
        )
        return parts, numbers


# -- budget and shape --------------------------------------------------------


class ClipShapeTests(WalkChecksMixin, unittest.TestCase):
    """The clipper's budget and shape contract, and the walk that pins it."""

    def test_text_at_or_under_the_cap_is_returned_unchanged(self):
        """At or under :data:`_OUTPUT_TAIL_CAP`, nothing happens at all."""
        for text in (
            "",
            "short log line\n",
            "x" * _OUTPUT_TAIL_CAP,
            "\n".join("log line {}".format(index) for index in range(50)),
        ):
            self.assertEqual(clip_output(text), text)

    def test_a_longer_text_is_never_clipped_past_the_cap(self):
        """The cap holds for every shape the clipper can meet."""
        cases = [
            "z" * 10000,
            "z" * 10000 + "\n",
            "".join(_plain_line(index) for index in range(200)),
            "".join(_assertion_line(index) for index in range(200)),
            "g" * 1500 + "".join(_plain_line(index) for index in range(50)),
            "".join(
                CAUSE_LINE if index == 100 else _plain_line(index)
                for index in range(200)
            ),
            "".join(
                [_assertion_line(index) for index in range(120)]
                + [_tail_line(index) for index in range(5)]
            ),
        ]
        for text in cases:
            self.assertGreater(len(text), _OUTPUT_TAIL_CAP)
            self.assertLessEqual(len(clip_output(text)), _OUTPUT_TAIL_CAP)

    def test_no_cut_lands_inside_a_line(self):
        """Every result line is a whole input line or a marker line.

        Checked two ways: each non-marker line of the result is one of the
        input's lines verbatim, and the walk matches every part against the
        input line by line at the position the walk has reached.
        """
        text = "".join(
            CAUSE_LINE if index in (30, 100, 160) else _plain_line(index)
            for index in range(200)
        )
        result = self._clip(text)
        whole_lines = set(text.splitlines(keepends=True))
        for line in result.splitlines(keepends=True):
            if _marker_number(line) is None:
                self.assertIn(line, whole_lines)
        self._walk(text, result)

    def test_the_walk_ends_exactly_at_len_text(self):
        """The walk clause, performed literally: it ends at ``len(text)``.

        For each shape, the parts the result keeps plus the numbers its
        markers name must account for every character of the input -- and
        ``_walk`` has already checked each part sits at the position the walk
        reached, so the sum is not a coincidence.
        """
        cases = [
            "".join(_plain_line(index) for index in range(200)),
            "".join(_assertion_line(index) for index in range(200)),
            "g" * 1500 + "".join(_plain_line(index) for index in range(50)),
            "".join(
                CAUSE_LINE if index in (30, 100, 160) else _plain_line(index)
                for index in range(200)
            ),
            "".join(
                CAUSE_LINE if index == 100 else _plain_line(index)
                for index in range(200)
            ),
        ]
        for text in cases:
            parts, numbers = self._walk(text, self._clip(text))
            self.assertEqual(
                sum(len(part) for part in parts) + sum(numbers), len(text)
            )

    def test_a_log_without_diagnosis_lines_gives_head_marker_and_tail(self):
        """No rescue in sight: head, one exact marker, tail.

        The middle is dropped but not denied: its whole character count is
        what the one marker names, the head is the input's opening and the
        tail its closing, and the marker is the template's line -- the part
        before the gap supplies the newline the template opens with.
        """
        text = "".join(_plain_line(index) for index in range(200))
        result = self._clip(text)
        parts, numbers = self._walk(text, result)
        self.assertEqual(len(numbers), 1, "one gap, one marker")
        self.assertEqual(len(parts), 2)
        head, tail = parts
        self.assertTrue(head and tail, "head and tail both kept")
        self.assertTrue(text.startswith(head))
        self.assertTrue(text.endswith(tail))
        self.assertEqual(numbers[0], len(text) - len(head) - len(tail))
        self.assertIn(_ELISION_TEMPLATE.format(numbers[0]), result)
        self.assertEqual(
            result, head + _ELISION_TEMPLATE.format(numbers[0])[1:] + tail
        )

    def test_several_gaps_get_several_exact_markers(self):
        """Several gaps mean several markers, each naming its own gap.

        A cause far from both ends leaves TWO gaps -- head to cause, cause to
        tail -- because the tail's budget runs out before it reaches the
        cause; each gap gets its own marker, and the walk has verified each
        number against the input at the position it stands for.
        """
        text = "".join(
            CAUSE_LINE if index == 100 else _plain_line(index)
            for index in range(200)
        )
        result = self._clip(text)
        parts, numbers = self._walk(text, result)
        self.assertEqual(len(numbers), 2)
        self.assertEqual(len(parts), 3)
        for number in numbers:
            self.assertIn(_ELISION_TEMPLATE.format(number), result)
        self.assertIn(CAUSE_LINE, result)
        head, middle, tail = parts
        self.assertTrue(text.startswith(head))
        self.assertTrue(text.endswith(tail))
        self.assertEqual(middle, CAUSE_LINE)

    def test_a_first_line_longer_than_the_head_budget_is_never_cut(self):
        """An oversized first line is dropped whole, never cut to fit.

        Line alignment forbids cutting it to fit the head budget, so the head
        starts empty and the result opens with the marker naming the loss --
        the giant first line appears nowhere, not even as a fragment.
        """
        text = "g" * 1500 + "".join(_plain_line(index) for index in range(50))
        result = self._clip(text)
        self.assertTrue(result.startswith("... ["))
        self.assertNotIn("gggg", result)
        self._walk(text, result)

    def test_a_single_line_longer_than_the_cap_keeps_a_partial_line(self):
        """The pinned exception: one line, longer than the cap.

        Nothing whole can be kept from such an input, so a partial line is
        the honest answer: the character-level cut this function made before
        it learned to read lines -- first quarter, marker, last three
        quarters, the template used verbatim because the cuts land mid-line
        and need the template's newlines. The marker still names exactly what
        the two partial stretches dropped.
        """
        for text in ("z" * 10000, "z" * 10000 + "\n"):
            result = self._clip(text)
            self.assertNotEqual(result, text)
            self.assertTrue(result.startswith("z"))
            self.assertEqual(text[:100], result[:100])
            self.assertEqual(text[-100:], result[-100:])
            result_lines = result.splitlines(keepends=True)
            markers = [
                line for line in result_lines if _marker_number(line) is not None
            ]
            self.assertEqual(len(markers), 1)
            number = _marker_number(markers[0])
            self.assertIn(_ELISION_TEMPLATE.format(number), result)
            kept = len(result) - len(_ELISION_TEMPLATE.format(number))
            self.assertEqual(number, len(text) - kept)
            whole_lines = text.splitlines()
            self.assertEqual(len(whole_lines), 1)
            # the kept head is a PARTIAL line: not one of the input's lines
            self.assertNotIn(result.splitlines()[0], whole_lines)


# -- the diagnosis rescue ----------------------------------------------------


class DiagnosisRescueTests(WalkChecksMixin, unittest.TestCase):
    """What counts as a diagnosis line, and what a rescued line is worth."""

    def test_the_marker_tuple_holds_the_named_vocabulary(self):
        """The constant is a tuple holding at least the six named markers."""
        self.assertIsInstance(DIAGNOSIS_MARKERS, tuple)
        for marker in ("Error", "error:", "assert", "FAILED", "Traceback", "E "):
            self.assertIn(marker, DIAGNOSIS_MARKERS)

    def test_a_cause_line_in_the_middle_of_a_huge_log_survives(self):
        """The failure that bought the rescue: the cause is shown.

        The middle of this log holds ``intent must be one of (...)`` -- the
        entire diagnosis of the real run that burned every retry without ever
        being shown it. Clipped, the cause line survives whole.
        """
        text = "".join(
            CAUSE_LINE if index == 100 else _plain_line(index)
            for index in range(200)
        )
        result = self._clip(text)
        self.assertIn("intent must be one of", result)
        self.assertIn(CAUSE_LINE, result)
        self._walk(text, result)

    def test_every_marker_in_the_tuple_rescues_its_line(self):
        """Each entry pulls its own weight: a line carrying only it survives."""
        probes = {
            "Error": "ValueError: bad value\n",
            "error:": "error: file not found\n",
            "assert": "assert 5 == 4\n",
            "FAILED": "FAILED tests/test_clip_output.py::test_x\n",
            "Traceback": "Traceback (most recent call last):\n",
            "E ": "E       the value was wrong\n",
        }
        for marker, line in probes.items():
            self.assertIn(marker, DIAGNOSIS_MARKERS)
            text = "".join(
                line if index == 100 else _plain_line(index)
                for index in range(200)
            )
            result = self._clip(text)
            self.assertIn(
                line, result, "marker {!r} did not rescue its line".format(marker)
            )
            self._walk(text, result)

    def test_rescued_lines_keep_their_input_order(self):
        """Rescued lines appear in the result in their input order."""
        causes = {
            40: "ValueError: the context slice was empty\n",
            100: CAUSE_LINE,
            160: "AssertionError: expected the marker to survive\n",
        }
        text = "".join(
            causes.get(index, _plain_line(index)) for index in range(200)
        )
        result = self._clip(text)
        places = [result.index(cause) for cause in causes.values()]
        self.assertEqual(places, sorted(places))
        for cause in causes.values():
            self.assertIn(cause, result)
        self._walk(text, result)

    def test_diagnoses_are_filled_before_the_tail_spends_the_rest(self):
        """Causes are worth more than the tail's lines.

        The middle of the first log is saturated with diagnosis lines, and
        they spend the post-head budget before the tail spends the rest: the
        tail keeps only what the causes left, and its FIRST line is dropped
        while every cause survives. Beside it, the same-shaped log with a
        plain middle keeps every one of the tail's five lines -- the contrast
        is the priority.
        """
        diagnosis_log = "".join(
            [_assertion_line(index) for index in range(120)]
            + [_tail_line(index) for index in range(5)]
        )
        plain_log = "".join(
            [_plain_line(index, width=26) for index in range(120)]
            + [_tail_line(index) for index in range(5)]
        )
        diagnosis_result = self._clip(diagnosis_log)
        plain_result = self._clip(plain_log)
        # the causes the budget could hold all survived the clip
        self.assertIn("assertion 0119 failed", diagnosis_result)
        # the tail spent only what the causes left: its first line is gone...
        self.assertNotIn("tail summary line 0000", diagnosis_result)
        # ...its last is kept...
        self.assertIn("tail summary line 0004", diagnosis_result)
        # ...and with a plain middle the same tail keeps ALL five lines.
        self.assertIn("tail summary line 0000", plain_result)
        self.assertIn("tail summary line 0004", plain_result)
        self._walk(diagnosis_log, diagnosis_result)
        self._walk(plain_log, plain_result)

    def test_a_diagnosis_line_too_large_for_the_budget_is_left_dropped(self):
        """A cause is rescued whole or not at all.

        A middle line carrying ``Error`` but larger than the budget that is
        left cannot be rescued without breaking either the cap or line
        alignment, so it stays in the middle it came from -- absent from the
        result entirely, named only by the marker that covers it.
        """
        giant = "g" * 3440 + "ValidationError: the giant cause\n"
        text = "".join(
            giant if index == 20 else _plain_line(index) for index in range(40)
        )
        result = self._clip(text)
        self.assertNotIn("the giant cause", result)
        self.assertNotIn("gggg", result)
        self._walk(text, result)


if __name__ == "__main__":
    unittest.main()
