"""Pure formatting of one archived Morph run report into the lines a CLI prints.

An archived run lives at ``.morph/runs/<deck-id>/report.json``.  Once the
caller has read that file and parsed its JSON into a plain dict, this module
turns the dict into the exact lines the CLI prints, one per card.  It exists
so the CLI wiring in ``flows/morph.py`` stays a handful of lines: read the
file, call :func:`format_run_report`, print the returned lines verbatim.

This module deliberately does none of the following; all of it stays in the
caller:

* no argument parsing,
* no filesystem access (it never opens ``report.json``; it only sees a dict),
* no git (it never runs git and never looks at a repository),
* no printing (it returns strings; it never writes to stdout).

The report shape it understands:

* ``report["generations"]`` is a list of generations, each a list of
  ``custom_id`` strings.  Flattened in order, that is the order of the
  returned lines.
* ``report["outcomes"]`` is a dict keyed by ``custom_id`` (not a list).  Each
  outcome carries at least ``custom_id``, ``status``, ``paths``, ``reason``,
  ``attempts`` and ``winning_variant`` and -- for a card that became a commit
  -- the two newer keys ``commit`` (a sha string) and ``diffstat`` (a list of
  ``{"path", "insertions", "deletions"}`` entries where the counts are ints,
  or ``None`` for a binary file).

Each returned line has the shape::

    <custom_id>   <status>   +<insertions> -<deletions>   <sha[:10]>   <paths>

with the five fields joined by three spaces, where the numbers are the sums
over the card's ``diffstat`` entries (``None`` counts skipped) and the paths
are the diffstat paths joined with ``", "`` -- falling back to the outcome's
own ``paths`` when there is no diffstat.  A line whose paths field would be
empty is trimmed, so no returned line ends with whitespace.

Three formatting rules, all mandatory:

* A card whose deletions exceed its insertions gets a mark appended at the
  end of its line (``_DELETION_HEAVY_MARK``) -- a visible warning that the
  executor removed more than it added.  It is a mark, never a refusal: the
  card is still listed normally.
* A card with no commit (a run made with nogit, or a card that changed
  nothing) prints ``-`` where the numbers and the sha would go, and does not
  raise.
* Reports written before ``commit`` and ``diffstat`` existed have neither
  key.  Every field is therefore read with ``.get(...)``, so an old report
  formats as the no-commit case instead of raising ``KeyError``; a
  ``custom_id`` listed in ``generations`` but missing from ``outcomes``
  formats without raising too (with ``missing`` as its status).
"""

from typing import List

__all__ = ["format_run_report"]

#: Separator between the fields of a printed line.
_SEPARATOR = "   "

#: Printed where the diff numbers and the sha would go for a card with no commit.
_NO_COMMIT = "-"

#: Printed in place of ``status`` when an outcome is missing (or status-less).
_MISSING_STATUS = "missing"

#: Appended to the line of a card whose deletions exceed its insertions.
_DELETION_HEAVY_MARK = "   !! removed more than added"


def format_run_report(report: dict) -> List[str]:
    """Format one archived run report into the lines a CLI prints.

    ``report`` is the parsed content of one archived
    ``.morph/runs/<deck-id>/report.json``.  The function is pure: it reads
    nothing but ``report``, mutates nothing, and never parses arguments,
    touches the filesystem, runs git or prints -- all of that stays in the
    caller (the CLI wiring in ``flows/morph.py``).

    One line is returned per card, in generation order (``generations``
    flattened in order), shaped::

        <custom_id>   <status>   +<insertions> -<deletions>   <sha[:10]>   <paths>

    with the fields joined by three spaces, where:

    * the numbers are the sums over the card's ``diffstat`` entries, with
      entries whose ``insertions``/``deletions`` are ``None`` (binary files)
      skipped;
    * the sha is the card's ``commit`` truncated to 10 characters;
    * the paths are the diffstat paths joined with ``", "``, falling back to
      the outcome's own ``paths`` when there is no diffstat.

    The three mandatory rules:

    * a card whose deletions exceed its insertions gets
      ``_DELETION_HEAVY_MARK`` appended at the end of its line -- a warning
      mark, never a refusal: the card is still listed normally;
    * a card with no ``commit`` (a run made with nogit, or a card that
      changed nothing) prints ``-`` where the numbers and the sha would go;
    * nothing raises on report-shaped input: every field is read with
      ``.get(...)``, so reports written before ``commit``/``diffstat``
      existed format as the no-commit case instead of raising ``KeyError``,
      and a ``custom_id`` listed in ``generations`` but missing from
      ``outcomes`` (or an outcome without a status) formats with
      ``missing`` as its status.

    Args:
        report: the parsed ``report.json`` dict of one archived run.

    Returns:
        One line per card, in generation order, ready to print verbatim.
    """
    generations = report.get("generations") or []
    raw_outcomes = report.get("outcomes")
    outcomes = raw_outcomes if isinstance(raw_outcomes, dict) else {}

    lines: List[str] = []
    for generation in generations:
        for custom_id in generation or []:
            # The outcome may be missing entirely; degrade, never raise.
            outcome = outcomes.get(custom_id)
            if not isinstance(outcome, dict):
                outcome = {}

            status = str(outcome.get("status") or _MISSING_STATUS)
            commit = outcome.get("commit")
            diffstat = outcome.get("diffstat") or []

            # Sums over the diffstat; None counts (binary files) are skipped.
            insertions = 0
            deletions = 0
            for entry in diffstat:
                if not isinstance(entry, dict):
                    continue
                count = entry.get("insertions")
                if isinstance(count, int):
                    insertions += count
                count = entry.get("deletions")
                if isinstance(count, int):
                    deletions += count

            if commit:
                numbers = f"+{insertions} -{deletions}"
                sha = str(commit)[:10]
            else:
                # No commit (nogit run, or nothing changed) -> dashes.
                numbers = _NO_COMMIT
                sha = _NO_COMMIT

            if diffstat:
                paths = ", ".join(
                    str(entry["path"])
                    for entry in diffstat
                    if isinstance(entry, dict) and entry.get("path")
                )
            else:
                # No diffstat: fall back to the outcome's own paths.
                paths = ", ".join(str(path) for path in (outcome.get("paths") or []))

            line = _SEPARATOR.join([str(custom_id), status, numbers, sha, paths])
            line = line.rstrip()
            if commit and deletions > insertions:
                # A mark, never a refusal: the card stays listed.
                line += _DELETION_HEAVY_MARK
            lines.append(line)
    return lines
