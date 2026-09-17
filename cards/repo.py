"""
The git substrate: the few working-copy operations a deck run needs.

Phase 7 of ``documentation/DEVELOPMENT_PLAN.md``. Until now a run wrote morphs
straight into whatever tree it found: an accepted card and a hand edit landed in
the same undifferentiated diff, and the provenance of a generated line lived
only in ``.morph/state.json``. The phase's answer is a branch per run and a
commit per accepted card, with the card's provenance in the commit trailers --
so ``git log`` answers "where did this line come from" and ``git checkout``
undoes a run without hand-cleaning the tree.

This module is ONLY the plumbing: pure functions over a working copy, with no
knowledge of decks, cards, generations or ``.morph/``. The run integrates them
(branch name in the state file, trailers built from a :class:`CardOutcome`);
keeping the two apart means the git behaviour is testable against throwaway
repositories, with no deck machinery in the way, and that a change of policy
upstream ("commit per generation instead of per card") touches no code here.

No GitPython: ``git`` is shelled out to via :mod:`subprocess`, which is the
project's no-new-dependencies rule and also the honest interface -- every call
is a command an engineer can paste into a terminal to see the same result.

THE MODULE DEGRADES, IT DOES NOT DEMAND. :func:`is_git_repo` is false both for a
plain directory and for a machine with no ``git`` on PATH, and the caller then
skips the whole git path with a warning rather than refusing to run (the phase's
acceptance criterion: "in a non-git directory the run degrades to the current
behaviour with a warning"). So no other function assumes a repository exists;
each fails loudly, carrying git's own message, when called anyway.

WHAT IT REFUSES TO DO, deliberately: no ``git config --global`` and no identity
of its own (a commit uses whatever the repository already has configured, and an
unconfigured repository gets git's own readable refusal); no push, no force, no
history rewrite, no ``git add -A`` and no ``git add .`` -- staging is always the
explicit path list a card declares, so a run can never sweep up an engineer's
unrelated edit.

Pure stdlib. No network, no imports from ``flows`` or ``processors``.
"""

import os
import subprocess
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Every git invocation is bounded. These commands are local and finish in
# milliseconds; a call that has not returned in half a minute is a wedged index
# lock or a hook waiting on input, and a batch orchestrator that blocks forever
# on it is exactly the failure mode Phase 5 spent its effort removing.
GIT_TIMEOUT_SECONDS = 30.0


class GitError(RuntimeError):
    """A git command failed, or was asked for something it must refuse.

    One exception type for the whole module -- a caller wrapping a run in
    ``except GitError`` catches both "git said no" (non-zero exit, message
    included) and "that request is not allowed" (a path outside the working
    copy), because at the call site both mean the same thing: this card does not
    get committed, and the operator needs to read why.
    """


def _git_env() -> Dict[str, str]:
    """The environment for a git child process.

    Only one thing is forced: ``GIT_TERMINAL_PROMPT=0``, so a repository with a
    credential helper or a hook that asks a question fails fast instead of
    blocking on a terminal nobody is watching. Everything else is inherited --
    the user's identity, hooks and config are theirs, and this module does not
    quietly rewrite them.
    """
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _git(root: str, args: Sequence[str], check: bool = True) -> Tuple[int, str, str]:
    """Run one git command in ``root`` and return ``(returncode, stdout, stderr)``.

    Arguments are passed as a list, never a shell string: a file name with a
    space, a quote or a ``$`` in it is data here, not syntax.

    With ``check`` (the default) a non-zero exit raises :class:`GitError`
    carrying git's own stderr -- git explains its refusals well ("a branch named
    'x' already exists", "Author identity unknown"), and paraphrasing it would
    only lose detail. ``check=False`` is for the calls whose non-zero exit is an
    ANSWER rather than a failure (a detached HEAD, a directory that is not a
    repository).
    """
    if not os.path.isdir(root):
        raise GitError(f"not a directory: {root}")
    try:
        completed = subprocess.run(
            ["git"] + list(args),
            cwd=root,
            env=_git_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        # No git binary at all. ``is_git_repo`` turns this into a plain False so
        # the run degrades; anything else has no honest fallback.
        raise GitError("git is not installed or not on PATH")
    except subprocess.TimeoutExpired:
        raise GitError(
            f"git {' '.join(args)} did not finish within "
            f"{GIT_TIMEOUT_SECONDS:.0f}s in {root}")

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if check and completed.returncode != 0:
        detail = (stderr.strip() or stdout.strip() or "no output")
        raise GitError(f"git {' '.join(args)} failed: {detail}")
    return completed.returncode, stdout, stderr


# -- inspection --------------------------------------------------------------


def is_git_repo(root: str) -> bool:
    """Is ``root`` inside a git working copy?

    The one function in the module that never raises: it is the predicate a run
    consults BEFORE deciding whether git exists for it at all, so "no repository
    here", "that path is not a directory" and "no git binary on this machine"
    all have to arrive as the same plain False. Everything else in the module
    may then assume the caller checked.
    """
    try:
        code, stdout, _stderr = _git(
            root, ["rev-parse", "--is-inside-work-tree"], check=False)
    except GitError:
        return False
    return code == 0 and stdout.strip() == "true"


def is_dirty(root: str) -> bool:
    """Does the working copy hold changes a run would bury?

    True for an uncommitted change to a tracked file (staged or not) and for an
    untracked file that is not ignored; false for ignored files, which are
    build output and virtualenvs and say nothing about the engineer's work in
    progress.

    This is the predicate that decides whether a run may start. A run that
    begins on a dirty tree makes its commits indistinguishable from the hand
    edits already sitting there, and the phase's promise -- ``git checkout``
    returns the tree without manual cleanup -- silently stops holding.

    ``--untracked-files=normal`` and ``--ignored=no`` are passed explicitly
    rather than left to default: a repository configured with
    ``status.showUntrackedFiles=no`` would otherwise report a tree full of
    unsaved new files as clean, and blind us exactly where it matters most.
    """
    _code, stdout, _stderr = _git(
        root, ["status", "--porcelain", "--untracked-files=normal", "--ignored=no"])
    return bool(stdout.strip())


def current_branch(root: str) -> Optional[str]:
    """The checked-out branch, or ``None`` on a detached HEAD.

    ``symbolic-ref`` rather than ``rev-parse --abbrev-ref HEAD``: the latter
    answers the literal string ``"HEAD"`` when detached, which is a string
    dangerously close to a valid branch name, and fails outright on a repository with no
    commits yet. ``symbolic-ref --quiet`` exits 1 for "not a branch" and names
    the unborn branch of a fresh repository correctly.
    """
    code, stdout, stderr = _git(
        root, ["symbolic-ref", "--quiet", "--short", "HEAD"], check=False)
    if code == 0:
        return stdout.strip() or None
    if code == 1:
        return None  # detached HEAD: a real answer, not a failure
    raise GitError(
        f"git symbolic-ref failed: {stderr.strip() or stdout.strip() or 'no output'}")


# -- mutation ----------------------------------------------------------------


def create_branch(root: str, name: str) -> None:
    """Create ``name`` from the current HEAD and check it out.

    A name that already exists is an error, not a silent switch: the run that
    asked for this branch expects to own it, and quietly appending commits to an
    existing ``morph/<deck-id>`` would mix two runs' histories. git's own
    message ("a branch named 'x' already exists") reaches the caller through
    :class:`GitError`, so the operator can rename or delete and retry.
    """
    if not name or name.strip() != name:
        raise GitError(f"invalid branch name: {name!r}")
    _git(root, ["checkout", "-b", name])


def _relative_to_root(root: str, path: str) -> str:
    """Normalise one path into a root-relative one, refusing anything outside.

    A card's target comes from a deck file, which is data -- so ``../../etc`` or
    an absolute path to somewhere else on the machine has to be refused HERE,
    before it reaches a git command that would happily stage it (or fail with a
    message about pathspecs that tells the operator nothing). Symlinks are
    resolved first: the check has to hold for where the path actually lands, not
    for how it is spelled.
    """
    real_root = os.path.realpath(root)
    candidate = path if os.path.isabs(path) else os.path.join(real_root, path)
    real_path = os.path.realpath(candidate)
    if real_path != real_root and not real_path.startswith(real_root + os.sep):
        raise GitError(f"path is outside the repository at {root}: {path}")
    relative = os.path.relpath(real_path, real_root)
    if relative == ".":
        raise GitError(f"refusing to commit the repository root itself: {path}")
    return relative


def _trailer_line(key: str, value: object) -> Optional[str]:
    """One ``Key: value`` footer line, flattened to a single line.

    A trailer value is often an acceptance command's output, and that output is
    multi-line by nature. Written verbatim it would not be a trailer at all --
    git's footer ends at the first line that does not parse as one, so a
    pytest traceback in the middle of the footer silently truncates every
    trailer after it. Newlines, tabs and runs of spaces therefore collapse to
    single spaces. A key that is empty (or a value that is empty once flattened)
    yields nothing rather than a malformed ``: `` line.
    """
    clean_key = " ".join(str(key).split())
    if not clean_key:
        return None
    clean_value = " ".join(str(value).split())
    if not clean_value:
        return None
    return f"{clean_key}: {clean_value}"


def _build_message(message: str, trailers: Optional[Dict[str, str]]) -> List[str]:
    """The ``-m`` arguments for the commit: body, then the trailer block.

    Two ``-m`` arguments rather than one string with ``\\n\\n`` in it: git joins
    repeated ``-m`` with a blank line itself, which is exactly the separation a
    footer needs, and it keeps the body untouched whatever it contains.
    """
    args = ["-m", message]
    lines = [line for line in
             (_trailer_line(key, value) for key, value in (trailers or {}).items())
             if line]
    if lines:
        args += ["-m", "\n".join(lines)]
    return args


def commit_paths(
    root: str,
    paths: Iterable[str],
    message: str,
    trailers: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Stage exactly ``paths``, commit them with ``message`` + trailers, return the sha.

    Returns ``None`` -- committing nothing -- when staging those paths produces
    no change. This is not an edge case but the common one: a card whose morph
    rewrites a file byte-identically has genuinely changed nothing, and an empty
    commit for it would put a provenance record in the history for a line of
    code that was never generated. The caller distinguishes the two by the
    return value.

    Only ``paths`` are committed. The staging is an explicit ``git add --
    <paths>`` (never ``-A``, never ``.``) and the commit repeats the pathspec,
    so a file an engineer had already staged by hand is neither included nor
    disturbed, and an unrelated modified file stays modified and unstaged.

    Hooks are NOT bypassed: a repository with a ``pre-commit`` hook gets it run,
    and its refusal arrives as a :class:`GitError`. Silently passing
    ``--no-verify`` would let a run write past a safety net its owner installed
    on purpose -- which is precisely the kind of surprise this phase exists to
    remove.
    """
    relative = [_relative_to_root(root, path) for path in paths]
    if not relative:
        return None

    _git(root, ["add", "--"] + relative)

    # What is actually staged for these paths, compared with HEAD. ``--`` keeps
    # a path that happens to look like a revision from being read as one.
    _code, staged, _stderr = _git(
        root, ["diff", "--cached", "--name-only", "--"] + relative)
    if not staged.strip():
        return None

    _git(root, ["commit"] + _build_message(message, trailers) + ["--"] + relative)
    _code, head, _stderr = _git(root, ["rev-parse", "HEAD"])
    return head.strip()
