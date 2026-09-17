"""
Phase 7 tests: reading a commit back (:func:`cards.repo.diffstat`).

The same discipline as ``tests/test_repo.py``: against REAL repositories,
because the value here is in what git actually prints -- which counts
``--numstat`` gives a text file, what it prints instead of a count for a binary
one, which order the paths arrive in -- and a faked subprocess would only assert
that we typed the command we meant to type. So every test builds a throwaway
repository exactly the way ``tests/test_repo.py`` does (``tempfile.mkdtemp()``,
identity written LOCAL to that repository, ``init.defaultBranch`` pinned) and
tears it down again, and the whole module skips when there is no ``git`` on
PATH -- the same condition :func:`cards.repo.is_git_repo` degrades on.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

from cards.repo import (
    GitError,
    commit_paths,
    diffstat,
)


HAVE_GIT = shutil.which("git") is not None
REQUIRES_GIT = unittest.skipUnless(HAVE_GIT, "git is not on PATH")

# Identity written into the throwaway repository only. Anything committed here
# is deleted seconds later; the address is deliberately unroutable.
TEST_EMAIL = "morph-test@example.invalid"
TEST_NAME = "Morph Test"


def _git(root, *args):
    """Run a git command in a test repository, failing the test on any error."""
    return subprocess.run(
        ["git"] + list(args), cwd=root,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, check=True, timeout=30).stdout


def _write(root, relative, text):
    path = os.path.join(root, relative)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _init_repo(root):
    """A repository with a local identity, one commit, and a .gitignore.

    ``init.defaultBranch`` is pinned so the branch assertions do not depend on
    the git version's idea of a default name; ``commit.gpgsign=false`` so an
    engineer who signs every commit by default does not have this suite
    prompting them for a passphrase.
    """
    _git(root, "-c", "init.defaultBranch=main", "init", "-q", ".")
    _git(root, "config", "user.email", TEST_EMAIL)
    _git(root, "config", "user.name", TEST_NAME)
    _git(root, "config", "commit.gpgsign", "false")
    _write(root, ".gitignore", "build/\n*.log\n")
    _write(root, "a.py", "print('a')\n")
    _git(root, "add", ".gitignore", "a.py")
    _git(root, "commit", "-q", "-m", "initial")


class _RepoCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-repo-")
        # macOS hands out /var/... symlinked to /private/var/...; realpath here
        # so assertions compare the same spelling git resolves to.
        self.tmp = os.path.realpath(self.tmp)
        _init_repo(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


@REQUIRES_GIT
class DiffstatTests(_RepoCase):
    def test_two_files_report_their_own_insertions_and_deletions(self):
        # One edit to the tracked a.py (1 line swapped for 1 line) and one new
        # three-line file: each path must carry its own counts, not a blended
        # total for the commit.
        _write(self.tmp, "a.py", "print('changed')\n")
        _write(self.tmp, "new.py", "one\ntwo\nthree\n")

        sha = commit_paths(self.tmp, ["a.py", "new.py"], "morph: two files", {})

        self.assertIsNotNone(sha)
        entries = {entry["path"]: entry for entry in diffstat(self.tmp, sha)}
        self.assertEqual(sorted(entries), ["a.py", "new.py"])
        self.assertEqual(
            set(entries["a.py"]), {"path", "insertions", "deletions"})
        self.assertEqual(entries["a.py"]["insertions"], 1)
        self.assertEqual(entries["a.py"]["deletions"], 1)
        self.assertEqual(entries["new.py"]["insertions"], 3)
        self.assertEqual(entries["new.py"]["deletions"], 0)
        # A genuine 0 is an int, not a stand-in for "unknown": new.py deleted
        # nothing and git counted nothing -- a different fact from the binary
        # file's "-" in the test below.
        self.assertIsInstance(entries["new.py"]["deletions"], int)

    def test_a_binary_file_reports_none_not_zero(self):
        # A NUL byte makes git read the file as binary, and git refuses to
        # count lines it cannot see: both fields come back as "-" and must
        # arrive here as None. A 0 would claim a certainty git does not have.
        blob = os.path.join(self.tmp, "blob.bin")
        with open(blob, "wb") as handle:
            handle.write(b"\x00\x01not really text\x00")

        sha = commit_paths(self.tmp, ["blob.bin"], "morph: binary", {})

        self.assertIsNotNone(sha)
        entries = diffstat(self.tmp, sha)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["path"], "blob.bin")
        self.assertIsNone(entries[0]["insertions"])
        self.assertIsNone(entries[0]["deletions"])

    def test_an_unknown_sha_raises_giterror(self):
        with self.assertRaises(GitError) as raised:
            diffstat(self.tmp, "f" * 40)
        # git's own words reach the caller, as they do for every other refusal
        # in this module.
        self.assertIn("git show", str(raised.exception))

    def test_entries_come_back_in_git_s_order_not_the_caller_s(self):
        # The commit lists zeta before alpha; git's diff walks paths in its own
        # order, and the list must follow git, not the order the call happened
        # to pass. Cross-checked against the raw numstat itself, so the test
        # does not hard-code more of git than it has to.
        _write(self.tmp, "zeta.py", "z\n")
        _write(self.tmp, "alpha.py", "a\n")
        _write(self.tmp, "mid.py", "m\n")

        sha = commit_paths(
            self.tmp, ["zeta.py", "alpha.py", "mid.py"], "morph: order", {})

        entries = diffstat(self.tmp, sha)
        self.assertEqual(
            [entry["path"] for entry in entries],
            ["alpha.py", "mid.py", "zeta.py"])
        raw = _git(self.tmp, "show", "--numstat", "--format=", sha)
        self.assertEqual(
            [entry["path"] for entry in entries],
            [line.split("\t", 2)[2] for line in raw.strip().splitlines()
             if line.strip()])


if __name__ == "__main__":
    unittest.main()
