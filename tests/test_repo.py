"""
Phase 7 tests: the git substrate (:mod:`cards.repo`).

Against REAL repositories, not mocks. The value of this module is entirely in
what git actually does -- which paths ``git add`` picks up, whether an ignored
file shows in ``git status``, what a partial commit leaves unstaged -- and a
faked subprocess would only assert that we typed the commands we meant to type.
So every test builds a throwaway repository under ``tempfile.mkdtemp()`` and
tears it down again.

The identity is LOCAL to those repositories (``git -c user.email=...`` at init
time), never ``--global``: a test suite that writes to the engineer's own git
config is a test suite that breaks their machine. The whole module skips when
there is no ``git`` on PATH, which is the same condition
:func:`cards.repo.is_git_repo` degrades on.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

from cards.repo import (
    GitError,
    commit_paths,
    create_branch,
    current_branch,
    is_dirty,
    is_git_repo,
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

    def _head(self):
        return _git(self.tmp, "rev-parse", "HEAD").strip()

    def _status(self):
        return _git(self.tmp, "status", "--porcelain")


@REQUIRES_GIT
class IsGitRepoTests(_RepoCase):
    def test_a_repository_is_recognised(self):
        self.assertTrue(is_git_repo(self.tmp))

    def test_a_subdirectory_of_a_repository_is_recognised(self):
        nested = os.path.join(self.tmp, "pkg")
        os.makedirs(nested)
        self.assertTrue(is_git_repo(nested))

    def test_a_plain_directory_answers_false_rather_than_raising(self):
        plain = tempfile.mkdtemp(prefix="morph-plain-")
        try:
            self.assertFalse(is_git_repo(plain))
        finally:
            shutil.rmtree(plain, ignore_errors=True)

    def test_a_missing_directory_answers_false_rather_than_raising(self):
        self.assertFalse(is_git_repo(os.path.join(self.tmp, "nope", "deeper")))


@REQUIRES_GIT
class IsDirtyTests(_RepoCase):
    def test_a_fresh_checkout_is_clean(self):
        self.assertFalse(is_dirty(self.tmp))

    def test_a_modified_tracked_file_is_dirty(self):
        _write(self.tmp, "a.py", "print('changed')\n")
        self.assertTrue(is_dirty(self.tmp))

    def test_a_staged_change_is_dirty(self):
        _write(self.tmp, "a.py", "print('changed')\n")
        _git(self.tmp, "add", "a.py")
        self.assertTrue(is_dirty(self.tmp))

    def test_an_untracked_file_is_dirty(self):
        _write(self.tmp, "new.py", "print('new')\n")
        self.assertTrue(is_dirty(self.tmp))

    def test_an_ignored_file_is_not_dirty(self):
        # Both .gitignore rules: a whole directory and a suffix.
        _write(self.tmp, "build/artifact.bin", "binary-ish\n")
        _write(self.tmp, "run.log", "noise\n")
        self.assertFalse(is_dirty(self.tmp))

    def test_untracked_stays_visible_when_the_repo_hides_it(self):
        # A repository configured to hide untracked files must not be able to
        # tell a run that a tree full of unsaved work is clean.
        _git(self.tmp, "config", "status.showUntrackedFiles", "no")
        _write(self.tmp, "new.py", "print('new')\n")
        self.assertTrue(is_dirty(self.tmp))

    def test_an_excluded_directory_does_not_make_the_tree_dirty(self):
        # The case the caller needs: the orchestrator's own bookkeeping was
        # written seconds ago (appending a card IS a change to it), and a run
        # that counted it would refuse every deck that had just been planned.
        _write(self.tmp, ".morph/deck.json", "[]\n")
        self.assertTrue(is_dirty(self.tmp))
        self.assertFalse(is_dirty(self.tmp, exclude=(".morph",)))

    def test_excluding_one_directory_hides_nothing_else(self):
        _write(self.tmp, ".morph/deck.json", "[]\n")
        _write(self.tmp, "a.py", "print('an engineer was here')\n")
        self.assertTrue(is_dirty(self.tmp, exclude=(".morph",)))

    def test_a_plain_directory_raises(self):
        plain = tempfile.mkdtemp(prefix="morph-plain-")
        try:
            with self.assertRaises(GitError):
                is_dirty(plain)
        finally:
            shutil.rmtree(plain, ignore_errors=True)


@REQUIRES_GIT
class BranchTests(_RepoCase):
    def test_current_branch_names_the_checked_out_branch(self):
        self.assertEqual(current_branch(self.tmp), "main")

    def test_create_branch_creates_and_checks_out(self):
        create_branch(self.tmp, "morph/deck-1")

        self.assertEqual(current_branch(self.tmp), "morph/deck-1")
        # From the current HEAD: the new branch points at the same commit.
        self.assertEqual(
            _git(self.tmp, "rev-parse", "morph/deck-1").strip(),
            _git(self.tmp, "rev-parse", "main").strip())

    def test_create_branch_refuses_an_existing_name(self):
        create_branch(self.tmp, "morph/deck-1")
        _git(self.tmp, "checkout", "-q", "main")

        with self.assertRaises(GitError) as raised:
            create_branch(self.tmp, "morph/deck-1")
        # git's own words reach the caller, not a paraphrase.
        self.assertIn("already exists", str(raised.exception))
        # And the refusal left us where we were.
        self.assertEqual(current_branch(self.tmp), "main")

    def test_create_branch_refuses_a_malformed_name(self):
        with self.assertRaises(GitError):
            create_branch(self.tmp, "")

    def test_current_branch_is_none_on_a_detached_head(self):
        _git(self.tmp, "checkout", "-q", "--detach", "HEAD")
        self.assertIsNone(current_branch(self.tmp))


@REQUIRES_GIT
class CommitPathsTests(_RepoCase):
    def test_commits_only_the_given_paths(self):
        _write(self.tmp, "one.py", "one\n")
        _write(self.tmp, "two.py", "two\n")
        _write(self.tmp, "a.py", "print('edited by hand')\n")

        sha = commit_paths(self.tmp, ["one.py", "two.py"], "card gen-a", {})

        self.assertEqual(sha, self._head())
        committed = _git(self.tmp, "show", "--name-only", "--format=", "HEAD").split()
        self.assertEqual(sorted(committed), ["one.py", "two.py"])
        # The third file is untouched: still modified, still unstaged.
        self.assertEqual(self._status().strip(), "M a.py")

    def test_a_hand_staged_file_is_not_swept_into_the_commit(self):
        _write(self.tmp, "a.py", "print('edited by hand')\n")
        _git(self.tmp, "add", "a.py")
        _write(self.tmp, "one.py", "one\n")

        commit_paths(self.tmp, ["one.py"], "card gen-a", {})

        committed = _git(self.tmp, "show", "--name-only", "--format=", "HEAD").split()
        self.assertEqual(committed, ["one.py"])
        self.assertIn("a.py", self._status())

    def test_trailers_appear_in_the_commit_message(self):
        _write(self.tmp, "one.py", "one\n")

        commit_paths(self.tmp, ["one.py"], "morph: gen-a", {
            "Custom-Id": "gen-a",
            "Model": "z-ai/glm-5.3-flash",
            "Variant": "2",
            "Acceptance": "pytest -q tests/test_one.py",
            "Acceptance-Exit": "0",
        })

        body = _git(self.tmp, "log", "-1", "--format=%B")
        lines = body.strip().splitlines()
        self.assertEqual(lines[0], "morph: gen-a")
        self.assertEqual(lines[1], "")  # the footer is separated from the body
        self.assertIn("Custom-Id: gen-a", lines)
        self.assertIn("Model: z-ai/glm-5.3-flash", lines)
        self.assertIn("Acceptance: pytest -q tests/test_one.py", lines)
        # git itself agrees these are trailers, not stray prose.
        parsed = _git(self.tmp, "log", "-1", "--format=%(trailers:key=Custom-Id)")
        self.assertIn("gen-a", parsed)

    def test_a_multiline_trailer_value_is_flattened(self):
        # An acceptance output is multi-line by nature. Written verbatim it
        # would end git's footer early and swallow every later trailer.
        _write(self.tmp, "one.py", "one\n")
        output = "FAILED tests/test_one.py::test_x\nE   AssertionError\n\n1 failed"

        commit_paths(self.tmp, ["one.py"], "morph: gen-a", {
            "Acceptance-Output": output,
            "Custom-Id": "gen-a",
        })

        body = _git(self.tmp, "log", "-1", "--format=%B")
        lines = body.strip().splitlines()
        self.assertIn(
            "Acceptance-Output: FAILED tests/test_one.py::test_x "
            "E AssertionError 1 failed",
            lines)
        # The trailer that came after the multi-line one survived.
        self.assertIn("Custom-Id: gen-a", lines)
        self.assertIn(
            "gen-a", _git(self.tmp, "log", "-1", "--format=%(trailers:key=Custom-Id)"))

    def test_an_empty_trailer_is_dropped(self):
        _write(self.tmp, "one.py", "one\n")

        commit_paths(self.tmp, ["one.py"], "morph: gen-a",
                     {"Model": "", "Custom-Id": "gen-a"})

        body = _git(self.tmp, "log", "-1", "--format=%B")
        self.assertNotIn("Model:", body)
        self.assertIn("Custom-Id: gen-a", body)

    def test_no_trailers_leaves_a_plain_message(self):
        _write(self.tmp, "one.py", "one\n")

        commit_paths(self.tmp, ["one.py"], "morph: gen-a", None)

        self.assertEqual(
            _git(self.tmp, "log", "-1", "--format=%B").strip(), "morph: gen-a")

    def test_a_byte_identical_rewrite_makes_no_commit(self):
        # The case that must not produce an empty commit: a card whose morph
        # reproduced the file exactly as it already was.
        before = self._head()
        with open(os.path.join(self.tmp, "a.py"), "r", encoding="utf-8") as handle:
            same = handle.read()
        _write(self.tmp, "a.py", same)

        sha = commit_paths(self.tmp, ["a.py"], "morph: gen-a", {"Custom-Id": "gen-a"})

        self.assertIsNone(sha)
        self.assertEqual(self._head(), before)
        self.assertEqual(self._status().strip(), "")

    def test_no_paths_makes_no_commit(self):
        before = self._head()
        self.assertIsNone(commit_paths(self.tmp, [], "morph: nothing", {}))
        self.assertEqual(self._head(), before)

    def test_paths_with_spaces_are_committed(self):
        _write(self.tmp, "src/my module.py", "spaces\n")
        _write(self.tmp, "a file with 'quotes'.txt", "quotes\n")

        sha = commit_paths(
            self.tmp, ["src/my module.py", "a file with 'quotes'.txt"],
            "morph: odd names", {})

        self.assertIsNotNone(sha)
        committed = _git(
            self.tmp, "show", "--name-only", "--format=", "-z", "HEAD").split("\0")
        self.assertEqual(
            sorted(name for name in committed if name),
            ["a file with 'quotes'.txt", "src/my module.py"])
        self.assertFalse(is_dirty(self.tmp))

    def test_a_nested_path_is_committed_from_the_root(self):
        _write(self.tmp, "pkg/sub/deep.py", "deep\n")

        commit_paths(self.tmp, ["pkg/sub/deep.py"], "morph: nested", {})

        self.assertEqual(
            _git(self.tmp, "show", "--name-only", "--format=", "HEAD").strip(),
            "pkg/sub/deep.py")

    def test_a_path_outside_the_root_is_refused(self):
        outside = tempfile.mkdtemp(prefix="morph-outside-")
        try:
            _write(outside, "elsewhere.py", "not ours\n")
            before = self._head()

            with self.assertRaises(GitError) as raised:
                commit_paths(self.tmp, ["../elsewhere.py"], "morph: escape", {})
            self.assertIn("outside", str(raised.exception))

            with self.assertRaises(GitError):
                commit_paths(
                    self.tmp, [os.path.join(outside, "elsewhere.py")],
                    "morph: escape", {})

            # Nothing was staged on the way to the refusal.
            self.assertEqual(self._head(), before)
            self.assertEqual(self._status().strip(), "")
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_the_repository_root_itself_is_refused(self):
        _write(self.tmp, "one.py", "one\n")
        with self.assertRaises(GitError):
            commit_paths(self.tmp, ["."], "morph: everything", {})

    def test_a_deletion_is_committed(self):
        os.remove(os.path.join(self.tmp, "a.py"))

        sha = commit_paths(self.tmp, ["a.py"], "morph: remove a", {})

        self.assertIsNotNone(sha)
        self.assertFalse(is_dirty(self.tmp))
        self.assertEqual(
            _git(self.tmp, "show", "--name-status", "--format=", "HEAD").split()[0],
            "D")

    def test_commits_land_on_the_run_branch(self):
        create_branch(self.tmp, "morph/deck-7")
        _write(self.tmp, "one.py", "one\n")

        sha = commit_paths(self.tmp, ["one.py"], "morph: gen-a", {})

        self.assertEqual(
            _git(self.tmp, "rev-parse", "morph/deck-7").strip(), sha)
        # main is untouched -- a checkout of it returns the pre-run tree.
        self.assertNotEqual(_git(self.tmp, "rev-parse", "main").strip(), sha)

    def test_the_local_identity_is_used(self):
        _write(self.tmp, "one.py", "one\n")
        commit_paths(self.tmp, ["one.py"], "morph: gen-a", {})
        self.assertEqual(
            _git(self.tmp, "log", "-1", "--format=%ae").strip(), TEST_EMAIL)


if __name__ == "__main__":
    unittest.main()
