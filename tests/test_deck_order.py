"""
The deck a run is about to execute, committed where git can keep it.

``.morph/deck.json`` -- the only copy of the order a run executes its cards in
-- sits in the directory a project's ``.gitignore`` covers. The archive under
``.morph/runs/<deck-id>/deck.json`` is written only when a run FINISHES, which
is exactly the case that did not happen when the nightly machine died at 02:00:
the accepted cards were on the branch as commits, the unfinished ones nowhere,
and nothing recorded what the night was supposed to do. So ``begin_run`` now
writes the deck to ``decks/<deck-id>.json`` and commits it on the run's branch
at the moment the run starts, before the first card is compiled.

As in ``tests/test_run_git.py``, every assertion here is made against a REAL
throwaway repository built from ``tests/fixtures/miniproject`` under
``tempfile.mkdtemp()``, with a LOCAL identity (never ``--global``) -- a faked
subprocess would only prove we typed the commands we meant to type. Nothing
here touches the repository these tests live in or its ``.morph/``, and the
module skips entirely without ``git`` on PATH.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest

from cards.deck import load_deck
from cards.generations import run_deck
from cards.store import DeckStore, begin_run, record_run


MINIPROJECT = os.path.join("tests", "fixtures", "miniproject")

HAVE_GIT = shutil.which("git") is not None
REQUIRES_GIT = unittest.skipUnless(HAVE_GIT, "git is not on PATH")

TEST_EMAIL = "morph-test@example.invalid"
TEST_NAME = "Morph Test"


def _code_block(body):
    return f"```python\n{body}\n```"


class _FakeBatchBackend:
    """A scripted submit/status/collect backend (the pattern used across tests).

    ``scripts`` maps a custom_id to the text ``collect`` returns; ids absent
    from the map fall back to ``default_response``. ``status`` always reports
    ``completed``, so one poll suffices.
    """

    def __init__(self, scripts=None, default_response=None):
        self.scripts = scripts or {}
        self.default_response = default_response
        self.submissions = []
        self._counter = 0

    def submit(self, requests):
        self.submissions.append(requests)
        self._counter += 1
        return f"fake-batch-{self._counter}"

    def status(self, _batch_id):
        return "completed"

    def collect(self, _batch_id):
        results = {}
        for request in self.submissions[-1]:
            custom_id = request["custom_id"]
            results[custom_id] = self.scripts.get(custom_id, self.default_response)
        return results


@REQUIRES_GIT
class DeckOrderTests(unittest.TestCase):
    """A throwaway git repository holding a copy of the miniproject fixture."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-deckorder-")
        self.root = os.path.join(self.tmp, "miniproject")
        shutil.copytree(MINIPROJECT, self.root,
                        ignore=shutil.ignore_patterns("node_modules"))
        self.store = DeckStore(project_root=self.root)
        self._init_repo()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- repository helpers -------------------------------------------------

    def git(self, *args):
        return subprocess.run(
            ["git"] + list(args), cwd=self.root,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, check=True, timeout=30).stdout

    def _init_repo(self):
        """A repository with a local identity and one commit holding the fixture.

        ``init.defaultBranch`` is pinned so the branch assertions do not depend
        on the git version's default name; ``commit.gpgsign=false`` so an
        engineer who signs everything is not prompted by this suite.
        """
        self.git("-c", "init.defaultBranch=main", "init", "-q", ".")
        self.git("config", "user.email", TEST_EMAIL)
        self.git("config", "user.name", TEST_NAME)
        self.git("config", "commit.gpgsign", "false")
        with open(os.path.join(self.root, ".gitignore"), "w",
                  encoding="utf-8") as handle:
            handle.write("*.pyc\n")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "the project before any run")

    def subjects(self):
        return [line for line in
                self.git("log", "--format=%s").splitlines() if line]

    def revision(self, subject):
        """The sha of the one commit with this exact subject line."""
        matches = [line.split(" ", 1)[0] for line in
                   self.git("log", "--format=%H %s").splitlines()
                   if line.split(" ", 1)[1:] == [subject]]
        self.assertEqual(len(matches), 1, f"no single commit titled {subject!r}")
        return matches[0]

    def porcelain(self):
        """``git status --porcelain`` of the PROJECT -- ``.morph/`` excluded.

        The run's own bookkeeping (the backlog and the state file) is never
        committed and is untracked here; counting it would make every assertion
        about the tree a claim about the orchestrator rather than about what
        git was given to keep.
        """
        return self.git("status", "--porcelain", "--", ".", ":(exclude).morph")

    def branch(self):
        return self.git("rev-parse", "--abbrev-ref", "HEAD").strip()

    def commit_count(self):
        return int(self.git("rev-list", "--count", "HEAD").strip())

    def trailers_of(self, revision):
        """The ``Key: value`` footer of one commit, as a dict."""
        body = self.git("log", "-1", "--format=%B", revision)
        trailers = {}
        for line in body.splitlines():
            if ": " in line and not line.startswith(" "):
                key, _sep, value = line.partition(": ")
                trailers[key] = value.strip()
        return trailers

    # -- deck helpers -------------------------------------------------------

    def _seed(self):
        """Two cards in the backlog, in the shape ``MorphCard.from_dict`` accepts.

        ``intent`` is the literal flow name ``"generate"``, not a description of
        what the card does. The cards go through the store, so ``record_run``'s
        archive later finds the same deck ``begin_run`` was handed.
        """
        self.store.add_card({
            "custom_id": "card-a",
            "meta": {"intent": "generate", "target": "gen_a.py",
                     "context_slice": ["util.py"]},
            "instruction": "make a",
        })
        self.store.add_card({
            "custom_id": "card-b",
            "meta": {"intent": "generate", "target": "gen_b.py",
                     "depends_on": ["card-a"], "context_slice": ["gen_a.py"]},
            "instruction": "make b",
        })
        return self.store.load_cards()

    def order_path(self, deck_id):
        return os.path.join(self.root, "decks", deck_id + ".json")

    # -- the tests ----------------------------------------------------------

    def test_the_deck_is_in_the_repository_the_moment_the_run_starts(self):
        cards = self._seed()
        state = begin_run(self.store, cards, root=self.root,
                          backend_label="fake", log=lambda _l: None)
        deck_id = state["deck_id"]

        # Nothing else has run -- no batch submitted, no card compiled, the run
        # merely begun -- and the order it will execute is already on disk.
        self.assertEqual(state["phase"], "idle")
        self.assertTrue(os.path.exists(self.order_path(deck_id)))
        with open(self.order_path(deck_id), encoding="utf-8") as handle:
            written = json.load(handle)
        self.assertEqual([card["custom_id"] for card in written],
                         ["card-a", "card-b"])
        self.assertEqual(written[0]["meta"]["intent"], "generate")
        self.assertEqual(written[0]["meta"]["target"], "gen_a.py")
        self.assertEqual(written[1]["meta"]["depends_on"], ["card-a"])

    def test_the_deck_commit_is_on_the_runs_branch(self):
        cards = self._seed()
        state = begin_run(self.store, cards, root=self.root,
                          backend_label="fake", log=lambda _l: None)
        deck_id = state["deck_id"]
        subject = f"morph run {deck_id}: the deck as submitted"

        self.assertEqual(self.branch(), "morph/" + deck_id)
        # Committed, not merely written: git status has nothing to say about it.
        self.assertEqual(self.porcelain(), "")
        sha = self.revision(subject)
        self.assertEqual(
            self.git("show", "--name-only", "--format=", sha).split(),
            [f"decks/{deck_id}.json"])
        self.assertEqual(self.trailers_of(sha)["Morph-Run"], deck_id)

    def test_load_deck_reads_the_committed_deck_back_into_the_same_cards(self):
        cards = self._seed()
        state = begin_run(self.store, cards, root=self.root,
                          backend_label="fake", log=lambda _l: None)

        reread = load_deck(self.order_path(state["deck_id"]))

        # Same ids in the same order -- the run's composition -- and the same
        # targets and dependencies behind them.
        self.assertEqual([card.custom_id for card in reread],
                         [card.custom_id for card in cards])
        self.assertEqual([card.targets for card in reread],
                         [card.targets for card in cards])
        self.assertEqual([card.depends_on for card in reread],
                         [card.depends_on for card in cards])
        self.assertEqual([card.instruction for card in reread],
                         [card.instruction for card in cards])

    def test_a_nogit_run_writes_the_file_and_commits_nothing(self):
        cards = self._seed()
        notes = []
        state = begin_run(self.store, cards, root=self.root, use_git=False,
                          log=notes.append)
        deck_id = state["deck_id"]

        self.assertIsNone(state["branch"])
        self.assertTrue(os.path.exists(self.order_path(deck_id)))
        self.assertEqual(self.branch(), "main")
        self.assertEqual(self.commit_count(), 1)
        # Untracked: the record is on disk, git was simply left alone.
        self.assertIn("decks/", self.porcelain())
        self.assertTrue(any("git: off for this run" in line for line in notes))

    def test_the_finished_runs_archive_is_unaffected(self):
        cards = self._seed()
        state = begin_run(self.store, cards, root=self.root,
                          backend_label="fake", log=lambda _l: None)
        deck_id = state["deck_id"]
        with open(self.order_path(deck_id), encoding="utf-8") as handle:
            at_start = handle.read()

        backend = _FakeBatchBackend(scripts={
            "card-a": _code_block("A = 1"),
            "card-b": _code_block("B = 2"),
        })
        result = run_deck(cards, backend, root=self.root, poll_interval=0,
                          log=lambda _l: None)
        record_run(self.store, result, backend_label="fake", root=self.root,
                   run_state=state, log=lambda _l: None)

        # The archive holds the deck as executed -- the same cards, in the same
        # order, in the same format the run committed at its start.
        with open(os.path.join(self.store.run_dir(deck_id), "deck.json"),
                  encoding="utf-8") as handle:
            archived = json.load(handle)
        self.assertEqual([card["custom_id"] for card in archived],
                         ["card-a", "card-b"])
        self.assertEqual(archived[1]["meta"]["depends_on"], ["card-a"])
        self.assertEqual(archived, json.loads(at_start))
        # ...and the run's end neither consumed nor rewrote the start's file.
        with open(self.order_path(deck_id), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), at_start)
        self.assertIn(f"morph run {deck_id}: deck and report", self.subjects())


if __name__ == "__main__":
    unittest.main()
