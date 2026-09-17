"""
Phase 7 acceptance: a run is a branch, a card is a commit, a run is archived.

Where ``tests/test_repo.py`` tests the git plumbing on its own, this is the two
halves put to work: the store opening a branch for a run, committing each
accepted card with its provenance in the trailers, and archiving the finished
run under ``.morph/runs/<deck-id>/``. So every assertion here is made against a
REAL repository -- ``git log`` output, ``git status`` output -- for the same
reason ``test_repo.py`` gives: a faked subprocess would only prove we typed the
commands we meant to type, and what is being claimed is what git ends up holding.

Each test builds a throwaway repository from ``tests/fixtures/miniproject`` under
``tempfile.mkdtemp()``, with a LOCAL identity (never ``--global``), and deletes
it again. The module skips entirely without ``git`` on PATH -- except the
no-repository case, which is precisely the one that must work anyway.

The development plan's acceptance criteria for the phase, one test each:

* a 2-card deck creates a branch and two commits whose trailers ``git log``
  shows, and ``git checkout`` of the original branch returns the tree;
* a 3-target card whose acceptance fails leaves ``git status`` clean and adds no
  commit;
* a non-git directory runs exactly as it did before, with one warning line.

Plus the two halves the plan's prose asks for and the criteria do not name: the
dirty-tree refusal, the ``nogit`` opt-out, and the archive (written, append-only,
listed by ``/deck runs``, committed onto the run's branch).
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest

from cards.generations import run_deck
from cards.store import (
    DeckStore,
    StoreError,
    begin_run,
    collect_generation,
    list_runs,
    make_card_committer,
    make_deck_id,
    record_run,
    submit_generation,
)
from flows.morph import MorphBot


MINIPROJECT = os.path.join("tests", "fixtures", "miniproject")

HAVE_GIT = shutil.which("git") is not None
REQUIRES_GIT = unittest.skipUnless(HAVE_GIT, "git is not on PATH")

TEST_EMAIL = "morph-test@example.invalid"
TEST_NAME = "Morph Test"


def _code_block(body):
    return f"```python\n{body}\n```"


def _file_block(path, body):
    """One file of a multi-file answer, in the format the directive asks for."""
    return f"FILE: {path}\n```python\n{body}\n```\n"


class _FakeBatchBackend:
    """A scripted submit/status/collect backend (the pattern used across tests).

    ``scripts`` maps a (variant/retry) custom_id to the text ``collect`` returns;
    ids absent from the map fall back to ``default_response``. ``status`` always
    reports ``completed``, so one ``collect`` poll suffices.
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


class _GitCase(unittest.TestCase):
    """A throwaway git repository holding a copy of the miniproject fixture."""

    # Whether the test repository ignores ``.morph/``. Most do not: the archive
    # commit is part of what is being tested. One subclass flips it, because a
    # project that DOES ignore it is the common real-world case.
    IGNORE_MORPH = False

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-git-")
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
            handle.write(".morph/\n" if self.IGNORE_MORPH else "*.pyc\n")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "the project before any run")

    def log(self, *extra):
        return self.git("log", "--format=%H%n%B%n--END--", *extra)

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

    def status(self):
        """``git status`` of the PROJECT -- ``.morph/`` excluded.

        The run's own bookkeeping (the backlog and the state file) is never
        committed and is untracked in a repository that does not ignore it;
        counting it here would make "the tree is clean" a claim about the
        orchestrator rather than about the engineer's code, which is the thing
        every assertion in this module actually means.
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

    def add(self, custom_id, target, **meta):
        instruction = meta.pop("instruction", "do it")
        self.store.add_card({
            "custom_id": custom_id,
            "meta": {"intent": meta.pop("intent", "generate"),
                     "target": target, **meta},
            "instruction": instruction,
        })

    def add_changeset(self, custom_id, targets, **meta):
        instruction = meta.pop("instruction", "do it")
        self.store.add_card({
            "custom_id": custom_id,
            "meta": {"intent": meta.pop("intent", "patch"),
                     "targets": targets, **meta},
            "instruction": instruction,
        })

    def read(self, name):
        with open(os.path.join(self.root, name), "r", encoding="utf-8") as handle:
            return handle.read()

    def run_split_step(self, backend, use_git=True, **kwargs):
        """Drive one deck through /submit + /collect until the run is done."""
        notes = []
        while True:
            submitted = submit_generation(
                self.store, backend, root=self.root, backend_label="fake",
                use_git=use_git, log=notes.append)
            if submitted.done:
                return notes
            for _ in range(200):
                collected = collect_generation(
                    self.store, backend, root=self.root, poll_interval=0,
                    log=notes.append, **kwargs)
                if not collected.in_progress:
                    break
            else:
                self.fail("collect never completed")
            if collected.phase == "done":
                return notes


@REQUIRES_GIT
class RunOnABranchTests(_GitCase):
    """The plan's headline criterion: two cards, one branch, two commits."""

    def test_a_two_card_deck_is_a_branch_and_two_trailered_commits(self):
        self.add("card-a", "gen_a.py", context_slice=["util.py"],
                 instruction="make a", model="tiny-model")
        self.add("card-b", "gen_b.py", context_slice=["util.py"],
                 instruction="make b")
        backend = _FakeBatchBackend(scripts={
            "card-a": _code_block("A = 1"),
            "card-b": _code_block("B = 2"),
        })

        self.run_split_step(backend)

        # The run owns a branch named after its deck id, taken off main.
        state = self.store.load_state()
        self.assertTrue(state["deck_id"])
        self.assertEqual(state["branch"], "morph/" + state["deck_id"])
        self.assertEqual(self.branch(), state["branch"])

        # One commit per accepted card, plus the archive, on top of the
        # project's own history.
        subjects = self.subjects()
        self.assertIn("morph card-a: gen_a.py", subjects)
        self.assertIn("morph card-b: gen_b.py", subjects)
        self.assertEqual(subjects[-1], "the project before any run")

        # The provenance git log shows. card-a pinned its own model; card-b
        # inherits the run's backend label, because "which model wrote this" is
        # a fact about the run when the card does not pin one.
        trailers = self.trailers_of(self.revision("morph card-a: gen_a.py"))
        self.assertEqual(trailers["Morph-Card"], "card-a")
        self.assertEqual(trailers["Morph-Model"], "tiny-model")
        self.assertEqual(self.trailers_of(self.revision("morph card-b: gen_b.py"))["Morph-Model"], "fake")

        # Each commit carries exactly its own card's file -- not the other's.
        touched = self.git("show", "--name-only", "--format=",
                              self.revision("morph card-a: gen_a.py"))
        self.assertEqual(touched.split(), ["gen_a.py"])

        # Nothing is left behind, and checking main out returns the tree.
        self.assertEqual(self.status(), "")
        self.git("checkout", "-q", "main")
        self.assertFalse(os.path.exists(os.path.join(self.root, "gen_a.py")))
        self.assertFalse(os.path.exists(os.path.join(self.root, "gen_b.py")))

    def test_the_acceptance_command_and_its_exit_code_are_trailers(self):
        self.add("card-a", "gen_a.py", context_slice=["util.py"],
                 acceptance="python3 -c \"import sys; sys.exit(0)\"",
                 instruction="make a")
        backend = _FakeBatchBackend(scripts={"card-a": _code_block("A = 1")})

        self.run_split_step(backend)

        trailers = self.trailers_of(self.revision("morph card-a: gen_a.py"))
        self.assertEqual(trailers["Morph-Acceptance"],
                         "python3 -c \"import sys; sys.exit(0)\"")
        self.assertEqual(trailers["Morph-Acceptance-Exit"], "0")
        # A single-variant card held no contest, so no variant is named: its
        # "winner" would be its own custom_id, which Morph-Card already says.
        self.assertNotIn("Morph-Variant", trailers)

    def test_a_best_of_n_card_names_the_variant_that_won(self):
        self.add("card-a", "gen_a.py", context_slice=["util.py"], variants=2,
                 acceptance="python3 -c \"import gen_a; assert gen_a.A == 2\"",
                 instruction="make a")
        # The first variant fails the card's own acceptance; the second passes.
        backend = _FakeBatchBackend(scripts={
            "card-a.v1": _code_block("A = 1"),
            "card-a.v2": _code_block("A = 2"),
        })

        self.run_split_step(backend)

        trailers = self.trailers_of(self.revision("morph card-a: gen_a.py"))
        self.assertEqual(trailers["Morph-Variant"], "card-a.v2")
        self.assertEqual(self.read("gen_a.py").strip(), "A = 2")

    def test_a_long_acceptance_command_is_capped_not_sprawled(self):
        command = "python3 -c \"print(1)\" # " + "x" * 400
        self.add("card-a", "gen_a.py", context_slice=["util.py"],
                 acceptance=command, instruction="make a")
        backend = _FakeBatchBackend(scripts={"card-a": _code_block("A = 1")})

        self.run_split_step(backend)

        trailer = self.trailers_of(self.revision("morph card-a: gen_a.py"))["Morph-Acceptance"]
        self.assertLessEqual(len(trailer), 120)
        self.assertTrue(trailer.endswith("..."))
        # Capped, not lost: the readable head of the command is still there, and
        # the other trailers after it survived (a sprawling value would have
        # ended git's footer and taken them with it).
        self.assertTrue(trailer.startswith("python3 -c \"print(1)\""))
        self.assertEqual(self.trailers_of(self.revision("morph card-a: gen_a.py"))["Morph-Acceptance-Exit"],
                         "0")

    def test_a_card_that_changes_nothing_makes_no_commit(self):
        # The card rewrites util.py with exactly the bytes already committed.
        original = self.read("util.py")
        self.add("card-a", "util.py", intent="patch", context_slice=[],
                 instruction="leave it alone")
        backend = _FakeBatchBackend(scripts={"card-a": f"```python\n{original}```"})

        before = self.commit_count()
        notes = self.run_split_step(backend)

        self.assertEqual(self.read("util.py"), original)
        # Only the archive commit was added -- no provenance record for a line
        # that was never generated.
        self.assertEqual(self.commit_count(), before + 1)
        self.assertNotIn("morph card-a: util.py", self.subjects())
        self.assertTrue(any("changed nothing" in line for line in notes))
        self.assertEqual(self.store.load_outcomes()["card-a"].status, "written")

    def test_a_second_submit_of_the_same_run_does_not_open_a_second_branch(self):
        self.add("card-a", "gen_a.py", context_slice=["util.py"],
                 instruction="make a")
        self.add("card-b", "gen_b.py", depends_on=["card-a"],
                 context_slice=["gen_a.py"], instruction="make b")
        backend = _FakeBatchBackend(default_response=_code_block("X = 1"))

        self.run_split_step(backend)

        state = self.store.load_state()
        branches = [line.strip("* ").strip()
                    for line in self.git("branch").splitlines()]
        self.assertEqual(sorted(branches), sorted(["main", state["branch"]]))
        self.assertEqual(state["generations"], [["card-a"], ["card-b"]])


@REQUIRES_GIT
class FailedCardLeavesNothingTests(_GitCase):
    """The plan's second criterion, at the git level."""

    # PYTHONPATH=. so the freshly written test can import the freshly written
    # module; running the file directly keeps the check stdlib-only.
    ACCEPTANCE = "PYTHONPATH=. python3 check.py"

    def test_a_three_target_card_that_fails_leaves_a_clean_tree_and_no_commit(self):
        self.add_changeset(
            "trio",
            ["one.py", "two.py", "check.py"],
            context_slice=[],
            acceptance=self.ACCEPTANCE,
            instruction="write three files that agree with each other",
        )
        # check.py demands a name the other two files do not define, so the
        # card's own acceptance fails it.
        backend = _FakeBatchBackend(scripts={
            "trio": (_file_block("one.py", "ONE = 1")
                       + _file_block("two.py", "TWO = 2")
                       + _file_block("check.py",
                                     "from one import ONE, MISSING\n")),
        })

        notes = self.run_split_step(backend, max_regenerations=0)

        outcome = self.store.load_outcomes()["trio"]
        self.assertEqual(outcome.status, "failed")
        # Failed because its OWN acceptance ran and rejected it -- not because
        # the batch came back empty, which would prove nothing about rollback.
        self.assertIn("MISSING", outcome.acceptance_output)
        # None of the three files is on disk, and git agrees the tree is clean:
        # the rollback is complete, not merely "mostly".
        for name in ("one.py", "two.py", "check.py"):
            self.assertFalse(os.path.exists(os.path.join(self.root, name)), name)
        self.assertEqual(self.status(), "")
        self.assertNotIn("morph trio: one.py, two.py, check.py", self.subjects())
        self.assertFalse(any("committed as" in line for line in notes
                             if "trio" in line))


@REQUIRES_GIT
class NightlyOnABranchTests(_GitCase):
    """``/nightly`` commits per card too -- the same hook, one blocking pass."""

    def test_nightly_commits_each_card_as_it_is_accepted(self):
        self.add("card-a", "gen_a.py", context_slice=["util.py"],
                 instruction="make a")
        self.add("card-b", "gen_b.py", depends_on=["card-a"],
                 context_slice=["gen_a.py"], instruction="make b")
        backend = _FakeBatchBackend(scripts={
            "card-a": _code_block("A = 1"),
            "card-b": _code_block("B = 2"),
        })
        notes = []

        cards = self.store.load_cards()
        state = begin_run(self.store, cards, root=self.root,
                          backend_label="fake", log=notes.append)
        result = run_deck(cards, backend, root=self.root, poll_interval=0,
                          log=notes.append,
                          on_accepted=make_card_committer(self.root, state,
                                                          notes.append))
        record_run(self.store, result, backend_label="fake", root=self.root,
                   run_state=state, log=notes.append)

        self.assertEqual(self.branch(), state["branch"])
        subjects = self.subjects()
        self.assertIn("morph card-a: gen_a.py", subjects)
        self.assertIn("morph card-b: gen_b.py", subjects)
        # Order matters: card-b's commit is the later one, as the run ran them.
        self.assertLess(subjects.index("morph card-b: gen_b.py"),
                        subjects.index("morph card-a: gen_a.py"))
        self.assertEqual(self.status(), "")
        # Every batch the run submitted is recorded for the archive.
        self.assertEqual(result.batch_ids, ["fake-batch-1", "fake-batch-2"])

    def test_a_patch_chain_attributes_each_version_to_its_own_card(self):
        """The reason the commit is taken WHERE the card is accepted.

        Two cards write the same file in successive generations. Committing
        afterwards, from the finished outcomes, would stage the final content
        under the first card's name and then find nothing left for the second.
        """
        self.add("gen-mod", "mod.py", context_slice=["util.py"],
                 instruction="write it")
        self.add("patch-mod", "mod.py", intent="patch", depends_on=["gen-mod"],
                 context_slice=[], instruction="extend it")
        backend = _FakeBatchBackend(scripts={
            "gen-mod": _code_block("VALUE = 1"),
            "patch-mod": _code_block("VALUE = 1\nEXTRA = 2"),
        })

        cards = self.store.load_cards()
        state = begin_run(self.store, cards, root=self.root,
                          backend_label="fake", log=lambda _l: None)
        result = run_deck(cards, backend, root=self.root, poll_interval=0,
                          log=lambda _l: None,
                          on_accepted=make_card_committer(self.root, state,
                                                          lambda _l: None))
        record_run(self.store, result, backend_label="fake", root=self.root,
                   run_state=state, log=lambda _l: None)

        first = self.git("show", "--format=",
                         self.revision("morph gen-mod: mod.py") + ":mod.py")
        second = self.git("show", "--format=",
                          self.revision("morph patch-mod: mod.py") + ":mod.py")
        self.assertIn("VALUE = 1", first)
        self.assertNotIn("EXTRA", first)
        self.assertIn("EXTRA = 2", second)


@REQUIRES_GIT
class DirtyTreeTests(_GitCase):
    def test_a_dirty_tree_refuses_to_start_a_run_and_names_the_way_out(self):
        with open(os.path.join(self.root, "app.py"), "a", encoding="utf-8") as handle:
            handle.write("\n# an engineer's unfinished thought\n")
        self.add("card-a", "gen_a.py", context_slice=["util.py"],
                 instruction="make a")
        backend = _FakeBatchBackend(default_response=_code_block("A = 1"))

        with self.assertRaises(StoreError) as caught:
            submit_generation(self.store, backend, root=self.root,
                              log=lambda _l: None)

        message = str(caught.exception)
        self.assertIn("uncommitted changes", message)
        self.assertIn("stash", message)
        self.assertIn("nogit", message)
        # Refused BEFORE anything was paid for or written: no batch, no state.
        self.assertEqual(backend.submissions, [])
        self.assertEqual(self.store.load_state()["generations"], [])
        self.assertEqual(self.branch(), "main")

    def test_an_untracked_file_counts_as_dirty(self):
        with open(os.path.join(self.root, "notes.txt"), "w",
                  encoding="utf-8") as handle:
            handle.write("scratch\n")
        self.add("card-a", "gen_a.py", context_slice=["util.py"],
                 instruction="make a")

        with self.assertRaises(StoreError):
            submit_generation(self.store, _FakeBatchBackend(), root=self.root,
                              log=lambda _l: None)

    def test_nogit_runs_on_a_dirty_tree_and_touches_no_branch(self):
        with open(os.path.join(self.root, "app.py"), "a", encoding="utf-8") as handle:
            handle.write("\n# an engineer's unfinished thought\n")
        self.add("card-a", "gen_a.py", context_slice=["util.py"],
                 instruction="make a")
        backend = _FakeBatchBackend(scripts={"card-a": _code_block("A = 1")})

        notes = self.run_split_step(backend, use_git=False)

        # The morph is on disk exactly as it would be without git at all.
        self.assertTrue(os.path.exists(os.path.join(self.root, "gen_a.py")))
        self.assertIsNone(self.store.load_state()["branch"])
        self.assertEqual(self.branch(), "main")
        self.assertEqual(self.commit_count(), 1)
        self.assertTrue(any("git: off for this run" in line for line in notes))
        # The run is still archived -- opting out of git is not opting out of
        # keeping a record.
        self.assertEqual(len(list_runs(self.store)), 1)


@REQUIRES_GIT
class RunArchiveTests(_GitCase):
    def _run_one(self, target, response="X = 1"):
        custom_id = "card-" + target.split(".")[0]
        self.add(custom_id, target, context_slice=["util.py"], instruction="make it")
        backend = _FakeBatchBackend(scripts={custom_id: _code_block(response)})
        self.run_split_step(backend)
        self.store.clear()
        return custom_id

    def test_a_completed_run_leaves_its_deck_and_report_on_disk(self):
        self.add("card-a", "gen_a.py", context_slice=["util.py"],
                 instruction="make a")
        self.add("card-b", "gen_b.py", depends_on=["card-a"],
                 context_slice=["gen_a.py"], instruction="make b")
        backend = _FakeBatchBackend(default_response=_code_block("X = 1"))

        self.run_split_step(backend)

        deck_id = self.store.load_state()["deck_id"]
        directory = self.store.run_dir(deck_id)
        with open(os.path.join(directory, "deck.json"), encoding="utf-8") as handle:
            deck = json.load(handle)
        with open(os.path.join(directory, "report.json"), encoding="utf-8") as handle:
            report = json.load(handle)

        # The deck AS EXECUTED, in deck order, in the backlog's own format.
        self.assertEqual([card["custom_id"] for card in deck], ["card-a", "card-b"])
        self.assertEqual(deck[1]["meta"]["depends_on"], ["card-a"])

        self.assertEqual(report["deck_id"], deck_id)
        self.assertEqual(report["branch"], "morph/" + deck_id)
        self.assertEqual(report["backend_label"], "fake")
        self.assertEqual(report["generations"], [["card-a"], ["card-b"]])
        self.assertEqual(report["batch_ids"], ["fake-batch-1", "fake-batch-2"])
        self.assertEqual(report["outcomes"]["card-a"]["status"], "written")
        self.assertEqual(report["outcomes"]["card-b"]["paths"],
                         [os.path.join(self.root, "gen_b.py")])

    def test_the_archive_is_committed_as_the_runs_final_commit(self):
        self.add("card-a", "gen_a.py", context_slice=["util.py"],
                 instruction="make a")
        backend = _FakeBatchBackend(default_response=_code_block("A = 1"))

        self.run_split_step(backend)

        deck_id = self.store.load_state()["deck_id"]
        self.assertEqual(self.subjects()[0], f"morph run {deck_id}: deck and report")
        files = self.git("show", "--name-only", "--format=", "HEAD").split()
        self.assertEqual(sorted(files), sorted([
            f".morph/runs/{deck_id}/deck.json",
            f".morph/runs/{deck_id}/report.json",
        ]))
        self.assertEqual(self.trailers_of("HEAD")["Morph-Written"], "1")
        self.assertEqual(self.status(), "")

    def test_a_second_run_never_rewrites_the_first(self):
        first_id = None
        self._run_one("gen_a.py")
        first_id = self.store.load_state()["deck_id"]
        first_report = os.path.join(self.store.run_dir(first_id), "report.json")
        with open(first_report, encoding="utf-8") as handle:
            first_before = handle.read()

        self.store.reset_state()
        self._run_one("gen_b.py")
        second_id = self.store.load_state()["deck_id"]

        self.assertNotEqual(first_id, second_id)
        with open(first_report, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), first_before)
        self.assertEqual(sorted(os.listdir(self.store.runs_dir)),
                         sorted([first_id, second_id]))

    def test_an_existing_archive_directory_is_left_alone(self):
        # The append-only rule, asserted directly: archiving twice under the
        # same id is a no-op, whatever the second run would have written.
        self.add("card-a", "gen_a.py", context_slice=["util.py"],
                 instruction="make a")
        backend = _FakeBatchBackend(default_response=_code_block("A = 1"))
        self.run_split_step(backend)

        deck_id = self.store.load_state()["deck_id"]
        report_path = os.path.join(self.store.run_dir(deck_id), "report.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write('{"deck_id": "hand-edited"}\n')

        from cards.store import archive_run
        again = archive_run(self.store, self.store.load_state(),
                            self.store.load_cards(), root=self.root,
                            log=lambda _l: None)

        self.assertIsNone(again)
        with open(report_path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), {"deck_id": "hand-edited"})

    def test_deck_runs_renders_every_archived_run_newest_first(self):
        self._run_one("gen_a.py")
        first_id = self.store.load_state()["deck_id"]
        self.store.reset_state()
        self._run_one("gen_b.py")
        second_id = self.store.load_state()["deck_id"]

        previous = os.getcwd()
        os.chdir(self.root)
        try:
            text = MorphBot._runs_text()
        finally:
            os.chdir(previous)

        self.assertIn("2 archived run(s)", text)
        self.assertIn(first_id, text)
        self.assertIn(second_id, text)
        self.assertLess(text.index(second_id), text.index(first_id))
        self.assertIn("1 written, 0 failed, 0 skipped", text)
        self.assertIn("morph/" + second_id, text)

    def test_a_run_that_ends_in_skips_is_archived_too(self):
        # Every card of generation 2 is blocked by a failure in generation 1, so
        # the run ends inside /submit rather than /collect -- and that run,
        # above all, is one worth having a record of.
        self.add("card-a", "gen_a.py", context_slice=["util.py"],
                 instruction="make a")
        self.add("card-b", "gen_b.py", depends_on=["card-a"],
                 context_slice=[], instruction="make b")
        backend = _FakeBatchBackend(scripts={"card-a": None})

        self.run_split_step(backend)

        reports = list_runs(self.store)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].counts,
                         {"written": 0, "failed": 1, "skipped": 1})
        self.assertEqual(reports[0].outcomes["card-b"].reason, "card-a")


@REQUIRES_GIT
class IgnoredMorphDirTests(_GitCase):
    """A project that ignores ``.morph/`` keeps its ignore; it is told, not overruled."""

    IGNORE_MORPH = True

    def test_the_archive_is_written_but_not_committed_and_the_run_is_told(self):
        self.add("card-a", "gen_a.py", context_slice=["util.py"],
                 instruction="make a")
        backend = _FakeBatchBackend(default_response=_code_block("A = 1"))

        notes = self.run_split_step(backend)

        deck_id = self.store.load_state()["deck_id"]
        self.assertTrue(os.path.exists(
            os.path.join(self.store.run_dir(deck_id), "report.json")))
        # The card's own commit landed; only the archive commit was refused, and
        # the refusal is reported rather than forced past the engineer's ignore.
        self.assertIn("morph card-a: gen_a.py", self.subjects())
        self.assertNotIn(f"morph run {deck_id}: deck and report", self.subjects())
        self.assertTrue(any("not committed" in line for line in notes))
        self.assertEqual(self.status(), "")


class NoRepositoryTests(unittest.TestCase):
    """The plan's third criterion: outside a repository, nothing changes.

    Deliberately NOT skipped when git is missing -- "no git on this machine" is
    one of the two ways to land here, and the behaviour has to hold for both.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-nogit-")
        self.root = os.path.join(self.tmp, "miniproject")
        shutil.copytree(MINIPROJECT, self.root,
                        ignore=shutil.ignore_patterns("node_modules"))
        self.store = DeckStore(project_root=self.root)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_plain_directory_runs_exactly_as_before_with_one_warning(self):
        self.store.add_card({
            "custom_id": "card-a",
            "meta": {"intent": "generate", "target": "gen_a.py",
                     "context_slice": ["util.py"]},
            "instruction": "make a",
        })
        backend = _FakeBatchBackend(scripts={"card-a": _code_block("A = 1")})
        notes = []

        submit_generation(self.store, backend, root=self.root,
                          backend_label="fake", log=notes.append)
        for _ in range(200):
            collected = collect_generation(self.store, backend, root=self.root,
                                           poll_interval=0, log=notes.append)
            if not collected.in_progress:
                break

        self.assertEqual(collected.phase, "done")
        self.assertTrue(os.path.exists(os.path.join(self.root, "gen_a.py")))
        self.assertIsNone(self.store.load_state()["branch"])
        warnings = [line for line in notes if line.startswith("mrph> git:")]
        self.assertEqual(len(warnings), 1)
        self.assertIn("not a git working copy", warnings[0])
        # The run is still archived: the archive is a file, not a commit.
        self.assertEqual(len(list_runs(self.store)), 1)

    def test_no_archived_runs_reads_as_an_explanation_not_an_error(self):
        previous = os.getcwd()
        os.chdir(self.root)
        try:
            text = MorphBot._runs_text()
        finally:
            os.chdir(previous)
        self.assertIn("No archived runs yet", text)
        self.assertEqual(list_runs(self.store), [])


class DeckIdTests(unittest.TestCase):
    def test_the_id_leads_with_a_timestamp_and_ends_with_a_backlog_digest(self):
        from cards.schema import MorphCard

        card = MorphCard.from_dict({
            "custom_id": "card-a",
            "meta": {"intent": "generate", "target": "gen_a.py"},
            "instruction": "make a",
        })
        other = MorphCard.from_dict({
            "custom_id": "card-a",
            "meta": {"intent": "generate", "target": "gen_a.py"},
            "instruction": "make something else entirely",
        })

        deck_id = make_deck_id([card])
        stamp, _dash, digest = deck_id.rpartition("-")
        self.assertRegex(stamp, r"^\d{8}-\d{6}$")
        self.assertRegex(digest, r"^[0-9a-f]{8}$")
        # Same backlog, same digest; a changed instruction, a changed digest.
        self.assertEqual(make_deck_id([card]).rpartition("-")[2], digest)
        self.assertNotEqual(make_deck_id([other]).rpartition("-")[2], digest)
        # And it is a legal branch name component.
        self.assertNotIn(" ", deck_id)


class _FakeBot:
    """The slice of ``MorphBot`` the ``/submit`` transition uses.

    The transition itself is the code under test here, so every method it calls
    is the real one; only the processor registry is stubbed, because resolving a
    backend is not what the flag is about.
    """

    report_unexpected = staticmethod(MorphBot.report_unexpected)
    git_notes = staticmethod(MorphBot.git_notes)
    deck_notes = staticmethod(MorphBot.deck_notes)
    split_run_flags = staticmethod(MorphBot.split_run_flags)

    def __init__(self, backend):
        self._active_backend = None
        self._backend = backend

    def resolve_batch_backend(self, _text):
        return self._backend, "fake"


class _FakeBotContext:
    """Collects what a transition sends, in place of a chat."""

    def __init__(self):
        self.messages = []
        self.bot = self

    async def send_message(self, chat_id=None, text=None, **_kwargs):
        self.messages.append(text)


@REQUIRES_GIT
class SubmitTransitionFlagTests(_GitCase):
    """The opt-out where the user actually types it: on ``/submit``."""

    def _submit(self, text):
        import asyncio

        bot = _FakeBot(_FakeBatchBackend(default_response=_code_block("A = 1")))
        context = _FakeBotContext()
        action = {"update": {"effective_chat": {"id": 1}},
                  "context": context, "text": text}

        async def nested(_action):
            return None

        previous = os.getcwd()
        os.chdir(self.root)
        try:
            asyncio.new_event_loop().run_until_complete(
                MorphBot.build_submit_transition(bot, nested)(action))
        finally:
            os.chdir(previous)
        return "\n".join(message for message in context.messages if message)

    def test_a_bare_submit_opens_a_branch_and_says_so(self):
        self.add("card-a", "gen_a.py", context_slice=["util.py"],
                 instruction="make a")

        text = self._submit("/submit")

        state = self.store.load_state()
        self.assertEqual(state["branch"], "morph/" + state["deck_id"])
        self.assertEqual(self.branch(), state["branch"])
        self.assertIn("git: the run is on branch", text)

    def test_submit_nogit_runs_and_opens_nothing(self):
        self.add("card-a", "gen_a.py", context_slice=["util.py"],
                 instruction="make a")

        text = self._submit("/submit nogit")

        # The generation went out -- "nogit" was not mistaken for a processor id
        # -- and the run holds no branch.
        self.assertIn("Submitted generation 1/1", text)
        self.assertIn("git: off for this run", text)
        self.assertIsNone(self.store.load_state()["branch"])
        self.assertEqual(self.branch(), "main")

    def test_submit_reports_the_dirty_tree_refusal_instead_of_raising(self):
        with open(os.path.join(self.root, "app.py"), "a", encoding="utf-8") as handle:
            handle.write("\n# unfinished\n")
        self.add("card-a", "gen_a.py", context_slice=["util.py"],
                 instruction="make a")

        text = self._submit("/submit")

        self.assertIn("uncommitted changes", text)
        self.assertIn("nogit", text)
        self.assertNotIn("Submitted generation", text)
        self.assertEqual(self.store.load_state()["phase"], "idle")


class RunFlagTests(unittest.TestCase):
    """The opt-out's parsing: ``nogit`` must not be read as a processor id."""

    def test_nogit_is_stripped_and_the_processor_spec_survives(self):
        self.assertEqual(MorphBot.split_run_flags("/submit nogit"),
                         ("/submit", False))
        # The reason it must be stripped BEFORE the backend is resolved: every
        # remaining token is read as a processor id, so a bare "/submit nogit"
        # would otherwise be answered with "no matching processor".
        self.assertEqual(MorphBot.parse_processor_spec("/submit nogit"), ["nogit"])
        self.assertIsNone(MorphBot.parse_processor_spec(
            MorphBot.split_run_flags("/submit nogit")[0]))
        self.assertEqual(MorphBot.split_run_flags("/submit @gpt4 nogit"),
                         ("/submit @gpt4", False))
        self.assertEqual(MorphBot.split_run_flags("/nightly NoGit @all"),
                         ("/nightly @all", False))

    def test_a_line_without_the_flag_is_unchanged_and_uses_git(self):
        self.assertEqual(MorphBot.split_run_flags("/submit @gpt4"),
                         ("/submit @gpt4", True))
        self.assertEqual(MorphBot.split_run_flags("/submit"), ("/submit", True))
        self.assertEqual(MorphBot.split_run_flags(""), ("", True))
        self.assertEqual(MorphBot.split_run_flags(None), (None, True))

    def test_git_notes_selects_only_the_git_commentary(self):
        lines = [
            "mrph> [generation 1/2] submitting 1 card(s): card-a",
            "mrph> git: the run is on branch 'morph/x'",
            "mrph> [generation 1/2] 'card-a' written: gen_a.py",
            "mrph> git: 'card-a' committed as deadbeef01.",
        ]
        self.assertEqual(MorphBot.git_notes(lines), [lines[1], lines[3]])


if __name__ == "__main__":
    unittest.main()
