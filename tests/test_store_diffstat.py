"""
The ``CardOutcome`` commit fields reach disk: ``commit`` and ``diffstat``.

``cards.generations.CardOutcome`` carries two git-mode fields -- the sha of the
commit an accepted card became, and what that commit changed -- and
``cards.store.make_card_committer`` fills them at the moment it commits. This
module checks the rest of that promise:

* the fields are SERIALIZED (``_outcome_to_dict`` / ``_outcome_from_dict``), and
  a record in the shape of an older ``state.json`` / ``report.json`` -- written
  before the fields existed -- still loads, with both ``None``. Backward
  compatibility is the requirement, not a nicety;
* the fields are FILLED after a real commit, and read back out of the
  repository rather than out of what the hook was told;
* a :func:`cards.repo.diffstat` that raises cannot undo an accepted card: the
  hook does not propagate the error, both fields stay ``None``, and the commit
  itself stands;
* the filled fields SURVIVE TO DISK on both run routes, with no plumbing beyond
  the hook's writes -- because both routes save their outcomes only after the
  hook has fired: the split-step route (``submit_generation`` ->
  ``collect_generation`` -> ``_store_outcomes`` -> ``state.json``) and the
  nightly route (``run_deck`` -> ``DeckResult`` -> ``record_run``). The archived
  ``report.json`` is written from the same outcome dicts, so it is checked too.

Every assertion about a commit is made against a REAL repository, built the way
``tests/test_run_git.py`` builds its ``_GitCase``: a copy of
``tests/fixtures/miniproject`` under a temporary directory, a LOCAL identity
(never ``--global``), one initial commit -- all deleted again afterwards. The
git-requiring tests skip when ``git`` is not on PATH; the serialization tests
run anywhere.
"""

import json
import os
import shutil
import subprocess

import pytest

from cards import repo
from cards.generations import CardOutcome, run_deck
from cards.store import (
    DeckStore,
    _outcome_from_dict,
    _outcome_to_dict,
    begin_run,
    collect_generation,
    make_card_committer,
    record_run,
    submit_generation,
)


MINIPROJECT = os.path.join("tests", "fixtures", "miniproject")
TEST_EMAIL = "morph-test@example.invalid"
TEST_NAME = "Morph Test"


def _code_block(body):
    return f"```python\n{body}\n```"


def _git(root, *args):
    return subprocess.run(
        ["git"] + list(args), cwd=root,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, check=True, timeout=30).stdout


def _revision(root, subject):
    """The sha of the one commit with this exact subject line."""
    matches = [line.split(" ", 1)[0] for line in
               _git(root, "log", "--format=%H %s").splitlines()
               if line.split(" ", 1)[1:] == [subject]]
    assert len(matches) == 1, f"no single commit titled {subject!r}"
    return matches[0]


def add_card(store, custom_id, target, **meta):
    """Append one card to the backlog (the shape ``/card`` writes)."""
    instruction = meta.pop("instruction", "do it")
    store.add_card({
        "custom_id": custom_id,
        "meta": {"intent": meta.pop("intent", "generate"),
                 "target": target, **meta},
        "instruction": instruction,
    })


class _FakeBatchBackend:
    """A scripted submit/status/collect backend (the tests/test_run_git.py idiom).

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


def run_split_step(store, root, backend, **kwargs):
    """Drive one deck through /submit + /collect until the run is done."""
    notes = []
    while True:
        submitted = submit_generation(
            store, backend, root=root, backend_label="fake", log=notes.append)
        if submitted.done:
            break
        for _ in range(200):
            collected = collect_generation(
                store, backend, root=root, poll_interval=0,
                log=notes.append, **kwargs)
            if not collected.in_progress:
                break
        else:
            pytest.fail("collect never completed")
        if collected.phase == "done":
            break
    return notes


# -- fixtures: tests/test_run_git.py's repository idiom, as pytest ------------


@pytest.fixture
def store(tmp_path):
    """A store over a plain directory -- all the serialization tests need."""
    return DeckStore(project_root=str(tmp_path))


@pytest.fixture
def git_root(tmp_path):
    """A throwaway git repository holding a copy of the miniproject fixture.

    ``_GitCase.setUp`` / ``_init_repo`` from ``tests/test_run_git.py``, as a
    fixture: a local identity (never ``--global``), ``init.defaultBranch`` and
    ``commit.gpgsign`` pinned so the assertions do not depend on the machine's
    git config, one commit holding the project. ``tmp_path`` removes it all.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")
    root = tmp_path / "miniproject"
    shutil.copytree(MINIPROJECT, root,
                    ignore=shutil.ignore_patterns("node_modules"))
    root = str(root)
    _git(root, "-c", "init.defaultBranch=main", "init", "-q", ".")
    _git(root, "config", "user.email", TEST_EMAIL)
    _git(root, "config", "user.name", TEST_NAME)
    _git(root, "config", "commit.gpgsign", "false")
    with open(os.path.join(root, ".gitignore"), "w", encoding="utf-8") as handle:
        handle.write("*.pyc\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "the project before any run")
    return root


@pytest.fixture
def git_store(git_root):
    return DeckStore(project_root=git_root)


# -- serialization ------------------------------------------------------------


def test_round_trip_carries_both_fields():
    """_outcome_to_dict writes both fields; _outcome_from_dict reads both back."""
    outcome = CardOutcome(
        "card-a", "written", paths=["gen_a.py"],
        commit="0123456789abcdef",
        diffstat=[{"path": "gen_a.py", "insertions": 3, "deletions": 1}],
    )

    data = _outcome_to_dict(outcome)
    assert data["commit"] == "0123456789abcdef"
    assert data["diffstat"] == [
        {"path": "gen_a.py", "insertions": 3, "deletions": 1}]

    read_back = _outcome_from_dict(data)
    assert read_back.commit == "0123456789abcdef"
    assert read_back.diffstat == [
        {"path": "gen_a.py", "insertions": 3, "deletions": 1}]
    # The fields the outcome had all along keep round-tripping too.
    assert read_back.custom_id == "card-a"
    assert read_back.status == "written"
    assert read_back.paths == ["gen_a.py"]

    # And a card that never became a commit keeps both None through the trip.
    plain = _outcome_from_dict(_outcome_to_dict(CardOutcome("card-b", "failed")))
    assert plain.commit is None
    assert plain.diffstat is None


def test_a_dict_written_before_the_fields_existed_still_loads():
    # Exactly the shape an older state.json / report.json holds: no "commit",
    # no "diffstat". It must load as the not-a-commit it was -- not raise, and
    # not invent values.
    legacy = {
        "custom_id": "card-a",
        "status": "written",
        "paths": ["gen_a.py"],
        "reason": None,
        "attempts": 1,
        "winning_variant": None,
        "acceptance_output": None,
    }

    outcome = _outcome_from_dict(legacy)

    assert outcome.status == "written"
    assert outcome.commit is None
    assert outcome.diffstat is None


def test_a_state_file_written_before_the_fields_existed_still_loads(store):
    # The same guarantee one level up, through the store: a state.json on disk,
    # as a version of the code from before the two fields would have written it.
    os.makedirs(store.morph_dir)
    with open(store.state_path, "w", encoding="utf-8") as handle:
        json.dump({
            "phase": "done",
            "generation_index": 1,
            "generations": [["card-a"]],
            "outcomes": {
                "card-a": {
                    "custom_id": "card-a",
                    "status": "written",
                    "paths": ["gen_a.py"],
                    "reason": None,
                    "attempts": 1,
                    "winning_variant": None,
                    "acceptance_output": None,
                },
            },
        }, handle)

    outcome = store.load_outcomes()["card-a"]
    assert outcome.status == "written"
    assert outcome.commit is None
    assert outcome.diffstat is None


# -- the committer fills the fields -------------------------------------------


def test_after_a_real_commit_the_hook_fills_both_fields(git_root, git_store):
    add_card(git_store, "card-a", "gen_a.py", context_slice=["util.py"],
             instruction="make a")
    cards = git_store.load_cards()
    state = begin_run(git_store, cards, root=git_root, backend_label="fake",
                      log=lambda _line: None)
    hook = make_card_committer(git_root, state, lambda _line: None)

    with open(os.path.join(git_root, "gen_a.py"), "w", encoding="utf-8") as handle:
        handle.write("A = 1\n")
    outcome = CardOutcome("card-a", "written",
                          paths=[os.path.join(git_root, "gen_a.py")])

    hook(cards[0], outcome)

    # Both read back out of the repository -- the sha names a commit git
    # actually holds, and the diffstat is what git says that commit changed --
    # not out of what the hook was handed.
    assert outcome.commit == _revision(git_root, "morph card-a: gen_a.py")
    assert outcome.diffstat == [
        {"path": "gen_a.py", "insertions": 1, "deletions": 0}]


def test_a_failing_diffstat_leaves_the_card_accepted_with_both_fields_none(
        git_root, git_store, monkeypatch):
    add_card(git_store, "card-a", "gen_a.py", context_slice=["util.py"],
             instruction="make a")
    cards = git_store.load_cards()
    state = begin_run(git_store, cards, root=git_root, backend_label="fake",
                      log=lambda _line: None)
    notes = []
    hook = make_card_committer(git_root, state, notes.append)

    def unreadable(_root, _sha):
        raise repo.GitError("numstat unavailable")

    monkeypatch.setattr(repo, "diffstat", unreadable)

    with open(os.path.join(git_root, "gen_a.py"), "w", encoding="utf-8") as handle:
        handle.write("A = 1\n")
    outcome = CardOutcome("card-a", "written",
                          paths=[os.path.join(git_root, "gen_a.py")])

    # Must not raise: the card is accepted and on disk, and bookkeeping may not
    # undo that.
    hook(cards[0], outcome)

    assert outcome.status == "written"
    assert os.path.exists(os.path.join(git_root, "gen_a.py"))
    assert outcome.commit is None
    assert outcome.diffstat is None
    # The commit itself went through -- only its measurement is missing.
    assert _revision(git_root, "morph card-a: gen_a.py")
    # ...and the hook said so in exactly one line, not a sprawl.
    said = [line for line in notes if "card-a" in line]
    assert len(said) == 1
    assert "committed" in said[0]
    assert "diffstat" in said[0]


# -- both run routes persist the fields ---------------------------------------


def test_the_split_step_route_persists_both_fields_to_disk(git_root, git_store):
    # /submit -> /collect -> _store_outcomes -> state.json: the hook mutates the
    # outcome object the route is holding, and the route's own save carries the
    # fields to disk. No plumbing beyond the hook's writes.
    add_card(git_store, "card-a", "gen_a.py", context_slice=["util.py"],
             instruction="make a")
    backend = _FakeBatchBackend(scripts={"card-a": _code_block("A = 1")})

    run_split_step(git_store, git_root, backend)

    outcome = git_store.load_outcomes()["card-a"]
    assert outcome.status == "written"
    assert outcome.commit == _revision(git_root, "morph card-a: gen_a.py")
    assert outcome.diffstat == [
        {"path": "gen_a.py", "insertions": 1, "deletions": 0}]
    # On disk, not merely in the returned object: the raw state file is what a
    # restarted session and the /deck view read.
    with open(git_store.state_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    assert raw["outcomes"]["card-a"]["commit"] == outcome.commit
    assert raw["outcomes"]["card-a"]["diffstat"] == outcome.diffstat
    # The run archive is written from the same outcome dicts, so it has them.
    deck_id = git_store.load_state()["deck_id"]
    with open(os.path.join(git_store.run_dir(deck_id), "report.json"),
              encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["outcomes"]["card-a"]["commit"] == outcome.commit
    assert report["outcomes"]["card-a"]["diffstat"] == outcome.diffstat


def test_the_nightly_route_persists_both_fields_to_disk(git_root, git_store):
    # run_deck -> DeckResult -> record_run: the hook fires inside run_deck, the
    # mutated outcomes travel out on the result, and record_run writes them.
    add_card(git_store, "card-a", "gen_a.py", context_slice=["util.py"],
             instruction="make a")
    cards = git_store.load_cards()
    state = begin_run(git_store, cards, root=git_root, backend_label="fake",
                      log=lambda _line: None)
    backend = _FakeBatchBackend(scripts={"card-a": _code_block("A = 1")})

    result = run_deck(cards, backend, root=git_root, poll_interval=0,
                      log=lambda _line: None,
                      on_accepted=make_card_committer(git_root, state,
                                                      lambda _line: None))
    record_run(git_store, result, backend_label="fake", root=git_root,
               run_state=state, log=lambda _line: None)

    outcome = git_store.load_outcomes()["card-a"]
    assert outcome.status == "written"
    assert outcome.commit == _revision(git_root, "morph card-a: gen_a.py")
    assert outcome.diffstat == [
        {"path": "gen_a.py", "insertions": 1, "deletions": 0}]
    with open(git_store.state_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    assert raw["outcomes"]["card-a"]["commit"] == outcome.commit
    assert raw["outcomes"]["card-a"]["diffstat"] == outcome.diffstat
    deck_id = git_store.load_state()["deck_id"]
    with open(os.path.join(git_store.run_dir(deck_id), "report.json"),
              encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["outcomes"]["card-a"]["commit"] == outcome.commit
    assert report["outcomes"]["card-a"]["diffstat"] == outcome.diffstat
