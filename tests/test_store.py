"""
Phase 5 tests: the ``.morph/`` store (backlog + run-state persistence).

Pure and offline: every test runs against a throwaway ``project_root`` tempdir,
so nothing touches the repo tree and no ``.morph`` directory is left behind. The
split-step submit/collect execution is exercised end to end in
``tests/test_nightly_e2e.py``; here we cover persistence, validation and the
display-status derivation in isolation.
"""

import json
import os
import shutil
import tempfile
import unittest

from cards.deck import DeckError
from cards.generations import CardOutcome, DeckResult
from cards.schema import CardError, MorphCard
from cards.store import (
    DeckStore,
    StoreError,
    build_deck_status,
    card_to_dict,
    record_run,
    recover_orphaned_local_batch,
)


def _card_dict(custom_id, target, **meta):
    instruction = meta.pop("instruction", "do it")
    return {"custom_id": custom_id,
            "meta": {"intent": meta.pop("intent", "generate"), "target": target, **meta},
            "instruction": instruction}


class DeckStoreBacklogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-store-")
        self.store = DeckStore(project_root=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_cards_empty_when_no_file(self):
        self.assertEqual(self.store.load_cards(), [])
        self.assertFalse(os.path.exists(self.store.deck_path))

    def test_add_card_roundtrips_all_fields(self):
        self.store.add_card(_card_dict(
            "gen-a", "a.py",
            context_slice=["util.py"], acceptance="pytest -q",
            model="claude-opus-4", variants=3, depends_on=[],
            instruction="make a"))
        # Reload through a *fresh* store instance: the backlog is really on disk.
        reloaded = DeckStore(project_root=self.tmp).load_cards()
        self.assertEqual(len(reloaded), 1)
        card = reloaded[0]
        self.assertEqual(card.custom_id, "gen-a")
        self.assertEqual(card.target, "a.py")
        self.assertEqual(card.context_slice, ["util.py"])
        self.assertEqual(card.acceptance, "pytest -q")
        self.assertEqual(card.model, "claude-opus-4")
        self.assertEqual(card.variants, 3)
        self.assertEqual(card.instruction, "make a")

    def test_add_card_persists_valid_json_list(self):
        self.store.add_card(_card_dict("gen-a", "a.py"))
        with open(self.store.deck_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        self.assertIsInstance(raw, list)
        self.assertEqual(raw[0]["custom_id"], "gen-a")

    def test_add_card_rejects_malformed_card_and_saves_nothing(self):
        with self.assertRaises(CardError):
            self.store.add_card(_card_dict("gen-a", "a.py", intent="frobnicate"))
        self.assertEqual(self.store.load_cards(), [])
        self.assertFalse(os.path.exists(self.store.deck_path))

    def test_add_card_rejects_duplicate_custom_id(self):
        self.store.add_card(_card_dict("dup", "a.py"))
        with self.assertRaises(DeckError):
            self.store.add_card(_card_dict("dup", "b.py"))
        # The first card is still the only one stored.
        self.assertEqual([c.custom_id for c in self.store.load_cards()], ["dup"])

    def test_add_card_rejects_dangling_dependency(self):
        with self.assertRaises(DeckError):
            self.store.add_card(_card_dict("gen-b", "b.py", depends_on=["missing"]))
        self.assertEqual(self.store.load_cards(), [])

    def test_add_cards_fragment_all_or_nothing(self):
        added = self.store.add_cards([
            _card_dict("frag-a", "a.py"),
            _card_dict("frag-b", "b.py", depends_on=["frag-a"]),
        ])
        self.assertEqual([c.custom_id for c in added], ["frag-a", "frag-b"])
        self.assertEqual(len(self.store.load_cards()), 2)

    def test_add_cards_fragment_rejected_leaves_backlog_unchanged(self):
        self.store.add_card(_card_dict("keep", "keep.py"))
        with self.assertRaises(DeckError):
            self.store.add_cards([
                _card_dict("ok", "ok.py"),
                _card_dict("keep", "clash.py"),  # duplicate against existing
            ])
        self.assertEqual([c.custom_id for c in self.store.load_cards()], ["keep"])

    def test_remove_card(self):
        self.store.add_card(_card_dict("a", "a.py"))
        self.store.add_card(_card_dict("b", "b.py"))
        self.store.remove_card("a")
        self.assertEqual([c.custom_id for c in self.store.load_cards()], ["b"])

    def test_remove_missing_card_raises(self):
        with self.assertRaises(StoreError):
            self.store.remove_card("nope")

    def test_remove_card_with_dependents_is_refused(self):
        self.store.add_card(_card_dict("a", "a.py"))
        self.store.add_card(_card_dict("b", "b.py", depends_on=["a"]))
        with self.assertRaises(DeckError):
            self.store.remove_card("a")
        self.assertEqual(len(self.store.load_cards()), 2)

    def test_clear_empties_backlog(self):
        self.store.add_card(_card_dict("a", "a.py"))
        self.store.clear()
        self.assertEqual(self.store.load_cards(), [])

    def test_card_to_dict_is_reloadable(self):
        card = MorphCard(custom_id="x", intent="patch", target="x.py",
                         instruction="fix", context_slice=["x.py"], variants=2)
        rebuilt = MorphCard.from_dict(card_to_dict(card))
        self.assertEqual(rebuilt, card)

    def test_a_changeset_card_round_trips_as_a_changeset(self):
        # The backlog is where a card waits between CLI sessions: a card that
        # writes a SET must not come back as a single-target one.
        card = MorphCard(custom_id="x", intent="patch",
                         targets=["x.py", "tests/test_x.py"],
                         instruction="fix", acceptance="pytest -q")
        as_dict = card_to_dict(card)
        self.assertEqual(as_dict["meta"]["targets"], ["x.py", "tests/test_x.py"])
        self.assertNotIn("target", as_dict["meta"])
        self.assertEqual(MorphCard.from_dict(as_dict), card)


class DeckStoreStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-state-")
        self.store = DeckStore(project_root=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_default_state_when_no_file(self):
        state = self.store.load_state()
        self.assertEqual(state["phase"], "idle")
        self.assertEqual(state["generation_index"], 0)
        self.assertEqual(state["generations"], [])
        self.assertIsNone(state["batch_id"])
        self.assertEqual(state["outcomes"], {})

    def test_state_survives_a_fresh_store_instance(self):
        # This is the whole point of state.json: a CLI restart resumes the run.
        state = self.store.load_state()
        state["phase"] = "submitted"
        state["generation_index"] = 1
        state["generations"] = [["a"], ["b"]]
        state["batch_id"] = "batch-xyz"
        state["submitted_ids"] = ["b"]
        self.store.save_state(state)

        resumed = DeckStore(project_root=self.tmp).load_state()
        self.assertEqual(resumed["phase"], "submitted")
        self.assertEqual(resumed["generation_index"], 1)
        self.assertEqual(resumed["generations"], [["a"], ["b"]])
        self.assertEqual(resumed["batch_id"], "batch-xyz")
        self.assertEqual(resumed["submitted_ids"], ["b"])

    def test_load_state_merges_missing_keys_over_defaults(self):
        # A partial (older) state file still loads with every key present.
        self.store._ensure_dir()
        with open(self.store.state_path, "w", encoding="utf-8") as handle:
            json.dump({"phase": "done"}, handle)
        state = self.store.load_state()
        self.assertEqual(state["phase"], "done")
        self.assertEqual(state["generations"], [])   # default filled in

    def test_reset_state_removes_file(self):
        self.store.save_state(self.store.load_state())
        self.assertTrue(os.path.exists(self.store.state_path))
        self.store.reset_state()
        self.assertFalse(os.path.exists(self.store.state_path))
        # Backlog is untouched by a state reset.
        self.assertEqual(self.store.load_cards(), [])


class DeckStatusDerivationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-status-")
        self.store = DeckStore(project_root=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_deck(self):
        view = build_deck_status(self.store)
        self.assertTrue(view.empty)
        self.assertEqual(view.card_status, [])

    def test_all_pending_before_any_run(self):
        self.store.add_card(_card_dict("a", "a.py"))
        self.store.add_card(_card_dict("b", "b.py", depends_on=["a"]))
        view = build_deck_status(self.store)
        self.assertFalse(view.empty)
        self.assertEqual(dict(view.card_status), {"a": "pending", "b": "pending"})
        # With no run started the composition is a fresh preview.
        self.assertEqual(view.generations, [["a"], ["b"]])

    def test_in_flight_cards_reflect_submitted_state(self):
        self.store.add_card(_card_dict("a", "a.py"))
        self.store.add_card(_card_dict("b", "b.py"))
        state = self.store.load_state()
        state["phase"] = "submitted"
        state["generations"] = [["a", "b"]]
        state["submitted_ids"] = ["a", "b"]
        self.store.save_state(state)

        view = build_deck_status(self.store)
        self.assertEqual(dict(view.card_status), {"a": "in_flight", "b": "in_flight"})
        self.assertEqual(view.phase, "submitted")

    def test_recorded_outcomes_win_over_in_flight(self):
        self.store.add_card(_card_dict("a", "a.py"))
        self.store.add_card(_card_dict("b", "b.py"))
        state = self.store.load_state()
        state["phase"] = "idle"
        state["generations"] = [["a", "b"]]
        state["outcomes"] = {
            "a": {"custom_id": "a", "status": "written", "paths": ["a.py"],
                  "reason": None, "attempts": 1, "winning_variant": None,
                  "acceptance_output": None},
            "b": {"custom_id": "b", "status": "failed", "paths": [],
                  "reason": None, "attempts": 3, "winning_variant": None,
                  "acceptance_output": "boom"},
        }
        self.store.save_state(state)

        view = build_deck_status(self.store)
        self.assertEqual(dict(view.card_status), {"a": "written", "b": "failed"})
        outcomes = self.store.load_outcomes()
        self.assertEqual(outcomes["b"].attempts, 3)
        self.assertEqual(outcomes["b"].acceptance_output, "boom")


class RecordRunTests(unittest.TestCase):
    """``/nightly`` ran the deck in memory; the run must end up in state.json.

    The nightly path used to throw its :class:`DeckResult` away, so the very next
    ``/deck`` reported ``idle`` with every card ``pending`` for morphs that were
    already on disk. :func:`record_run` leaves exactly what the split-step
    ``/collect`` path leaves.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-record-")
        self.store = DeckStore(project_root=self.tmp)
        self.store.add_card(_card_dict("a", "a.py"))
        self.store.add_card(_card_dict("b", "b.py", depends_on=["a"]))
        self.store.add_card(_card_dict("c", "c.py", depends_on=["b"]))
        self.result = DeckResult(
            outcomes={
                "a": CardOutcome("a", "written", paths=["a.py"]),
                "b": CardOutcome("b", "failed", attempts=3,
                                 acceptance_output="boom"),
                "c": CardOutcome("c", "skipped", reason="b"),
            },
            generations=[["a"], ["b"], ["c"]],
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_deck_status_reports_the_recorded_outcomes(self):
        record_run(self.store, self.result)
        # Read back through a fresh store: the run really is on disk.
        view = build_deck_status(DeckStore(project_root=self.tmp))
        self.assertEqual(dict(view.card_status),
                         {"a": "written", "b": "failed", "c": "skipped"})
        self.assertEqual(view.phase, "done")
        self.assertEqual(view.generations, [["a"], ["b"], ["c"]])

    def test_outcome_detail_survives(self):
        record_run(self.store, self.result)
        outcomes = DeckStore(project_root=self.tmp).load_outcomes()
        self.assertEqual(outcomes["a"].paths, ["a.py"])
        self.assertEqual(outcomes["b"].attempts, 3)
        self.assertEqual(outcomes["b"].acceptance_output, "boom")
        self.assertEqual(outcomes["c"].reason, "b")

    def test_nothing_is_left_in_flight(self):
        record_run(self.store, self.result, backend_label="node-a")
        state = self.store.load_state()
        self.assertEqual(state["phase"], "done")
        self.assertEqual(state["generation_index"], 3)
        self.assertIsNone(state["batch_id"])
        self.assertEqual(state["submitted_ids"], [])
        self.assertEqual(state["backend_label"], "node-a")

    def test_a_nightly_run_replaces_an_earlier_split_step_run(self):
        # A stale in-flight state from an abandoned /submit must not survive a
        # nightly pass, which computes its own composition from the backlog.
        state = self.store.load_state()
        state["phase"] = "submitted"
        state["batch_id"] = "local-deadbeef"
        state["submitted_ids"] = ["a"]
        state["generations"] = [["a", "b", "c"]]
        self.store.save_state(state)

        record_run(self.store, self.result)

        state = self.store.load_state()
        self.assertEqual(state["phase"], "done")
        self.assertIsNone(state["batch_id"])
        self.assertEqual(state["generations"], [["a"], ["b"], ["c"]])


class OrphanedLocalBatchTests(unittest.TestCase):
    """A local batch dies with the process that fired it; the run must not.

    Restarting the CLI between ``/submit`` and ``/collect`` on a local backend
    left the deck wedged in phase ``"submitted"`` with no way out. Recovery is
    strictly for LOCAL batch ids -- a cloud batch is still running on a
    provider's server and is genuinely collectable later.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="morph-orphan-")
        self.store = DeckStore(project_root=self.tmp)
        self.store.add_card(_card_dict("a", "a.py"))
        self.store.add_card(_card_dict("b", "b.py"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _submitted(self, batch_id, **extra):
        state = self.store.load_state()
        state["phase"] = "submitted"
        state["generations"] = [["a", "b"]]
        state["batch_id"] = batch_id
        state["backend_label"] = "node-a"
        state["submitted_ids"] = ["a", "b"]
        state.update(extra)
        self.store.save_state(state)

    def test_local_batch_is_recovered_to_idle_with_cards_pending(self):
        self._submitted("local-0123456789abcdef")
        self.assertTrue(recover_orphaned_local_batch(self.store))

        state = self.store.load_state()
        self.assertEqual(state["phase"], "idle")
        self.assertIsNone(state["batch_id"])
        self.assertIsNone(state["backend_label"])
        self.assertEqual(state["submitted_ids"], [])

        view = build_deck_status(self.store)
        self.assertEqual(dict(view.card_status), {"a": "pending", "b": "pending"})

    def test_cloud_batch_is_never_touched(self):
        self._submitted("batch_abc123")
        self.assertFalse(recover_orphaned_local_batch(self.store))

        state = self.store.load_state()
        self.assertEqual(state["phase"], "submitted")
        self.assertEqual(state["batch_id"], "batch_abc123")
        self.assertEqual(state["submitted_ids"], ["a", "b"])
        view = build_deck_status(self.store)
        self.assertEqual(dict(view.card_status), {"a": "in_flight", "b": "in_flight"})

    def test_recovery_keeps_composition_and_earlier_outcomes(self):
        # A second-generation batch orphaned: generation 1's morphs stay written.
        self._submitted(
            "local-cafe",
            generations=[["a"], ["b"]],
            generation_index=1,
            submitted_ids=["b"],
            outcomes={"a": {"custom_id": "a", "status": "written",
                            "paths": ["a.py"], "reason": None, "attempts": 1,
                            "winning_variant": None, "acceptance_output": None}},
        )
        self.assertTrue(recover_orphaned_local_batch(self.store))

        state = self.store.load_state()
        self.assertEqual(state["generations"], [["a"], ["b"]])
        self.assertEqual(state["generation_index"], 1)
        view = build_deck_status(self.store)
        self.assertEqual(dict(view.card_status), {"a": "written", "b": "pending"})

    def test_nothing_to_recover_when_no_batch_is_in_flight(self):
        self.assertFalse(recover_orphaned_local_batch(self.store))   # fresh state
        self.store.save_state(dict(self.store.load_state(), phase="done"))
        self.assertFalse(recover_orphaned_local_batch(self.store))



if __name__ == "__main__":
    unittest.main()
