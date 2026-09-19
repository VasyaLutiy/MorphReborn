"""
Tests for the runner job (``.github/workflows/nightly-deck.yml``).

What a test can prove here and what it cannot. It cannot prove the job runs:
that is settled by a runner, once, by a human. It CAN pin the handful of
decisions whose absence turns a two-hour run into a silent loss, and every
assertion below names one that has already cost something:

* the token is read-only by default in this repository, so a job without
  ``contents: write`` finishes the whole deck and then fails to push it;
* ``ubuntu-latest`` ships Python 3.12, where ``pkg_resources`` -- imported
  unconditionally by ``flows/morph.py`` -- is gone, so an unpinned job dies on
  import;
* the acceptance commands of every card are written as ``venv/bin/python ...``
  by the deck's author, so the environment must sit at that exact path;
* the job's own limit must stay under the six-hour runner limit, because a job
  killed by the runner archives nothing.

The YAML is parsed when PyYAML is available and read as text otherwise: the
project does not depend on PyYAML, and a criterion that silently does nothing
when an optional import is missing is worse than no criterion.
"""

import os
import re
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "nightly-deck.yml")


class WorkflowTextTest(unittest.TestCase):
    """Assertions that hold whether or not a YAML parser is installed."""

    @classmethod
    def setUpClass(cls):
        with open(WORKFLOW, encoding="utf-8") as handle:
            cls.text = handle.read()

    def test_the_file_exists_and_is_not_empty(self):
        self.assertTrue(self.text.strip())

    def test_the_runner_may_write_the_branch(self):
        self.assertRegex(self.text, r"permissions:\s*\n\s*contents:\s*write")

    def test_python_is_pinned(self):
        match = re.search(r'python-version:\s*"?([\d.]+)"?', self.text)
        self.assertIsNotNone(match, "no python-version in the job")
        self.assertTrue(match.group(1).startswith("3.9"),
                        f"pinned to {match.group(1)}, baseline is 3.9")

    def test_the_environment_sits_where_acceptances_expect_it(self):
        self.assertIn("python -m venv venv", self.text)
        self.assertIn("venv/bin/mrph run", self.text)

    def test_the_deck_arrives_as_a_file_from_the_repository(self):
        self.assertIn("deck add --file", self.text)
        self.assertIn("inputs.deck", self.text)

    def test_the_run_carries_every_budget_limit(self):
        # Все три предела, а не один: колода, зациклившаяся на перегенерациях,
        # упирается в max_regenerations раньше, чем в часы.
        for flag in ("--max-cards", "--max-regenerations", "--deadline"):
            self.assertIn(flag, self.text)

    def test_the_nightly_run_starts_itself(self):
        # Колода называется ночной; без расписания она ночная только на словах.
        self.assertIn("schedule:", self.text)
        # Комментарии между ключом и элементом списка допустимы, поэтому ищем
        # само расписание, а не соседство строк.
        self.assertRegex(self.text, r"-\s*cron:\s*[\"']?[\d*/ ,-]+[\"']?")

    def test_every_input_has_a_literal_fallback(self):
        # На событии schedule контекст inputs ПУСТ. Значение без литерала
        # приедет пустой строкой, и ночной запуск умрёт на разборе аргументов,
        # тогда как ручной диспатч будет работать -- отказ, который видно
        # только ночью.
        for name in ("deck", "processor", "max_cards", "max_regenerations",
                     "deadline"):
            match = re.search(r"inputs\." + name + r"\s*\|\|", self.text)
            self.assertIsNotNone(
                match, f"inputs.{name} используется без литерального фолбэка")

    def test_pytest_is_installed_where_acceptances_look_for_it(self):
        # Каждая приёмка карты -- это "venv/bin/python -m pytest ...". Раннер
        # без pytest в этом venv проваливает КАЖДУЮ карту, и структурная
        # проверка файла этого не видит.
        self.assertRegex(self.text, r"venv/bin/python -m pip install[^\n]*pytest")

    def test_the_job_limit_stays_under_the_runner_limit(self):
        match = re.search(r"timeout-minutes:\s*(\d+)", self.text)
        self.assertIsNotNone(match, "the job declares no timeout-minutes")
        self.assertLess(int(match.group(1)), 360,
                        "a job killed by the runner archives nothing")

    def test_only_the_run_branch_is_pushed(self):
        self.assertIn("morph/*", self.text)
        self.assertNotRegex(self.text, r"git push origin\s+(HEAD:)?refs/heads/main\b")
        self.assertNotIn("git merge", self.text)

    def test_a_wrong_head_stops_the_job_instead_of_warning(self):
        # Предупреждение в логе ночного прогона не читает никто. Если HEAD не
        # ветка прогона -- job обязан упасть.
        tail = self.text[self.text.index("Запушить ветку"):]
        self.assertIn("exit 1", tail)

    def test_the_summary_line_is_the_one_morph_already_builds(self):
        # cards.notify.build_run_message renders it; the job only names the
        # channel. A second assembly here would be a second format to keep
        # correct.
        self.assertIn("MORPH_NOTIFY_CMD", self.text)
        self.assertIn("GITHUB_STEP_SUMMARY", self.text)

    def test_credentials_come_from_secrets_only(self):
        for line in self.text.splitlines():
            if "API_KEY" in line and "secrets." not in line:
                self.assertNotRegex(
                    line, r"API_KEY\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{16,}",
                    f"a credential looks hard-coded: {line.strip()}")

    def test_the_archive_leaves_as_an_artifact(self):
        self.assertIn("upload-artifact", self.text)
        self.assertIn(".morph/runs/**", self.text)


class WorkflowSchemaTest(unittest.TestCase):
    """The same file, read as YAML, when a parser happens to be installed."""

    def setUp(self):
        self.yaml = __import__("pytest").importorskip("yaml")
        with open(WORKFLOW, encoding="utf-8") as handle:
            self.document = self.yaml.safe_load(handle)

    def test_it_parses_into_a_job_with_steps(self):
        jobs = self.document["jobs"]
        self.assertEqual(1, len(jobs), "в файле ровно один job")
        only = next(iter(jobs.values()))
        self.assertTrue(only["steps"])

    def test_it_is_started_by_hand(self):
        # ``on`` is YAML's boolean True once parsed -- the well-known trap.
        trigger = self.document.get("on", self.document.get(True))
        self.assertIn("workflow_dispatch", trigger)
        self.assertIn("deck", trigger["workflow_dispatch"]["inputs"])

    def test_two_decks_do_not_race(self):
        self.assertIn("concurrency", self.document)
        self.assertFalse(self.document["concurrency"].get("cancel-in-progress"))


if __name__ == "__main__":
    unittest.main()
