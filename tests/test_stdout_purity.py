"""
Контрактные тесты на утверждения, которые проверка 18.09 опровергла.

Оба теста написаны ПО ТЕКСТУ КОНТРАКТА (`documentation/TASK_headless.md`), а не
по реализации, и оба падали до правки. Это важно: 583 теста существовали и не
видели ни одного из двух дефектов, потому что проверяли пути, не доходящие до
компиляции карты и не открывающие ветку.

* ``stdout`` — ровно один JSON-документ, **включая путь, где карта реально
  компилируется**. Ломал это `print()` в ``context_folder_dialog``: одна строка
  на каждый файл среза, прямо в stdout.
* Пустой бэклог — не прогон: ``submit`` не открывает ветку и не коммитит пустой
  архив. ``run`` на том же входе всегда был честным no-op.
"""

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from cards.store import DeckStore, submit_generation


def _card(custom_id, target, slice_):
    return {"custom_id": custom_id,
            "meta": {"intent": "generate", "target": target,
                     "context_slice": list(slice_)},
            "instruction": "сделай"}


class StdoutIsOneJsonDocumentTests(unittest.TestCase):
    """Контракт B: stdout парсится как JSON на ЛЮБОМ исходе любой подкоманды."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="morph-stdout-")
        self.previous = os.getcwd()
        with open(os.path.join(self.root, "README.md"), "w") as handle:
            handle.write("# проект\n")
        with open(os.path.join(self.root, "util.py"), "w") as handle:
            handle.write("A = 1\n")
        os.chdir(self.root)

    def tearDown(self):
        os.chdir(self.previous)
        shutil.rmtree(self.root, ignore_errors=True)

    def test_submit_stdout_parses_even_when_a_card_compiles(self):
        # Срез из двух файлов: до правки это давало две строки
        # "Folder context file: ..." в stdout ПЕРЕД JSON-ответом.
        from cards.cli import main

        DeckStore(".").add_card(
            _card("probe", "scratch_probe.py", ["README.md", "util.py"]))

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["submit", "--nogit"])

        payload = out.getvalue()
        try:
            json.loads(payload)
        except json.JSONDecodeError as error:
            self.fail(f"stdout не парсится как JSON ({error}); "
                      f"получено: {payload[:200]!r}")
        # Провайдера в тесте нет, поэтому ответ — ошибка; важен не код, а то,
        # что он один и в JSON.
        self.assertNotEqual(code, 0)

    def test_the_compiler_progress_goes_to_stderr(self):
        # То же утверждение на уровне компонента: сборка контекста пишет
        # прогресс, но не в stdout.
        from cards.compiler import compile_card
        from cards.schema import MorphCard

        card = MorphCard(custom_id="p", intent="generate",
                         target="out.py", instruction="x",
                         context_slice=["README.md", "util.py"])
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            compile_card(card, ".")

        self.assertEqual(out.getvalue(), "")
        self.assertIn("Folder context file", err.getvalue())


class EmptyDeckIsNotARunTests(unittest.TestCase):
    """Пустой бэклог: ни ветки, ни коммита, ни архива."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="morph-empty-")
        self.store = DeckStore(project_root=self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_submit_on_an_empty_backlog_opens_nothing(self):
        class _Unused:
            def submit(self, requests):
                raise AssertionError("бэкенд не должен быть вызван")

        result = submit_generation(self.store, _Unused(), root=self.root,
                                   use_git=True, log=lambda _line: None)

        self.assertFalse(result.submitted)
        self.assertTrue(result.done)
        # Главное: состояние прогона не заведено, ветки нет, архива нет.
        self.assertIsNone(self.store.load_state().get("branch"))
        self.assertFalse(os.path.exists(self.store.runs_dir))


if __name__ == "__main__":
    unittest.main()


class SubmitBudgetIsCappedTests(unittest.TestCase):
    """Отправка не имеет права съесть всю ночь.

    18.09 `mrph run` без `.env` повторял `Missing credentials` против всего
    шестичасового бюджета: ожидание устроено правильно, но ошибка, которая
    повторами не лечится, выглядела для него как обычный сетевой сбой.
    Классификация исключений провайдера здесь не спасает — в openai SDK
    `OpenAIError` является базовым и для обрыва связи, и для отсутствия ключа,
    так что по классу их не различить. Поэтому предохранитель структурный:
    отправка получает свой короткий срез общего бюджета.
    """

    class _Clock:
        def __init__(self, start=1000.0):
            self.time = float(start)
            self.sleeps = []

        def now(self):
            return self.time

        def sleep(self, delay):
            self.sleeps.append(delay)
            self.time += delay

    class _AlwaysRefuses:
        """Бэкенд, который отвечает одинаково и навсегда."""

        def __init__(self):
            self.calls = 0

        def submit(self, requests):
            self.calls += 1
            raise RuntimeError("Missing credentials. Please pass an api_key")

        def status(self, batch_id):
            return "completed"

    def test_submit_gives_up_inside_its_own_budget_not_the_whole_wait(self):
        from cards.cli_wait import ResilientBackend, WaitTimeout

        clock = self._Clock()
        backend = self._AlwaysRefuses()
        resilient = ResilientBackend(
            backend, timeout=6 * 3600.0, submit_timeout=120.0,
            sleep=clock.sleep, now=clock.now, log=lambda _line: None)

        with self.assertRaises(WaitTimeout):
            resilient.submit([{"custom_id": "x"}])

        spent = clock.time - 1000.0
        self.assertLessEqual(spent, 130.0,
                             f"отправка потратила {spent:.0f}с вместо ~120с")
        # И общий бюджет ожидания при этом почти цел: ночь не потеряна.
        self.assertGreater(resilient.remaining, 6 * 3600.0 - 200.0)

    def test_polling_still_gets_the_whole_budget(self):
        # Обратная сторона: опрос — это то, что законно длится часами,
        # и короткий колпак отправки его не трогает.
        from cards.cli_wait import ResilientBackend

        clock = self._Clock()
        resilient = ResilientBackend(
            self._AlwaysRefuses(), timeout=6 * 3600.0, submit_timeout=120.0,
            sleep=clock.sleep, now=clock.now, log=lambda _line: None)
        self.assertEqual(resilient.status("batch-1"), "completed")
        self.assertGreater(resilient.remaining, 6 * 3600.0 - 1.0)


class HeadlessCanUnstickItselfTests(unittest.TestCase):
    """Р4: у безголовой поверхности должен быть свой выход из клина.

    До правки `mrph deck` знал только `add`, `check`, `status`, а тексты ошибок
    советовали `/deck reset` — команду, которая живёт только в REPL. Раннер под
    кроном переключиться в интерактив не может, поэтому единственным выходом
    было удалить `.morph/state.json` руками.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="morph-unstick-")
        self.previous = os.getcwd()
        os.chdir(self.root)

    def tearDown(self):
        os.chdir(self.previous)
        shutil.rmtree(self.root, ignore_errors=True)

    def test_deck_reset_discards_the_run_and_keeps_the_backlog(self):
        from cards.cli import main

        store = DeckStore(".")
        store.add_card(_card("a", "a.py", ["a.py"]))
        state = store.load_state()
        state.update(phase="submitted", generations=[["a"]],
                     batch_id="batch-1", submitted_ids=["a"])
        store.save_state(state)

        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = main(["deck", "reset"])

        payload = json.loads(out.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(payload["reset"])
        self.assertEqual(payload["phase_before"], "submitted")
        # Прогон сброшен, заказ на месте.
        self.assertEqual(DeckStore(".").load_state().get("phase"), "idle")
        self.assertEqual(len(DeckStore(".").load_cards()), 1)

    def test_deck_clear_empties_the_backlog(self):
        from cards.cli import main

        DeckStore(".").add_card(_card("a", "a.py", ["a.py"]))
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = main(["deck", "clear"])

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["cards_removed"], 1)
        self.assertEqual(DeckStore(".").load_cards(), [])

    def test_deck_clear_refuses_while_a_batch_is_in_flight(self):
        from cards.cli import main

        store = DeckStore(".")
        store.add_card(_card("a", "a.py", ["a.py"]))
        state = store.load_state()
        state.update(phase="submitted", batch_id="batch-1")
        store.save_state(state)

        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = main(["deck", "clear"])

        payload = json.loads(out.getvalue())
        self.assertEqual(code, 2)
        self.assertIn("deck reset", payload["error"]["message"])
        # Заказ не тронут: результаты летящего батча ещё могут на него лечь.
        self.assertEqual(len(DeckStore(".").load_cards()), 1)
