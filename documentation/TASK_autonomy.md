# Прогон без машины оператора

> Спека по [`TASK_TEMPLATE.md`](./TASK_TEMPLATE.md). Шесть препятствий между
> сегодняшним прогоном (человек сидит у макбука и смотрит) и ночным прогоном на
> чужой машине: четыре боли из [`Head_Pains.md`](../Head_Pains.md) и два пункта
> эпика #1 из [`ROADMAP.md`](./ROADMAP.md).
>
> Разбивки на карты здесь нет — см. §6.

---

## 1. Зачем это

Автономность упирается не в идею, а в шесть мест, каждое с замером.

**2.15 — диагноз упавшей попытки не доживает до архива.** `CardOutcome.earlier_failures`
заполняется (`cards/generations.py:1303`), но `_outcome_to_dict` его не
сериализует, а через него пишутся **оба** артефакта — `state.json` и
`report.json`. *Улика 19.09:* карта `run-budget-wiring` прошла со второй попытки
(`attempts: 2`, вариант `.r1`), и почему упала первая — узнать уже нельзя.
Контур самопочинки диагноз получил и потратил; оператор утром не получил
ничего. Это приоритет №1 сводки `Head_Pains`.

**2.16 — бюджет судится только на границе поколения.** `run_deck` спрашивает
`BudgetLedger` один раз на поколение, до сабмита (`cards/generations.py:1117`), и
никогда внутри пути перегенераций. Замер прямой: три попытки внутри поколения
отработают при любом пределе. При `max_regenerations` карт 20 и трёх попытках
это до 60 неучтённых обращений; `deadline_seconds`, единственный предел, ради
которого ночной прогон вообще снабжают бюджетом, внутри поколения не смотрят
вовсе — то есть прогон, ушедший в перегенерации на исходе шестичасового окна
раннера, будет убит раннером, а не собой.

**2.17 — безголовый путь не умеет пул локальных узлов.** `resolve_backend`
принимает один идентификатор, `registry.batch(id)` для локального узла отдаёт
`LocalBatchBackend(self, [id])` — один слот. *Улика 19.09:* пул существует только
в REPL (`/submit @all`), и `mrph run --processor qwen` пошёл бы по картам
последовательно. Для фермы K80 из восьми узлов это восьмикратная потеря
конкурентности ровно на том пути, который единственный переживает обрыв.

**3.6 — реестр подхватывает облачный слот из окружения.** *Улика 19.09:* стенд
объявил восемь локальных узлов, а `registry.ids` вернул девять; девятым пришёл
`openai`, потому что в оболочке оператора выставлен `AZURE_OPENAI_API_KEY`, и
даже без `OPENAI_MODEL_NAME` — то есть слот, который при обращении к провайдеру
всё равно упал бы. `registry.batch_pool(registry.ids)` отправил бы часть карт к
модели, которой оператор не выбирал: ломает и правило «одна модель на колоду», и
счёт.

**Эпик #1, пункт 3 — `entry_points` вместо копирующего `scripts=`.** `setup.py`
ставит `scripts=["bin/mrph"]`, то есть установка **копирует** файл. После любой
самоправки установленная команда устаревает молча. Для машины, которая правит
себя, это мина: ночью на раннере выполнится не тот код, который лежит в дереве.

*Замер 19.09, настоящая установка в пустой `venv` (`pip install .`, 20 секунд):*
установленная команда **не работает вовсе**. `py_modules` перечисляет 13 модулей
из примерно тридцати, поэтому в `site-packages` приезжает `cards/` из **7 файлов
против 20**, и первый же безголовый вызов падает на
`ModuleNotFoundError: No module named 'cards.cli'`. Не импортируются 18 модулей,
включая `cards.cli*`, `cards.budget`, `cards.hazards`, `cards.repo`,
`cards.runs_view` и — транзитивно — `flows.morph`, то есть и REPL. Локально это
не видно: в рабочем `venv` пакет стоит в режиме develop, и всё разрешается из
дерева. Ни один из 674 тестов этого не ловит **по устройству** — они гоняют код
из дерева, а не установленную команду. Для раннера, который ставит проект с
нуля, это не косметика, а неработающий `mrph`.

**Эпик #1, пункт 7 — job на GitHub Actions.** Ночная колода не живёт на
засыпающем макбуке: cron его не разбудит, а уснувшая машина роняет соединения —
улика `ConnectionResetError` на 23-й минуте. Лимит job'а 6 часов покрывает
замеренную длительность колод (2–4 поколения по 20–45 минут).

Общая цена бездействия: ночной прогон сегодня невозможен, а разбор утреннего
провала невозможен вдвойне — диагноза в архиве нет.

## 2. Контракт: что должно стать правдой

### 2.1. Формы данных на ВХОД

Всё, что код обязан **построить** (а не просто вызвать), с адресом определения.

| форма | где определена | кто её строит |
|---|---|---|
| `CardOutcome` — датакласс, поля `custom_id, status, paths, reason, attempts, winning_variant, acceptance_output, earlier_failures, commit, diffstat` | `cards/generations.py:154-166` | тесты сериализации исхода; код пути перегенераций |
| словарь исхода (ключи `_outcome_to_dict`) | `cards/store.py:181-198` | тесты круга сериализации |
| `RunBudget(max_cards, max_regenerations, deadline_seconds)`, `BudgetLedger(budget, now=)`, `BudgetExceeded(limit, allowed, reached)`, `STATUS_BUDGET_EXCEEDED` | `cards/budget.py` | тесты бюджета в пути перегенераций |
| `MorphCard` — поля карты, допустимые `intent` | `cards/schema.py` | любой тест, которому нужна валидная карта |
| `ProcessorConfig(identifier, kind, params)`; `kind ∈ {llama_cpp, ollama, openai, anthropic, openrouter}`; `params` — ключи `endpoint_uri, model, api_key, base_url` | `processors/registry.py:59-66`, `KNOWN_TYPES` там же | тесты реестра и тесты пула |
| `ProcessorRegistry(configs: Dict[str, ProcessorConfig])` — конструктор берёт готовый словарь, порядок вставки = приоритет | `processors/registry.py:87-93` | тесты, которым нужен реестр без окружения |
| `LocalBatchBackend(registry, processor_ids)` | `processors/batch.py:709-740` | тесты пула |
| сигнатура `_run_retries(pending, index, total, root, backend, poll_interval, log, acceptance_timeout, max_regenerations, outcomes, blocked, on_accepted, batch_ids)`; `pending` — список пар `(MorphCard, AcceptanceResult|None)` | `cards/generations.py:1330-1344` | код и тесты бюджета в перегенерациях |

**Фикстуру морф-карты брать копией** из `.morph/runs/*/deck.json`, а не сочинять
(`TASK_TEMPLATE.md` §5).

### 2.2. Формы данных на ВЫХОД

1. **Словарь исхода.** Ровно одна сериализация — `_outcome_to_dict` /
   `_outcome_from_dict` в `cards/store.py`. Через них пишутся и `state.json`, и
   `report.json` (`cards/store.py:244` и `:840`); второй формы завести нельзя.
   К существующим ключам добавляется **один**: `"earlier_failures"` — строка или
   `null`. Чтение — через `.get("earlier_failures")`, чтобы файл, записанный до
   появления ключа, продолжал загружаться (`None`), как уже сделано для
   `commit` и `diffstat`.
2. **Исход карты, которую остановил бюджет внутри поколения**, по форме не
   отличается от остановленной на границе: `status == cards.budget.STATUS_BUDGET_EXCEEDED`,
   `reason == BudgetExceeded.reason` (то же предложение «`<limit>` exceeded:
   allowed `<A>`, reached `<R>`»). Новых статусов и новых причин не заводить.
3. **`resolve_backend`** возвращает `(backend, resolved_label)` — как сейчас. Для
   пула `resolved_label` — идентификаторы через `+` в порядке реестра
   (`"k80-a+k80-b"`), та же строка, что кладёт REPL (`flows/morph.py:832`).
4. **Файл job'а** — `.github/workflows/nightly-deck.yml`, единственный новый
   YAML. Ключи и шаги перечислены в §2.3.

### 2.3. Имена

**2.15.** Ключ словаря и поле — `earlier_failures`, оба места в
`cards/store.py`. Новых функций нет.

**2.16.** `cards/generations.py`:

- `process_generation` получает два новых **именованных параметра со значением
  по умолчанию**: `ledger: Optional[BudgetLedger] = None` и
  `regenerations_before: int = 0`, и передаёт их в `_run_retries`;
- `_run_retries` получает те же два, с теми же умолчаниями;
- перед сабмитом **каждой** попытки перегенерации (`attempt` считается с 1)
  вызывается `ledger.check(len(outcomes) + len(pending), regenerations_before + attempt)`;
- `run_deck` передаёт свой `ledger` и `regenerations_before=len(batch_ids) - generations_submitted`
  — то самое число уже потраченных батчей перегенерации, которое он уже считает
  для гейта на границе (`cards/generations.py:1096-1101`).

**2.17.** `cards/cli_backend.py`:

- константа `LOCAL_KINDS = ("llama_cpp", "ollama")`;
- функция `resolve_pool_ids(registry, label) -> List[str]`, добавляется в
  `__all__`;
- `resolve_backend` остаётся единственной точкой входа CLI и сохраняет свою
  сигнатуру `(label: Optional[str] = None) -> Tuple[BatchBackend, str]`.

**3.6.** `processors/registry.py`: правится `_add_legacy` (слот `openai`) и
`batch_pool`. Новых имён нет.

**Эпик #1.3.** Новый модуль в корне — `mrph_console.py`, единственная публичная
функция `main(argv: Optional[List[str]] = None) -> int`. `argv=None` означает
`sys.argv[1:]`: сгенерированный setuptools console-script зовёт `main()` **без
аргументов**, и умолчание — единственное место, где argv может быть прочитан.
Тело — сегодняшнее тело `bin/mrph`: `settings.load_settings()`, затем при
непустом argv `cards.cli.main(argv)`, при пустом — `flows.morph.MorphBot(None).run()`
и `return 0`. Импорт `flows` остаётся **внутри** ветки: безголовый путь не должен
грузить консольного бота.

В `setup.py`:

- `entry_points={"console_scripts": ["mrph = mrph_console:main"]}`;
- `scripts=["bin/mrph"]` удалён;
- **пакеты объявлены целиком, а не списком из 13 модулей**: `packages` покрывает
  `cards`, `processors`, `flows`, `morph_mcp`, а `py_modules` — корневые модули
  `settings`, `scheduler`, `llm_dialog`, `context_folder_dialog`, `mrph_console`.
  Частичный пакет ломается уже на `cards/__init__.py`, который импортирует
  `cards.repo`;
- `version` поднята до `1.0.56`.

`bin/mrph` остаётся исполняемым входом из дерева и становится тонкой обёрткой:
`from mrph_console import main` и `sys.exit(main(sys.argv[1:]))`.

**Эпик #1.7.** `.github/workflows/nightly-deck.yml`; job называется `run`;
входы `workflow_dispatch`: `deck`, `processor`, `max_cards`,
`max_regenerations`, `deadline`.

### 2.4. Что нельзя сломать

1. **674 теста остаются зелёными.**
2. **`ledger=None` и `regenerations_before=0` дают сегодняшнее поведение
   байт в байт.** `cards/store.py: collect_generation` зовёт
   `process_generation` без этих аргументов и не правится вовсе: бюджет в
   расщеплённом пути `/collect` — вне области (§7).
3. **Один идентификатор в `--processor` ведёт себя как сегодня**, включая
   облачный: `registry.batch(id)`, тот же лейбл, те же тексты ошибок
   (`BackendError` — подкласс `ValueError`, код выхода 4).
4. **REPL не трогаем.** `flows/morph.py: resolve_batch_backend` и `@all` в нём
   остаются как есть; `flows/` не входит в область ни одной карты.
5. **`registry.batch(id)` для облачного слота не меняется** — правится только
   появление легаси-слота `openai` и только `batch_pool`.
6. **Настоящая установка обязана работать целиком**: после `pip install .` в
   пустом `venv` импортируются все модули проекта, а не подмножество. Список
   `py_modules` из 13 имён — источник дефекта, а не то, что нужно дополнить одним
   именем.
7. **`bin/mrph` продолжает работать из дерева** — и с аргументами, и без них;
   отложенный импорт `flows.morph` внутри ветки сохраняется (безголовый путь не
   должен грузить консольного бота), что проверяет `tests/test_stdout_purity.py`.
8. **Проза не выпиливается** (боль 4.2): докстринги и комментарии файлов,
   которых карта касается, сохраняются; удаления допустимы только там, где они
   названы в §2.3 явно (`scripts=` и список `py_modules` в `setup.py`,
   диспетчеризация в `bin/mrph`).

## 3. Приёмка

Базовая линия на момент постановки: **674 теста, 20 с**
(`venv/bin/python -m pytest -q --tb=no`).

Критерий ступенчатый, от узкого к широкому — в перегенерацию уезжает вывод
**первого** упавшего звена, поэтому узкое идёт первым. Везде `-q --tb=line`:
плотный формат переживает обрезку в 4000 символов, `-q` — нет, `--tb=no` не
использовать (замер в `TASK_TEMPLATE.md` §3).

Ступень 1 — **свой тест карты** (путь выбирает оркестратор):

```bash
venv/bin/python -m pytest <свой тест> -q --tb=line
```

Ступень 2 — **тесты задетых модулей**, по областям:

| область | команда ступени 2 |
|---|---|
| 2.15 | `venv/bin/python -m pytest tests/test_store.py tests/test_failed_attempt_diagnostics.py tests/test_outcome_diffstat_fields.py tests/test_store_diffstat.py -q --tb=line` |
| 2.16 | `venv/bin/python -m pytest tests/test_generations.py tests/test_budget.py tests/test_run_budget.py tests/test_budget_flags.py tests/test_nightly_e2e.py -q --tb=line` |
| 2.17 | `venv/bin/python -m pytest tests/test_cli_backend.py tests/test_batch_backends.py tests/test_cli_run.py -q --tb=line` |
| 3.6 | `venv/bin/python -m pytest tests/test_batch_backends.py tests/test_cli_backend.py -q --tb=line` |
| #1.3 | **настоящая установка** (ниже), затем `venv/bin/python -m pytest tests/test_cli_main.py tests/test_cli_contract.py tests/test_stdout_purity.py -q --tb=line` |
| #1.7 | ступени 2 нет: файл новый и ничьих тестов не задевает |

Ступень 3 — **полный прогон**, обязателен для каждой карты, которая правит
существующий файл (правило 3):

```bash
venv/bin/python -m pytest -q --tb=line
```

Ступень 4 — **страж прозы** для каждой карты, правящей существующий файл: число
докстрингов в файле после правки не меньше, чем до неё, и файл не стал короче.
Печатает числа, а не только код возврата:

```bash
venv/bin/python - <<'PY'
import sys
was = {"<файл>": (<строк>, <докстрингов>)}
bad = False
for path, (lines, docs) in was.items():
    text = open(path, encoding="utf-8").read()
    now_lines, now_docs = text.count("\n") + 1, text.count('"""')
    print(f"{path}: {now_lines} строк (было {lines}), "
          f"{now_docs} кавычек-троек (было {docs})")
    if now_lines < lines or now_docs < docs:
        print(f"ПРОЗА: {path} усох"); bad = True
sys.exit(1 if bad else 0)
PY
```

Исходные числа на момент постановки: `cards/store.py` 1688 / 86,
`cards/generations.py` 1396 / 64, `cards/cli_backend.py` 152 / 10,
`processors/registry.py` 301 / 18, `setup.py` 12 / 0, `bin/mrph` 21 / 0. Для
`setup.py` и `bin/mrph` страж прозы не применяется — там правка удаляющая по
условию (§2.4.8).

**Область #1.3 доказывается установкой, а не тестами.** Зелёные 674 теста здесь
— слабый критерий: они гоняют код из дерева, а не установленную команду, и
`entry_points` можно объявить неправильно, не уронив ни одного. Поэтому в приёмку
карты идёт настоящая установка в пустой `venv` — четыре утверждения, каждое со
своим диагнозом. Замер: **20 секунд** при умолчании `acceptance_timeout` 300 с, и
на текущем дереве три утверждения из четырёх **красные** — критерий не полый:

```sh
set -u
T=$(mktemp -d)
python3 -m venv "$T/v" > "$T/log" 2>&1 || { echo "УСТАНОВКА: venv не создался"; tail -20 "$T/log"; rm -rf "$T"; exit 1; }
"$T/v/bin/pip" -q install . >> "$T/log" 2>&1 || { echo "УСТАНОВКА: pip install . упал"; tail -30 "$T/log"; rm -rf "$T"; exit 1; }
rc=0
if [ -x "$T/v/bin/mrph" ]; then echo "1 OK  команда установлена"
else echo "1 ПРОВАЛ  $T/v/bin/mrph не появился"; rc=1; fi
if grep -q mrph_console "$T/v/bin/mrph" 2>/dev/null && ! grep -q MorphBot "$T/v/bin/mrph" 2>/dev/null; then
  echo "2 OK  сгенерированный console_script через mrph_console"
else
  echo "2 ПРОВАЛ  установленная команда не ссылается на mrph_console или несёт тело bin/mrph:"
  sed -n '1,15p' "$T/v/bin/mrph" 2>/dev/null; rc=1
fi
mkdir -p "$T/root"
out=$("$T/v/bin/mrph" deck status --root "$T/root" 2>&1); drc=$?
if [ $drc -eq 0 ] && printf '%s' "$out" | grep -q '"phase"'; then echo "3 OK  установленная команда работает: $out"
else echo "3 ПРОВАЛ  rc=$drc, вывод: $out"; rc=1; fi
miss=$(cd "$T" && "$T/v/bin/python" - <<'PY'
import importlib
mods = ["cards.cli","cards.cli_backend","cards.cli_run","cards.cli_cycle","cards.cli_deck",
        "cards.cli_json","cards.cli_views","cards.cli_wait","cards.budget","cards.hazards",
        "cards.notify","cards.repo","cards.runs_view","cards.store","cards.generations",
        "cards.compiler","cards.acceptance","cards.schema","cards.deck",
        "processors.registry","processors.batch","flows.morph","mrph_console"]
bad = []
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: bad.append(f"{m} ({type(e).__name__}: {e})")
print("; ".join(bad))
PY
)
if [ -z "$miss" ]; then echo "4 OK  весь пакет доехал в установку"
else echo "4 ПРОВАЛ  в установке не хватает модулей: $miss"; rc=1; fi
rm -rf "$T"
exit $rc
```

Два места здесь неочевидны и оплачены проверкой:

- **чистый `python3 -m venv`, без `--system-site-packages`.** Дочерний venv с
  системными пакетами видит develop-установку рабочего `venv` и разрешает
  `cards.cli` **из дерева** — утверждение 3 становится полым (проверено);
- **`cd "$T"` перед проверкой импортов.** Heredoc-питон кладёт в `sys.path`
  текущий каталог, то есть корень проекта, и 18 отсутствующих модулей
  «находятся» в дереве. Первая версия этой приёмки так и соврала.

Приёмка области #1.7 — только питоновский тест над текстом workflow: `pyyaml` в
`venv` **нет**, поэтому структурная проверка через `yaml.safe_load` допустима
лишь под `pytest.importorskip("yaml")` и не может быть единственной. Текстовые
утверждения обязаны печатать, чего именно не хватило.

Каждая команда запускается руками до сабмита колоды: команда, которая не
стартует, сжигает три попытки вслепую.

## 4. Ограничения

1. **Конверт правки существующего файла — 20–60 строк.** Не влезает — делить
   карту, а не расширять срез.
2. **Карта аддитивна**, кроме названных в §2.4.8 удалений.
3. **Код и его тест — одной картой** через `targets`.
4. **Один файл — одна карта-владелец в поколении** (`/deck check`).
5. **Файл, который пишет карта, не лежит в срезах её соседей по поколению** —
   иначе `stale-context` выбросит ответ непрочитанным.
6. **Починка не действует на текущий прогон.** Модули уже в памяти: ни новый
   `earlier_failures` в архиве, ни бюджет в перегенерациях, ни пул не появятся в
   том прогоне, который их пишет. `venv/bin/mrph` — **копия**, сделанная старым
   `scripts=`; после колоды её обновляет человек переустановкой.
7. **Срез называть явно.** Пустой срез = весь проект: дорого и даёт
   `implicit-read`.
8. **Ширина дешевле глубины.** Каждое поколение — отдельное стояние в очереди
   (20–45 минут) плюс прирост контекста оркестратора.

## 5. Приёмы, которые уже сработали

- Чистая логика — в новый файл, в большой файл — только проводка.
  `cards/store.py` (1688 строк) и `cards/generations.py` (1396) — большие файлы:
  туда идёт проводка, и ничего больше.
- Идиому карты брать из реального принятого прогона `.morph/runs/*/deck.json`.
- Образец формата — настоящий артефакт (`report.json` из архива), а не код,
  который его производит.
- Фикстура домена — копия реального артефакта, а не сочинение.

## 6. Что намеренно НЕ задано

Разбивка на карты, их число и границы, срезы, порядок, формулировки приёмок,
имена тестовых файлов.

## 7. Что вне области этой колоды

1. **Пуш и мерж ветки прогона.** Ветка `morph/<deck-id>` этой колоды остаётся
   **локальной**: не пушится и не мержится. То, что job в §2.3 пушит ветку, —
   содержимое файла, а не действие колоды.
2. **Сам ночной запуск на GitHub Actions.** Колода производит файл workflow.
   Первый реальный запуск на раннере, секреты репозитория и отдельный ключ
   провайдера с лимитом трат — человеческие шаги после мержа.
3. **Файл колоды `decks/nightly.json`.** Job принимает путь к закоммиченной
   колоде входом `deck` и засевает бэклог через `mrph deck add --file <path>`;
   саму ночную колоду эта колода не сочиняет. Пункт 5 эпика («колода как
   коммит», боль 2.6) остаётся открытым — job обходит его существующей
   подкомандой, а не закрывает.
4. **Бюджет в расщеплённом пути `/collect`.** `cards/store.py: collect_generation`
   персистит батч перегенерации и возвращает управление; бюджет там не судится
   ни до, ни после этой колоды.
5. **Пул для облачных провайдеров.** `batch_pool` остаётся смыслом только для
   локальных узлов; смешанный пул отвергается, а не поддерживается.
6. **Удаление легаси-переменных окружения.** `LLAMA_CPP_*`, `OLLAMA_*`,
   `ANTHROPIC_API_KEY` продолжают заводить слоты как сегодня; правится один
   слот `openai` и только требованием имени модели.
7. **`ResilientBackend` в REPL** (пункт 2 «Ближайшего» в `ROADMAP`), **`mrph primer`**,
   **нотификатор**, **перекройка MCP** — другие пункты эпика, не эти.
8. **Переустановка `venv`** после смены `setup.py`.

## 8. Как запускать

```bash
cd /Users/kyrylo/Documents/Pers/python_gptmorph_cli
venv/bin/mrph deck check
venv/bin/mrph run --processor @glm
```

Исполнитель: `@glm`. Дерево перед стартом чистое, иначе прогон не начнётся.
Безголовым путём, не REPL: устойчивость к обрыву есть в `mrph run`.

## 9. Предрегистрация прогнозов

| величина | прогноз |
|---|---|
| карт в колоде | 6 |
| поколений | 1 |
| счёт исполнителя | < $0.15 |
| счёт сессии-оркестратора | не оптимизируем, пишем фактом |
| карт с перегенерацией | 2 |
| конфликтов `write-write` на префлайте | 0 |
| тестов | > 674 |

**Проверяемое утверждение:** все шесть областей независимы по владению файлами,
поэтому колода уложится в **одно** поколение — ни одна карта не читает то, что
пишет другая. Опровергается любым ребром `depends_on`, которое найдёт
`mrph deck check`.

## 10. Что записать по итогу

`/cost` сессии целиком, счёт провайдера, время в очереди, число перегенераций,
итоговое число тестов и доллар за принятую карту.

## 11. Факт

`<Заполняется после прогона.>`
