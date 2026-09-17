# Фаза 7: сценарий демонстрации и приёмки

> Пошаговый сценарий для живого показа. Все приведённые выводы — **настоящие**,
> сняты с прогонов 17.09.2026, а не придуманы для слайда.

## Что доказываем

Одно предложение: **ночная колода — это ветка, где каждая принятая карта стала
коммитом с доказательством, а непринятая не оставила следов.**

Три тезиса, которые видно глазами:

1. Карта пишет **набор файлов** — модуль и его тест одним коммитом.
2. Упавшая карта **не оставляет мусора**: `git status` чист, коммита нет.
3. Прогон **архивируется**: колода и отчёт ложатся в историю той же ветки.

---

## Подготовка

**Никогда не показывайте на рабочем репозитории.** Прогон создаёт ветку и
коммиты; на демо это лишний риск. Делайте одноразовую копию:

```bash
rsync -a --exclude .git --exclude .morph /path/to/python_gptmorph_cli/ /tmp/demo/
cd /tmp/demo && git init -q . && git add -A && git commit -qm baseline
git config user.email demo@morph.local && git config user.name "Morph Demo"
```

Зависимости: Python 3.9+, пакеты `openai` и `python-dotenv` (нужны только для
запуска CLI), `pytest`. Никаких других.

### Два контура

| контур | чем исполняет | время цикла | когда использовать |
|---|---|---|---|
| **быстрый** | локальные stub-узлы | секунды | живой показ механики |
| **боевой** | OpenRouter, `z-ai/glm-5.3-flash:batch` | ~20 минут на поколение | доказательство, что это не симуляция |

Очередь провайдера съедает около двадцати минут **независимо от размера батча**
(замерено: 1 запрос — 21.6 мин, 3 — 21.1, 8 — 22.0). Поэтому боевой прогон
**запекайте заранее**, до начала презентации, и показывайте на нём результат и
`git log`, а механику крутите на быстром контуре.

### Быстрый контур: поднять stub-узлы

```bash
python3 tests/stub_node.py --port 8080 --answers tests/stub_answers.json --name stub-a &
python3 tests/stub_node.py --port 8081 --answers tests/stub_answers.json --name stub-b &
```

Это минимальный OpenAI-совместимый сервер: отдаёт заготовленный ответ по маркеру
`[[STUB:<ключ>]]` в инструкции карты. Ответы лежат в `tests/stub_answers.json` —
сценарий демо правит их, не трогая код сервера. Ключ вида `<ключ>@2` отвечает
иначе на второй вызов: так показывается best-of-N и перегенерация.
Добавьте `--log-dir /tmp/morph-prompts`, чтобы поймать скомпилированные промпты
и показать, что именно уехало исполнителю.

`.env` для этого контура:

```
MRPH_PROCESSORS=stub-a,stub-b
MRPH_PROCESSOR_stub-a_TYPE=llama_cpp
MRPH_PROCESSOR_stub-a_ENDPOINT_URI=http://127.0.0.1:8080/v1
MRPH_PROCESSOR_stub-a_MODEL=stub-a
MRPH_PROCESSOR_stub-b_TYPE=llama_cpp
MRPH_PROCESSOR_stub-b_ENDPOINT_URI=http://127.0.0.1:8081/v1
MRPH_PROCESSOR_stub-b_MODEL=stub-b
```

Карты демо ссылаются на готовые ключи: `cs-calc` (модуль плюс его тест),
`cs-bad` (три файла, приёмка провалится), `cs-retry` (первая попытка красная,
вторая зелёная), `cs-single` (один файл).

---

## Сценарий, восемь тактов

### Такт 1. Пустая колода — это план, а не ошибка

```
/deck
```
```
mrph> The deck is empty -- this is the planning phase, not an error.
```

Говорите: *колода существует до кода. Это артефакт, который можно прочитать и
отревьюить до того, как потрачен хоть один токен.*

### Такт 2. Карта, которая пишет код и его тест

```
/card
{"custom_id":"cs-calc","intent":"generate","targets":["pkg_demo/calc.py","tests/test_calc_demo.py"],
 "context_slice":["settings.py"],"acceptance":"python3 -m pytest tests/test_calc_demo.py -q",
 "variants":1,"depends_on":[],"instruction":"Write add(a, b) and its pytest."}
```

Подчеркните `targets` и `acceptance`: **два файла, одна приёмка**. До Фазы 7
такая задача была невыразима — карта писала ровно один файл.

### Такт 3. Карта, которой суждено упасть

```
/card
{"custom_id":"cs-bad","intent":"generate","targets":["pkg_bad/one.py","pkg_bad/two.py","pkg_bad/three.py"],
 "context_slice":["settings.py"],"acceptance":"python3 -c \"import pkg_bad.one; assert pkg_bad.one.VALUE == 999\"",
 "variants":1,"depends_on":[],"instruction":"Write three modules."}
```

Три цели, заведомо непроходимая приёмка. Она понадобится в такте 6.

### Такт 4. Сабмит создаёт ветку

```
/submit @all
```
```
mrph> git: the run is on branch 'morph/20260917-113712-ef288981' (branched off main);
      each accepted card becomes one commit.
mrph> Submitted generation 1/1 on "stub-a+stub-b" (batch local-28084316...):
    cards: cs-calc, cs-bad
```

Скажите: *с этой секунды всё, что произойдёт ночью, изолировано в ветке. Утром
её либо мержат, либо выбрасывают одной командой.*

### Такт 5. Ожидание с прогрессом

```
/collect wait
```
```
mrph> git: 'cs-calc' committed as c3febcc559.
mrph> cs-bad failed acceptance -- regeneration 1/2 submitted as batch local-161cf345...; still waiting.
mrph> Waiting for regeneration 1/2 of cs-bad in generation 1/1 (batch local-161cf345...) -- 0s elapsed.
mrph> cs-bad failed acceptance -- regeneration 2/2 submitted as batch local-d812d646...; still waiting.
mrph> Collected generation 1/1:
    cs-calc: written -> ./pkg_demo/calc.py, ./tests/test_calc_demo.py
    cs-bad: failed after 3 attempt(s)
```

Два наблюдения для зала: принятая карта коммитится **сразу**, не дожидаясь
конца колоды; упавшая уходит на перегенерацию **с выводом ошибки в промпте**,
и лимит попыток — три.

### Такт 6. Провенанс и чистота

```bash
git log -1 --format='%B'
```
```
morph cs-calc: pkg_demo/calc.py, tests/test_calc_demo.py

Morph-Card: cs-calc
Morph-Model: stub-a+stub-b
Morph-Acceptance: python3 -m pytest tests/test_calc_demo.py -q
Morph-Acceptance-Exit: 0
```

```bash
git show --stat --format="" HEAD    # два файла в одном коммите
ls pkg_bad                          # каталога нет
git status --porcelain              # пусто
```

Это кульминация. Говорите: *у каждой строки есть автор, спецификация и
доказательство. А карта, не прошедшая проверку, не оставила ни одного из трёх
своих файлов — атомарность.*

### Такт 7. Архив прогона

```
/deck runs
```
```bash
git log --oneline main..HEAD
```
```
7c99084 morph run 20260917-114046-e0c36aee: deck and report
cca726c morph cs-four: pkg_demo/calc4.py, tests/test_calc_demo4.py
```

Ветка читается как документ: коммит на карту, затем — что было заказано и чем
кончилось. Архив в `.morph/runs/<id>/` (`deck.json` + `report.json`),
append-only.

> Чтобы архив попадал в git, в `.gitignore` нужен именно такой вид:
> `.morph/*` и следом `!.morph/runs/`. Git не заходит внутрь исключённого
> каталога, поэтому `.morph/` + `!.morph/runs/` **не работает**.

### Такт 8. Защита от себя

Испачкайте дерево и попробуйте стартовать:

```bash
echo x >> README.md
```
```
/submit @all
```
```
mrph> the working tree has uncommitted changes, and a run started here could not be
      undone with git checkout. Commit or stash them (git stash -u), or start the run
      with "nogit" (/submit nogit, /nightly nogit) to leave git alone entirely.
```

Затем покажите выход:

```
/submit @all nogit
```
```
mrph> git: off for this run -- no branch, no commits (the morphs still land in the working tree).
```

---

## Приёмочные критерии Фазы 7

Те же такты, но как чек-лист (все пройдены 17.09.2026 на одноразовом репозитории):

- [x] колода из двух карт даёт ветку и коммит на принятую карту;
- [x] `git log` показывает трейлеры с картой, моделью, командой приёмки и её кодом возврата;
- [x] карта с тремя целями и упавшей приёмкой оставляет `git status` чистым и не создаёт коммита;
- [x] карта пишет модуль и его тест одним коммитом;
- [x] `nogit` отключает git полностью, морфы всё равно ложатся в дерево;
- [x] отказ на грязном дереве называет оба выхода;
- [x] архив `.morph/runs/<id>/` создаётся, не перезаписывается и коммитится последним коммитом ветки;
- [x] не-git-каталог работает как раньше.

Автоматическая часть: `python3 -m pytest -q` → **311 passed**, из них 30 — на
настоящих временных репозиториях (`tests/test_run_git.py`).

---

## Цифры для слайда

| | |
|---|---|
| MCP-сервер, написанный Морфом | 1082 строки, $0.0223, два поколения по 21 мин |
| Правка файла в 1286 строк | +12 строк, 0 потерянных символов, приёмка с первого раза |
| Цена карты (узкий срез / весь проект) | $0.0007 / $0.0083 |
| Колода 20 карт × 3 варианта, весь проект | ~$0.50 |
| Время в очереди | ~20 мин независимо от размера батча |

---

## Что может пойти не так на показе

| симптом | причина и что делать |
|---|---|
| `no matching processor` | не настроен `.env`; покажите `/settings` |
| прогон отказывается стартовать | грязное дерево — это by design; `git stash -u` или `nogit` |
| батч висит дольше 25 минут | очередь провайдера; переключитесь на запечённый прогон |
| `/collect` говорит «still in progress» | так и задумано: голый `/collect` — один опрос, ждёт `/collect wait` |
| архив не коммитится | `.gitignore` без `!.morph/runs/`; см. такт 7 |

## Оснастка

Всё нужное лежит в репозитории: `tests/stub_node.py` (сервер),
`tests/stub_answers.json` (ответы демо), `tests/test_stub_node.py` (17 тестов на
саму оснастку). Никаких зависимостей сверх стандартной библиотеки; сервер пишет
баннер и лог запросов в stderr, оставляя stdout под протокол.
