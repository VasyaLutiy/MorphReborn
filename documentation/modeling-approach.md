# GPT Morph's Modeling Approach: Whole-Project Context

## The core idea

Most LLM coding assistants (Claude Code, Cline, and similar tools) treat the
project as something too large to fit in a model's context window. They cope
with that by retrieving snippets (RAG / embeddings search), letting the model
call tools to read files on demand, or periodically summarizing/compressing
the conversation so it doesn't grow without bound. All of these are workarounds
for the same constraint: only part of the project is ever visible to the model
at once.

GPT Morph takes a different position: **the whole project is the context**.
Every morph (`/generate` or `/patch`) walks the current project directory and
loads the full contents of every recognized source file into the LLM
conversation before the user's instruction is sent. There is no retrieval
step, no embedding index, no chunking, and no context compression — the
project *is* the prompt.

## Why this matters: de-hallucination, not just convenience

Partial-context tools have to guess which files are relevant to a given
change, either through retrieval heuristics or by asking the model to request
files iteratively. Every one of those steps is a place where a relevant file
can be missed, and a missed file is exactly the situation in which a model
hallucinates: it invents a function signature, assumes a module exists that
was renamed, or duplicates logic that already lives elsewhere in the project
because it never saw it.

By putting the entire project in context up front:

- The model sees the *actual* signatures, types, and conventions used
  elsewhere in the codebase instead of inferring them statistically.
- There is nothing to "forget" mid-session — GPT Morph never needs to
  summarize or evict older turns to make room, because each morph starts a
  fresh, complete `LLMDialog` built straight from the filesystem.
- Cross-file consistency (naming, error handling style, existing helpers) is
  enforced by exposure, not by instructing the model to "follow the existing
  style" and hoping it read the right file.

This is only tractable because of the constraint described below.

## The low-coupling precondition

Loading an entire project into a single prompt scales with the project's
*token footprint*, not its line count. A project that is properly designed
around low coupling — small, focused files; clear module boundaries; minimal
duplication; no giant generated or vendored files mixed into source
directories — stays well within practical context sizes even as it grows in
feature count. A tightly coupled, monolithic codebase (huge files, sprawling
cross-cutting state, vendored dependencies checked into the tree) defeats this
approach simply by being too large to fit, long before the LLM's reasoning
would be the limiting factor.

In other words, GPT Morph's context model is not just a technical choice, it
is also an architectural incentive: projects that want good morphs are
naturally pushed toward the same low-coupling, high-cohesion structure that
makes them easier for human maintainers, too. `filter_source_code_file_names`
(`flows/morph.py`) already excludes the usual sources of bloat that would
break this assumption — `node_modules/`, `venv/`/`venvy/`, and
`typechain-types/` — on the assumption that everything else in the tree is
project-authored and worth the model seeing.

## How the context is built

### `LLMDialog` — the conversation accumulator

`llm_dialog.py` defines `LLMDialog`, a thin wrapper around a list of
`{role, content, time}` messages compatible with the OpenAI-style chat
completion format used by every processor. Two dialogs can be concatenated
with `+`, which is how the project context and the user's own instructions
are combined into one conversation before it is sent to a processor.

### `ContextFolderDialog` — the whole-project loader

`context_folder_dialog.py` defines `ContextFolderDialog`, an `LLMDialog`
subclass that walks a directory tree (`os.walk`) and, for every file that
passes a filter callback, appends a `user` message:

```
Contents for another file "<path>" in this project:

---
<file contents>
---
```

Each morph starts by building this dialog fresh from `.` (the project root)
via `build_current_project_context()` in `flows/morph.py` — there is no
caching of a stale context between morphs; the filesystem is re-read every
time, so the model always sees the project's current state, not a snapshot
from session start.

### File selection

`filter_source_code_file_names` (`flows/morph.py`) is an allow-list by
extension (`.py`, `.js`/`.jsx`/`.ts`/`.tsx`, `.go`, `.rs`, `.sol`, `.fc`,
`.sql`, `.proto`, `.yaml`/`.yml`, `.sh`, `.md`, `.dot`, `Dockerfile`,
`package.json`, `requirements.txt`, etc.), with explicit excludes for
`node_modules`, `typechain-types`, and Python virtualenvs. `.env` is
deliberately excluded (commented out in the filter) to keep secrets out of
the prompt sent to remote processors.

### Extending the corpus: `.corpora`

For projects using rare languages, internal frameworks, or conventions an LLM
wouldn't otherwise know, a `.corpora` folder (documented in `corpora.md`) can
hold Markdown reference material. No special-casing is needed for it to take
effect: because it is just another folder of `.md` files under the project
root, `ContextFolderDialog`'s walk picks it up automatically as part of the
whole-project context, alongside the actual source files.

## From context to morph: `/generate` and `/patch`

Both flows (`flows/morph.py`) build an `LLMDialog`, add the whole-project
context, then layer the task-specific instruction on top:

- **`/generate`** — `dialog = LLMDialog() + build_current_project_context()`,
  then the user's free-text description of the new file is assigned as the
  final `user` message. Two specialized variants exist: unit-test generation
  (prepends "Please, generate a unit test for my project.") and statement-flow
  Graphviz generation.
- **`/patch`** — the target file's current contents are injected as an
  `assistant` message *before* the whole-project context is appended, then the
  user's change instructions follow. This ordering lets the model treat "the
  file to edit" as the most recently stated fact, with the rest of the project
  providing corroborating context for what "correct" looks like.
- **`/todo`** — same shape as `/patch`, but the instruction is fixed
  ("criticize this file and add TODO: comments").

In every case the response is expected as one or more fenced code blocks;
`response_to_file_body` extracts and concatenates them (falling back to the
raw response for `/todo`-style prose, appended rather than overwriting the
target file).

## Multi-processor morphing is orthogonal to the context model

GPT Morph can fan a single morph out across several named processor
instances at once (`processors/registry.py`, `/generate @k80-a,@gpt4`,
`/generate @all`) — for example several local llama.cpp nodes plus a hosted
model, run concurrently via `asyncio.gather` over a thread pool
(`MorphBot.run_morphers`). Every processor instance receives the *same*
whole-project `LLMDialog`; multi-processor mode is purely about comparing how
different models morph identical, complete context, not about splitting the
project across processors. Each processor writes its own output file
(`<name>.<processor_id>.<ext>`) so results can be diffed by hand.

## Trade-offs of this approach

- **Context window is the hard ceiling.** There is no fallback to
  retrieval or summarization if the project's filtered file set exceeds the
  target model's context window — the morph will simply fail or truncate at
  the processor level. This is the direct cost of the low-coupling
  precondition above: it must actually hold for the project in question.
- **No incremental context reuse.** Because context is rebuilt from disk on
  every morph, there is no persistent session memory to manage or to leak
  stale state from — but it also means every morph re-pays the full cost of
  reading and transmitting the whole project, rather than reusing a warm
  context.
- **Whole-file granularity.** Morphs are written as complete files, not
  diffs/patches against specific lines, which keeps the output format simple
  at the cost of not being able to target a narrow edit within a very large
  file.

These trade-offs are the mirror image of the benefit: a small, low-coupling
codebase pays an almost-negligible context cost for consistently
hallucination-resistant morphs; a large or tightly-coupled one pays for the
same design in reliability, and eventually in raw feasibility.
