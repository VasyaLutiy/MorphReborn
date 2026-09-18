# Morph 2.0: The Batch Orchestrator

> **Status: built, and running.** This document was written as a design
> proposal; it is kept as the *rationale* — why the machine is shaped this way —
> and no longer describes anything unbuilt. As of 2026-09-18 phases 0-7 of
> [`DEVELOPMENT_PLAN.md`](./DEVELOPMENT_PLAN.md) are shipped: morph cards,
> the deck compiler, batch backends, generations, mechanical acceptance with
> best-of-N and rollback, the CLI orchestrator, the git substrate (a branch per
> run, one provenance commit per card, changeset cards) and a headless
> command surface. 583 tests. Morph has written several of its own features as
> decks, including the one that lets it report what each card changed.
>
> **Where the current state actually lives**, because prose about a running
> machine rots faster than the machine changes:
> [`README_Morph2.md`](../README_Morph2.md) — how it is built today and every
> measurement taken; [`Head_Pains.md`](../Head_Pains.md) — what hurts, with
> evidence; [`Morph_SKILL.md`](./Morph_SKILL.md) — how to operate it;
> [`TASK_TEMPLATE.md`](./TASK_TEMPLATE.md) — how to write a deck's
> specification.
>
> It builds directly on two shipped foundations:
> [`modeling-approach.md`](./modeling-approach.md) (stateless whole-context
> morphs) and [`parallel-generate-scheduling.md`](./parallel-generate-scheduling.md)
> (the job/slot/queue scheduler in `scheduler.py`).

## Positioning: a serious machine for serious programmers

Morph 2.0 is deliberately **not** an interactive coding agent. Tools like
Claude Code or Replit optimize for conversational immediacy: type, watch,
correct, retype. That loop is powerful, but it rewards under-specified
requests — the machine answers in seconds, so thinking hard up front costs
more than iterating.

Batch mode inverts the economics, the way the PDP-11 era did. When results
come back in minutes to hours rather than seconds, a job submitted with a
sloppy specification is a job wasted. The engineer is pushed to state, per
change: what exactly should change, which files are relevant, and how success
will be verified — *before* submitting. The deck of punch cards is back, and
that is a feature: machine turnaround time once again buys human deliberation.

This is the same philosophical move as Morph 1.0's whole-project context
(see `modeling-approach.md`): the tool's constraint is also an architectural
incentive. Whole-project context pushed codebases toward low coupling; batch
turnaround pushes engineers toward complete, verifiable specifications.

## What batch APIs offer, and why Morph fits them natively

Modern LLM providers expose **batch endpoints** alongside interactive chat
completions (Anthropic Message Batches, OpenAI Batch API). The contract:

- You submit a file of many independent requests (JSONL, each with a
  `custom_id`), and collect results asynchronously — typically within
  minutes to hours, with a 24-hour completion window.
- Each request must be **fully self-contained**: no tool-use round trips,
  no mid-flight clarification, no shared session state.
- In exchange, batched tokens cost **roughly half** the interactive price.

An interactive agent cannot use this: its read-file/edit-file loop is
inherently conversational. But a Morph 1.0 morph already *is* a batch
request — a fresh, complete `LLMDialog` built from disk, one instruction,
one response, one output file, no session state. Morph has always been a
batch system that happened to run its jobs interactively. Morph 2.0 stops
pretending.

## Architecture: the orchestrator plans, the batch executes

Morph 1.0 solved context selection by brute force: the whole project in
every prompt, because GPT-3.5-era models could not be trusted to ask for
files. Morph 2.0 splits the work between two very different kinds of model
usage:

### 1. The Orchestrator (interactive, frontier model, writes no code)

A single agentic session that holds a *lightweight* project representation —
a repository map, public signatures, a dependency graph — never the full
source of everything. Its sole outputs are **morph cards** (below). Its
intellectual work is:

- decomposing the user's backlog of change requests into independent jobs;
- **compiling the context slice** for each job: exactly which files this
  job's executor must see;
- ordering jobs into **generations** so that no job depends on a file
  another job in the same batch is still producing (a topological sort of
  the deck);
- after each batch returns: verifying morphs (tests, linters, acceptance
  criteria), integrating the ones that pass, and recompiling the failures —
  with the error output added to their context — into the next generation's
  deck.

### 2. The Executors (batched, per-card model choice, write all the code)

Each morph card becomes one request in the batch file. The executor model
receives the card's context slice and instruction, and returns a complete
file — exactly Morph 1.0's output contract (`response_to_file_body`,
whole-file granularity, fenced code blocks). Executors are cheap,
half-price, and massively parallel; they never converse.

## The morph card: JCL for LLM jobs

The PDP-11 era prefixed every job with Job Control Language: metadata that
told the operator what the job was, what resources it needed, and where the
output went. A **morph card** is the same artifact for an LLM batch:

```json
{
  "custom_id": "gen-042-scheduler-priorities",
  "meta": {
    "intent": "patch",
    "target": "scheduler.py",
    "context_slice": [
      "scheduler.py",
      "tests/test_scheduler.py",
      "documentation/parallel-generate-scheduling.md"
    ],
    "acceptance": "pytest tests/test_scheduler.py passes",
    "model": "claude-sonnet-5",
    "variants": 3,
    "generation": 1,
    "depends_on": []
  },
  "instruction": "Add priority levels to queued jobs: ..."
}
```

Field notes:

- **`intent`** — `generate` / `patch` / `todo`, matching today's flows.
- **`context_slice`** — the orchestrator-compiled file list. For a small,
  low-coupling project the slice may legitimately be *the whole project*
  (Morph 1.0 behaviour, now at half price); for larger projects it is a
  dependency-graph cut around the target.
- **`acceptance`** — a machine-checkable criterion. This is what makes the
  post-batch verification loop mechanical instead of vibes-based, and it is
  the field that forces specification discipline.
- **`variants`** — how many independent samples of this card to place in
  the batch. Morph 1.0's multi-processor fan-out (`/generate @all`) mutates
  into cheap **best-of-N**: the orchestrator picks the variant that passes
  acceptance, or diffs survivors for the engineer.
- **`generation` / `depends_on`** — batch requests cannot see each other's
  output, so any job that reads or writes a file another job writes must wait
  for the next generation. `depends_on` declares that ordering and a
  topological sort applies it; cards with unmet dependencies stay in the deck
  for generation N+1.

  The declaration alone is not the guarantee, and treating it as one was this
  design's one silent failure mode: two cards of a generation writing the same
  file produced a lost update — the second write replaced the first, both cards
  reported success, and the run was green. The invariant is therefore *derived*
  rather than trusted (`cards/hazards.py`): a card's write set is its
  `targets`, its read set its `context_slice`, and every run is preflighted
  against them. A missing edge between a reader and a writer is added
  automatically (the reader serializes into the next generation); two cards
  writing one file refuse the run, because there is no correct automatic answer
  to which of them should win. A second, independent check runs at acceptance
  time: each card records what it was compiled from, and an answer written
  against files that have since changed is discarded unread and regenerated
  from the current tree, which also covers what no deck check can know — a
  human editing the project while a batch sits in a provider's queue.

## The generation cycle: nightly builds for code generation

```
backlog of change requests
        │
        ▼
  ORCHESTRATOR ── compiles deck (cards, slices, ordering)
        │
        ▼
  BATCH SUBMIT ── generation N (half-price, massively parallel)
        │                                        ┌──────────────┐
        ▼                                        │  turnaround: │
  RESULTS ── verify each card's acceptance       │ minutes—hours│
        │                                        └──────────────┘
        ├── pass → integrate morph, close card
        └── fail → recompile card + error context into generation N+1
```

The natural usage rhythm is the **nightly morph**: the engineer spends the
day appending well-specified cards to the backlog, submits the deck in the
evening, and reviews integrated morphs plus a failure report in the morning.
Failed cards carry their test output into the next generation automatically —
an agentic repair loop, but batched and asynchronous instead of interactive.

## Where cards come from: the cold start

The deck describes how cards are *executed*; this section describes where they
come from when a session starts with zero cards. An empty deck is not a
degenerate state — it is the planning phase, the way an empty sprint backlog
precedes refinement. Cards enter the deck from three sources:

1. **Dialogue.** The primary path. The engineer states an intent to the
   orchestrator ("add priorities to the scheduler", "migrate the storage
   layer"); the orchestrator *decomposes* it into cards — proposing targets,
   context slices, and acceptance criteria — and the engineer reviews the deck
   before submitting. One intent may unfold into one card or twenty.
2. **A deck file.** A hand-written or generated `deck.json`, for repeatable
   runs and CI. The power-user path, not the default.
3. **The machine itself.** The generation cycle already breeds cards: a card
   that fails its acceptance is recompiled into the next generation with the
   error context attached. The deck is partially self-reproducing.

### Existing projects

There is no true cold start here: the context already exists — it is the
project. The orchestrator builds its lightweight representation (repository
map, signatures — the research question above), the engineer states a goal,
and the first cards appear through dialogue. An empty deck over a live project
simply means "the backlog has not been articulated yet."

### New projects: the genesis deck

A brand-new project has neither code nor anything to slice. The machinery
already handles this: a `generate` card with an empty `context_slice` gets the
whole-project context of an empty directory — which is *nothing* (or only
`.corpora`) — and that is correct for a project's first file. The pattern is
that **the first generation generates context, not code**:

- Generation 0: one card → `ARCHITECTURE.md` (a spec, a manifest), distilled
  from the engineer's interview with the orchestrator.
- Generation 1: skeleton-module cards, each slicing on
  `["ARCHITECTURE.md"]` and depending on the genesis card.
- Generation 2+: code and tests, slicing on the files already born.

The project **grows its own context generation by generation**. No special
mode is needed: `depends_on` ordering plus the compile-after-write rule (a
generation is compiled only after the previous generation's morphs are on
disk) carry the whole pattern.

This is not hypothetical — it is how Morph 2.0 itself is being built. The
session that produced this repository started with zero cards: generation 0
wrote the manifests (context, not code), generation 1 the development plan
sliced on them, and every implementation phase since has been a card carrying
a `context_slice` of the documentation plus prior phases' code, with a pytest
acceptance criterion — executed by a separate coding agent. The cold-start
protocol described here is the protocol Morph 2.0 is using to build itself.

## What survives from the current codebase

The migration is an evolution of existing components, not a rewrite:

- **`LLMDialog` / `ContextFolderDialog`** — already build self-contained
  conversations; `ContextFolderDialog` gains a file-list mode (the context
  slice) alongside its walk-everything mode.
- **`JobScheduler`** (`scheduler.py`) — the pinned/rotation/queue semantics
  become the **deck compiler**: slots map to per-provider batch size and
  rate limits, the FIFO queue to generation ordering, pinning (`@id`) to the
  card's `model` field.
- **`processors/registry.py`** — grows a batch backend per provider type
  (submit deck, poll, collect) next to the existing synchronous `run`.
  `llama_cpp`/`ollama` instances can emulate a batch endpoint locally by
  draining a deck through the existing pool — the K80 nodes become,
  literally, the department minicomputer running overnight jobs.
- **`response_to_file_body`, output naming** — unchanged; a variant writes
  `<name>.<custom_id>.<ext>` exactly as multi-processor fan-out does today.

## Trade-offs, stated plainly

- **Latency is the price of the discount.** This is a tool for a backlog of
  independent, specified changes — never for "fix this typo now." The
  interactive `/generate` flow remains for that.
- **Context slicing is the hard problem.** A mis-compiled slice reproduces
  exactly the hallucination mode whole-project context was designed to kill
  (`modeling-approach.md`). Mitigations: whole-project slices as the default
  for small projects; dependency-graph slicing only where the project is too
  large; failed acceptance feeds the widened slice for the retry generation.
- **No mid-course correction.** A batch, once submitted, runs to completion.
  Under-specified cards fail their acceptance and cost a generation. This is
  the discipline mechanism working as intended, but it must be understood as
  such by the operator.
- **Dependent changes serialize.** A deep chain of dependent edits degrades
  to one card per generation — interactive agents genuinely win that shape
  of work. Morph 2.0's sweet spot is a *wide* backlog of independent morphs.
