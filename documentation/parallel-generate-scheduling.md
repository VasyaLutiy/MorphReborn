# Parallel `/generate`: Round-Robin Processor Scheduling

> **Status: implemented** in `scheduler.py` (the pool/queue/round-robin
> bookkeeping) and `flows/morph.py` (wiring it into `/generate` and
> `/patch`). This document remains the behavioural spec, kept in sync with
> the shipped wording; see `tests/test_scheduler.py` for the scenarios it is
> checked against.

## The problem this solves

`processors/registry.py` already supports **multi-processor fan-out**:
`/generate @k80-a,@gpt4` or `/generate @all` sends *one* file's prompt to
*several* processors at once, so you can compare how different models morph
identical context. That feature is about **one file, many processors**.

This document describes the opposite shape: **many files, a fixed pool of
processors**. A user working through a backlog of files wants to type a
*sequence* of independent `/generate` calls — one per file — back to back,
and have each call automatically picked up by whichever configured processor
is free, without waiting for earlier files to finish first. With three
processors configured, the first three `/generate` calls should start
morphing concurrently; a fourth call should wait in a queue until one of the
first three finishes, then run automatically.

The two features are orthogonal and compose: a queued job can itself be a
fan-out job (`/generate @all` still occupies every processor's slot for the
duration of that one job).

## Core concepts

- **Processor pool** — the ordered set of processor ids from `/settings`
  (`registry.ids`). The worked examples below assume a pool of three:
  `k80-a`, `k80-b`, `gpt4`, configured exactly as shown in
  [`README.md`](../README.md#25-multi-agent-setup-many-processor-instances):

  ```
  MRPH_PROCESSORS=k80-a,k80-b,gpt4
  ```

- **Slot** — a processor's capacity to run exactly one job at a time. Pool
  size == number of slots == number of processors (3 in the examples).
- **Job** — one `/generate` (or `/patch`) request: a file name plus a
  prompt. Each job is independent of every other job — different files,
  potentially different prompts — unlike fan-out, where one job spans
  several processors.
- **Round-robin cursor** — a pointer that advances through the pool in
  configured order (`k80-a` → `k80-b` → `gpt4` → `k80-a` → …) each time a job
  is *dispatched* to a free slot. It guarantees that when several slots are
  free at once, new jobs are spread evenly across processors instead of
  always landing on the first configured id.
- **Queue** — a FIFO holding jobs that arrived while every slot was busy.
  When a slot frees, the job at the head of the queue is dispatched to it.

Explicit processor selection (`/generate @gpt4`) still works exactly as
today — it pins the job to that one processor instead of consulting the
round-robin cursor — but it now also respects that processor's slot: if
`gpt4` is already busy, a pinned job queues for `gpt4` specifically rather
than running immediately.

## Scheduling algorithm

1. A `/generate` request is parsed as it is today (`parse_processor_spec`),
   yielding either an explicit processor list or "use the rotation."
2. **If the request targets the rotation** (no `@id` given):
   - If at least one slot is free, the job is assigned to the *next*
     free processor in round-robin order and starts immediately; the
     cursor advances past it.
   - If every slot is busy, the job is appended to the queue. Its
     eventual processor is not fixed yet — it will take whichever
     processor becomes free first.
3. **If the request pins specific processor(s)** (`@id`, `@id,@id`, `@all`):
   - Each pinned processor that is currently free starts immediately.
   - Each pinned processor that is currently busy causes the job to wait on
     that processor's own queue position; the job does not run until *all*
     of its pinned processors are free (matching today's fan-out semantics,
     where one job's results are collected together).
4. Whenever a processor finishes a job (success or failure), it looks at the
   FIFO queue: if non-empty, it immediately dequeues and starts the next
   job on that processor and announces the hand-off in the chat.
5. The round-robin cursor is only advanced by step 2's immediate-dispatch
   case, where it is genuinely choosing among several free processors. A
   step 4 dequeue does not move it: the processor that just freed is the
   only candidate for the job at the head of its queue, so no rotation
   choice is actually being made. This matches the worked example below,
   where dequeuing job 4 onto `k80-a` does not change which processor the
   *next fresh* `/generate` would land on.

## Telling the user which processor is next

Every `/generate` reply — the same message that already asks for a file
name — is extended to name the processor (or queue position) the job has
been given, decided at intake time, before the file name/prompt are even
collected. This directly answers "which processor will run *this* call":

**A slot is free — dispatched immediately** (`describe_assignment` in
`flows/morph.py`; "processors" and a comma-joined id list when the job is
pinned to more than one, e.g. `@k80-a,@gpt4` or `@all`):
```
mrph> [job 3] Assigned to processor "gpt4" (slot 3/3 now busy).
mrph> Enter the file name for saving the generated file:
```

**All slots busy, rotation job — queued:**
```
mrph> [job 4] All 3 processors busy (k80-a, k80-b, gpt4). Queued at position 1 -- will run on whichever processor frees first.
mrph> Enter the file name for saving the generated file:
```

**A pinned job (`@gpt4`) whose target is busy — queued on that processor
specifically, not the general rotation queue:**
```
mrph> [job 7] Processor(s) "gpt4" busy. Queued at position 2 for gpt4.
mrph> Enter the file name for saving the generated file:
```

**A queued job is picked up once a processor frees** (printed directly to
the console the moment the job's payload -- file name, prompt -- is both
known and dispatched; in the overwhelmingly common case both are already
true by the time the processor frees, since typing a file name and prompt
is far faster than a morph):
```
mrph> [job 4] Processor(s) k80-a now free -- starting your queued generate for "foo4.py".
```

**Completion keeps the job/processor tag so it's traceable in a busy chat**
(sent out of band via `self.send_message`, not as a reply to whatever the
user is typing next; a pinned/fan-out job's tag joins every processor it
ran on with `+`, e.g. `[job 6 · k80-a+gpt4]`):
```
mrph> [job 4 · k80-a] Your "foo4.py" file was saved.
```

If the run itself raises (a processor error not already swallowed by
`run_morphers`), the same tag is used: `mrph> [job N · <ids>] Processor
failed: <error>.` Either way the processor(s) are released and whatever is
queued for them is dispatched next, so one bad run never leaves a slot
stuck.

The `[job N]` tag is a simple per-session incrementing counter assigned when
the `/generate` command is first parsed, so the user (and the transcript)
can match an eventual completion message back to the request that queued
it, even after several other `/generate` calls and completions have
interleaved in between.

## Worked example (pool of 3, five files queued back to back)

| # | Command                        | Immediate reply                                          | What happens                                                    |
|---|--------------------------------|------------------------------------------------------------|-----------------------------------------------------------------|
| 1 | `/generate foo1.py`            | `Assigned to processor "k80-a" (slot 1/3 now busy)`         | starts now                                                      |
| 2 | `/generate foo2.py`            | `Assigned to processor "k80-b" (slot 2/3 now busy)`         | starts now                                                      |
| 3 | `/generate foo3.py`            | `Assigned to processor "gpt4" (slot 3/3 now busy)`          | starts now                                                      |
| 4 | `/generate foo4.py`            | `All 3 processors busy (...). Queued at position 1 ...`    | waits                                                           |
| 5 | `/generate foo5.py`            | `All 3 processors busy (...). Queued at position 2 ...`    | waits                                                           |
| — | *(`k80-a` finishes `foo1.py`)* | `[job 4] Processor(s) k80-a now free -- starting your queued generate for "foo4.py".` | job 4 dequeued, cursor unaffected (dequeue, not fresh rotation) |
| — | *(`gpt4` finishes `foo3.py`)*  | `[job 5] Processor(s) gpt4 now free -- starting your queued generate for "foo5.py".`  | job 5 dequeued                                                  |
| — | *(`k80-b` finishes `foo2.py`)* | *(queue empty, nothing dispatched)*                         | `k80-b` goes idle, available for the next fresh `/generate`     |

Each of rows 1-5 is also followed, once its background morph finishes, by
its own out-of-band completion line, e.g. `mrph> [job 1 · k80-a] Your
"foo1.py" file was saved.` — omitted above to keep the table focused on the
scheduling decisions.

At steady state, three files are always morphing concurrently (bounded by
pool size) while the rest of the backlog waits its turn, and the chat always
tells the user, at the moment they type a `/generate`, exactly which
processor got it or where in line it landed.

## Letting the chat state machine accept overlapping `/generate` calls

Before this change, `/generate` walked a strictly linear chain of states in
`flows/morph.py` — `/start` → `/generate_file_name_input` →
`/generate_prompt_input` → back to `/start` — and the final step,
`morph_and_save`, blocked that chain until the LLM call finished and the
file was written. That is fine for one job at a time, but it is exactly what
prevented the user from firing off a second `/generate` while the first was
still morphing: the conversation state machine had no notion of a job
running "in the background" of the main menu.

Supporting the round-robin pool above requires the state machine to treat
job **intake** (parsing the processor spec/rotation slot, collecting a file
name and a prompt) as separate from job **execution** (running the LLM call
against the assigned processor and writing the file). Concretely, this maps
onto `flows/morph.py` and `scheduler.py` as follows:

- Intake still happens through the existing linear conversation
  (`/generate` → file name → prompt), exactly as today, so the user
  experience of *starting* a `/generate` is unchanged. `intake_job` calls
  `scheduler.submit(...)` the moment `/generate`/`/patch` is parsed —
  before the file name is even asked for — creating a `Job` that already
  knows whether it is running, queued for the rotation, or queued for one
  or more pinned processors.
- Once the prompt is collected, `morph_and_save` builds the dialog and
  calls `scheduler.attach_launch(job, launch)`, where `launch` is a
  zero-argument callback that fires exactly once — immediately if the job
  already has processor(s) assigned, or later, from inside `release()`,
  the moment a slot it is waiting on frees up. Either way,
  `morph_and_save` itself `await`s only `attach_launch` and the nested
  transition, not the morph, so the conversation returns to `/start`
  right away and the user is free to type another `/generate` immediately
  — pool permitting, it starts on a different processor; otherwise it
  queues.
- The morph itself runs as a detached `asyncio` task (`asyncio.ensure_future`
  inside `launch`). Job completion (or a queued job's dispatch) is reported
  to the chat as an out-of-band `print`/`send_message` call tagged with its
  `[job N]` or `[job N · processor(s)]` id, rather than as the reply to
  whatever command the user happens to be typing next. This is why every
  status message above is self-contained and names the job explicitly —
  several may interleave with unrelated `/start` menu traffic or with the
  intake of yet another `/generate`.
- Because intake no longer waits on execution, the same chat can have
  multiple jobs in flight simultaneously (up to the pool size actually
  running, plus any number queued behind them). `/settings` shows this
  directly: each configured processor's description line is followed by
  `idle`, `idle (N waiting)`, `running job N for Ns` or `running job N for
  Ns (M waiting)`, and, when anything is queued, a trailing `"K job(s)
  waiting for a free processor."` line.

## Interaction with existing fan-out

- `/generate @all` with a 3-processor pool behaves as it does today — one
  job occupies all three slots for its duration — except now it also
  correctly reports "all processors busy" to any *other* `/generate` typed
  while it runs, and queues that other request rather than silently
  round-robining onto a processor that is actually occupied by the fan-out
  job.
- `/generate @k80-a,@gpt4` occupies two of the three slots; `k80-b` remains
  available for the rotation to hand to the very next plain `/generate`.
- `/patch` follows the same pool/queue/rotation rules as `/generate`, since
  it goes through the same `morph_and_save` finisher today.

## Configuration

No new environment variables are introduced. Pool size is simply
`len(registry.ids)` — however many `MRPH_PROCESSOR_<ID>_*` blocks (or legacy
single-instance variables) are configured, as already documented in
[`README.md`](../README.md#25-multi-agent-setup-many-processor-instances) and
[`documentation/modeling-approach.md`](./modeling-approach.md). A pool of one
configured processor degrades this whole scheme to today's behaviour: every
`/generate` simply queues behind the previous one on that single processor.
