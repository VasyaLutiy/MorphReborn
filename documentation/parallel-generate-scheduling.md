# Parallel `/generate`: Round-Robin Processor Scheduling

> **Status: implemented** in `scheduler.py` (the pool/queue/round-robin
> bookkeeping) and `flows/morph.py` (wiring it into `/generate` and
> `/patch`). This document remains the behavioural spec; see
> `tests/test_scheduler.py` for the scenarios it is checked against.

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

**A slot is free — dispatched immediately:**
```
mrph> [job 3] Assigned to processor "gpt4" (slot 3/3 now busy).
mrph> Enter the file name for saving the generated file:
```

**All slots busy — queued:**
```
mrph> [job 4] All 3 processors busy (k80-a, k80-b, gpt4). Queued at
position 1 — will run on whichever processor frees first.
mrph> Enter the file name for saving the generated file:
```

**A queued job is picked up once a processor frees:**
```
mrph> [job 4] Processor "k80-a" is now free — starting your queued
generate for "foo4.py".
```

**Completion keeps the job/processor tag so it's traceable in a busy chat:**
```
mrph> [job 4 · k80-a] Your "foo4.py" file was saved.
```

The `[job N]` tag is a simple per-session incrementing counter assigned when
the `/generate` command is first parsed, so the user (and the transcript)
can match an eventual completion message back to the request that queued
it, even after several other `/generate` calls and completions have
interleaved in between.

## Worked example (pool of 3, five files queued back to back)

| # | Command                        | Immediate reply                                    | What happens                                                    |
|---|--------------------------------|----------------------------------------------------|-----------------------------------------------------------------|
| 1 | `/generate foo1.py`            | `Assigned to "k80-a" (slot 1/3 busy)`              | starts now                                                      |
| 2 | `/generate foo2.py`            | `Assigned to "k80-b" (slot 2/3 busy)`              | starts now                                                      |
| 3 | `/generate foo3.py`            | `Assigned to "gpt4" (slot 3/3 busy)`               | starts now                                                      |
| 4 | `/generate foo4.py`            | `Queued at position 1`                             | waits                                                           |
| 5 | `/generate foo5.py`            | `Queued at position 2`                             | waits                                                           |
| — | *(`k80-a` finishes `foo1.py`)* | `[job 4] "k80-a" is now free — starting "foo4.py"` | job 4 dequeued, cursor unaffected (dequeue, not fresh rotation) |
| — | *(`gpt4` finishes `foo3.py`)*  | `[job 5] "gpt4" is now free — starting "foo5.py"`  | job 5 dequeued                                                  |
| — | *(`k80-b` finishes `foo2.py`)* | *(queue empty, nothing dispatched)*                | `k80-b` goes idle, available for the next fresh `/generate`     |

At steady state, three files are always morphing concurrently (bounded by
pool size) while the rest of the backlog waits its turn, and the chat always
tells the user, at the moment they type a `/generate`, exactly which
processor got it or where in line it landed.

## Letting the chat state machine accept overlapping `/generate` calls

Today (`flows/morph.py`) a `/generate` walks a strictly linear chain of
states — `/start` → `/generate_file_name_input` →
`/generate_prompt_input` → back to `/start` — and the final step,
`morph_and_save`, blocks that chain until the LLM call finishes and the file
is written. That is fine for one job at a time, but it is exactly what
prevents the user from firing off a second `/generate` while the first is
still morphing: the conversation state machine has no notion of a job
running "in the background" of the main menu.

Supporting the round-robin pool above requires the state machine to treat
job **intake** (parsing the processor spec/rotation slot, collecting a file
name and a prompt) as separate from job **execution** (running the LLM call
against the assigned processor and writing the file):

- Intake still happens through the existing linear conversation
  (`/generate` → file name → prompt), exactly as today, so the user
  experience of *starting* a `/generate` is unchanged.
- Once the prompt is collected, execution is handed off to the scheduler
  described above and the conversation returns to `/start` immediately
  (rather than waiting on `morph_and_save`), so the user is free to type
  another `/generate` right away — pool permitting, it starts on a
  different processor; otherwise it queues.
- Job completion (or a queued job's dispatch) is reported to the chat as an
  out-of-band notification tagged with its `[job N · processor]` id, rather
  than as the reply to whatever command the user happens to be typing next.
  This is why every status message above is self-contained and names the
  job explicitly — several may interleave with unrelated `/start` menu
  traffic or with the intake of yet another `/generate`.
- Because intake no longer waits on execution, the same chat can have
  multiple jobs in flight simultaneously (up to the pool size actually
  running, plus any number queued behind them); `/settings` should show,
  alongside each processor's description, whether it is idle or currently
  running a job and for how long the queue is.

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
