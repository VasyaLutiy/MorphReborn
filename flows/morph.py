import os
import re
import sys
import copy
import json
import time
import asyncio
import functools
import traceback
import pkg_resources

from pysyun.conversation.flow.console_bot import ConsoleBot

from context_folder_dialog import ContextFolderDialog
from llm_dialog import LLMDialog
from scheduler import JobScheduler
from settings import load_settings, load_registry

from cards.schema import CardError
from cards.deck import DeckError
from cards.generations import ensure_parent_dir, is_truncated_response, run_deck
from cards.store import (
    DeckStore,
    StoreError,
    begin_run,
    build_deck_status,
    collect_generation,
    list_runs,
    make_card_committer,
    record_run,
    recover_orphaned_local_batch,
    submit_generation,
)


# The orchestrator prompt for /card decomposition mode: turn one goal into a
# deck of morph cards. Kept verbatim here (not in cards/, which must stay free of
# prompt/presentation concerns) so a reviewer can read exactly what the model is
# asked. See documentation/batch-orchestrator.md, "Where cards come from".
DECOMPOSE_INSTRUCTION = '''You are the ORCHESTRATOR of a batch code-morphing system. \
Decompose the following goal into a deck of morph cards. You write NO code \
yourself -- each card is a self-contained job a separate executor will run.

GOAL:
{goal}

A morph card is a JSON object with these fields:
  custom_id      - unique id, characters [A-Za-z0-9._-]; names the job and its output file
  intent         - one of "generate", "patch", "todo"
  target         - the single file this card writes
  targets        - INSTEAD of "target": the list of files this card writes as ONE
                   atomic set (a module AND its test, say). Use it whenever a change
                   only makes sense applied to several files together -- the set is
                   accepted or rolled back whole. Give exactly one of target/targets
  context_slice  - list of files the executor must see; use paths that EXIST in the
                   project OR are the "target" of an EARLIER card in this deck;
                   [] means "the whole project"
  acceptance     - a runnable shell command that exits 0 when the morph is correct
                   (e.g. "pytest tests/test_foo.py -q"); optional but strongly preferred
  variants       - integer >= 1: how many samples to try (best-of-N)
  depends_on     - list of custom_ids that must complete first; a card may only read
                   another card's target if it depends_on that card
  instruction    - the natural-language instruction for the executor

RULES:
  - Output ONLY a JSON array of card objects -- no commentary. A single fenced
    ```json code block wrapping the array is acceptable.
  - Every "acceptance" value must be a real, runnable shell command.
  - A card names exactly one of "target" and "targets", never both.
  - Every "context_slice" path must already exist in the project OR be the target
    of another card in this array.
  - Use depends_on so no card reads a file another card in the same generation is
    still writing (dependent changes serialize into later generations).
  - Use the flat card shape, e.g.:
    {{"custom_id":"gen-foo","intent":"generate","target":"foo.py",
      "context_slice":["bar.py"],"acceptance":"pytest tests/test_foo.py -q",
      "variants":1,"depends_on":[],"instruction":"..."}}
'''


def filter_source_code_file_names(file_path):

    if 'node_modules' in file_path:
        return False

    if 'typechain-types' in file_path:
        return False

    if 'venv/' in file_path:
        return False

    if 'venvy/' in file_path:
        return False

    return (
            file_path.endswith('Dockerfile') or
            file_path.endswith('package.json') or
            file_path.endswith('requirements.txt') or
            file_path.endswith('.md') or
            file_path.endswith('.dot') or
            # file_path.endswith('.env') or
            file_path.endswith('.py') or
            file_path.endswith('.sol') or
            file_path.endswith('.sh') or
            file_path.endswith('.rs') or
            file_path.endswith('.js') or
            file_path.endswith('.jsx') or
            file_path.endswith('.go') or
            file_path.endswith('.fc') or
            file_path.endswith('.yaml') or
            file_path.endswith('.yml') or
            file_path.endswith('.sql') or
            file_path.endswith('.ino') or
            file_path.endswith('.proto') or
            file_path.endswith('.txt') or
            file_path.endswith('.ts') or
            file_path.endswith('.tsx')
    )


# ``/collect wait`` polling. WHY these numbers. A cloud batch queue runs 10-40
# minutes, and a card with the default two regenerations chains up to three of
# them, so the timeout is sized for the worst case a single generation can
# honestly produce -- past that, something is wrong and an operator should be
# told rather than kept waiting. The interval is the granularity that actually
# helps: a batch that lands is noticed within half a minute, and an hour of
# waiting costs 120 log lines instead of 3600. Nothing is lost at the timeout:
# the batch is still in flight and still collectable by the next /collect.
COLLECT_WAIT_POLL_SECONDS = 30.0
COLLECT_WAIT_TIMEOUT_SECONDS = 2 * 60 * 60.0


def format_elapsed(seconds):
    """``seconds`` as a compact ``1h02m``/``12m30s``/``45s`` for a progress line."""
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


def describe_collect_wait(result, batch_id, elapsed):
    """The progress line one poll of ``/collect wait`` prints.

    Names what is being waited for -- the generation, or the regeneration and
    the cards in it -- which batch id is in the queue, and how long the wait has
    been going. An operator reading this can tell a slow batch from a hung one,
    and can look the batch up on the provider's side; the silent hour that this
    replaces could do neither.
    """
    waited = format_elapsed(elapsed)
    where = f"generation {result.generation_number}/{result.total_generations}"
    if result.retry_in_flight:
        cards = ", ".join(result.retry_card_ids) or "?"
        where = (f"regeneration {result.retry_attempt}/{result.retry_limit} of "
                 f"{cards} in {where}")
    return f"mrph> Waiting for {where} (batch {batch_id}) -- {waited} elapsed."


def describe_collect_progress(result):
    """What a single-poll ``/collect`` says when the generation is not in yet.

    Two different pieces of news: a batch still in a queue (nothing happened,
    ask again later), or a regeneration this very call submitted (cards failed
    acceptance and are being rewritten -- ``/collect`` again picks THAT batch up,
    it is never re-sent).
    """
    if not result.retry_in_flight:
        return (f"mrph> Generation {result.generation_number}/"
                f"{result.total_generations} is still in progress -- "
                f"try /collect again in a moment.")
    cards = ", ".join(result.retry_card_ids)
    verb = "submitted" if result.retry_submitted else "still in flight"
    return (f"mrph> Generation {result.generation_number}/"
            f"{result.total_generations}: regeneration {result.retry_attempt}/"
            f"{result.retry_limit} for {cards} {verb} "
            f"(batch {result.retry_batch_id}) -- run /collect again to pick it "
            f"up. Running it while it is in flight costs nothing.")


def build_current_project_context():
    # Load the current folder context
    context_folder = ContextFolderDialog(".", filter_callback=filter_source_code_file_names)
    context_folder.process([])

    return context_folder


class MorphBot(ConsoleBot):

    def __init__(self, token):

        super().__init__(token)

        load_settings()

        # The registry of every named processor instance configured in ``.env``.
        # This is the backbone of the multi-agent behaviour: a single command can
        # fan a morph out across many of these instances at once.
        self.registry = load_registry()

        # One slot per configured processor; queues/round-robins a sequence of
        # independent /generate or /patch jobs across the pool. See
        # documentation/parallel-generate-scheduling.md.
        self.scheduler = JobScheduler(self.registry.ids)

        # The welcome banner is only shown once, on the very first /start;
        # later returns to the main menu show just the menu.
        self.shown_welcome = False

        # The batch backend a /submit is in flight on, held so the matching
        # /collect in the same session reuses it. A LocalBatchBackend keeps its
        # in-flight batch in memory (worker threads), so /collect MUST use the
        # very instance that /submit fired; a cloud backend is reconstructable
        # from state after a restart (see build_collect_transition).
        self._active_backend = None

    # -- processor selection ----------------------------------------------

    @staticmethod
    def parse_processor_spec(text):
        """Extract the processor identifiers requested on a command line.

        Accepts forms such as ``/generate @k80``, ``/generate @k80,@gpt4`` or
        ``/generate @all``. Returns a list of raw tokens (``@`` stripped), or
        ``None`` when no identifier was supplied.
        """
        if not text:
            return None
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            return None
        tokens = [token.lstrip("@") for token in re.split(r"[\s,]+", parts[1].strip())]
        tokens = [token for token in tokens if token]
        return tokens or None

    @staticmethod
    def split_run_flags(text):
        """Split a ``/submit`` / ``/nightly`` line into ``(text, use_git)``.

        The one flag is ``nogit``: this run opens no branch and writes no
        commits. WHY a word on the command that STARTS a run, rather than a
        setting or a new command (the ``/deck reset`` and ``/collect wait``
        pattern): the decision belongs to one run, is taken at the moment the run
        begins, and is exactly the answer to the refusal a dirty tree produces --
        so the message that refuses can name the words that proceed anyway.

        The flag is stripped BEFORE :meth:`resolve_batch_backend` sees the line,
        because that reads every remaining token as a processor id and would
        answer "no matching processor" to a bare ``/submit nogit``.
        """
        if not text:
            return text, True
        tokens = text.split()
        remaining = [token for token in tokens[1:] if token.lower() != "nogit"]
        use_git = len(remaining) == len(tokens) - 1
        return " ".join(tokens[:1] + remaining), use_git

    @staticmethod
    def git_notes(lines):
        """The git lines a store call logged, for the chat.

        The store logs a running commentary the CLI otherwise discards (it
        renders its own summary from the returned result), but the git lines
        report things no result object carries -- which branch the run opened,
        which card became which commit, what git refused. Selecting them by
        prefix keeps that one channel open without reopening the rest.
        """
        return [line for line in lines if line.startswith("mrph> git:")]

    @staticmethod
    def resolve_processor_ids(registry, spec):
        """Turn a raw spec into a concrete, de-duplicated list of known ids.

        ``None``/empty -> the single default processor (legacy behaviour).
        ``all``/``*`` -> every configured processor (full multi-agent fan-out).
        """
        available = registry.ids
        if not spec:
            default = registry.default_id()
            return [default] if default else []

        resolved = []
        for token in spec:
            if token.lower() in ("all", "*"):
                return list(available)
            if token in available and token not in resolved:
                resolved.append(token)
        return resolved

    @staticmethod
    def clean_conversation(dialog):
        # Make a deep copy of the conversation and strip the transport-only time fields.
        conversation = copy.deepcopy(dialog.conversation)
        for message in conversation:
            if 'time' in message:
                del message['time']
        return conversation

    @staticmethod
    async def run_morphers(registry, processor_ids, dialog):
        """Run every selected processor concurrently and collect their morphs.

        Each processor call is blocking (network I/O), so it is dispatched to a
        thread; ``asyncio.gather`` then lets all of them run in parallel. Returns
        an ``{id: response_or_None}`` mapping (``None`` marks a failed instance).
        """
        conversation = MorphBot.clean_conversation(dialog)
        loop = asyncio.get_event_loop()

        async def run_one(processor_id):
            try:
                value = await loop.run_in_executor(None, registry.run, processor_id, conversation)
                return processor_id, value
            except Exception as error:
                print(f"llm[{processor_id}]> processor failed: {error}")
                return processor_id, None

        pairs = await asyncio.gather(*[run_one(pid) for pid in processor_ids])
        return dict(pairs)

    @staticmethod
    def output_file_name(file_name, processor_id, multi):
        """Single processor keeps the target name; parallel morphs get suffixed."""
        if not multi:
            return file_name
        root, ext = os.path.splitext(file_name)
        return f"{root}.{processor_id}{ext}"

    @staticmethod
    def response_to_file_body(response, append_if_plain=False):
        """Extract the file body from a response and choose the write mode.

        Mirrors ``cards.generations.response_to_file_body`` (which has no append
        mode). Callers must reject a CUT-OFF answer first -- one that opened a
        ``` fence and never closed it -- with
        :func:`cards.generations.is_truncated_response`: the "no fenced block"
        branch below cannot tell that from a model answering in bare code, and
        would write the literal ```` ```python ```` line into the file. That
        detector is IMPORTED rather than mirrored a second time: the layering
        rule forbids ``cards`` importing ``flows``, not the reverse, and one copy
        of the rule is what keeps the two paths honest about the same defect.
        """
        code_blocks = re.findall(r"```(.*?)\n(.*?)\n```", response, re.DOTALL)
        if 0 < len(code_blocks):
            body = "".join(f"{code_block}\n" for _, code_block in code_blocks)
            return body, 'w'
        return response, ('a' if append_if_plain else 'w')

    async def morph_and_save(self, action, dialog, file_name, nested_transition,
                             append_if_plain=False):
        """Shared finisher: hand the dialog off to whichever processor(s) the
        scheduler already assigned (or is about to, once they free up), then
        return to the nested transition immediately -- the actual morph runs
        in the background so the user can start another /generate right away.
        See documentation/parallel-generate-scheduling.md."""
        job = self.context_get(action, "job")
        chat_id = action["update"]["effective_chat"]["id"]

        if job is None:
            text = "mrph> No matching processor. Configure one (see \"/settings\") " \
                   "or pick a valid id, e.g. \"/generate @all\"."
            await action["context"].bot.send_message(chat_id=chat_id, text=text)
            await nested_transition(action)
            return

        async def run():
            processor_ids = job.assigned
            multi = 1 < len(processor_ids)
            tag = f"[job {job.id} · {'+'.join(processor_ids)}]"
            try:
                results = await self.run_morphers(self.registry, processor_ids, dialog)

                saved = []
                cut_off = []
                for processor_id in processor_ids:
                    response = results.get(processor_id)
                    print(f"llm[{processor_id}]> {response}")
                    if response is None:
                        continue
                    if is_truncated_response(response):
                        # An answer that ran out of output budget inside a code
                        # fence is a corrupt response, not a file body. Saving it
                        # verbatim used to put the ```` ```python ```` line on
                        # disk and hand the user a file that will not even parse;
                        # the honest report is that the answer was cut off.
                        cut_off.append(processor_id)
                        continue
                    out_name = self.output_file_name(file_name, processor_id, multi)
                    body, mode = self.response_to_file_body(response, append_if_plain)
                    # A target may name a directory that does not exist yet
                    # ("morph_mcp/jsonrpc.py"); open() alone would fail on it.
                    ensure_parent_dir(out_name)
                    with open(out_name, mode, encoding='utf-8') as file:
                        file.write(body)
                    saved.append(out_name)

                if saved:
                    if multi:
                        text = f"mrph> {tag} Saved (parallel):\n  " + "\n  ".join(saved)
                    else:
                        text = f"mrph> {tag} Your \"{saved[0]}\" file was saved."
                    if cut_off:
                        text += (f"\n  cut off mid-file, nothing saved for: "
                                 f"{', '.join(cut_off)}")
                elif cut_off:
                    text = (f"mrph> {tag} The answer was cut off mid-file -- it opened a "
                            f"``` code fence and never closed it, so nothing was saved. "
                            f"Ask again (a shorter file, or a smaller context).")
                else:
                    text = f"mrph> {tag} No morph was produced (all selected processors failed)."
            except Exception as error:
                text = f"mrph> {tag} Processor failed: {error}"

            await self.send_message(chat_id=chat_id, text=text)
            self.scheduler.release(processor_ids)

        def launch():
            if job.was_queued:
                pids = ", ".join(job.assigned)
                print(f"mrph> [job {job.id}] Processor(s) {pids} now free -- "
                      f"starting your queued generate for \"{file_name}\".")
            asyncio.ensure_future(run())

        self.scheduler.attach_launch(job, launch)
        await nested_transition(action)

    @staticmethod
    def context_get(action, name, default=None):
        try:
            return action["context"].get(name)
        except KeyError:
            return default

    def intake_job(self, action):
        """Parse the processor spec for a `/generate` or `/patch` command,
        reserve its slot or queue position on the scheduler right away, and
        return the status line to show the user. The processor(s) this job
        will actually run on are decided here -- before the file name or
        prompt are even collected. See
        documentation/parallel-generate-scheduling.md."""
        spec = self.parse_processor_spec(action.get("text"))

        if spec:
            pinned_ids = self.resolve_processor_ids(self.registry, spec)
            if not pinned_ids:
                action["context"].add("job", None)
                return "mrph> No matching processor. Please, configure one as stated in \"/settings\"."
        else:
            pinned_ids = None
            if not self.registry.ids:
                action["context"].add("job", None)
                return "mrph> No matching processor. Please, configure one as stated in \"/settings\"."

        job = self.scheduler.submit(pinned_ids)
        action["context"].add("job", job)
        return self.describe_assignment(job)

    def describe_assignment(self, job):
        """Render the "[job N] Assigned to ..." / "Queued at position ..."
        status line for a just-submitted job."""
        pool_size = len(self.scheduler)

        if job.assigned:
            ids = job.assigned
            if 1 < len(ids):
                return f"mrph> [job {job.id}] Assigned to processors {', '.join(ids)} " \
                       f"(slot {self.scheduler.busy_count()}/{pool_size} now busy)."
            return f"mrph> [job {job.id}] Assigned to processor \"{ids[0]}\" " \
                   f"(slot {self.scheduler.busy_count()}/{pool_size} now busy)."

        position = self.scheduler.queue_position(job)
        if job.is_pinned:
            pins = ", ".join(job.pinned_ids)
            return f"mrph> [job {job.id}] Processor(s) \"{pins}\" busy. " \
                   f"Queued at position {position} for {pins}."
        pool_desc = ", ".join(self.scheduler.pool)
        return f"mrph> [job {job.id}] All {pool_size} processors busy ({pool_desc}). " \
               f"Queued at position {position} -- will run on whichever processor frees first."

    def build_settings_transition(self):

        async def transition(action):

            text = "mrph>\n"

            if len(self.registry):
                default_id = self.registry.default_id()
                text += f"Configured processors ({len(self.registry)}), " \
                        f"default is \"{default_id}\":\n\n"
                for identifier, description in zip(self.registry.ids, self.registry.describe_all()):
                    status = self.scheduler.describe_status(identifier)
                    text += f"    {description} -- {status}\n"
                queued = self.scheduler.queue_length()
                if queued:
                    text += f"\n{queued} job(s) waiting for a free processor.\n"
                text += "\n"
                text += "Address them by id: \"/generate @<id>\" or \"/patch @<id>\".\n"
                text += "Run several in parallel (multi-agent): " \
                        "\"/generate @a,@b\" or \"/generate @all\".\n" \
                        "A plain \"/generate\" (no @id) rides the round-robin pool: " \
                        "back-to-back calls fan out one file per processor, queuing once " \
                        "all slots are busy.\n\n"
            else:
                text += '''No processors are configured yet.
--------------------------------------------------
        HOW TO CONFIGURE PROCESSORS?

    Each processor is a named instance in your ".env" file. Define as many as
    you like - many instances of each backend are allowed (multi-agent):

          MRPH_PROCESSORS=k80-a,k80-b,gpt4

          MRPH_PROCESSOR_k80-a_TYPE=llama_cpp
          MRPH_PROCESSOR_k80-a_ENDPOINT_URI=http://192.168.0.14:8080/v1
          MRPH_PROCESSOR_k80-a_MODEL=k80-model

          MRPH_PROCESSOR_k80-b_TYPE=llama_cpp
          MRPH_PROCESSOR_k80-b_ENDPOINT_URI=http://192.168.0.15:8080/v1
          MRPH_PROCESSOR_k80-b_MODEL=k80-model

          MRPH_PROCESSOR_gpt4_TYPE=openai
          MRPH_PROCESSOR_gpt4_API_KEY=<YOUR_API_KEY>
          MRPH_PROCESSOR_gpt4_MODEL=gpt-4o

    Supported TYPE values: llama_cpp, ollama, openai.
    (The classic LLAMA_CPP_*, OLLAMA_*, OPENAI_* variables still work and map
    to the default ids "llama_cpp", "ollama" and "openai".)
--------------------------------------------------
'''

            nested_transition = self.build_menu_response_transition(text, ["Start", "Exit"])
            await nested_transition(action)

        return transition

    @staticmethod
    def build_help_transition():
        async def transition(action):
            # Updated help text to reflect current bot features
            text = '''/generate - Generate a new file for your project based on a description or selection.
/patch - Update an existing file by providing instructions on what needs to be changed.
/settings - List your configured processor instances and the default one.
/graph - Display a Graphviz representation of this bot's API for better visualization.
/help - Show this help message with the list of available commands.
/exit - Exit the application gracefully.

Morph 2.0 batch orchestrator (see documentation/batch-orchestrator.md):
/deck - Show the backlog, its generations and each card's status ("/deck reset" discards the run state, keeping the backlog; "/deck clear" empties the backlog, keeping the run state; "/deck runs" lists the archived runs).
/card - Add a card: "/card" pastes one as JSON; "/card <goal>" decomposes a goal into cards.
/submit - Compile and submit the current generation ("@id" pins a processor, "@all" the local pool).
/collect - Fetch, verify and integrate the submitted generation, then advance ("/collect wait" polls until it lands, printing progress).
/nightly - Run the whole deck generation by generation in one blocking pass.

Git (a run is a branch, a card is a commit):
    In a git working copy a run opens "morph/<deck-id>" from the current HEAD
    and commits each accepted card on its own -- exactly the files that card
    wrote, with its custom_id, model, winning variant and acceptance command in
    the commit trailers, so "git log" answers where a line came from and
    "git checkout" undoes the run. A dirty tree refuses to start the run.
    /submit nogit  (or /nightly nogit) - run without touching git at all: no
    branch, no commits. Decided when the run STARTS; a later /submit of the same
    run follows what the first one chose. Outside a git repository, or with no
    git on PATH, a run degrades to exactly this with one warning line.
    When a run finishes it is archived under .morph/runs/<deck-id>/ (the deck as
    executed plus a report of every card's outcome) and "/deck runs" lists what
    has run.

Choosing processors (multi-agent):
    /generate            - ride the round-robin pool (whichever processor is free next).
    /generate @gpt4      - use the processor with id "gpt4".
    /generate @k80,@gpt4 - run both in parallel; each writes its own file (foo.<id>.ext).
    /generate @all       - fan out across every configured processor.
    (The same @id syntax works for /patch.)

Queuing: fire off several /generate calls back to back -- one per file -- and
each is picked up by whichever processor is free, queuing automatically once
every slot is busy. /settings shows what is idle, busy or queued.
            '''

            await action["context"].bot.send_message(chat_id=action["update"]["effective_chat"]["id"], text=text)

        return transition

    def build_generate_transition(self):

        async def transition(action):

            # Decides this job's processor(s)/queue position right away.
            status = self.intake_job(action)
            job = self.context_get(action, "job")

            text = status
            if job is not None:
                text += "\nmrph> Enter the file name for saving the generated file:"

            await action["context"].bot.send_message(chat_id=action["update"]["effective_chat"]["id"], text=text)

        return transition

    def build_patch_transition(self):

        async def transition(action):

            status = self.intake_job(action)
            job = self.context_get(action, "job")

            text = status
            if job is not None:
                text += "\nmrph> Enter the file name to be patched:"

            await action["context"].bot.send_message(chat_id=action["update"]["effective_chat"]["id"], text=text)

        return transition

    def build_generate_file_name_input_transition(self):

        async def transition(action):
            file_name = action['text']
            action["context"].add("generate_file_name", file_name)
            text = f"mrph> Ok, I will create a file \"{file_name}\" when finished. What should be in this file?"
            await self.build_menu_response_transition(text, ['Unit test', 'Statement flow', '*'])(action)

        return transition

    def build_patch_file_name_input_transition(self):

        async def transition(action):

            file_name = action['text']
            action["context"].add("patch_file_name", file_name)
            text = f"mrph> Loaded \"{file_name}\". How to augment that?"

            await self.build_menu_response_transition(text, ['/todo (criticize and add comments)', '*'])(action)

        return transition

    def build_generate_prompt_input_transition(self, nested_transition):

        async def transition(action):

            prompt = action['text']

            # The dialog
            dialog = LLMDialog()
            dialog += build_current_project_context()
            dialog.assign("user", prompt)

            file_name = action["context"].get("generate_file_name")
            await self.morph_and_save(action, dialog, file_name, nested_transition)

        return transition

    def build_generate_unit_test_prompt_input_transition(self, nested_transition):

        async def transition(action):

            prompt = action['text']

            # The dialog
            dialog = LLMDialog()
            dialog += build_current_project_context()
            dialog.assign("user", "Please, generate a unit test for my project.")
            dialog.assign("user", prompt)

            file_name = action["context"].get("generate_file_name")
            await self.morph_and_save(action, dialog, file_name, nested_transition)

        return transition

    def build_generate_statement_flow_prompt_input_transition(self, nested_transition):

        async def transition(action):

            prompt = action['text']

            # The dialog
            dialog = LLMDialog()
            dialog += build_current_project_context()
            message = f"Please, generate a logical statement flow graph in the Graphviz format for the \"{prompt}\" " \
                      f"method. When necessary, expand methods being called."
            dialog.assign("user", message)

            file_name = action["context"].get("generate_file_name")
            await self.morph_and_save(action, dialog, file_name, nested_transition)

        return transition

    @staticmethod
    def build_generate_prompt_input_unit_test_choice_transition():

        async def transition(action):

            file_name = action["context"].get("generate_file_name")

            text = f"mrph> Will generate the \"{file_name}\" unit test. Please, describe, what should " \
                   f"be checked in this test."
            await action["context"].bot.send_message(chat_id=action["update"]["effective_chat"]["id"], text=text)

        return transition

    @staticmethod
    def build_generate_prompt_input_statement_flow_choice_transition():

        async def transition(action):

            file_name = action["context"].get("generate_file_name")

            text = f"mrph> Will generate the \"{file_name}\" statement flow Graphviz representation. Please, " \
                   f"describe, which ClassName.MethodName needs to be analyzed."
            await action["context"].bot.send_message(chat_id=action["update"]["effective_chat"]["id"], text=text)

        return transition

    def build_patch_prompt_input_transition(self, nested_transition):
        async def transition(action):

            file_name = action['context'].get("patch_file_name")

            try:

                # Load file contents
                with open(file_name, 'r', encoding='utf-8') as file:
                    file_contents = file.read()

                prompt = action['text']

                # The dialog
                dialog = LLMDialog()
                dialog.assign("assistant", f"Let's update the {file_name} file provided.") # Original: "system"
                dialog.assign("assistant", f"Original file:\n\n---\n{file_contents}\n---\n")
                dialog += build_current_project_context()
                dialog.assign("user", prompt)

                # Augment the file contents based on the user's prompt, fanning out
                # across the selected processor(s). For a plain (non-code-block)
                # response the morph is appended to the original file.
                await self.morph_and_save(action, dialog, file_name, nested_transition,
                                          append_if_plain=True)
            except Exception as e:
                text = f"mrph> An error occurred while processing the file: {str(e)}"
                await action["context"].bot.send_message(chat_id=action["update"]["effective_chat"]["id"], text=text)

        return transition

    def build_todo_transition(self, nested_transition):
        async def transition(action):

            file_name = action['context'].get("patch_file_name")

            try:
                # Load file contents
                with open(file_name, 'r', encoding='utf-8') as file:
                    file_contents = file.read()

                # The dialog
                dialog = LLMDialog()
                dialog.assign("assistant", f"Let's update the {file_name} file provided.") # Original: "system"
                dialog.assign("assistant", f"Original file:\n\n---\n{file_contents}\n---\n")
                dialog.assign("user", "Please, criticize this file contents and add \"TODO:\" comments, saying, "
                                      "what can be improved.")

                await self.morph_and_save(action, dialog, file_name, nested_transition)
            except Exception as e:
                text = f"mrph> An error occurred while processing the file: {str(e)}"
                await action["context"].bot.send_message(chat_id=action["update"]["effective_chat"]["id"], text=text)

        return transition

    # -- Morph 2.0: batch orchestrator ------------------------------------

    def resolve_batch_backend(self, text):
        """Resolve a ``/submit``/``/nightly`` spec to a batch backend + label.

        ``None`` spec (bare command) -> the default processor's batch backend.
        ``@<id>`` -> that processor's backend (``registry.batch``). ``@all`` ->
        one local pool over every llama_cpp/ollama id (``registry.batch_pool``).
        Returns ``(backend, label)`` on success, or ``(None, error_message)``.
        """
        if not self.registry.ids:
            return None, "mrph> No matching processor. Configure one (see \"/settings\")."

        spec = self.parse_processor_spec(text)
        if not spec:
            default = self.registry.default_id()
            return self.registry.batch(default), default

        for token in spec:
            if token.lower() in ("all", "*"):
                local_ids = [
                    identifier for identifier in self.registry.ids
                    if self.registry.get(identifier).kind in ("llama_cpp", "ollama")
                ]
                if not local_ids:
                    return None, "mrph> \"@all\" needs at least one local " \
                                 "(llama_cpp/ollama) processor; none is configured."
                return self.registry.batch_pool(local_ids), "+".join(local_ids)

        resolved = self.resolve_processor_ids(self.registry, spec)
        if not resolved:
            return None, "mrph> No matching processor. Please, configure one as " \
                         "stated in \"/settings\"."
        chosen = resolved[0]
        return self.registry.batch(chosen), chosen

    @staticmethod
    def extract_json_array(response):
        """Parse a JSON array out of a model response (fenced or bare).

        Reuses the fenced-code-block convention of :meth:`response_to_file_body`:
        if the response carries a ```` ``` ```` block, its body is parsed, else
        the whole response is. Raises ``ValueError``/``json.JSONDecodeError`` on
        anything that is not a JSON array.
        """
        blocks = re.findall(r"```[a-zA-Z0-9]*\n(.*?)\n```", response, re.DOTALL)
        candidate = blocks[0] if blocks else response
        data = json.loads(candidate)
        if not isinstance(data, list):
            raise ValueError("expected a JSON array of morph cards")
        return data

    def _deck_text(self):
        """Render the ``/deck`` view from :func:`cards.store.build_deck_status`."""
        store = DeckStore(".")
        view = build_deck_status(store)
        if view.empty:
            return (
                "mrph> The deck is empty -- this is the planning phase, not an error.\n"
                "  Add cards two ways:\n"
                "    /card                 -- paste one morph card as JSON (nested or flat)\n"
                "    /card <goal text>     -- let the orchestrator decompose a goal into cards\n"
                "  A brand-new project starts from a genesis deck (a first card that writes\n"
                "  a spec, later cards sliced on it). See documentation/batch-orchestrator.md,\n"
                "  \"Where cards come from\"."
            )

        outcomes = store.load_outcomes()
        total = len(view.generations)
        lines = [f"mrph> Deck: {len(view.card_status)} card(s), phase: {view.phase}"]
        # A "submitted" generation is a batch sitting in a provider's queue right
        # now -- possibly for twenty minutes or more. Name the batch and the
        # processor it was submitted on, so the operator can check it on the
        # provider's side or report it: both ids are already in
        # .morph/state.json, this only renders them.
        if view.phase == "submitted":
            state = store.load_state()
            batch_id = state.get("batch_id")
            backend_label = state.get("backend_label")
            if batch_id or backend_label:
                lines.append(f"  in flight: batch \"{batch_id}\" on \"{backend_label}\" "
                             f"-- fetch it with /collect")
            # A regeneration is a batch like any other, but it is NOT the
            # generation's first attempt, and an operator reading "in flight"
            # deserves to know which it is looking at.
            retries = state.get("retries") or {}
            if retries:
                attempt = max(int(entry.get("attempt", 1)) for entry in retries.values())
                lines.append(f"    that batch is regeneration {attempt} of "
                             f"{', '.join(sorted(retries))} (acceptance failed)")
        lines.append("  generations:")
        for number, generation in enumerate(view.generations, start=1):
            marker = ""
            if view.phase != "done" and number - 1 == view.current_generation:
                marker = "   <- current"
            lines.append(f"    [{number}/{total}] {', '.join(generation)}{marker}")
        lines.append("  cards:")
        for custom_id, status in view.card_status:
            detail = ""
            outcome = outcomes.get(custom_id)
            if outcome is not None:
                if outcome.status == "written" and outcome.paths:
                    detail = " -> " + ", ".join(outcome.paths)
                elif outcome.status == "failed":
                    detail = f" (after {outcome.attempts} attempt(s))"
                elif outcome.status == "skipped" and outcome.reason:
                    detail = f" (dependency {outcome.reason})"
            lines.append(f"    {custom_id:24} {status}{detail}")
        return "\n".join(lines)

    @staticmethod
    def _runs_text():
        """Render ``/deck runs``: the archived runs, newest first.

        The backlog view answers "what is the deck doing now"; this one answers
        "what has this project run", which until the archive existed nothing
        could -- each run overwrote the last. One line per run, carrying the id
        (which is also the archive directory and the branch suffix), when it
        finished, how big it was, how it went, and the branch to check out to see
        it.
        """
        reports = list_runs(DeckStore("."))
        if not reports:
            return ("mrph> No archived runs yet. A run is archived under "
                    ".morph/runs/<deck-id>/ the moment it finishes -- the deck "
                    "as executed, plus a report of what happened to each card.")
        lines = [f"mrph> {len(reports)} archived run(s), newest first:"]
        for report in reports:
            counts = report.counts
            lines.append(
                # The stored timestamp carries milliseconds so two runs of the
                # same second still sort; a reader does not need them.
                f"    {report.deck_id}  {report.completed_at[:19]}  "
                f"{len(report.outcomes)} card(s): "
                f"{counts['written']} written, {counts['failed']} failed, "
                f"{counts['skipped']} skipped"
                f"{'  branch ' + report.branch if report.branch else '  (no branch)'}")
        lines.append("  Each run's deck and report: .morph/runs/<deck-id>/")
        return "\n".join(lines)

    def build_deck_transition(self):
        """``/deck`` shows the backlog; ``/deck reset`` and ``/deck runs`` do more.

        The reset is the only way out of a run the user wants to abandon (a deck
        already ``done``, or a batch that cannot be collected any more): the
        backlog stays, every card goes back to ``pending``. ``runs`` lists the
        archived runs instead of the backlog -- the history the backlog view
        cannot show, since ``deck.json`` only ever holds the current one. Any
        other argument to ``/deck`` is ignored -- bare ``/deck`` is a status view
        and stays one.
        """
        async def transition(action):
            chat_id = action["update"]["effective_chat"]["id"]
            arguments = (action.get("text") or "").split()
            if len(arguments) > 1 and arguments[1].lower() == "runs":
                await action["context"].bot.send_message(
                    chat_id=chat_id, text=self._runs_text())
                return
            if len(arguments) > 1 and arguments[1].lower() == "reset":
                store = DeckStore(".")
                store.reset_state()
                self._active_backend = None
                await action["context"].bot.send_message(
                    chat_id=chat_id,
                    text="mrph> Run state discarded (.morph/state.json). The backlog "
                         "is kept; every card is pending again.")

            # ``clear`` empties the BACKLOG the way ``reset`` empties the run
            # state: until now the only way to start a fresh deck was to delete
            # .morph/deck.json by hand, from outside the CLI that owns it. It
            # answers with one line and returns (as ``runs`` does): the line
            # already says what happened to each file, and a deck view of an
            # empty backlog would repeat it at greater length. A run IN FLIGHT
            # is refused rather than served -- the batch is out in a queue
            # already and paid for, and its results can only land on cards
            # that still exist, so clearing the backlog first would strand
            # them; /deck reset returns the cards to pending without touching
            # the batch, and the clear works once it has run.
            if len(arguments) > 1 and arguments[1].lower() == "clear":
                store = DeckStore(".")
                if store.load_state().get("phase") == "submitted":
                    await action["context"].bot.send_message(
                        chat_id=chat_id,
                        text="mrph> A generation is still in flight -- clearing the "
                             "backlog now would strand its results. Run /deck reset "
                             "first to return the cards to pending, then /deck clear.")
                    return
                store.clear()
                await action["context"].bot.send_message(
                    chat_id=chat_id,
                    text="mrph> Backlog emptied (.morph/deck.json). The run state is "
                         "untouched.")
                return
            await action["context"].bot.send_message(chat_id=chat_id, text=self._deck_text())

        return transition

    def build_card_prompt_transition(self):
        async def transition(action):
            chat_id = action["update"]["effective_chat"]["id"]
            text = (
                "mrph> Paste ONE morph card as JSON (a single line), nested or flat form:\n"
                "    {\"custom_id\":\"gen-foo\",\"meta\":{\"intent\":\"generate\","
                "\"target\":\"foo.py\",\"context_slice\":[\"bar.py\"],"
                "\"acceptance\":\"pytest tests/test_foo.py -q\"},"
                "\"instruction\":\"...\"}\n"
                "mrph> (or /start to cancel)"
            )
            await action["context"].bot.send_message(chat_id=chat_id, text=text)

        return transition

    def build_card_manual_input_transition(self, nested_transition):
        async def transition(action):
            chat_id = action["update"]["effective_chat"]["id"]
            store = DeckStore(".")
            try:
                data = json.loads(action["text"])
            except json.JSONDecodeError as error:
                await action["context"].bot.send_message(
                    chat_id=chat_id, text=f"mrph> Not valid JSON: {error}")
                await nested_transition(action)
                return
            try:
                card = store.add_card(data)
            except (CardError, DeckError) as error:
                await action["context"].bot.send_message(
                    chat_id=chat_id, text=f"mrph> {error}")
                await nested_transition(action)
                return
            await action["context"].bot.send_message(
                chat_id=chat_id,
                text=f"mrph> Added card \"{card.custom_id}\" -> {card.target} "
                     f"({card.intent}).")
            await action["context"].bot.send_message(chat_id=chat_id, text=self._deck_text())
            await nested_transition(action)

        return transition

    def build_card_decompose_transition(self, nested_transition):
        async def transition(action):
            chat_id = action["update"]["effective_chat"]["id"]
            parts = action["text"].split(maxsplit=1)
            goal = parts[1].strip() if len(parts) > 1 else ""

            default = self.registry.default_id()
            if not default:
                await action["context"].bot.send_message(
                    chat_id=chat_id,
                    text="mrph> No matching processor. Configure one (see \"/settings\").")
                await nested_transition(action)
                return

            dialog = LLMDialog()
            dialog += build_current_project_context()
            dialog.assign("user", DECOMPOSE_INSTRUCTION.format(goal=goal))
            conversation = self.clean_conversation(dialog)

            await action["context"].bot.send_message(
                chat_id=chat_id,
                text=f"mrph> Decomposing the goal into morph cards via \"{default}\"...")

            loop = asyncio.get_event_loop()
            try:
                response = await loop.run_in_executor(
                    None, self.registry.run, default, conversation)
            except Exception as error:
                await action["context"].bot.send_message(
                    chat_id=chat_id, text=f"mrph> Processor failed: {error}")
                await nested_transition(action)
                return

            store = DeckStore(".")
            try:
                fragment = self.extract_json_array(response)
                added = store.add_cards(fragment)
            except (ValueError, CardError, DeckError, json.JSONDecodeError) as error:
                os.makedirs(".morph", exist_ok=True)
                raw_path = os.path.join(".morph", "last_decompose.txt")
                with open(raw_path, "w", encoding="utf-8") as handle:
                    handle.write(response)
                await action["context"].bot.send_message(
                    chat_id=chat_id,
                    text=f"mrph> Could not add the proposed cards: {error}\n"
                         f"mrph> The model's raw output was saved to \"{raw_path}\" "
                         f"for inspection. Nothing was added.")
                await nested_transition(action)
                return

            lines = [f"mrph> Added {len(added)} card(s) from the decomposition:"]
            for card in added:
                lines.append(f"    {card.custom_id} -> {card.target} ({card.intent})")
            await action["context"].bot.send_message(chat_id=chat_id, text="\n".join(lines))
            await action["context"].bot.send_message(chat_id=chat_id, text=self._deck_text())
            await nested_transition(action)

        return transition

    @staticmethod
    def report_unexpected(error, doing, aftermath=""):
        """Log an unanticipated exception and return its ``mrph>`` chat line.

        WHY this exists. ``/submit``, ``/collect`` and ``/nightly`` used to catch
        only ``CardError`` / ``DeckError`` / ``StoreError``. Anything else -- a
        ``FileNotFoundError`` from a card targeting a directory that did not
        exist yet, a provider transport error, a bug inside a morph body --
        propagated out of the transition, through the state machine, and
        terminated the whole ``mrph`` process with a traceback: an operator
        collecting an overnight deck lost the session to one bad card.

        The chat gets a compact line naming the exception CLASS and message
        (``FileNotFoundError: no such file...`` tells an operator what to fix;
        "something went wrong" does not) followed by ``aftermath`` -- what is now
        true of the run state, so the user knows whether to retry. The traceback
        itself goes to stderr, where the console log lives: available for the bug
        report, out of the conversation.
        """
        traceback.print_exc()
        text = (f"mrph> Unexpected failure while {doing}: "
                f"{type(error).__name__}: {error}")
        if aftermath:
            text += f"\n{aftermath}"
        return text

    def build_submit_transition(self, nested_transition):
        async def transition(action):
            chat_id = action["update"]["effective_chat"]["id"]
            spec, use_git = self.split_run_flags(action.get("text"))
            backend, label = self.resolve_batch_backend(spec)
            if backend is None:
                await action["context"].bot.send_message(chat_id=chat_id, text=label)
                await nested_transition(action)
                return

            store = DeckStore(".")
            # No live backend means this session did not fire what state.json
            # calls in flight; a LOCAL batch died with the process that fired it,
            # so its cards go back to pending rather than blocking /submit
            # forever. A cloud batch is left alone -- it is still collectable.
            if self._active_backend is None and recover_orphaned_local_batch(store):
                await action["context"].bot.send_message(
                    chat_id=chat_id,
                    text="mrph> The previous local batch was lost with the CLI "
                         "restart; its cards are pending again.")

            loop = asyncio.get_event_loop()
            notes = []
            try:
                result = await loop.run_in_executor(
                    None,
                    functools.partial(submit_generation, store, backend,
                                      root=".", backend_label=label,
                                      use_git=use_git, log=notes.append))
            except StoreError as error:
                await action["context"].bot.send_message(
                    chat_id=chat_id, text=f"mrph> {error}")
                await nested_transition(action)
                return
            except (CardError, DeckError) as error:
                await action["context"].bot.send_message(
                    chat_id=chat_id, text=f"mrph> Backlog is invalid: {error}")
                await nested_transition(action)
                return
            except Exception as error:
                # Anything the store did not classify. The run state is only
                # advanced to "submitted" by a successful submit, so failing
                # here leaves the generation pending and the session alive.
                await action["context"].bot.send_message(
                    chat_id=chat_id,
                    text=self.report_unexpected(
                        error, "submitting the generation",
                        "mrph> The run state was not advanced -- nothing is marked "
                        "in flight, so /submit can be retried."))
                await nested_transition(action)
                return

            if result.submitted:
                self._active_backend = backend
                lines = self.git_notes(notes) + [
                    f"mrph> Submitted generation {result.generation_number}/"
                    f"{result.total_generations} on \"{label}\" (batch {result.batch_id}):",
                    f"    cards: {', '.join(result.card_ids)}",
                ]
                for custom_id, dependency in result.skipped:
                    lines.append(f"    skipped {custom_id} (dependency {dependency})")
                lines.append("mrph> Run /collect to fetch the results.")
                text = "\n".join(lines)
            else:
                self._active_backend = None
                lines = self.git_notes(notes) + [
                    "mrph> Nothing to submit -- the deck run is complete."]
                for custom_id, dependency in result.skipped:
                    lines.append(f"    skipped {custom_id} (dependency {dependency})")
                text = "\n".join(lines)

            await action["context"].bot.send_message(chat_id=chat_id, text=text)
            await nested_transition(action)

        return transition

    def build_collect_transition(self, nested_transition):
        """``/collect`` polls once; ``/collect wait`` polls until the work lands.

        WHY an argument rather than a new command (the ``/deck reset`` pattern):
        it is the same operation with the same preconditions, differing only in
        who does the re-running -- the operator or the loop. A bare ``/collect``
        keeps its exact single-poll behaviour, which is what a script or an
        impatient human wants.

        WHY wait at all. A cloud queue runs 10-40 minutes, so ``/collect`` used
        to be a coin toss ("still in progress") and ``/nightly`` an hour of
        silence: nothing distinguished a slow batch from a hung one. The loop
        prints a line per poll naming the generation, the batch and the elapsed
        time, and -- since a regeneration is now a persisted batch of its own
        (:func:`cards.store.collect_generation`) -- carries on across it instead
        of stopping there. Every poll goes through the same single-poll call, so
        waiting cannot cost more than not waiting.
        """
        async def transition(action):
            chat_id = action["update"]["effective_chat"]["id"]
            arguments = (action.get("text") or "").split()
            waiting = len(arguments) > 1 and arguments[1].lower() == "wait"
            store = DeckStore(".")

            backend = self._active_backend
            if backend is None:
                # No live backend (e.g. a fresh session after a restart). A local
                # batch cannot survive a restart at all, so recover it: its cards
                # go back to pending and the user re-submits. A cloud batch is
                # server-side and retrievable by id, so rebuild its backend from
                # the stored label instead.
                if recover_orphaned_local_batch(store):
                    await action["context"].bot.send_message(
                        chat_id=chat_id,
                        text="mrph> The previous local batch was lost with the CLI "
                             "restart; its cards are pending again. Run /submit to "
                             "send that generation once more.")
                    await nested_transition(action)
                    return
                label = store.load_state().get("backend_label")
                if label and label in self.registry.ids:
                    backend = self.registry.batch(label)
                else:
                    await action["context"].bot.send_message(
                        chat_id=chat_id,
                        text="mrph> No in-flight batch to collect in this session. "
                             "Run /submit first, or /deck reset to discard the run "
                             "state.")
                    await nested_transition(action)
                    return

            async def send(text):
                await action["context"].bot.send_message(chat_id=chat_id, text=text)

            loop = asyncio.get_event_loop()
            started = time.monotonic()
            while True:
                notes = []
                try:
                    result = await loop.run_in_executor(
                        None,
                        functools.partial(collect_generation, store, backend,
                                          root=".", log=notes.append))
                except StoreError as error:
                    await send(f"mrph> {error}")
                    break
                except Exception as error:
                    # One bad card must not kill the CLI -- and a waiting loop
                    # must not swallow the failure into another poll either.
                    # ``collect_generation`` saves the advanced state only after
                    # every card is processed, so a failure here leaves the
                    # generation marked in flight: nothing is silently written
                    # off, and the batch (cloud batches live on the provider's
                    # side, local ones in this process) is still there to be
                    # collected again. ``self._active_backend`` is deliberately
                    # left set so the retry needs no re-resolution.
                    await send(self.report_unexpected(
                        error, "collecting the generation",
                        "mrph> The generation is still in flight and nothing was "
                        "marked done -- the results are not lost, so run /collect "
                        "again (fix the card first if the error names one)."))
                    break

                if not result.in_progress:
                    self._active_backend = None
                    lines = [
                        f"mrph> Collected generation {result.generation_number}/"
                        f"{result.total_generations}:"
                    ]
                    for custom_id, outcome in result.outcomes.items():
                        if outcome.status == "written":
                            lines.append(f"    {custom_id}: written -> {', '.join(outcome.paths)}")
                        elif outcome.status == "failed":
                            lines.append(f"    {custom_id}: failed after {outcome.attempts} attempt(s)")
                        elif outcome.status == "skipped":
                            lines.append(f"    {custom_id}: skipped (dependency {outcome.reason})")
                        else:
                            lines.append(f"    {custom_id}: {outcome.status}")
                    lines.extend(self.git_notes(notes))
                    if result.phase == "done":
                        lines.append("mrph> The deck run is complete.")
                    else:
                        lines.append("mrph> Run /submit to send the next generation.")
                    await send("\n".join(lines))
                    break

                # A generation is not always collected in one go: a card that
                # passed acceptance is committed even when a generation-mate is
                # being regenerated. Those commits are news now, not when the
                # generation finally lands.
                interim = self.git_notes(notes)
                if interim:
                    await send("\n".join(interim))

                if not waiting:
                    await send(describe_collect_progress(result))
                    break

                if result.retry_submitted:
                    # A poll that SPENT money is news of its own, distinct from
                    # the "still queued" lines around it: name the cards that
                    # failed acceptance at the moment the regeneration goes out.
                    await send(
                        f"mrph> {', '.join(result.retry_card_ids)} failed "
                        f"acceptance -- regeneration {result.retry_attempt}/"
                        f"{result.retry_limit} submitted as batch "
                        f"{result.retry_batch_id}; still waiting.")

                elapsed = time.monotonic() - started
                if COLLECT_WAIT_TIMEOUT_SECONDS <= elapsed:
                    await send(
                        f"mrph> Gave up waiting after "
                        f"{format_elapsed(elapsed)} -- nothing was lost: the "
                        f"batch is still in flight and the next /collect picks "
                        f"it up. A queue this slow is worth checking on the "
                        f"provider's side.")
                    break

                batch_id = result.retry_batch_id or store.load_state().get("batch_id")
                await send(describe_collect_wait(result, batch_id, elapsed))
                await asyncio.sleep(COLLECT_WAIT_POLL_SECONDS)

            await nested_transition(action)

        return transition

    def build_nightly_transition(self, nested_transition):
        async def transition(action):
            chat_id = action["update"]["effective_chat"]["id"]
            spec, use_git = self.split_run_flags(action.get("text"))
            backend, label = self.resolve_batch_backend(spec)
            if backend is None:
                await action["context"].bot.send_message(chat_id=chat_id, text=label)
                await nested_transition(action)
                return

            store = DeckStore(".")
            cards = store.load_cards()
            if not cards:
                await action["context"].bot.send_message(
                    chat_id=chat_id,
                    text="mrph> The deck is empty. Add cards with /card first.")
                await nested_transition(action)
                return

            # The branch has to exist before the first morph is written, and
            # /nightly writes them all inside one blocking call -- so the run is
            # opened here, and the commit hook built from what it returns.
            notes = []
            try:
                run_state = begin_run(store, cards, root=".", use_git=use_git,
                                      backend_label=label, log=notes.append)
            except StoreError as error:
                await action["context"].bot.send_message(
                    chat_id=chat_id, text=f"mrph> {error}")
                await nested_transition(action)
                return
            git_lines = self.git_notes(notes)

            await action["context"].bot.send_message(
                chat_id=chat_id,
                text="\n".join(git_lines + [
                    f"mrph> Nightly run of {len(cards)} card(s) on \"{label}\" -- "
                    f"submitting and polling each generation to completion..."]))
            del notes[:]  # reported; what follows is the run's own commentary

            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(
                    None,
                    functools.partial(run_deck, cards, backend, root=".",
                                      log=notes.append,
                                      on_accepted=make_card_committer(
                                          ".", run_state, notes.append)))
            except (CardError, DeckError) as error:
                await action["context"].bot.send_message(
                    chat_id=chat_id, text=f"mrph> Backlog is invalid: {error}")
                await nested_transition(action)
                return
            except Exception as error:
                # An overnight run is exactly where a crash costs the most. No
                # outcome is recorded (``record_run`` below never ran) and the
                # run is not archived, so the user can fix the offending card and
                # start again -- but the run WAS opened, so whatever cards were
                # accepted before the crash are already commits on its branch,
                # and the branch is still what is checked out.
                await action["context"].bot.send_message(
                    chat_id=chat_id,
                    text=self.report_unexpected(
                        error, "running the deck",
                        "mrph> No outcome was recorded and the run was not "
                        "archived; /nightly (or /submit) can be run again. "
                        "/deck reset discards the run state -- any cards "
                        "accepted before the failure are already commits, and "
                        "that branch stays checked out."))
                await nested_transition(action)
                return

            # Persist what just happened: /nightly writes every morph to disk,
            # so the run state must say so too -- otherwise the next /deck reads
            # "idle, everything pending" for work that is finished. The same call
            # archives the run under .morph/runs/<deck-id>/ and commits that
            # record as the branch's final commit.
            record_run(store, result, backend_label=label, root=".",
                       run_state=run_state, log=notes.append)
            self._active_backend = None
            await action["context"].bot.send_message(
                chat_id=chat_id,
                text="\n".join(self.git_notes(notes) + ["mrph> " + str(result)]))
            await nested_transition(action)

        return transition

    def build_version_transition(self):
        menu = self.build_menu([['Generate', 'Patch'], ['Settings', 'Help', 'Exit'], ['Graph'], ['Version']])

        async def transition(action):
            version_morph = pkg_resources.get_distribution("mrph").version

            text = f"GPT Morph CLI Bot: {version_morph}"

            await action["context"].bot.send_message(chat_id=action["update"]["effective_chat"]["id"], text=text,
                                                     reply_markup=menu)

        return transition

    @staticmethod
    def build_exit_transition():

        async def transition(_):
            sys.exit()

        return transition

    def build_menu(self, menu_items):
        buttons = [[self.build_button(item) for item in row] for row in menu_items]
        return {"keyboard": buttons, "resize_keyboard": True}

    @staticmethod
    def build_button(label):
        return label

    def build_state_machine(self, builder):
        menu_items = [["Generate", "Patch"], ["Deck"], ["Settings", "Help", "Exit"], ["Graph", "Version"]]

        welcome_transition = self.build_menu_response_transition(
            r'''┌────────────────────────────────────────────────────────────────────────────┐
│ GPT Morph :: GRANDPA v1.0.55          THE GRANDPA OF CLAUDE CODE           │
├────────────────────────────────────────────────────────────────────────────┤
│      .----------------.          > HOW CAN I HELP YOU, KIDDO?              │
│     /   _        _     \                                                   │
│    |   [ ]      [ ]     |         Grandpa writes clean code.               │
│    |       ___          |         No frameworks. No fluff.                 │
│    |      /___\         |         Memory: 64K   Wisdom: ∞                  │
│     \    .____.        /                                                   │
│      '---|____|-------'          > _                                       │
│          /|  |\                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│ "WE DEBUGGED WITH PRINT STATEMENTS."                                       │
└────────────────────────────────────────────────────────────────────────────┘
''',
            menu_items)
        short_menu_transition = self.build_menu_response_transition("mrph> Main menu:", menu_items)

        async def main_menu_transition(action):
            if self.shown_welcome:
                await short_menu_transition(action)
            else:
                self.shown_welcome = True
                await welcome_transition(action)

        return builder \
            .edge(
                "/start",
                "/start",
                "/graph",
                on_transition=self.build_graphviz_response_transition()) \
            .edge("/start", "/start", "/version", on_transition=self.build_version_transition()) \
            .edge("/start", "/start", "/start", on_transition=main_menu_transition) \
            .edge("/start", "/settings", "/settings", on_transition=self.build_settings_transition()) \
            .edge("/settings", "/start", "/exit", on_transition=self.build_exit_transition()) \
            .edge("/settings", "/start", "/start", on_transition=main_menu_transition) \
            .edge("/start", "/start", "/help", on_transition=self.build_help_transition()) \
            .edge("/start", "/start", "/exit", on_transition=self.build_exit_transition()) \
            .edge(
                "/start",
                "/start",
                "/deck",
                matcher=re.compile(r"^/deck"),
                on_transition=self.build_deck_transition()) \
            .edge(
                "/start",
                "/start",
                None,
                matcher=re.compile(r"^/card\s+\S"),
                on_transition=self.build_card_decompose_transition(main_menu_transition)) \
            .edge(
                "/start",
                "/card_input",
                None,
                matcher=re.compile(r"^/card\s*$"),
                on_transition=self.build_card_prompt_transition()) \
            .edge(
                "/start",
                "/start",
                None,
                matcher=re.compile(r"^/submit"),
                on_transition=self.build_submit_transition(main_menu_transition)) \
            .edge(
                "/start",
                "/start",
                None,
                matcher=re.compile(r"^/collect"),
                on_transition=self.build_collect_transition(main_menu_transition)) \
            .edge(
                "/start",
                "/start",
                None,
                matcher=re.compile(r"^/nightly"),
                on_transition=self.build_nightly_transition(main_menu_transition)) \
            .edge("/card_input", "/start", "/start", on_transition=main_menu_transition) \
            .edge("/card_input", "/start", "/exit", on_transition=self.build_exit_transition()) \
            .edge(
                "/card_input",
                "/start",
                None,
                matcher=re.compile("^.*$"),
                on_transition=self.build_card_manual_input_transition(main_menu_transition)) \
            .edge(
                "/start",
                "/generate_file_name_input",
                "/generate",
                matcher=re.compile(r"^/generate"),
                on_transition=self.build_generate_transition()) \
            .edge("/generate_file_name_input", "/start", "/start", on_transition=main_menu_transition) \
            .edge("/generate_file_name_input", "/start", "/exit", on_transition=self.build_exit_transition()) \
            .edge(
                "/generate_file_name_input",
                "/settings",
                "/settings",
                on_transition=self.build_settings_transition()) \
            .edge(
                "/generate_file_name_input",
                "/generate_prompt_input",
                None,
                matcher=re.compile("^.*$"),
                on_transition=self.build_generate_file_name_input_transition()) \
            .edge("/generate_prompt_input", "/start", "/start", on_transition=main_menu_transition) \
            .edge("/generate_prompt_input", "/start", "/exit", on_transition=self.build_exit_transition()) \
            .edge(
                "/generate_prompt_input",
                "/generate_unit_test_prompt_input",
                "/unit_test",
                on_transition=self.build_generate_prompt_input_unit_test_choice_transition()) \
            .edge(
                "/generate_prompt_input",
                "/generate_statement_flow_prompt_input",
                "/statement_flow",
                on_transition=self.build_generate_prompt_input_statement_flow_choice_transition()) \
            .edge(
                "/generate_prompt_input",
                "/start",
                None,
                matcher=re.compile("^.*$"),
                on_transition=self.build_generate_prompt_input_transition(main_menu_transition)) \
            .edge("/generate_statement_flow_prompt_input", "/start", "/start", on_transition=main_menu_transition) \
            .edge("/generate_statement_flow_prompt_input", "/start", "/exit", on_transition=self.build_exit_transition()) \
            .edge("/generate_unit_test_prompt_input", "/start", "/start", on_transition=main_menu_transition) \
            .edge("/generate_unit_test_prompt_input", "/start", "/exit", on_transition=self.build_exit_transition()) \
            .edge(
                "/generate_unit_test_prompt_input",
                "/start",
                None,
                matcher=re.compile("^.*$"),
                on_transition=self.build_generate_unit_test_prompt_input_transition(main_menu_transition)) \
            .edge(
                "/generate_statement_flow_prompt_input",
                "/start",
                None,
                matcher=re.compile("^.*$"),
                on_transition=self.build_generate_statement_flow_prompt_input_transition(main_menu_transition)) \
            .edge(
                "/start",
                "/patch_file_name_input",
                "/patch",
                matcher=re.compile(r"^/patch"),
                on_transition=self.build_patch_transition()) \
            .edge("/patch_file_name_input", "/start", "/start", on_transition=main_menu_transition) \
            .edge("/patch_file_name_input", "/start", "/exit", on_transition=self.build_exit_transition()) \
            .edge("/patch_file_name_input", "/settings", "/settings", on_transition=self.build_settings_transition()) \
            .edge(
                "/patch_file_name_input",
                "/patch_prompt_input",
                None,
                matcher=re.compile("^.*$"),
                on_transition=self.build_patch_file_name_input_transition()) \
            .edge("/patch_prompt_input", "/start", "/start", on_transition=main_menu_transition) \
            .edge("/patch_prompt_input", "/start", "/exit", on_transition=self.build_exit_transition()) \
            .edge(
                "/patch_prompt_input",
                "/start",
                "/todo",
                on_transition=self.build_todo_transition(main_menu_transition)) \
            .edge(
                "/patch_prompt_input",
                "/start",
                None,
                matcher=re.compile("^.*$"),
                on_transition=self.build_patch_prompt_input_transition(main_menu_transition))
