"""
Generations: splitting a deck by dependency depth and running it generation
by generation.

Phase 3 of ``documentation/DEVELOPMENT_PLAN.md``. A batch request cannot see
another request's output (``documentation/batch-orchestrator.md``, "The
generation cycle"), so any card that reads a file another card writes must wait
for the next batch. This module turns a validated deck into an ordered list of
*generations* -- each a set of cards with no dependency on one another -- and
then executes them in order through a single :class:`processors.batch.BatchBackend`:
compile generation N (reading the fresh files generation N-1 just wrote),
submit, poll, collect, write the morphs, then move to generation N+1.

Two responsibilities live here:

* :func:`split_into_generations` -- a pure topological layering of the deck by
  longest dependency path. The card's stored ``generation`` field is advisory
  author input and is ignored: computed placement always wins. Cards are never
  mutated.
* :func:`run_deck` -- the generation execution loop, plus its bookkeeping in
  :class:`DeckResult`.

Constraint (see the development plan): ``cards`` must not import from ``flows``
or ``processors``. The backend arrives as a parameter (duck-typed: only
``submit`` / ``status`` / ``collect`` are called), and the small response-to-file
helper is duplicated from ``flows.morph`` rather than imported -- the same
pattern as the source filter duplicated in :mod:`cards.compiler`.

Phase 4 adds mechanical acceptance on top of this loop (see
:mod:`cards.acceptance`): a card with an ``acceptance`` command has its variants
verified best-of-N, and a card that fails verification is regenerated -- its
instruction extended with the acceptance error output -- into a retry batch that
runs before its dependents. A card *without* an ``acceptance`` command keeps the
exact Phase 3 semantics below (write every surviving variant, no verification,
no retries), so ``verify=True`` is a no-op for it.

The retry machinery comes in three pieces on purpose -- :func:`build_retry_cards`
(prepare), :func:`process_retry_batch` (judge), :func:`_run_retries` (the
blocking submit/poll loop over the two). ``/nightly`` uses all three; the
split-step ``/collect`` (:func:`cards.store.collect_generation`) uses the first
two and PERSISTS the batch between them, so a regeneration survives a restart and
a second CLI session cannot pay for the same retry twice.

Phase 7 adds the *changeset* card: a card may name a SET of ``targets``, and
one response then carries several files (:func:`response_to_files`). Splitting
the answer lives here, next to the single-file splitter it extends; writing the
set atomically -- all of it or none -- is :mod:`cards.acceptance`'s job.

Known simplification, still in force: ``flows.morph``'s ``append_if_plain`` /
``todo`` append semantics are *not* reproduced. A response with no fenced code
block is written verbatim in mode ``'w'`` (never appended). Per-card backend
routing is Phase 5 and is not built here.
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from cards.compiler import compile_card
from cards.schema import MorphCard


# -- splitting ---------------------------------------------------------------


def split_into_generations(cards: List[MorphCard]) -> List[List[MorphCard]]:
    """Layer a deck into generations by longest dependency path.

    Generation 0 holds every card with no ``depends_on``; a card otherwise lands
    in ``1 + max(generation of its dependencies)``. Within a generation the deck
    order is preserved. The result is a list of generations, index ``0`` first.

    Assumes ``cards`` already passed :func:`cards.deck.validate_deck` (acyclic,
    every dependency present), so the recursion below always terminates. The
    stored ``generation`` field on each card is ignored -- computed placement
    wins -- and no card is mutated.
    """
    by_id = {card.custom_id: card for card in cards}

    # Memoized longest-path depth. Safe against re-computation and, on an acyclic
    # deck, guaranteed to terminate.
    depth: Dict[str, int] = {}

    def generation_of(custom_id: str) -> int:
        if custom_id in depth:
            return depth[custom_id]
        card = by_id[custom_id]
        if not card.depends_on:
            result = 0
        else:
            result = 1 + max(generation_of(dep) for dep in card.depends_on)
        depth[custom_id] = result
        return result

    highest = max((generation_of(card.custom_id) for card in cards), default=-1)
    generations: List[List[MorphCard]] = [[] for _ in range(highest + 1)]
    # A second pass in deck order fills each generation, so intra-generation
    # ordering follows the deck rather than the recursion order above.
    for card in cards:
        generations[generation_of(card.custom_id)].append(card)
    return generations


# -- run result --------------------------------------------------------------


@dataclass
class CardOutcome:
    """What happened to one card in a deck run.

    ``status`` is ``"written"`` (``paths`` lists the files written -- one for a
    single-variant card, one per surviving variant otherwise), ``"failed"``
    (every variant response was ``None``, or the whole batch failed, or -- for a
    card with acceptance -- verification failed through the last retry), or
    ``"skipped"`` (a dependency failed or was itself skipped; ``reason`` names
    the blocking dependency).

    Phase 4 adds three fields, meaningful only for a card with an ``acceptance``
    command: ``attempts`` is the number of generation-level tries (1 = original
    only, 2 = one retry, and so on); ``winning_variant`` is the batch custom_id
    of the variant that passed acceptance (``None`` unless ``"written"`` via
    verification); ``acceptance_output`` is the final failure's captured output
    (``None`` when the card passed or has no acceptance). A card without
    acceptance keeps ``attempts == 1`` and both others ``None``.

    ``earlier_failures`` completes the picture for a card that was accepted only
    after failed acceptance tries: it carries the captured acceptance output of
    the LAST of those failed attempts, so the run report keeps the evidence of
    WHY the earlier tries were rejected -- the diagnosis an operator needs to
    fix a bad criterion. ``acceptance_output`` cannot hold it: it is set only
    on a terminal failure, so a card that passed on its third attempt used to
    report nothing about the first two. ``None`` for a card accepted on its
    first attempt, for a card without acceptance, and when the last failed
    attempt left no captured output at all.
    """

    custom_id: str
    status: str
    paths: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    attempts: int = 1
    winning_variant: Optional[str] = None
    acceptance_output: Optional[str] = None
    # The captured acceptance output of the LAST failed attempt of a card that
    # was eventually accepted on a retry -- the diagnosis behind the success,
    # which ``acceptance_output`` (set only on a terminal failure) drops. None
    # unless a retry succeeded after a failure that produced captured output.
    earlier_failures: Optional[str] = None

    def __str__(self) -> str:
        if self.status == "written":
            return f"{self.custom_id}: written -> {', '.join(self.paths)}"
        if self.status == "skipped":
            return f"{self.custom_id}: skipped (blocked by {self.reason})"
        return f"{self.custom_id}: failed"


# The hook a run fires the instant one card is accepted -- ``(card, outcome)``,
# with the outcome already recorded and its files final on disk.
#
# WHY a hook rather than a sweep over the finished ``outcomes`` map: the run's
# git layer (:func:`cards.store.make_card_committer`) commits exactly the files
# one card wrote, and "exactly" only holds at THIS moment. A patch chain --
# card A writes ``foo.py`` in generation 1, card B rewrites it in generation 2 --
# has one true state of ``foo.py`` per card, and a commit taken afterwards would
# attribute B's content to A and then find nothing left to commit for B. So the
# accepted card is committed where it is accepted, before the next one runs.
#
# The hook is called for accepted cards only (a failed card has nothing on disk
# to record, a skipped one never ran), it may not raise -- a run must not die of
# its bookkeeping -- and a run that passes none behaves exactly as before.
AcceptedHook = Callable[["MorphCard", "CardOutcome"], None]


def _fire(on_accepted: Optional[AcceptedHook], card: "MorphCard",
          outcome: "CardOutcome") -> None:
    """Call the accepted-card hook, if any. Kept in one place so every
    ``"written"`` branch below announces itself identically."""
    if on_accepted is not None:
        on_accepted(card, outcome)


@dataclass
class DeckResult:
    """A summary of one :func:`run_deck` execution.

    ``outcomes`` maps ``custom_id -> CardOutcome`` in the order the cards were
    processed (generation order, deck order within a generation).
    ``generations`` records the composition as executed: the ``custom_id``\\ s of
    each generation, index ``0`` first. The Phase 5 ``/deck`` status view builds
    on this.

    ``batch_ids`` lists every batch this run submitted, in submission order --
    generation batches and regeneration batches alike. The split-step path keeps
    the same list in ``state.json`` (one id is appended per ``/submit`` and per
    persisted retry); recording it here gives ``/nightly`` the same facts, which
    is what the run archive (:func:`cards.store.archive_run`) writes down: the
    provider-side ids are the only way to go back and ask what a run actually
    cost or why a batch failed.
    """

    outcomes: Dict[str, CardOutcome] = field(default_factory=dict)
    generations: List[List[str]] = field(default_factory=list)
    batch_ids: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        total = len(self.generations)
        lines = [f"deck run: {total} generation(s), {len(self.outcomes)} card(s)"]
        for index, generation in enumerate(self.generations, start=1):
            lines.append(f"  [generation {index}/{total}] {', '.join(generation)}")
        for outcome in self.outcomes.values():
            lines.append(f"  {outcome}")
        return "\n".join(lines)


# -- response to file (mirrors flows.morph) ----------------------------------


def response_to_file_body(response: str) -> str:
    """Extract the file body from a response text.

    Mirrors ``flows.morph.MorphBot.response_to_file_body``: pull the fenced code
    blocks if any are present, else use the response verbatim. Duplicated (not
    imported) to keep ``cards`` free of any dependency on ``flows``; keep the two
    in step when either changes. Callers must reject a CUT-OFF response
    (:func:`is_truncated_response`) before calling this: the verbatim fallback
    below cannot tell "the model answered with bare code" from "the model was cut
    off inside a fence", and would write the ```` ```python ```` line to disk.
    Public so :mod:`cards.acceptance` reuses this one copy rather than a third.
    Simplification for this phase: the ``append_if_plain`` / ``todo`` append mode
    is dropped -- callers here always write mode ``'w'``.
    """
    code_blocks = re.findall(r"```(.*?)\n(.*?)\n```", response, re.DOTALL)
    if 0 < len(code_blocks):
        return "".join(f"{code_block}\n" for _, code_block in code_blocks)
    return response


# A fence marker only counts at the start of a line -- that is where both the
# opening ```` ```python ```` and the closing ```` ``` ```` are written, while a
# stray ```` ``` ```` quoted mid-sentence in prose is not a delimiter.
_FENCE_MARKER = re.compile(r"^```", re.MULTILINE)


# What a card that was cut off mid-file carries into its retry, in place of the
# acceptance output it never got to produce. Written AT the executor: it is
# pasted into the next attempt's instruction by :func:`_retry_card`.
TRUNCATED_RESPONSE_MESSAGE = (
    "The previous answer was cut off mid-file: it opened a ``` code fence and "
    "never closed it, so the file body could not be extracted. Answer again "
    "with the COMPLETE file, and close the fence."
)


def is_truncated_response(response: Optional[str]) -> bool:
    """Is this answer a code fence that was never closed -- a corrupt response?

    WHY this exists. :func:`response_to_file_body` needs the CLOSING fence to
    match a block; an answer that opened ```` ```python ```` and then ran out of
    output budget matches nothing, so the "no fenced block, use it verbatim"
    fallback wrote the literal ```` ```python ```` line into the file. The card
    then failed with ``SyntaxError: invalid syntax`` on every attempt -- a full
    paid batch each -- because the executor was never told what was wrong.

    A cut-off answer is a CORRUPT RESPONSE, not a file body: callers must reject
    it exactly the way they reject a missing (``None``) response, so the variant
    is unusable and the card is retried with :data:`TRUNCATED_RESPONSE_MESSAGE`
    as its error context. An answer with NO fence at all is not truncated -- a
    model that simply answers with bare code keeps being written verbatim.

    Detection is a parity count of line-leading fence markers: every opened block
    must be closed, so an odd count means the last one never was.
    """
    if not response:
        return False
    return len(_FENCE_MARKER.findall(response)) % 2 == 1


# The line that opens one file in a multi-file answer, as
# ``cards.compiler.MULTI_TARGET_DIRECTIVE`` asks for it: ``FILE: <path>``.
_FILE_MARKER_PREFIX = "FILE:"


class CorruptResponse(ValueError):
    """The answer cannot be turned into the files this card declared.

    Carries, as its message, the text the executor is shown on the next attempt
    -- exactly the role :data:`TRUNCATED_RESPONSE_MESSAGE` plays for an answer
    cut off mid-fence. Raised by :func:`response_to_files`; every caller treats
    it the way it treats a missing response: the variant is unusable, nothing is
    written, and the card is regenerated with the reason.
    """


# The three ways a multi-file answer can fail to match its card, written AT the
# executor: they are pasted into the next attempt's instruction.
MISSING_FILES_MESSAGE = (
    "The previous answer did not return every file this card writes -- missing: "
    "{missing}. This card writes exactly: {declared}. Answer again with ALL of "
    "them, each as a 'FILE: <path>' line followed by a fenced block holding the "
    "complete file body."
)

UNDECLARED_FILE_MESSAGE = (
    "The previous answer returned a file this card does not write: {path}. "
    "This card writes exactly: {declared}. Answer again with those files and no "
    "others, each as a 'FILE: <path>' line followed by a fenced block holding "
    "the complete file body."
)

DUPLICATE_FILE_MESSAGE = (
    "The previous answer returned {path} more than once, so there is no telling "
    "which body was meant. Answer again with exactly one 'FILE: <path>' block "
    "per file, for: {declared}."
)


def _path_list(paths: List[str]) -> str:
    """Render a list of paths for an executor-facing message."""
    return ", ".join(paths)


def _target_key(path: str) -> str:
    """A canonical identity for one declared/answered path, for comparison only.

    ``cards/schema.py`` and ``./cards/schema.py`` are the same file, and a model
    handed a list of paths will spell one of them either way. The key is never
    written anywhere: the CARD's spelling is what reaches disk.
    """
    return os.path.normpath(path.strip())


def _split_file_sections(response: str):
    """Split a response on its ``FILE: <path>`` marker lines, or ``None``.

    Returns ``[(answered path, the text under it), ...]`` in answer order, or
    ``None`` when the response carries no marker at all (Morph 1.0's shape --
    one file, no markers). Anything before the first marker is preamble and is
    dropped.

    A marker counts only at the start of a line and OUTSIDE a fenced block: a
    ``FILE:`` line inside a file body is that file's content, not the start of
    the next one. The fence parity this tracks is the same one
    :func:`is_truncated_response` counts.
    """
    sections = []
    current = None
    in_fence = False
    for line in response.split("\n"):
        if line.startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith(_FILE_MARKER_PREFIX):
            # Models wrap a path in backticks or quotes as readily as not.
            answered = line[len(_FILE_MARKER_PREFIX):].strip().strip("`\"' ")
            current = (answered, [])
            sections.append(current)
            continue
        if current is not None:
            current[1].append(line)
    if not sections:
        return None
    return [(answered, "\n".join(lines)) for answered, lines in sections]


def response_to_files(response: str, targets: List[str]) -> Dict[str, str]:
    """Split one response into ``{declared target: file body}``.

    Which shape an answer is in, the ANSWER decides -- not the card:

    * **No ``FILE:`` marker** -- Morph 1.0's shape, unchanged: the whole
      response is one file body (:func:`response_to_file_body`), which a
      single-target card writes to its one target. A card that writes a SET has
      nothing to split here, so for it that answer is corrupt.
    * **``FILE: <path>`` markers** -- each followed by that file's body, the
      format ``cards.compiler.MULTI_TARGET_DIRECTIVE`` asks for. Bodies go
      through the same fenced-block extraction as a single-file answer.

    Raises :class:`CorruptResponse` -- the treatment an unterminated fence
    already gets, so the card fails verification and is regenerated with the
    reason -- when the answer does not match the card: a declared target is
    MISSING from it, it names a path the card never declared, or it returns the
    same path twice. Each alternative is worse than paying for one retry:
    a half-written changeset, or a file nobody asked for landing in the tree.

    The returned keys are the CARD's own target spellings, never the model's.
    """
    declared = _path_list(targets)
    sections = _split_file_sections(response)

    if sections is None:
        if len(targets) > 1:
            raise CorruptResponse(
                MISSING_FILES_MESSAGE.format(missing=declared, declared=declared))
        return {targets[0]: response_to_file_body(response)}

    by_key = {_target_key(target): target for target in targets}
    bodies: Dict[str, str] = {}
    for answered, section in sections:
        target = by_key.get(_target_key(answered))
        if target is None:
            raise CorruptResponse(
                UNDECLARED_FILE_MESSAGE.format(path=answered, declared=declared))
        if target in bodies:
            raise CorruptResponse(
                DUPLICATE_FILE_MESSAGE.format(path=answered, declared=declared))
        bodies[target] = response_to_file_body(section)

    missing = [target for target in targets if target not in bodies]
    if missing:
        raise CorruptResponse(MISSING_FILES_MESSAGE.format(
            missing=_path_list(missing), declared=declared))
    return bodies


def target_paths(card: MorphCard, root: str) -> List[str]:
    """The real paths a card writes, relative to ``root``, in the card's order.

    One entry for a single-target card. Public so :mod:`cards.acceptance`
    snapshots, writes and rolls back exactly the set the card declared.
    """
    return [os.path.join(root, target) for target in card.targets]


def _variant_ids(card: MorphCard) -> List[str]:
    """The batch custom_ids a card compiles to, matching :func:`compile_card`."""
    if card.variants == 1:
        return [card.custom_id]
    return [f"{card.custom_id}.v{n}" for n in range(1, card.variants + 1)]


def _output_path(card: MorphCard, variant_custom_id: str, root: str) -> str:
    """Where one (variant) response is written, relative to ``root``.

    A single-variant card writes its ``target``; a multi-variant card writes
    ``<stem>.<variant_custom_id><ext>`` (the ``variant_custom_id`` already
    carries the ``.vN`` suffix), mirroring ``flows.morph.output_file_name``.

    For SINGLE-target cards only. A card that writes a set of files gets no
    suffixed per-variant copies -- N variants times M files is a pile of
    ``.v2`` debris nobody reads, and the winning set lands at the real paths
    anyway; see :func:`cards.acceptance.verify_card`.
    """
    if card.variants == 1:
        return os.path.join(root, card.target)
    stem, ext = os.path.splitext(card.target)
    return os.path.join(root, f"{stem}.{variant_custom_id}{ext}")


def ensure_parent_dir(path: str) -> None:
    """Create ``path``'s parent directory tree if it is missing.

    WHY every writer needs this. A card's ``target`` is an arbitrary project
    path, and creating a NEW package (``morph_mcp/jsonrpc.py``) is the most
    ordinary thing a card can ask for -- but ``open(path, "w")`` does not make
    directories, so such a card used to die with ``FileNotFoundError`` the
    moment its generation was collected, taking the whole run with it. Public so
    :mod:`cards.acceptance` and ``flows.morph`` reuse this one copy: every place
    a morph body reaches disk must call it first. Idempotent (``exist_ok``), so
    calling it next to each write costs a syscall and keeps the guard where a
    later reader cannot separate it from the write it protects.

    ``os.path.dirname`` is empty for a bare file name inside ``root`` -- every
    card this project had run until now, which is why 156 green tests never saw
    the bug -- and ``makedirs("")`` would raise, so that case is skipped.

    Rollback note: :func:`cards.acceptance._restore_original` deletes a file a
    rejected variant created, but a directory created here is deliberately left
    behind -- that is not an oversight. An empty directory is harmless, whereas
    removing it would race with whatever else may already have written into it.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# -- the generation cycle ----------------------------------------------------


def _submit_poll_collect(
    requests: List[dict], backend, poll_interval: float
) -> Tuple[str, Optional[Dict[str, Optional[str]]]]:
    """Submit one batch, poll to completion, and collect -- or ``None`` on failure.

    Returns ``(batch_id, results)``: the ``{custom_id: text|None}`` map on a
    completed batch, or ``None`` when the backend reports the whole batch
    ``"failed"`` (in which case there is nothing to collect). Sleeps
    ``poll_interval`` between polls. ``backend`` is duck-typed: only ``submit`` /
    ``status`` / ``collect`` are called.

    The batch id is returned alongside the results, rather than dropped once the
    batch is drained, because it is the run's receipt: ``/nightly`` records it in
    :class:`DeckResult` and the archive writes it down, so a finished run can
    still be asked what it submitted.
    """
    batch_id = backend.submit(requests)
    while True:
        status = backend.status(batch_id)
        if status in ("completed", "failed"):
            break
        time.sleep(poll_interval)
    if status == "failed":
        return batch_id, None
    return batch_id, backend.collect(batch_id)


def _write_variants(
    card: MorphCard,
    results: Dict[str, Optional[str]],
    root: str,
    log: Callable[[str], None] = print,
) -> List[str]:
    """Write every surviving variant of a card (Phase 3, no-acceptance path).

    A ``None`` response is a failed variant and is skipped; so is a response cut
    off inside an unclosed code fence (:func:`is_truncated_response`) -- it is a
    corrupt response, not a file body, and writing it verbatim used to put the
    literal ```` ```python ```` line on disk -- and so is one that does not carry
    the card's files (:class:`CorruptResponse`), which would otherwise leave a
    half-written changeset behind. Returns the paths written; an empty list means
    every variant response was missing or corrupt.

    A card that writes a SET of files has no suffixed per-variant copies to write
    (see :func:`_output_path`), so the FIRST surviving variant's set is written
    and the rest are dropped: without an acceptance command there is nothing to
    judge the others by, and letting each in turn write over the same real paths
    would just crown the last one silently.
    """
    written: List[str] = []
    for variant_id in _variant_ids(card):
        response = results.get(variant_id)
        if response is None:
            continue
        if is_truncated_response(response):
            log(f"mrph> variant {variant_id!r} was cut off mid-file "
                f"(unclosed code fence) -- discarded")
            continue
        try:
            bodies = response_to_files(response, card.targets)
        except CorruptResponse as error:
            log(f"mrph> variant {variant_id!r} did not return this card's "
                f"files -- discarded ({error})")
            continue
        single = len(card.targets) == 1
        for target, real_path in zip(card.targets, target_paths(card, root)):
            path = _output_path(card, variant_id, root) if single else real_path
            ensure_parent_dir(path)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(bodies[target])
            written.append(path)
        if not single:
            break
    return written


def _retry_card(card: MorphCard, attempt: int, result) -> MorphCard:
    """A copy of ``card`` for retry ``attempt``, its instruction carrying the error.

    The batch custom_id is suffixed ``.r<attempt>`` (variants then become
    ``.r<attempt>.v<m>`` via :func:`compile_card`). The instruction gains a
    clearly delimited block with the failed acceptance command and the tail of
    its output (empty when the previous attempt produced none, e.g. a timeout or
    all-``None`` responses) so the executor can see what went wrong.
    """
    output = result.output if result is not None else ""
    error_block = (
        "\n\n---\n"
        f"A previous attempt failed its acceptance check (`{card.acceptance}`):\n"
        f"{output}\n"
        "---\n"
        "Please fix the issues and produce the complete corrected file."
    )
    return MorphCard(
        custom_id=f"{card.custom_id}.r{attempt}",
        intent=card.intent,
        targets=list(card.targets),
        instruction=card.instruction + error_block,
        context_slice=list(card.context_slice),
        acceptance=card.acceptance,
        model=card.model,
        variants=card.variants,
        generation=card.generation,
        depends_on=list(card.depends_on),
    )


def resolve_runnable(
    generation: List[MorphCard],
    index: int,
    total: int,
    outcomes: Dict[str, CardOutcome],
    blocked: set,
    log: Callable[[str], None] = print,
) -> List[MorphCard]:
    """Split one generation into the cards that can run now versus the skipped.

    A card is skipped (and itself added to ``blocked``) when any of its
    dependencies is already ``blocked`` -- failed or skipped in an earlier
    generation. The skip is logged and recorded as a ``"skipped"``
    :class:`CardOutcome` naming the blocking dependency. Mutates ``outcomes`` and
    ``blocked`` in place; returns the runnable cards in deck order.

    Extracted from :func:`run_deck` so the split-step CLI path
    (:func:`cards.store.submit_generation`) resolves a generation exactly the way
    the synchronous loop does.
    """
    runnable: List[MorphCard] = []
    for card in generation:
        blocking = next((dep for dep in card.depends_on if dep in blocked), None)
        if blocking is not None:
            status = outcomes[blocking].status
            log(
                f"mrph> [generation {index}/{total}] skipping {card.custom_id!r}: "
                f"dependency {blocking!r} {status}"
            )
            outcomes[card.custom_id] = CardOutcome(
                card.custom_id, "skipped", reason=blocking, attempts=0
            )
            blocked.add(card.custom_id)
        else:
            runnable.append(card)
    return runnable


def process_generation(
    runnable: List[MorphCard],
    results: Optional[Dict[str, Optional[str]]],
    index: int,
    total: int,
    root: str,
    backend,
    poll_interval: float,
    log: Callable[[str], None],
    verify: bool,
    acceptance_timeout: float,
    max_regenerations: int,
    outcomes: Dict[str, CardOutcome],
    blocked: set,
    inline_retries: bool = True,
    on_accepted: Optional[AcceptedHook] = None,
    batch_ids: Optional[List[str]] = None,
) -> List[tuple]:
    """Turn one generation's collected ``results`` into outcomes (with retries).

    ``results`` is the ``{variant_custom_id: text|None}`` map ``backend.collect``
    returned, or ``None`` when the whole batch failed (every runnable card fails
    terminally, no retry). Otherwise each card is either verified best-of-N (when
    it has an ``acceptance`` command and ``verify`` is true) or has its surviving
    variants written (Phase 3 semantics); cards that fail acceptance with retries
    left are regenerated inline via :func:`_run_retries`, whose retry batches are
    submitted and collected through ``backend`` here. Mutates ``outcomes`` and
    ``blocked`` in place.

    This is the shared per-generation body of both :func:`run_deck` (which
    submits and polls the batch itself, then hands the results here) and
    :func:`cards.store.collect_generation` (which collects the in-flight batch
    across a CLI restart, then hands the results here) -- so the two paths keep
    identical best-of-N / rollback / retry / skip-cascade semantics.

    ``inline_retries`` is the one difference between them. ``True`` (``run_deck``
    /``/nightly``) runs :func:`_run_retries` here and blocks until every card is
    settled. ``False`` (``/collect``) returns the still-pending
    ``(card, verify_outcome)`` pairs instead, so the caller can submit ONE retry
    batch, persist it and return -- an unfinished generation whose retry survives
    a restart, rather than an hour of silence inside one call. The returned list
    is empty whenever retries ran here or nothing needs one.

    ``on_accepted`` (see :data:`AcceptedHook`) fires once per accepted card, at
    the moment its files are final -- that is where the run's git layer turns a
    card into a commit. ``batch_ids``, when given, is the list an inline retry
    batch's id is appended to: the run's receipt of everything it submitted.
    """
    from cards.acceptance import verify_card

    if results is None:
        log(
            f"mrph> [generation {index}/{total}] batch failed; "
            f"{len(runnable)} card(s) failed"
        )
        for card in runnable:
            outcomes[card.custom_id] = CardOutcome(card.custom_id, "failed")
            blocked.add(card.custom_id)
        return []

    # Cards that failed acceptance but have retries left, paired with the
    # failing :class:`cards.acceptance.AcceptanceResult` (or ``None`` when
    # nothing ran) that is their next attempt's error context. The result rather
    # than the whole verify outcome, because that pair is what survives into
    # ``.morph/state.json`` when the retry is persisted instead of run inline.
    retry_pending: List[tuple] = []

    for card in runnable:
        if verify and card.acceptance:
            outcome = verify_card(card, results, root, acceptance_timeout, log)
            if outcome.passed:
                outcomes[card.custom_id] = CardOutcome(
                    card.custom_id,
                    "written",
                    paths=outcome.paths,
                    attempts=1,
                    winning_variant=outcome.winning_custom_id,
                )
                log(
                    f"mrph> [generation {index}/{total}] {card.custom_id!r} "
                    f"written (variant {outcome.winning_custom_id!r} passed "
                    f"acceptance): {', '.join(outcome.paths)}"
                )
                _fire(on_accepted, card, outcomes[card.custom_id])
            elif max_regenerations > 0:
                retry_pending.append((card, outcome.result))
            else:
                outcomes[card.custom_id] = CardOutcome(
                    card.custom_id,
                    "failed",
                    attempts=1,
                    acceptance_output=_acceptance_output(outcome.result),
                )
                blocked.add(card.custom_id)
                log(
                    f"mrph> [generation {index}/{total}] {card.custom_id!r} "
                    f"failed acceptance (no retries)"
                )
        else:
            written = _write_variants(card, results, root, log)
            if written:
                outcomes[card.custom_id] = CardOutcome(
                    card.custom_id, "written", paths=written
                )
                log(
                    f"mrph> [generation {index}/{total}] {card.custom_id!r} "
                    f"written: {', '.join(written)}"
                )
                _fire(on_accepted, card, outcomes[card.custom_id])
            else:
                outcomes[card.custom_id] = CardOutcome(card.custom_id, "failed")
                blocked.add(card.custom_id)
                log(
                    f"mrph> [generation {index}/{total}] {card.custom_id!r} "
                    f"failed: all variants empty"
                )

    # Retry generations for the failed acceptance cards -- these run BEFORE
    # this generation's dependents (which live in later generations).
    if not inline_retries:
        return retry_pending

    _run_retries(
        retry_pending,
        index,
        total,
        root,
        backend,
        poll_interval,
        log,
        acceptance_timeout,
        max_regenerations,
        outcomes,
        blocked,
        on_accepted,
        batch_ids,
    )
    return []


def run_deck(
    cards: List[MorphCard],
    backend,
    root: str = ".",
    poll_interval: float = 1.0,
    log: Callable[[str], None] = print,
    verify: bool = True,
    acceptance_timeout: float = 300.0,
    max_regenerations: int = 2,
    on_accepted: Optional[AcceptedHook] = None,
) -> DeckResult:
    """Execute a deck generation by generation through one batch backend.

    For each generation, in order: skip any card whose dependency failed or was
    itself skipped (logged, never compiled or submitted); log the composition of
    what is being submitted *before* submitting; compile the remaining cards --
    which reads the fresh files earlier generations wrote, the whole point of
    generations -- submit them as one batch, poll ``backend.status`` until it
    reaches ``"completed"`` or ``"failed"`` (sleeping ``poll_interval`` between
    polls), collect, and process the morphs.

    A card WITHOUT an ``acceptance`` command keeps the Phase 3 contract: it is
    ``"written"`` if at least one variant response is non-``None`` (all surviving
    variants written), ``"failed"`` when every variant response is ``None``, and
    it is never retried.

    A card WITH an ``acceptance`` command (only when ``verify`` is true, the
    default) is verified best-of-N by :func:`cards.acceptance.verify_card`: the
    first variant to pass acceptance wins and the card is ``"written"``. A card
    that fails verification (or whose responses were all ``None``) is
    *regenerated* -- resubmitted with the acceptance error appended to its
    instruction -- into a retry batch that runs BEFORE this generation's
    dependents, up to ``max_regenerations`` times (default 2, so at most 3 total
    attempts). Its ``CardOutcome`` is recorded under the ORIGINAL custom_id with
    the attempt count and, on success, the winning variant. Only once a card
    exhausts its retries and finally ``"failed"`` are its dependents skipped.

    A whole batch reported ``"failed"`` (a transport/provider failure, no
    responses to judge) fails every card it carried terminally -- no retry. A
    failed or skipped card's transitive dependents are skipped.

    ``on_accepted`` (see :data:`AcceptedHook`) fires as each card is accepted, so
    a blocking ``/nightly`` pass produces the same per-card commits, in the same
    order, as the split-step route -- the hook is the ONLY thing the two paths
    need to share for that, since everything else about a commit is derived from
    the card and its outcome.

    ``backend`` is duck-typed: only ``submit(requests) -> batch_id``,
    ``status(batch_id) -> str`` and ``collect(batch_id) -> {custom_id: text|None}``
    are called. Returns a :class:`DeckResult` whose ``generations`` records the
    static generation composition (retry batches are extra submits, not extra
    generations) and whose ``batch_ids`` records every submission, retries
    included.
    """
    generations = split_into_generations(cards)
    total = len(generations)
    composition = [[card.custom_id for card in generation] for generation in generations]

    outcomes: Dict[str, CardOutcome] = {}
    # custom_ids that failed or were skipped: their dependents cannot run.
    blocked: set = set()
    batch_ids: List[str] = []

    for index, generation in enumerate(generations, start=1):
        runnable = resolve_runnable(generation, index, total, outcomes, blocked, log)
        if not runnable:
            continue

        ids = [card.custom_id for card in runnable]
        log(
            f"mrph> [generation {index}/{total}] submitting "
            f"{len(runnable)} card(s): {', '.join(ids)}"
        )

        # Compile only now -- generation N-1's morphs are already on disk, so a
        # dependent card's context slice sees the fresh files.
        requests: List[dict] = []
        for card in runnable:
            requests.extend(compile_card(card, root))

        batch_id, results = _submit_poll_collect(requests, backend, poll_interval)
        batch_ids.append(batch_id)

        process_generation(
            runnable,
            results,
            index,
            total,
            root,
            backend,
            poll_interval,
            log,
            verify,
            acceptance_timeout,
            max_regenerations,
            outcomes,
            blocked,
            inline_retries=True,
            on_accepted=on_accepted,
            batch_ids=batch_ids,
        )

    return DeckResult(outcomes=outcomes, generations=composition, batch_ids=batch_ids)


def _acceptance_output(result) -> Optional[str]:
    """The captured output of a failing acceptance result, or ``None`` if none ran."""
    return result.output if result is not None else None


def build_retry_cards(
    pending: List[tuple],
    attempt: int,
    index: int,
    total: int,
    max_regenerations: int,
    log: Callable[[str], None] = print,
) -> List[MorphCard]:
    """One retry card per still-pending card, logged as the retry is prepared.

    ``pending`` is a list of ``(card, previous AcceptanceResult or None)``; the
    retry card carries that error into its instruction (:func:`_retry_card`) and
    is suffixed ``.r<attempt>``. Split out of :func:`_run_retries` so the
    split-step path (:func:`cards.store.collect_generation`, which submits a
    retry batch and PERSISTS it instead of polling it inline) builds exactly the
    same retry cards the blocking loop does.
    """
    retry_cards: List[MorphCard] = []
    for card, prev_result in pending:
        log(
            f"mrph> [generation {index}/{total}] retry "
            f"{attempt}/{max_regenerations} for card {card.custom_id!r} "
            f"(acceptance failed)"
        )
        retry_cards.append(_retry_card(card, attempt, prev_result))
    return retry_cards


def process_retry_batch(
    retry_cards: List[MorphCard],
    pending: List[tuple],
    results: Optional[Dict[str, Optional[str]]],
    attempt: int,
    index: int,
    total: int,
    root: str,
    log: Callable[[str], None],
    acceptance_timeout: float,
    max_regenerations: int,
    outcomes: Dict[str, CardOutcome],
    blocked: set,
    on_accepted: Optional[AcceptedHook] = None,
) -> List[tuple]:
    """Judge one retry batch's results; return what is still pending after it.

    ``retry_cards`` and ``pending`` are positionally paired (the output and the
    input of :func:`build_retry_cards`). ``results`` is the collected
    ``{variant_custom_id: text|None}`` map, or ``None`` when the whole retry
    batch failed -- then every pending card fails terminally, carrying its last
    real acceptance output, and nothing is left pending.

    A card that passes is recorded ``"written"`` under its ORIGINAL custom_id
    with the attempt count; a card that fails with retries left is returned in
    the new pending list, paired with the fresh :class:`AcceptanceResult` that
    becomes the next attempt's error context; a card that fails on the last
    allowed attempt is
    recorded ``"failed"`` and blocks its dependents. Mutates ``outcomes`` and
    ``blocked`` in place.

    Shared by the blocking loop (:func:`_run_retries`) and the persisted one
    (:func:`cards.store.collect_generation`), so an outcome reads the same
    whichever route produced it -- ``on_accepted`` (see :data:`AcceptedHook`)
    included: a card accepted on its third attempt becomes a commit exactly as
    one accepted on its first does, and the hook is handed the ORIGINAL card, not
    the regenerated stand-in whose instruction carries an error message.

    A card accepted on a retry also records ``earlier_failures`` on its outcome:
    :func:`_acceptance_output` of the pending pair's result -- the captured
    output of the failed attempt this retry replaces, which is the card's LAST
    failed attempt, kept because ``acceptance_output`` reports it only on a
    terminal failure.
    """
    from cards.acceptance import verify_card

    if results is None:
        log(
            f"mrph> [generation {index}/{total}] retry {attempt} batch "
            f"failed; {len(pending)} card(s) failed"
        )
        for card, prev_result in pending:
            outcomes[card.custom_id] = CardOutcome(
                card.custom_id,
                "failed",
                attempts=1 + attempt,
                acceptance_output=_acceptance_output(prev_result),
            )
            blocked.add(card.custom_id)
        return []

    next_pending: List[tuple] = []
    for retry_card, (card, prev_result) in zip(retry_cards, pending):
        outcome = verify_card(retry_card, results, root, acceptance_timeout, log)
        if outcome.passed:
            outcomes[card.custom_id] = CardOutcome(
                card.custom_id,
                "written",
                paths=outcome.paths,
                attempts=1 + attempt,
                winning_variant=outcome.winning_custom_id,
                # ``prev_result`` is the attempt this retry just replaced -- the
                # card's LAST failed one -- and its captured output is the
                # evidence of why the earlier tries were rejected, which
                # ``acceptance_output`` reports only on a terminal failure.
                earlier_failures=_acceptance_output(prev_result),
            )
            log(
                f"mrph> [generation {index}/{total}] {card.custom_id!r} "
                f"written after retry {attempt} (variant "
                f"{outcome.winning_custom_id!r} passed): "
                f"{', '.join(outcome.paths)}"
            )
            _fire(on_accepted, card, outcomes[card.custom_id])
        elif attempt < max_regenerations:
            next_pending.append((card, outcome.result))
        else:
            outcomes[card.custom_id] = CardOutcome(
                card.custom_id,
                "failed",
                attempts=1 + attempt,
                acceptance_output=_acceptance_output(outcome.result),
            )
            blocked.add(card.custom_id)
            log(
                f"mrph> [generation {index}/{total}] {card.custom_id!r} "
                f"failed acceptance after {attempt} retr"
                f"{'y' if attempt == 1 else 'ies'}"
            )
    return next_pending


def _run_retries(
    pending: List[tuple],
    index: int,
    total: int,
    root: str,
    backend,
    poll_interval: float,
    log: Callable[[str], None],
    acceptance_timeout: float,
    max_regenerations: int,
    outcomes: Dict[str, CardOutcome],
    blocked: set,
    on_accepted: Optional[AcceptedHook] = None,
    batch_ids: Optional[List[str]] = None,
) -> None:
    """Regenerate cards that failed acceptance, up to ``max_regenerations`` times.

    The BLOCKING retry loop: each attempt builds the retry cards, submits them as
    one batch, polls it to completion and judges the results, until nothing is
    pending or the limit is reached. This is ``/nightly``'s contract -- one
    blocking pass does the whole deck -- and is deliberately NOT what the
    split-step ``/collect`` path does (see
    :func:`cards.store.collect_generation`: it persists the retry batch and
    returns, so a second ``/collect`` picks it up instead of a second CLI session
    submitting a second paid batch for the same card).

    Mutates ``outcomes`` and ``blocked`` in place.
    """
    attempt = 0
    while pending and attempt < max_regenerations:
        attempt += 1
        retry_cards = build_retry_cards(
            pending, attempt, index, total, max_regenerations, log)

        requests: List[dict] = []
        for retry_card in retry_cards:
            requests.extend(compile_card(retry_card, root))

        batch_id, results = _submit_poll_collect(requests, backend, poll_interval)
        if batch_ids is not None:
            batch_ids.append(batch_id)

        pending = process_retry_batch(
            retry_cards, pending, results, attempt, index, total, root, log,
            acceptance_timeout, max_regenerations, outcomes, blocked, on_accepted)
