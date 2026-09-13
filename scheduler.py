"""
Round-robin / pinned scheduler for ``/generate`` and ``/patch`` jobs.

Turns the processor registry (``processors/registry.py``) into a bounded
pool: one slot per configured processor id. A job either targets the
*rotation* (no ``@id`` given -- picked up by whichever processor is next
free, spread evenly) or is *pinned* to one or more explicit ids (``@id``,
``@id,@id`` or ``@all``). When every slot a job needs is busy, it waits in a
FIFO queue and is dispatched automatically once its processor(s) free.

See ``documentation/parallel-generate-scheduling.md`` for the full spec this
module implements. This module is purely synchronous bookkeeping -- it does
not run any LLM calls itself; callers hand it a zero-argument callback (via
``attach_launch``) that they want invoked exactly once, as soon as the job's
processor(s) are assigned.
"""

import time
from typing import Callable, Dict, List, Optional


class Job:
    """One `/generate`/`/patch` request being tracked by the scheduler."""

    def __init__(self, job_id: int, pinned_ids: Optional[List[str]]):
        self.id = job_id
        # ``None`` -> this job rides the rotation; otherwise the explicit
        # list of ids it is pinned to (order as requested).
        self.pinned_ids = pinned_ids
        # Set once the job is actually given processor(s) to run on.
        self.assigned: Optional[List[str]] = None
        # True as soon as this job spent any time waiting for a slot.
        self.was_queued = False
        # The caller's "go" callback, and whether it has fired yet.
        self._launch: Optional[Callable[[], None]] = None
        self._launched = False

    @property
    def is_pinned(self) -> bool:
        return self.pinned_ids is not None


class JobScheduler:
    """Round-robin/pinned scheduling across a fixed pool of processor ids."""

    def __init__(self, processor_ids: List[str]):
        self.pool: List[str] = list(processor_ids)
        self._busy: Dict[str, Optional[Job]] = {pid: None for pid in self.pool}
        self._busy_since: Dict[str, float] = {}
        # Jobs pinned to ``pid`` that are waiting specifically on it, FIFO.
        self._waiters: Dict[str, List[Job]] = {pid: [] for pid in self.pool}
        # Rotation-only jobs waiting for *any* processor to free.
        self._rotation_queue: List[Job] = []
        self._cursor = 0
        self._next_job_id = 1

    def __len__(self) -> int:
        return len(self.pool)

    # -- intake --------------------------------------------------------------

    def submit(self, pinned_ids: Optional[List[str]]) -> Job:
        """Reserve a slot (or a queue position) for a new job.

        ``pinned_ids`` is ``None`` for a rotation job, or the concrete list
        of ids (one or more) it must run on -- matching today's fan-out
        semantics when there is more than one.
        """
        job = Job(self._next_job_id, pinned_ids)
        self._next_job_id += 1

        if job.is_pinned:
            for pid in job.pinned_ids:
                self._waiters[pid].append(job)
            self._try_start_pinned(job)
        else:
            self._try_start_rotation(job)

        if job.assigned is None:
            job.was_queued = True

        return job

    def attach_launch(self, job: Job, launch: Callable[[], None]) -> None:
        """Record the callback to fire once ``job`` has processor(s) assigned.

        Fires immediately if the job was already dispatched at intake time.
        """
        job._launch = launch
        self._maybe_launch(job)

    # -- status/reporting ------------------------------------------------------

    def queue_position(self, job: Job) -> int:
        """1-based position in whichever queue(s) ``job`` is waiting in."""
        if job.is_pinned:
            positions = [
                self._waiters[pid].index(job) + 1
                for pid in job.pinned_ids
                if job in self._waiters[pid]
            ]
            return max(positions) if positions else 0
        if job in self._rotation_queue:
            return self._rotation_queue.index(job) + 1
        return 0

    def busy_count(self) -> int:
        return sum(1 for pid in self.pool if self._busy[pid] is not None)

    def queue_length(self) -> int:
        """Number of distinct jobs currently waiting for a slot."""
        seen = {job.id for job in self._rotation_queue}
        for pid in self.pool:
            seen.update(job.id for job in self._waiters[pid])
        return len(seen)

    def describe_status(self, pid: str) -> str:
        job = self._busy.get(pid)
        waiting = len(self._waiters.get(pid, []))
        if job is None:
            return f"idle ({waiting} waiting)" if waiting else "idle"
        elapsed = int(time.monotonic() - self._busy_since[pid])
        return f"running job {job.id} for {elapsed}s ({waiting} waiting)" if waiting \
            else f"running job {job.id} for {elapsed}s"

    # -- release/dispatch --------------------------------------------------------

    def release(self, processor_ids: List[str]) -> None:
        """Called once a job finishes on ``processor_ids``; frees each slot
        and dispatches whatever is queued for it, if anything."""
        for pid in processor_ids:
            self._busy[pid] = None
            self._busy_since.pop(pid, None)
            self._offer(pid)

    # -- internals ------------------------------------------------------------

    def _offer(self, pid: str) -> None:
        # A pinned job waiting specifically on ``pid`` reserves it: rotation
        # never steals a slot a fan-out job is still assembling.
        if self._waiters[pid]:
            self._try_start_pinned(self._waiters[pid][0])
            return
        if self._rotation_queue and self._busy[pid] is None:
            job = self._rotation_queue.pop(0)
            self._dispatch(job, [pid])

    def _try_start_rotation(self, job: Job) -> bool:
        n = len(self.pool)
        for step in range(n):
            idx = (self._cursor + step) % n
            pid = self.pool[idx]
            if self._busy[pid] is None and not self._waiters[pid]:
                self._cursor = (idx + 1) % n
                self._dispatch(job, [pid])
                return True
        self._rotation_queue.append(job)
        return False

    def _try_start_pinned(self, job: Job) -> bool:
        ready = all(
            self._busy[pid] is None and self._waiters[pid] and self._waiters[pid][0] is job
            for pid in job.pinned_ids
        )
        if ready:
            self._dispatch(job, list(job.pinned_ids))
        return ready

    def _dispatch(self, job: Job, processor_ids: List[str]) -> None:
        job.assigned = processor_ids
        for pid in processor_ids:
            self._busy[pid] = job
            self._busy_since[pid] = time.monotonic()
            if self._waiters[pid] and self._waiters[pid][0] is job:
                self._waiters[pid].pop(0)
        self._maybe_launch(job)

    @staticmethod
    def _maybe_launch(job: Job) -> None:
        if job._launched or job.assigned is None or job._launch is None:
            return
        job._launched = True
        job._launch()
