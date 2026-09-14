"""Durable completion outbox for background terminal processes (receipt schema 2).

A ``terminal(background=true, notify=true)`` process on a *finite* surface (an API/Wizard run
that ends the moment its HTTP turn returns) has no live parent to hand its exit to. The
per-process result receipts written by ``process_registry_results`` already survive that parent;
this module promotes them into a delivery outbox so a LATER process can pick the completion up
and resume the exact originating session.

Shape: one receipt file per process (``<HERMES_HOME>/logs/process-results/<proc_id>.json``),
written atomically, mutated only under a cross-process ``flock``/``msvcrt`` receipt lock so two
gateway processes cannot both claim the same completion. Schema 1 receipts (everything written
before this outbox existed) carry no ``completion`` block and are NEVER replayed — they were
produced under the old read-only contract and nobody is waiting on them.

State machine (``completion.state``)::

    armed ──(process exits)──> pending ──(claim)──> claimed ──(accepted delivery)──> delivered
      │                           ↑                    │
      │                           └──(release/retry)───┘
      └──(reader read the result in-turn, or attempts exhausted)──> consumed / dropped

``armed`` exists so an immediately-exiting process cannot win the race against its own
bookkeeping: the route metadata is on disk before the child can produce an exit, and a worker
that dies with the gateway is recovered as a ``pending`` unknown-outcome completion rather than
silently dropped.

**Known limitation (gateway restart).** ``survive_gateway_restart`` records whether a worker owns
a private systemd scope and could therefore genuinely outlive the gateway. Shutdown cleanup still
kills it, deliberately. ``GatewayShutdownMixin._stop_kill_tool_subprocesses`` is a static
phase-labelled catch-all invoked on every teardown (``post-interrupt`` and ``final-cleanup``
alike) and it goes through ``process_registry.kill_all()``, which has no per-job targeting and no
planned-vs-crash signal to key an exemption on. Exempting jobs there would weaken cleanup for
every shutdown path, including the crash one, so we fail closed. Nothing is lost: the kill settles
a ``pending`` completion with ``completion_reason="killed"``, and a worker whose scope is gone by
the next startup is recovered the same way — so the originating session is told the restart
stopped its job instead of hearing nothing at all.
"""

import contextlib
import errno
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from utils import atomic_json_write

logger = logging.getLogger("tools.process_registry")

RECEIPT_SCHEMA = 2

ARMED = "armed"
PENDING = "pending"
CLAIMED = "claimed"
DELIVERED = "delivered"
CONSUMED = "consumed"
DROPPED = "dropped"

# States whose receipt must outlive ordinary result pruning: the completion still owes someone
# a delivery, and pruning it is exactly the silent loss this module exists to prevent.
UNSETTLED_STATES = frozenset({ARMED, PENDING, CLAIMED})

# A claim is exclusive but not eternal: a consumer that dies mid-delivery must not wedge the
# completion forever, so a stale claim is reclaimable.
CLAIM_TTL_SECONDS = 300.0
MAX_DELIVERY_ATTEMPTS = 5
# Bounded wait for the receipt lock; a contended receipt is retried on the next replay tick.
_LOCK_TIMEOUT_SECONDS = 5.0
_LOCK_POLL_SECONDS = 0.05
_IS_WINDOWS = sys.platform == "win32"
_LOCK_CONTENTION_ERRNOS = {errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK, errno.EDEADLK}


def results_dir() -> Path:
    """Profile-local receipt directory (also the outbox)."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "logs" / "process-results"


def receipt_path(session_id: str) -> Path:
    return results_dir() / f"{session_id}.json"


def _lock_path(session_id: str) -> Path:
    return results_dir() / f"{session_id}.lock"


def _is_contention(exc: BaseException) -> bool:
    return isinstance(exc, BlockingIOError) or (
        isinstance(exc, OSError) and exc.errno in _LOCK_CONTENTION_ERRNOS)


def _try_lock(handle) -> None:
    if _IS_WINDOWS:
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle) -> None:
    if _IS_WINDOWS:
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def receipt_lock(session_id: str) -> Iterator[bool]:
    """Hold the cross-process lock for one receipt; yields False when it could not be taken.

    Fail closed: a caller that does not hold the lock must not mutate the delivery state (the
    completion stays where it is and the next pass retries it).
    """
    directory = results_dir()
    path = _lock_path(session_id)
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        handle = open(path, "a+b")
    except OSError as exc:
        logger.warning("Could not open completion receipt lock %s (%s); deferring", path, exc)
        yield False
        return
    acquired = False
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    try:
        while True:
            try:
                _try_lock(handle)
                acquired = True
                break
            except (BlockingIOError, OSError) as exc:
                if not _is_contention(exc) or time.monotonic() >= deadline:
                    break
                time.sleep(_LOCK_POLL_SECONDS)
        yield acquired
    finally:
        if acquired:
            with contextlib.suppress(OSError, ValueError):
                _unlock(handle)
        with contextlib.suppress(OSError):
            handle.close()


def read_receipt(path: Path) -> Optional[Dict[str, Any]]:
    """Parsed receipt, or None when unreadable/not JSON."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def write_receipt(record: Dict[str, Any]) -> None:
    directory = results_dir()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_json_write(directory / f"{record['id']}.json", record, mode=0o600)


def completion_block(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The delivery block of a schema-2 receipt; ``{}`` for legacy schema-1 receipts."""
    if not isinstance(record, dict) or record.get("schema", 1) < RECEIPT_SCHEMA:
        return {}
    block = record.get("completion")
    return block if isinstance(block, dict) else {}


def is_unsettled(record: Optional[Dict[str, Any]]) -> bool:
    """Whether this receipt still owes a delivery (pruning must leave it alone)."""
    return completion_block(record).get("state") in UNSETTLED_STATES


def new_claim_id(consumer: str) -> str:
    return f"{consumer}:{os.getpid()}:{uuid.uuid4().hex}"


# ----- Arming and terminal outcomes ------------------------------------------------------


def _block_for(session: Any, state: str) -> Dict[str, Any]:
    """A fresh delivery block describing *session*'s durable route."""
    return {
        "state": state,
        "mode": session.completion_delivery,
        "adapter_id": session.delivery_adapter_id,
        "origin_session_id": session.watcher_chat_id,
        "survive_gateway_restart": bool(session.survive_gateway_restart),
        "pid": session.pid,
        "host_start_time": session.host_start_time,
        "claim": None, "claimed_at": None, "attempts": 0, "updated_at": time.time(),
        "event": None,
    }


def arm(session: Any, base_record: Dict[str, Any]) -> None:
    """Write the ``armed`` receipt for a freshly tracked process.

    Called before the reader thread exists, so the route metadata is on disk in the FIRST
    receipt: an immediate exit cannot win the race and produce a routeless completion.
    """
    record = dict(base_record)
    record.update(schema=RECEIPT_SCHEMA, exited=False, completion=_block_for(session, ARMED))
    try:
        with receipt_lock(session.id) as held:
            if not held:
                logger.warning("Could not arm durable completion for %s: receipt lock busy", session.id)
                return
            write_receipt(record)
    except OSError:
        logger.warning("Could not arm durable completion for %s", session.id, exc_info=True)


def attach_outcome(record: Dict[str, Any], previous: Optional[Dict[str, Any]], *, session: Any,
                   event: Optional[Dict[str, Any]], consumed: bool) -> Dict[str, Any]:
    """Fold the terminal outcome into *record*'s delivery block (caller holds the receipt lock).

    A completion whose result the owning turn already read (wait/log/explicit kill) settles as
    ``consumed`` so no later replay can wake the session a second time with a result it has.
    A missing block means arming lost its lock; rebuild it here rather than dropping the
    completion — only the armed window (dead-worker recovery) is lost, not the delivery.
    """
    block = dict(completion_block(previous)) or _block_for(session, ARMED)
    if block.get("state") not in {DELIVERED, CONSUMED, DROPPED}:
        block.update(state=CONSUMED if consumed else PENDING, event=event, updated_at=time.time(),
                     claim=None, claimed_at=None)
    record.update(schema=RECEIPT_SCHEMA, exited=True, completion=block)
    return record


def consume(session_id: str) -> bool:
    """Settle a completion the owning turn read itself; True when a receipt changed state."""
    def _apply(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        block = completion_block(record)
        if block.get("state") not in UNSETTLED_STATES:
            return None
        block.update(state=CONSUMED, claim=None, claimed_at=None, updated_at=time.time())
        return record

    return _mutate(session_id, _apply)


def record_lost(entry: Dict[str, Any], *, reason: str = "lost") -> bool:
    """Turn an armed receipt whose worker is gone into a PENDING unknown-outcome completion.

    Used by checkpoint recovery and by the replay loop's reconcile pass. The outcome is honestly
    unknown (``exit_code`` None), which is still worth delivering: the originating session learns
    its job died instead of waiting forever on a completion that will never arrive.
    """
    session_id = str(entry.get("session_id") or entry.get("id") or "")
    if not session_id:
        return False

    def _apply(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        block = completion_block(record)
        if block.get("state") != ARMED:
            return None
        record.update(exited=True, exit_code=None, completion_reason=reason,
                      termination_source="gateway_restart")
        block.update(state=PENDING, updated_at=time.time(), claim=None, claimed_at=None,
                     event=_lost_event(record, block, reason))
        record["completion"] = block
        return record

    return _mutate(session_id, _apply)


def _lost_event(record: Dict[str, Any], block: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """A completion_queue-shaped event for a worker whose outcome nobody observed."""
    return {
        "type": "completion", "session_id": record.get("id", ""),
        "session_key": record.get("session_key", ""), "task_id": record.get("task_id", ""),
        "owner_task_id": record.get("owner_task_id") or record.get("task_id", ""),
        "command": record.get("command", ""), "exit_code": None, "completion_reason": reason,
        "termination_source": record.get("termination_source", ""),
        "output": record.get("output", ""), "started_at": record.get("started_at", 0.0),
        "platform": str(block.get("adapter_id") or ""),
        "origin_session_id": block.get("origin_session_id", ""),
        "parent_session_id": record.get("parent_session_id", ""),
    }


# ----- Delivery claims --------------------------------------------------------------------


def _mutate(session_id: str, apply) -> bool:
    """Read-modify-write one receipt under its lock; *apply* returns the new record or None."""
    path = receipt_path(session_id)
    try:
        with receipt_lock(session_id) as held:
            if not held:
                return False
            record = read_receipt(path)
            if record is None:
                return False
            updated = apply(record)
            if updated is None:
                return False
            write_receipt(updated)
            return True
    except OSError:
        logger.warning("Could not update completion receipt %s", session_id, exc_info=True)
        return False


def pending_completions(adapter_id: str) -> List[Dict[str, Any]]:
    """Unsettled receipts this adapter owns, oldest first. Legacy schema-1 receipts are skipped."""
    from tools.process_registry_results import retained_result_paths

    out: List[Dict[str, Any]] = []
    try:
        paths = retained_result_paths()
    except OSError:
        logger.warning("Could not scan the completion outbox", exc_info=True)
        return out
    for path in paths:
        record = read_receipt(path)
        block = completion_block(record)
        if not block or block.get("adapter_id") != adapter_id:
            continue
        if block.get("state") not in {ARMED, PENDING}:
            continue
        if record.get("id") != path.stem:
            continue
        out.append(record)
    out.sort(key=lambda r: (r.get("started_at") or 0.0, r.get("id", "")))
    return out


def claim(session_id: str, claim_id: str, *, adapter_id: str) -> Optional[Dict[str, Any]]:
    """Take the exclusive delivery claim for one pending completion.

    Returns the claimed record, or None when another consumer holds a live claim, the receipt
    belongs to a different adapter, or it is no longer pending.
    """
    claimed: Dict[str, Any] = {}

    def _apply(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        block = completion_block(record)
        if not block or block.get("adapter_id") != adapter_id:
            return None
        now = time.time()
        state = block.get("state")
        if state == CLAIMED:
            claimed_at = block.get("claimed_at") or 0.0
            if now - float(claimed_at) < CLAIM_TTL_SECONDS:
                return None
        elif state != PENDING:
            return None
        attempts = int(block.get("attempts") or 0) + 1
        if attempts > MAX_DELIVERY_ATTEMPTS:
            # Counted here, not only in release(): a consumer that dies mid-delivery never
            # releases, and without this the stale-claim takeover would retry forever.
            logger.warning("Background completion %s exhausted its %d delivery attempts; dropping "
                           "the replay (result remains queryable).", session_id, MAX_DELIVERY_ATTEMPTS)
            block.update(state=DROPPED, claim=None, claimed_at=None, updated_at=now)
            record["completion"] = block
            return record
        block.update(state=CLAIMED, claim=claim_id, claimed_at=now, attempts=attempts, updated_at=now)
        record["completion"] = block
        claimed.update(record)
        return record

    # A write without a populated ``claimed`` is the attempts-exhausted drop above, not a claim.
    return dict(claimed) if (_mutate(session_id, _apply) and claimed) else None


def release(session_id: str, claim_id: str) -> bool:
    """Return a failed delivery to ``pending`` for a later retry.

    Once the attempt budget is spent the receipt converges to terminal ``dropped``: the result
    stays queryable through ``process_manage``, but it stops waking the session forever.
    """
    def _apply(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        block = completion_block(record)
        if block.get("state") != CLAIMED or block.get("claim") != claim_id:
            return None
        exhausted = int(block.get("attempts") or 0) >= MAX_DELIVERY_ATTEMPTS
        if exhausted:
            logger.warning("Background completion %s exhausted its %d delivery attempts; dropping "
                           "the replay (result remains queryable).", session_id, MAX_DELIVERY_ATTEMPTS)
        block.update(state=DROPPED if exhausted else PENDING, claim=None, claimed_at=None,
                     updated_at=time.time())
        record["completion"] = block
        return record

    return _mutate(session_id, _apply)


def settle(session_id: str, claim_id: str, state: str = DELIVERED) -> bool:
    """Terminally settle a claimed completion after the delivery was ACCEPTED."""
    if state not in {DELIVERED, CONSUMED, DROPPED}:
        raise ValueError(f"not a terminal completion state: {state!r}")

    def _apply(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        block = completion_block(record)
        if block.get("state") != CLAIMED or block.get("claim") != claim_id:
            return None
        block.update(state=state, claim=None, claimed_at=None, updated_at=time.time())
        record["completion"] = block
        return record

    return _mutate(session_id, _apply)


def idempotency_key(record: Dict[str, Any]) -> str:
    """Stable per-completion key so a retried wake cannot run the same turn twice.

    ``started_at`` is the producer's spawn epoch, so a recycled process id is a distinct
    incarnation rather than a suppressed duplicate.
    """
    return f"hermes-proc-completion-{record.get('id', '')}-{int(float(record.get('started_at') or 0.0))}"
