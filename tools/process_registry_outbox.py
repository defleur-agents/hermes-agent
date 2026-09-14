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
silently dropped. Arming is a HARD precondition — :func:`arm` raises :class:`DurableArmingError`
rather than returning, because a spawn whose obligation was never written must not be reported to
the model as a notification that will arrive.

**Profile identity is immutable.** ``completion.profile`` records the multiplex profile
(``""`` = default) that spawned the worker. The receipt already lives in that profile's home, but
the value is also written into the block at arming time and never rewritten: it is what routes the
replay to ``/p/<profile>/v1/chat/completions`` with that profile's own credential, and a record
that could be re-homed could be delivered under the wrong principal.

**Exactly-once across restarts.** A claim gives one consumer the right to try; the durable
at-most-once gate is :func:`reserve_wake_delivery`, consulted by the API server BEFORE it runs a
wake turn. The mark lives in the receipt the wake key names, so a wake accepted by a process that
died before settling its claim is replayed — not re-run — by the next process.

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
TERMINAL_STATES = frozenset({DELIVERED, CONSUMED, DROPPED})

# Identity written once, at arming, from the spawning session — never rewritten by a later
# mutation. Delivery routes on these; a record that could be re-homed mid-flight could be
# delivered to another profile's session under another profile's credential.
_IDENTITY_FIELDS = ("mode", "adapter_id", "profile", "origin_session_id")

# A claim is exclusive but not eternal: a consumer that dies mid-delivery must not wedge the
# completion forever, so a stale claim is reclaimable.
CLAIM_TTL_SECONDS = 300.0
MAX_DELIVERY_ATTEMPTS = 5
# Bounded wait for the receipt lock; a contended receipt is retried on the next replay tick.
_LOCK_TIMEOUT_SECONDS = 5.0
_LOCK_POLL_SECONDS = 0.05
_IS_WINDOWS = sys.platform == "win32"
_LOCK_CONTENTION_ERRNOS = {errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK, errno.EDEADLK}

# ``reserve_wake_delivery`` outcomes.
WAKE_RUN = "run"                  # first observation — this caller owns the turn
WAKE_REPLAY = "replay"            # already served by some process; must NOT run a second turn
WAKE_UNAVAILABLE = "unavailable"  # could not read the ledger; transient, ask the caller to retry
WAKE_KEY_PREFIX = "hermes-proc-completion-"


class DurableArmingError(RuntimeError):
    """The durable completion obligation could not be written; the spawn must not be reported."""


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


def _locked_the_live_inode(path: Path, handle) -> bool:
    """Whether the handle we just locked is still the file *path* names.

    POSIX locks live on the INODE, not the name. ``drop_receipt`` unlinks a settled receipt's
    lock file while holding that lock, so a handle opened an instant BEFORE that unlink would
    lock a detached inode nobody else can see — and believe it holds a lock that excludes
    nothing. Re-opening on a mismatch closes that split. Windows cannot unlink an open file and
    ``st_ino`` is not dependable there, so the check is POSIX-only.
    """
    if _IS_WINDOWS:
        return True
    try:
        return os.stat(path).st_ino == os.fstat(handle.fileno()).st_ino
    except OSError:
        return False


def _acquire_receipt_lock(path: Path, deadline: float):
    """Open + flock *path*, re-opening on an inode split. Returns the held handle or None."""
    while True:
        try:
            handle = open(path, "a+b")
        except OSError as exc:
            logger.warning("Could not open completion receipt lock %s (%s); deferring", path, exc)
            return None
        try:
            _try_lock(handle)
        except (BlockingIOError, OSError) as exc:
            handle.close()
            if not _is_contention(exc) or time.monotonic() >= deadline:
                return None
            time.sleep(_LOCK_POLL_SECONDS)
            continue
        if _locked_the_live_inode(path, handle):
            return handle
        with contextlib.suppress(OSError, ValueError):
            _unlock(handle)
        handle.close()
        if time.monotonic() >= deadline:
            return None


@contextlib.contextmanager
def receipt_lock(session_id: str, *, timeout: float = _LOCK_TIMEOUT_SECONDS) -> Iterator[bool]:
    """Hold the cross-process lock for one receipt; yields False when it could not be taken.

    Fail closed: a caller that does not hold the lock must not mutate the delivery state (the
    completion stays where it is and the next pass retries it).
    """
    path = _lock_path(session_id)
    try:
        results_dir().mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create the completion outbox directory (%s); deferring", exc)
        yield False
        return
    handle = _acquire_receipt_lock(path, time.monotonic() + timeout)
    try:
        yield handle is not None
    finally:
        if handle is not None:
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


def drop_receipt(path: Path) -> bool:
    """Delete a SETTLED receipt and its lock file, both under that receipt's own lock.

    Two rules this enforces for the pruner, which is its only caller: the unsettled check is
    re-read under the lock (a completion armed between the scan and now is not collateral), and
    the ``.lock`` file is never unlinked by a process that does not hold it.
    """
    session_id = path.stem
    with receipt_lock(session_id) as held:
        if not held:
            return False
        record = read_receipt(path)
        if record is not None and is_unsettled(record):
            return False
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            _lock_path(session_id).unlink(missing_ok=True)
        return True


# ----- Arming and terminal outcomes ------------------------------------------------------


def _block_for(session: Any, state: str) -> Dict[str, Any]:
    """A fresh delivery block describing *session*'s durable route."""
    return {
        "state": state,
        "mode": session.completion_delivery,
        "adapter_id": session.delivery_adapter_id,
        # Multiplex profile that spawned the worker ("" = default). Immutable; see the module
        # docstring — it selects both the /p/<profile>/ wake path and the credential used on it.
        "profile": session.delivery_profile,
        "origin_session_id": session.watcher_chat_id,
        "survive_gateway_restart": bool(session.survive_gateway_restart),
        "pid": session.pid,
        "host_start_time": session.host_start_time,
        "claim": None, "claimed_at": None, "attempts": 0, "updated_at": time.time(),
        "consume_requested": False, "wake_key": "", "wake_served_at": None,
        "event": None,
    }


def arm(session: Any, base_record: Dict[str, Any]) -> None:
    """Write the ``armed`` receipt for a freshly tracked process.

    Called before the reader thread exists, so the route metadata is on disk in the FIRST
    receipt: an immediate exit cannot win the race and produce a routeless completion.

    Raises :class:`DurableArmingError` when the obligation could not be written. The caller
    (``ProcessRegistry._track_started``) reaps the already-spawned child and fails the tool call:
    a notified spawn reported as started, with nothing on disk to deliver it, is precisely the
    "Executing now" failure this module exists to prevent.
    """
    record = dict(base_record)
    record.update(schema=RECEIPT_SCHEMA, exited=False, completion=_block_for(session, ARMED))
    try:
        with receipt_lock(session.id) as held:
            if not held:
                raise DurableArmingError(
                    f"receipt lock for {session.id} was busy; the completion obligation was "
                    "not written")
            write_receipt(record)
    except OSError as exc:
        raise DurableArmingError(
            f"could not write the completion obligation for {session.id}: {exc}") from exc


def attach_outcome(record: Dict[str, Any], previous: Optional[Dict[str, Any]], *, session: Any,
                   event: Optional[Dict[str, Any]], consumed: bool) -> Dict[str, Any]:
    """Fold the terminal outcome into *record*'s delivery block (caller holds the receipt lock).

    A completion whose result the owning turn already read (wait/log/explicit kill) settles as
    ``consumed`` so no later replay can wake the session a second time with a result it has.
    The armed block's identity (mode/adapter/profile/origin) is carried over verbatim: this runs
    in the producing process, but the route was decided — and authorized — at arming time.
    """
    fresh = _block_for(session, ARMED)
    block = dict(completion_block(previous)) or fresh
    for key in _IDENTITY_FIELDS:
        block.setdefault(key, fresh[key])
    if block.get("state") not in TERMINAL_STATES:
        state = CONSUMED if (consumed or block.get("consume_requested")) else PENDING
        block.update(state=state, event=event, updated_at=time.time(),
                     claim=None, claimed_at=None, consume_requested=False)
    record.update(schema=RECEIPT_SCHEMA, exited=True, completion=block)
    return record


def consume(session_id: str) -> bool:
    """Settle a completion the owning turn read itself; True when it is terminally consumed.

    Mutually exclusive with delivery. A CLAIMED receipt has a consumer that may already have a
    wake in flight, and stealing it would let ``settle()`` fail silently while the session got
    woken anyway. Record the intent instead: whoever holds the claim converges the receipt to
    ``consumed`` when it releases, and :func:`claim` refuses to hand out a new claim for it.
    """
    def _apply(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        block = completion_block(record)
        state = block.get("state")
        if state not in UNSETTLED_STATES:
            return None
        if state == CLAIMED:
            if block.get("consume_requested"):
                return None
            block.update(consume_requested=True, updated_at=time.time())
        else:
            block.update(state=CONSUMED, claim=None, claimed_at=None, consume_requested=False,
                         updated_at=time.time())
        record["completion"] = block
        return record

    def _settled(record: Dict[str, Any]) -> bool:
        return completion_block(record).get("state") == CONSUMED

    return _mutate(session_id, _apply, verify=_settled)


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
        state = CONSUMED if block.get("consume_requested") else PENDING
        block.update(state=state, updated_at=time.time(), claim=None, claimed_at=None,
                     consume_requested=False, event=_lost_event(record, block, reason))
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


def _mutate(session_id: str, apply, *, verify=None) -> bool:
    """Read-modify-write one receipt under its lock; *apply* returns the new record or None.

    ``verify`` (optional) inspects the record after a no-op ``apply`` so a caller can distinguish
    "nothing to do because it is already in the state I wanted" from "nothing to do at all".
    """
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
                return bool(verify and verify(record))
            write_receipt(updated)
            return bool(verify(updated)) if verify else True
    except OSError:
        logger.warning("Could not update completion receipt %s", session_id, exc_info=True)
        return False


def _claim_is_stale(block: Dict[str, Any], now: float) -> bool:
    """Whether a CLAIMED receipt's holder is presumed dead (it never released)."""
    return (now - float(block.get("claimed_at") or 0.0)) >= CLAIM_TTL_SECONDS


def pending_completions(adapter_id: str, *, profile: str = "") -> List[Dict[str, Any]]:
    """Replay-eligible receipts this adapter/profile owns, oldest first.

    Eligible means ARMED (reconcile it against the live process table), PENDING, or CLAIMED with
    a claim older than :data:`CLAIM_TTL_SECONDS` — a consumer that crashed mid-delivery never
    releases, so without the stale-claim row its receipt would never be inspected again and both
    the TTL and the attempt budget would be dead code. Legacy schema-1 receipts are skipped.
    """
    from tools.process_registry_results import retained_result_paths

    out: List[Dict[str, Any]] = []
    try:
        paths = retained_result_paths()
    except OSError:
        logger.warning("Could not scan the completion outbox", exc_info=True)
        return out
    now = time.time()
    for path in paths:
        record = read_receipt(path)
        block = completion_block(record)
        if not block or block.get("adapter_id") != adapter_id:
            continue
        if str(block.get("profile") or "") != profile:
            continue
        state = block.get("state")
        if state not in {ARMED, PENDING} and not (state == CLAIMED and _claim_is_stale(block, now)):
            continue
        if record.get("id") != path.stem:
            continue
        out.append(record)
    out.sort(key=lambda r: (r.get("started_at") or 0.0, r.get("id", "")))
    return out


def claim(session_id: str, claim_id: str, *, adapter_id: str,
          profile: str = "") -> Optional[Dict[str, Any]]:
    """Take the exclusive delivery claim for one pending completion.

    Returns the claimed record, or None when another consumer holds a live claim, the receipt
    belongs to a different adapter/profile, the owning turn asked to consume it, or it is no
    longer deliverable. Taking over a stale claim is safe on its own: re-posting the same wake
    key is gated durably by :func:`reserve_wake_delivery`, so a crashed consumer's in-flight
    delivery cannot become a second turn.
    """
    claimed: Dict[str, Any] = {}

    def _apply(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        block = completion_block(record)
        if not block or block.get("adapter_id") != adapter_id:
            return None
        if str(block.get("profile") or "") != profile:
            return None
        now = time.time()
        state = block.get("state")
        if state not in UNSETTLED_STATES:
            return None
        if block.get("consume_requested"):
            # wait/log/kill read the result inside the owning turn while this receipt was
            # claimed. Converge terminally instead of waking the session with what it has.
            block.update(state=CONSUMED, claim=None, claimed_at=None, consume_requested=False,
                         updated_at=now)
            record["completion"] = block
            return record
        if state == CLAIMED:
            if not _claim_is_stale(block, now):
                return None
            logger.warning("Reclaiming background completion %s: its consumer held the claim for "
                           "more than %.0fs without releasing it", session_id, CLAIM_TTL_SECONDS)
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

    # A write without a populated ``claimed`` is the attempts-exhausted drop (or the consume
    # convergence) above, not a claim.
    return dict(claimed) if (_mutate(session_id, _apply) and claimed) else None


def release(session_id: str, claim_id: str) -> bool:
    """Return a failed delivery to ``pending`` for a later retry.

    Once the attempt budget is spent the receipt converges to terminal ``dropped``: the result
    stays queryable through ``process_manage``, but it stops waking the session forever. A
    consume requested while we held the claim wins outright — the turn already has the result.
    """
    def _apply(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        block = completion_block(record)
        if block.get("state") != CLAIMED or block.get("claim") != claim_id:
            return None
        if block.get("consume_requested"):
            block.update(state=CONSUMED, claim=None, claimed_at=None, consume_requested=False,
                         updated_at=time.time())
            record["completion"] = block
            return record
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
    """Terminally settle a claimed completion after the delivery was ACCEPTED.

    False means the claim was no longer ours (reclaimed after the TTL, or converged by a
    consume): the caller must treat the receipt as someone else's and say so, never assume the
    state it wanted is the state on disk.
    """
    if state not in TERMINAL_STATES:
        raise ValueError(f"not a terminal completion state: {state!r}")

    def _apply(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        block = completion_block(record)
        if block.get("state") != CLAIMED or block.get("claim") != claim_id:
            return None
        block.update(state=state, claim=None, claimed_at=None, consume_requested=False,
                     updated_at=time.time())
        record["completion"] = block
        return record

    return _mutate(session_id, _apply)


# ----- Restart-stable wake idempotency ----------------------------------------------------


def idempotency_key(record: Dict[str, Any]) -> str:
    """Stable per-completion key so a retried wake cannot run the same turn twice.

    ``started_at`` is the producer's spawn epoch, so a recycled process id is a distinct
    incarnation rather than a suppressed duplicate.
    """
    return f"{WAKE_KEY_PREFIX}{record.get('id', '')}-{int(float(record.get('started_at') or 0.0))}"


def reserve_wake_delivery(key: str) -> str:
    """Durable at-most-once gate for one completion wake. Called by the API server BEFORE the turn.

    The in-memory ``_IdempotencyCache`` only dedupes within one process and one 5-minute TTL, so
    a wake accepted and then followed by a gateway restart would re-run as a real second turn.
    The mark is written into the receipt the key names — the same file the claim lives in, in the
    same profile home the wake was routed to — so it survives the restart.

    Returns :data:`WAKE_RUN` (run the turn), :data:`WAKE_REPLAY` (already served — replay a
    response, do NOT run it again), or :data:`WAKE_UNAVAILABLE` (the ledger could not be read; the
    caller should answer transiently so the producer retries). Keys this ledger does not govern
    return :data:`WAKE_RUN`.
    """
    if not key.startswith(WAKE_KEY_PREFIX):
        return WAKE_RUN
    session_id = key[len(WAKE_KEY_PREFIX):].rpartition("-")[0]
    if not session_id:
        return WAKE_RUN
    path = receipt_path(session_id)
    try:
        with receipt_lock(session_id) as held:
            if not held:
                return WAKE_UNAVAILABLE
            record = read_receipt(path)
            block = completion_block(record)
            if not block or idempotency_key(record) != key:
                return WAKE_RUN  # not a live completion wake; nothing here governs it
            if block.get("wake_key") == key:
                logger.info("Suppressed a duplicate completion wake for %s: it was already "
                            "served at %s", session_id, block.get("wake_served_at"))
                return WAKE_REPLAY
            block.update(wake_key=key, wake_served_at=time.time())
            record["completion"] = block
            write_receipt(record)
            return WAKE_RUN
    except OSError:
        logger.warning("Could not reserve the completion wake for %s", session_id, exc_info=True)
        return WAKE_UNAVAILABLE
