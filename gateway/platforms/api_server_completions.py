"""Durable background-completion replay for API/Wizard sessions.

A finite ``/v1/runs`` (or ``/v1/chat/completions``) turn ends the moment its HTTP response is
written, so a ``terminal(background=true, notify=true)`` worker it spawned has no live parent to
report to. The worker's completion is persisted by ``tools.process_registry_outbox``; this loop is
the consumer that takes it out again — in THIS process or in whichever API process exists next —
and resumes the exact originating session with a new turn.

The adapter owns the loop (started from ``connect()``) rather than the GatewayRunner, so a
standalone ``wizard-api.service`` with no messaging gateway still delivers.

Exactly-once rests on three things, not on this loop being careful: the outbox claim is exclusive
across processes, a wake is only settled after the API accepted it, and every wake self-post
carries a per-completion ``Idempotency-Key`` so an accepted-but-unacknowledged retry replays the
stored response instead of running the turn twice.
"""

import asyncio
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Delivery is not latency-critical (the client is no longer waiting on that turn), but a
# still-open Wizard session should see its result within seconds of the worker exiting.
REPLAY_INTERVAL_SECONDS = 5.0
DELIVERY_ADAPTER_ID = "api_server"


def _reconcile_running(record: Dict[str, Any]) -> bool:
    """Promote an ``armed`` receipt whose worker is gone to a pending unknown-outcome completion.

    Armed means "still running" — true only while the producing process lives. A receipt still
    armed with no live worker is a job that died with its process (crash, SIGKILL, host reboot);
    delivering an unknown outcome beats leaving the session waiting forever.
    """
    from tools.process_registry import process_registry
    from tools.process_registry_outbox import ARMED, completion_block, record_lost

    block = completion_block(record)
    if block.get("state") != ARMED:
        return False
    session_id = record.get("id", "")
    if process_registry.get(session_id) is not None:
        return False
    pid, start = block.get("pid"), block.get("host_start_time")
    if pid and process_registry._host_pid_is_ours(pid, start):
        return False
    return record_lost({"session_id": session_id, "completion_delivery": block.get("mode", "")})


async def _deliver_one(adapter: Any, record: Dict[str, Any]) -> bool:
    """Claim, format, wake, settle. True when the completion was terminally settled."""
    from gateway.wake import deliver_wake
    from tools.process_registry_notifications import format_process_notification
    from tools.process_registry_outbox import (
        claim, completion_block, idempotency_key, new_claim_id, release, settle)

    session_id = record.get("id", "")
    claim_id = new_claim_id(DELIVERY_ADAPTER_ID)
    claimed = await asyncio.to_thread(claim, session_id, claim_id, adapter_id=DELIVERY_ADAPTER_ID)
    if claimed is None:
        return False  # another consumer holds it, or it settled between the scan and now
    block = completion_block(claimed)
    target = str(block.get("origin_session_id") or "")
    event = block.get("event") or {}
    text = format_process_notification(event) if event else None
    if not target or not text:
        logger.warning("Background completion %s has no resumable destination; dropping the replay "
                       "(the result remains queryable through process_manage).", session_id)
        await asyncio.to_thread(settle, session_id, claim_id, "dropped")
        return True
    try:
        await deliver_wake(adapter, text=text, session_id=target,
                           idempotency_key=idempotency_key(claimed))
    except Exception as exc:
        logger.warning("Background completion %s could not wake session %s (%s); will retry",
                       session_id, target, exc)
        await asyncio.to_thread(release, session_id, claim_id)
        return False
    await asyncio.to_thread(settle, session_id, claim_id, "delivered")
    logger.info("Background completion %s delivered to API session %s", session_id, target)
    return True


async def replay_pending_completions(adapter: Any) -> int:
    """One pass over this adapter's outbox; returns how many completions were settled."""
    from tools.process_registry_outbox import PENDING, completion_block, pending_completions

    records: List[Dict[str, Any]] = await asyncio.to_thread(pending_completions, DELIVERY_ADAPTER_ID)
    settled = 0
    for record in records:
        if completion_block(record).get("state") != PENDING:
            # Still armed: reconcile it against the live process table. A promoted receipt is
            # delivered on the next pass, reading its terminal outcome from disk.
            await asyncio.to_thread(_reconcile_running, record)
            continue
        settled += await _deliver_one(adapter, record)
    return settled


async def completion_replay_loop(adapter: Any, *, interval: float = REPLAY_INTERVAL_SECONDS) -> None:
    """Drain the durable completion outbox for as long as this adapter serves requests."""
    while True:
        try:
            await asyncio.sleep(interval)
            if not adapter.is_connected():
                continue
            await replay_pending_completions(adapter)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Background completion replay pass failed", exc_info=True)
