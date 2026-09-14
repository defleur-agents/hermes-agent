"""Durable background-completion replay for API/Wizard sessions.

A finite ``/v1/runs`` (or ``/v1/chat/completions``) turn ends the moment its HTTP response is
written, so a ``terminal(background=true, notify=true)`` worker it spawned has no live parent to
report to. The worker's completion is persisted by ``tools.process_registry_outbox``; this loop is
the consumer that takes it out again — in THIS process or in whichever API process exists next —
and resumes the exact originating session with a new turn.

The adapter owns the loop (started from ``connect()``, cancelled on bind failure and on
``disconnect()``) rather than the GatewayRunner, so a standalone ``wizard-api.service`` with no
messaging gateway still delivers.

**Every outbox, not just the default one.** Receipts are profile-local: they live under the
spawning profile's ``HERMES_HOME``. Under ``gateway.multiplex_profiles`` one adapter serves
``/v1/...`` and every ``/p/<profile>/v1/...`` mirror, so one pass must enumerate each served
profile's home in that profile's scope. The receipt's immutable ``completion.profile`` then
selects both the wake path (``/p/<profile>/v1/chat/completions``) and the credential presented on
it, which is that profile's own ``API_SERVER_KEY`` — never the owner's. A named profile whose key
cannot be resolved is skipped with a warning, never delivered under the default principal.

Exactly-once rests on four things, not on this loop being careful: the outbox claim is exclusive
across processes, a stale claim is only reclaimed after its TTL, a wake is settled only after the
API accepted it AND the settle itself succeeded, and every wake self-post carries a
per-completion ``Idempotency-Key`` that the receiving side reserves DURABLY
(``reserve_wake_delivery``) — so a wake accepted by a process that died before settling is
replayed by the next one, never re-run as a second turn.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Delivery is not latency-critical (the client is no longer waiting on that turn), but a
# still-open Wizard session should see its result within seconds of the worker exiting.
REPLAY_INTERVAL_SECONDS = 5.0
DELIVERY_ADAPTER_ID = "api_server"


def _served_outboxes(adapter: Any) -> List[Tuple[str, Path]]:
    """``(profile_identity, hermes_home)`` rows this adapter must drain.

    ``profile_identity`` is the value a turn bound as ``HERMES_SESSION_PROFILE`` and the outbox
    stored in the receipt; it is also the URL prefix (``""`` = the unprefixed default route).
    The default home answers to two identities because both reach it: no prefix at all, and an
    explicit ``/p/default/``.
    """
    from hermes_constants import get_hermes_home

    cfg = getattr(getattr(adapter, "gateway_runner", None), "config", None)
    if not getattr(cfg, "multiplex_profiles", False):
        # Single-profile gateway: the prefix middleware resolves even a self-referential
        # /p/<own-profile>/ to None, so every receipt it writes carries "".
        return [("", Path(get_hermes_home()))]
    try:
        from hermes_cli.profiles import profiles_to_serve
        served = profiles_to_serve(multiplex=True)
    except Exception:
        logger.warning("Could not enumerate multiplex profiles; draining the default outbox only",
                       exc_info=True)
        return [("", Path(get_hermes_home()))]
    rows: List[Tuple[str, Path]] = []
    for name, home in served:
        if name == "default":
            rows.extend((("", Path(home)), ("default", Path(home))))
        else:
            rows.append((name, Path(home)))
    return rows


def _wake_credential(adapter: Any, profile: str) -> str:
    """The API key a wake self-post must present on *profile*'s route; "" when unresolvable.

    Delegates to the adapter's own ``_expected_api_key`` so the wake authenticates exactly the
    way an external client on that route would: the default listener's key for the default
    profile, and the profile-scoped ``API_SERVER_KEY`` for a named one. Never logged.
    """
    resolve = getattr(adapter, "_expected_api_key", None)
    scope = getattr(adapter, "_profile_scope", None)
    if resolve is None or scope is None:
        # An adapter without the multiplex hooks knows exactly one key: the one it binds with,
        # which authorizes the default route only. A named profile must never borrow it.
        return "" if profile else str(getattr(adapter, "_api_key", "") or "")
    try:
        from gateway.platforms.api_server import _api_request_profile

        token = _api_request_profile.set(profile or None)
        try:
            with scope(profile or None):
                return str(resolve() or "")
        finally:
            _api_request_profile.reset(token)
    except Exception:
        logger.warning("Could not resolve the wake credential for profile %r",
                       profile or "default", exc_info=True)
        return ""


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


async def _deliver_one(adapter: Any, record: Dict[str, Any], *, profile: str, api_key: str) -> bool:
    """Claim, format, wake, settle. True when the completion was terminally settled."""
    from gateway.wake import deliver_wake
    from tools.process_registry_notifications import format_process_notification
    from tools.process_registry_outbox import (
        claim, completion_block, idempotency_key, new_claim_id, release, settle)

    session_id = record.get("id", "")
    claim_id = new_claim_id(DELIVERY_ADAPTER_ID)
    claimed = await asyncio.to_thread(
        claim, session_id, claim_id, adapter_id=DELIVERY_ADAPTER_ID, profile=profile)
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
        await deliver_wake(adapter, text=text, session_id=target, profile=profile, api_key=api_key,
                           idempotency_key=idempotency_key(claimed))
    except Exception as exc:
        logger.warning("Background completion %s could not wake session %s on profile %r (%s); "
                       "will retry", session_id, target, profile or "default", exc)
        await asyncio.to_thread(release, session_id, claim_id)
        return False
    if not await asyncio.to_thread(settle, session_id, claim_id, "delivered"):
        # The wake WAS accepted, so the turn ran; we simply no longer own the receipt (the claim
        # aged past its TTL and someone reclaimed it, or the owning turn consumed it meanwhile).
        # Say so rather than reporting a settlement that is not on disk — the durable wake-key
        # reservation is what stops the new owner from producing a second turn.
        logger.warning("Background completion %s was delivered to session %s but its claim was no "
                       "longer ours to settle; the duplicate-wake gate now owns the outcome.",
                       session_id, target)
        return False
    logger.info("Background completion %s delivered to API session %s on profile %r",
                session_id, target, profile or "default")
    return True


async def _replay_one_outbox(adapter: Any, profile: str, home: Path) -> int:
    """Drain one profile's outbox; returns how many completions were settled."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.process_registry_outbox import ARMED, completion_block, pending_completions

    token = set_hermes_home_override(str(home))
    try:
        records: List[Dict[str, Any]] = await asyncio.to_thread(
            pending_completions, DELIVERY_ADAPTER_ID, profile=profile)
        if not records:
            return 0
        api_key = await asyncio.to_thread(_wake_credential, adapter, profile)
        if not api_key:
            logger.warning("Holding %d background completion(s) for profile %r: no usable "
                           "API_SERVER_KEY on that route, and a wake must never be sent under "
                           "another profile's credential.", len(records), profile or "default")
            return 0
        settled = 0
        for record in records:
            if completion_block(record).get("state") == ARMED:
                # Still armed: reconcile it against the live process table. A promoted receipt is
                # delivered on the next pass, reading its terminal outcome from disk.
                await asyncio.to_thread(_reconcile_running, record)
                continue
            # PENDING, or CLAIMED past the TTL by a consumer that crashed without releasing —
            # claim() decides which, and takes the stale one over.
            settled += await _deliver_one(adapter, record, profile=profile, api_key=api_key)
        return settled
    finally:
        reset_hermes_home_override(token)


async def replay_pending_completions(adapter: Any) -> int:
    """One pass over every outbox this adapter serves; returns how many were settled."""
    settled = 0
    for profile, home in _served_outboxes(adapter):
        try:
            settled += await _replay_one_outbox(adapter, profile, home)
        except asyncio.CancelledError:
            raise
        except Exception:
            # One unreadable profile home must not stop the others from draining.
            logger.warning("Background completion replay failed for profile %r",
                           profile or "default", exc_info=True)
    return settled


async def completion_replay_loop(adapter: Any, *, interval: float = REPLAY_INTERVAL_SECONDS) -> None:
    """Drain the durable completion outboxes for as long as this adapter serves requests."""
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
