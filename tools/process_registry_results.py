"""Bounded, profile-local receipts for completed terminal processes.

Receipts are read through process_manage and never adopted as live PIDs. A receipt carrying a
durable ``completion`` block (schema 2) is additionally an outbox row — see
``tools.process_registry_outbox`` for the delivery state machine. Each producer writes its own
file so independent one-shot parents cannot overwrite each other's results in the running-PID
checkpoint.
"""

import json
import logging
import re
import sqlite3
import time

from tools import process_registry_outbox as outbox

logger = logging.getLogger("tools.process_registry")

RESULT_RETENTION_SECONDS = 7 * 24 * 60 * 60
MAX_RETAINED_RESULTS = 64
_RESULT_FIELDS = (
    "id", "command", "cwd", "task_id", "owner_task_id", "session_key",
    "parent_session_id", "started_at", "exit_code", "completion_reason",
    "termination_source", "notify_on_complete",
)


def _drop(path) -> None:
    path.unlink(missing_ok=True)
    path.with_suffix(".lock").unlink(missing_ok=True)


def retained_result_paths():
    """Prune settled receipts by completion time (jobs can take days), newest first.

    A completion that still owes a delivery is immune to the count cap — pruning it is exactly
    the silent loss the outbox exists to prevent — but not immortal: past the retention window
    nobody is waiting on it, and immunity without a bound would grow the directory forever.
    """
    directory = outbox.results_dir()
    cutoff = time.time() - RESULT_RETENTION_SECONDS
    retained = []
    for path in directory.glob("proc_*.json"):
        try:
            modified = path.stat().st_mtime
            if modified < cutoff:
                _drop(path)
            else:
                retained.append((modified, path))
        except FileNotFoundError:
            continue  # Another producer pruned it.
    retained.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    kept_over_cap = []
    for _, path in retained[MAX_RETAINED_RESULTS:]:
        if outbox.is_unsettled(outbox.read_receipt(path)):
            kept_over_cap.append(path)
        else:
            _drop(path)
    return [path for _, path in retained[:MAX_RETAINED_RESULTS]] + kept_over_cap


def save_completed_result(session, *, notification=None, consumed: bool = False) -> None:
    """Persist the terminal receipt for *session*.

    For an armed process this is the transition that makes its completion ``pending``: it runs
    while the session is still tracked as running, so no consumer can observe "finished, no
    receipt". ``notification`` is the completion_queue event the delivery replays verbatim.
    """
    from agent.redact import redact_sensitive_text, redact_terminal_output
    from tools.process_registry import MAX_OUTPUT_CHARS

    with session._lock:
        record = {key: getattr(session, key) for key in _RESULT_FIELDS}
        record["output"] = session.output_buffer[-MAX_OUTPUT_CHARS:]
    # Live-output opt-out must not persist raw credentials in durable receipts.
    record["output"] = redact_terminal_output(record["output"], record["command"], force=True)
    record["command"] = redact_sensitive_text(record["command"], code_file=True, force=True)
    try:
        if not session.completion_delivery:
            outbox.write_receipt(record)
        else:
            with outbox.receipt_lock(session.id) as held:
                if not held:
                    # Writing without the lock could clobber a concurrent consumer's claim; the
                    # completion is lost rather than duplicated, and the result stays in memory.
                    logger.error("Could not settle durable completion %s: receipt lock busy — its "
                                 "delivery will not be replayed", session.id)
                    return
                previous = outbox.read_receipt(outbox.receipt_path(session.id))
                outbox.write_receipt(outbox.attach_outcome(
                    record, previous, session=session, event=notification, consumed=consumed))
        retained_result_paths()
    except OSError:
        # Preserve live delivery on disk failure, but never silently claim durability.
        logger.warning("Could not retain completed process result %s", session.id, exc_info=True)


def _owns_result(owner: str, parent: str | None) -> bool:
    if not parent:
        return False
    if owner == parent:
        return True
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        return db.get_compression_tip(parent) == owner
    finally:
        db.close()


def load_completed_results(prefix: str = "") -> dict:
    """Restore read-only snapshots; no process handles, watchers, or queue events."""
    from tools.process_registry import ProcessSession

    from gateway.session_context import get_session_env

    owner = get_session_env("HERMES_SESSION_ID", "")
    if not owner:
        return {}
    results = {}
    try:
        paths = retained_result_paths()
    except OSError:
        logger.warning("Could not read retained process results", exc_info=True)
        return results
    for path in paths:
        if not path.stem.startswith(prefix):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["id"] != path.stem or not re.fullmatch(r"proc_[\w]+", record["id"]):
                continue
            # An armed receipt describes a process that is still RUNNING; only legacy receipts
            # (no flag) and terminal ones are restorable results.
            if not record.get("exited", True):
                continue
            if not _owns_result(owner, record.get("parent_session_id")):
                continue
            session = ProcessSession(
                **{key: record[key] for key in _RESULT_FIELDS},
                exited=True, output_buffer=record["output"],
            )
            session._completion_event.set()
            results[session.id] = session
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
            logger.debug("Skipping unreadable process result %s", path.name, exc_info=True)
    return results
