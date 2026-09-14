"""Background-process launch path: ``terminal(background=true)`` spawns a tracked
process via the process registry (Popen locally, ``env.execute`` in a sandbox),
stamps gateway routing metadata for completion / watch-pattern notifications and
returns the JSON result. Lazy ``tools.terminal_tool`` imports keep the origin's
monkeypatch points authoritative.
"""

import json
import logging
from typing import Any, List, Optional

logger = logging.getLogger("tools.terminal_tool")

# A silent background process (no notify_on_complete / watch_patterns) is right
# only for servers/watchers; for bounded tasks the agent almost always wanted a
# notification and forgot the flag, so nudge it (cheap false positive).
_SILENT_BACKGROUND_HINT = (
    'background=true without notify_on_complete=true means this process runs SILENTLY — you '
    'will not be told when it exits. If this is a bounded task (test suite, build, CI poller, '
    'deploy, anything with a defined end), you almost certainly wanted notify_on_complete=true '
    'so the system pings you on exit. Re-launch with notify_on_complete=true, or call '
    "process(action='poll') / process(action='wait') yourself to learn the outcome. Only "
    'ignore this hint for genuine long-lived processes that never exit (servers, watchers, '
    'daemons).'
)

# Homebrewed CI pollers built on `gh pr view --json statusCheckRollup` or
# `gh pr checks | jq` fail silently in known ways (block-buffered stdout never
# reaches capture, jq null-key edge cases exit the loop, conclusion-vs-status
# confusion declares all-green early, TTY-only banners never appear piped).
# Detector is deliberately narrow: the canonical column-2 awk poller is fine.
_HOMEBREW_CI_POLLER_HINT = (
    'This looks like a homebrewed CI poller built from `gh pr view --json statusCheckRollup` '
    'and/or `gh pr checks | jq`. That shape has burned us repeatedly in hermes-agent dev work '
    '(PRs #31329, #31448, #31695, #31709, #31745, #32264, #33131) — stdout buffering kills '
    'output capture, jq null-key edge cases silently exit the loop, conclusion-vs-status field '
    'confusion exits early with bogus all-green verdicts, TTY-only summary banners never '
    'appear when piped. Use the canonical snippets in the green-ci-policy skill instead: the '
    'exit-code-driven `gh pr checks $PR >/dev/null` (rc 0 = green, 8 = pending, else fail) for '
    'exit-on-first-fail behavior, or the column-2 awk-on-tabs poller (`awk -F"\\t" '
    '"$2==\\"pending\\""`) for sharded matrices. Load '
    "skill_view(name='github/hermes-agent-dev', file_path='references/green-ci-policy.md') for "
    'the verbatim snippets. If you must roll a custom loop with rich structured output, write '
    "each tick to a known file (`tee -a /tmp/ci.log`) and rely on `process(action='log')` to "
    'read THAT file — do not rely on background-process stdout capture for line-buffered shell '
    'loops.'
)

_ASYNC_UNSUPPORTED_NOTE = (
    'notify_on_complete / watch_patterns are not available in this session — it cannot receive '
    'an async completion after the turn ends (a one-shot runner such as `hermes -z`, a cron '
    'job, a Kanban worker, or a stateless HTTP endpoint). The process is running in the '
    "background; retrieve its result with process(action='poll') or process(action='wait')."
)

# A finite API run ENDS when its HTTP turn returns. Detaching a notify-on-complete worker there
# without a durable completion route is the "Executing now" failure: the worker dies or finishes
# after the run is gone and nothing ever resumes the session. Refuse the spawn outright so the
# model learns inside the turn instead of promising a notification nobody can deliver.
_API_DURABLE_UNSUPPORTED_ERROR = (
    'This API session cannot receive a background notification after the turn ends, so NO process '
    'was started (there is no PID to poll). This run is finite: it finishes when this response is '
    'returned, and its session id is not one the server can resume. Run the command in the '
    'FOREGROUND (background=false) and report the real result in this turn, or split the work so '
    'each turn finishes what it starts.'
)

# The durable API wake replays ONE terminal completion into the session. Streaming interim
# watch-pattern matches would each need their own authorized turn, which this route does not grant.
_API_WATCH_UNSUPPORTED_NOTE = (
    'watch_patterns are not available on API sessions: the durable completion route delivers a '
    "process's FINAL outcome only, never interim pattern matches. Use notify_on_complete=true to "
    "be told when it exits, or process(action='poll') to inspect it yourself."
)

# proc_session attribute -> HERMES_SESSION_* env var carrying it.
_ROUTING_FIELDS = (
    ("watcher_chat_id", "HERMES_SESSION_CHAT_ID"),
    ("watcher_user_id", "HERMES_SESSION_USER_ID"),
    ("watcher_user_name", "HERMES_SESSION_USER_NAME"),
    ("watcher_thread_id", "HERMES_SESSION_THREAD_ID"),
    ("watcher_message_id", "HERMES_SESSION_MESSAGE_ID"),
    # The spawning conversation's session-db id lets the gateway's completion
    # pre-flight drop the notification if the user closed this session (/new)
    # before the process finished, instead of injecting it into the NEW one.
    ("parent_session_id", "HERMES_SESSION_ID"),
)


def _looks_like_homebrew_ci_poller(command: str) -> bool:
    has_gh = "gh pr view" in command or "gh pr checks" in command
    has_jq = " jq " in command or "| jq" in command or "$(jq" in command
    # `gh pr checks` doesn't emit JSON, so piping it to jq is confused intent.
    return "statusCheckRollup" in command or (has_gh and has_jq)


def _gateway_routing_fields(get_session_env) -> dict:
    """The spawning chat's routing metadata, as ProcessSession field values, so completion /
    watch notifications reach the right chat/thread. Empty when no platform is bound."""
    platform = get_session_env("HERMES_SESSION_PLATFORM", "")
    if not platform:
        return {}
    return {"watcher_platform": platform,
            **{attr: get_session_env(var, "") for attr, var in _ROUTING_FIELDS}}


def _stamp_gateway_routing(proc_session, get_session_env) -> None:
    """Apply :func:`_gateway_routing_fields` to an already-created session (adoption path)."""
    for attr, value in _gateway_routing_fields(get_session_env).items():
        setattr(proc_session, attr, value)


def _spawn(process_registry, *, env, env_type, command, cwd, effective_task_id, task_id,
           session_key, effective_pty, session_fields):
    common = dict(command=command, cwd=cwd, task_id=effective_task_id,
                  owner_task_id=task_id or effective_task_id, session_key=session_key,
                  session_fields=session_fields)
    if env_type == "local":
        return process_registry.spawn_local(
            env_vars=env.env if hasattr(env, 'env') else None, use_pty=effective_pty, **common)
    return process_registry.spawn_via_env(env=env, **common)


class _UnsupportedCompletion(Exception):
    """The requested completion contract cannot be honored on this session — do not spawn."""


def _resolve_completion_support(notify_on_complete, watch_patterns, result_data):
    """Decide, BEFORE spawning, how this session can deliver a completion.

    Returns the ``session_fields`` the spawn must pre-bind (routing, notify flag, durable mode)
    and the effective ``watch_patterns``. Raises :class:`_UnsupportedCompletion` when a finite API
    run asks for a notification nothing could ever deliver — refusing before the spawn is what
    keeps an unsupported run from leaving a PID behind with a promise attached.
    """
    from gateway.session_context import (
        async_delivery_supported, completion_delivery_mode, get_session_env)

    if not (notify_on_complete or watch_patterns):
        return {}, watch_patterns
    routing = _gateway_routing_fields(get_session_env)
    if async_delivery_supported():
        return {**routing, "notify_on_complete": bool(notify_on_complete)}, watch_patterns
    mode = completion_delivery_mode()
    if mode:
        if watch_patterns:
            # Watch-pattern streaming needs an authorized turn per match; the durable API route
            # grants exactly one, for the terminal outcome.
            result_data["watch_patterns_unsupported"] = _API_WATCH_UNSUPPORTED_NOTE
            watch_patterns = None
        if not notify_on_complete:
            return {}, None
        # The resumable destination is the bound chat id (the real session), never the
        # run/approval session_key — that key is per-run and resumes no conversation.
        if not routing.get("watcher_chat_id"):
            raise _UnsupportedCompletion()
        return {**routing, "notify_on_complete": True, "completion_delivery": mode,
                "delivery_adapter_id": get_session_env("HERMES_SESSION_PLATFORM", "")}, None
    if get_session_env("HERMES_SESSION_PLATFORM", "") == "api_server":
        raise _UnsupportedCompletion()
    result_data["notify_on_complete"] = False
    result_data["notify_unsupported"] = _ASYNC_UNSUPPORTED_NOTE
    logger.info("background spawn: async delivery unsupported on this session; "
                "notify_on_complete/watch_patterns disabled")
    return {}, None


def _register_completion_watcher(process_registry, proc_session, session_key) -> None:
    """Gateway mode: register a fast watcher so completion triggers a new
    agent turn (CLI mode uses the completion_queue directly)."""
    proc_session.watcher_interval = 5
    process_registry.pending_watchers.append({
        "session_id": proc_session.id, "check_interval": 5, "session_key": session_key,
        "platform": proc_session.watcher_platform,
        **{attr.removeprefix("watcher_"): getattr(proc_session, attr)
           for attr, _ in _ROUTING_FIELDS[:-1]},
        "notify_on_complete": True, "parent_session_id": proc_session.parent_session_id,
    })


def spawn_background_process(
    *, command: str, env: Any, env_type: str, effective_task_id: str, task_id: Optional[str],
    session_key: str, workdir: Optional[str], cwd: str, effective_pty: bool,
    notify_on_complete: bool, watch_patterns: Optional[List[str]], approval_note: Optional[str],
    pty_disabled_reason: Optional[str],
) -> str:
    """Spawn *command* as a tracked background process and return the JSON result.

    Never inline-polls ``is_interrupted()``: the spawn detaches and returns
    exit_code 0 immediately, so the stale-interrupt kill cannot occur here.
    """
    from tools.process_registry import process_registry
    from tools.terminal_tool import (
        _redact_terminal_error_text, _resolve_command_cwd, _resolve_notification_flag_conflict,
    )

    effective_cwd = _resolve_command_cwd(
        workdir=workdir, default_cwd=cwd, session_key=session_key, env_type=env_type,
    )
    result_data = {"output": "Background process started", "exit_code": 0, "error": None}
    try:
        # Resolve the completion contract BEFORE spawning: an unsupported finite API run must
        # fail with no PID and no process entry, not with a detached promise nobody can keep.
        session_fields, watch_patterns = _resolve_completion_support(
            notify_on_complete, watch_patterns, result_data)
    except _UnsupportedCompletion:
        logger.warning("Refusing a detached notify-on-complete spawn: this API session has no "
                       "durable completion route")
        return json.dumps({"output": "", "exit_code": -1, "error": _API_DURABLE_UNSUPPORTED_ERROR},
                          ensure_ascii=False)
    notify_on_complete = bool(session_fields.get("notify_on_complete"))
    watch_patterns, conflict_note = _resolve_notification_flag_conflict(
        notify_on_complete=notify_on_complete, watch_patterns=watch_patterns, background=True,
    )
    if conflict_note:
        logger.warning("background spawn: %s", conflict_note)
        result_data["watch_patterns_ignored"] = conflict_note
    if watch_patterns:
        session_fields["watch_patterns"] = list(watch_patterns)
    try:
        proc_session = _spawn(
            process_registry, env=env, env_type=env_type, command=command, cwd=effective_cwd,
            effective_task_id=effective_task_id, task_id=task_id, session_key=session_key,
            effective_pty=effective_pty, session_fields=session_fields,
        )
        result_data.update(session_id=proc_session.id, pid=proc_session.pid)
        if approval_note:
            result_data["approval"] = approval_note
        if pty_disabled_reason:
            result_data["pty_note"] = pty_disabled_reason
        if not notify_on_complete and not watch_patterns:
            result_data["hint"] = _SILENT_BACKGROUND_HINT
        if command and _looks_like_homebrew_ci_poller(command):
            existing = result_data.get("hint", "")
            result_data["hint"] = (existing + "\n\n" + _HOMEBREW_CI_POLLER_HINT if existing
                                   else _HOMEBREW_CI_POLLER_HINT)
        if notify_on_complete:
            result_data["notify_on_complete"] = True
            # A durable route owns its own delivery (the outbox claim is the single arbiter);
            # registering a gateway watcher too would deliver the same completion twice.
            if proc_session.watcher_platform and not proc_session.completion_delivery:
                _register_completion_watcher(process_registry, proc_session, session_key)
            from agent.delegation_context import is_delegated_child_context
            if is_delegated_child_context():
                result_data["notify_on_complete"] = False
                result_data["subagent_note"] = _SUBAGENT_NOTIFY_NOTE
        if watch_patterns:
            result_data["watch_patterns"] = proc_session.watch_patterns
        return json.dumps(result_data, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "output": "", "exit_code": -1,
            "error": _redact_terminal_error_text(f"Failed to start background process: {e}"),
        }, ensure_ascii=False)


_SUBAGENT_NOTIFY_NOTE = (
    "You are a subagent: this process's completion notice will NOT reach your parent, and the process is killed when "
    "you finish. Before you finish, either wait for it (process_manage wait), kill it, or hand it to your parent with "
    "process_manage(action='handoff', session_id=..., data='<purpose>') so the parent receives its completion. For CI "
    "watchers prefer returning the fact (PR number, SHA) and letting the parent watch."
)

_YIELDED_NOTE = (
    "The user sent a message while this command was running, so it was moved to the "
    "background WITHOUT being killed and is still running. You will be notified when it "
    "exits (notify_on_complete). Read the user's message and respond to it now; use "
    "process(action='poll'|'wait'|'log', session_id=...) to check on this command."
)


def yield_to_background_handler(
    *, command: str, env_type: str, cwd: Optional[str], effective_task_id: str,
    task_id: Optional[str], session_key: str,
):
    """Build the ``yield_handler`` a foreground ``env.execute`` calls when the tool thread is
    asked to yield (a user message arrived mid-command). Local backend only: the live Popen
    is adopted by the process registry as a notify-on-complete background session and the
    partial output is returned to the model right away. Other backends return None (no
    adoptable host process) and the foreground wait continues."""
    if env_type != "local":
        return None

    def _handler(proc, output_so_far: str) -> dict:
        from tools.process_registry import process_registry
        session = process_registry.adopt_local(
            proc, command=command, cwd=cwd, task_id=effective_task_id,
            owner_task_id=task_id or effective_task_id, session_key=session_key,
            output_so_far=output_so_far)
        _stamp_routing_if_gateway(process_registry, session, session_key)
        logger.info("foreground command yielded to background as %s (pid %s)", session.id, session.pid)
        return {
            "output": output_so_far, "returncode": None, "yielded_session_id": session.id, "pid": session.pid,
        }
    return _handler


def _stamp_routing_if_gateway(process_registry, session, session_key) -> None:
    """Route the adopted session's completion like a normal notify_on_complete spawn."""
    from gateway.session_context import async_delivery_supported, get_session_env
    if not async_delivery_supported():
        session.notify_on_complete = False
        return
    _stamp_gateway_routing(session, get_session_env)
    if session.watcher_platform:
        _register_completion_watcher(process_registry, session, session_key)
