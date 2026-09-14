"""Completed work remains retrievable when its finite CLI owner exits."""

import http.server
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import textwrap
import threading
import time

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_headless_terminal_result_survives_cli_exit(tmp_path):
    """Real CLI, tool dispatch, shell child and fresh reader; only the LLM is local."""
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  provider: custom\n  api_mode: chat_completions\n"
        "terminal:\n  env: local\n  oneshot_completion_wait_seconds: 10\n"
        "memory:\n  memory_enabled: false\n  user_profile_enabled: false\n",
        encoding="utf-8",
    )
    release = tmp_path / "release"
    child = tmp_path / "review.py"
    child.write_text(textwrap.dedent('''
        import pathlib, sys, time
        deadline = time.monotonic() + 15
        while not pathlib.Path(sys.argv[1]).exists():
            if time.monotonic() > deadline:
                sys.exit(91)
            time.sleep(0.02)
        print("SYNTHETIC_REVIEW_COMPLETE")
        print("review stderr", file=sys.stderr)
        sys.exit(7)
    '''), encoding="utf-8")
    # The local terminal backend uses bash, including Git Bash on Windows.
    command = shlex.join(path.as_posix() for path in (Path(sys.executable), child, release))
    observed = []

    class Provider(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_error(404)

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "messages" not in request:
                self.send_error(404)
                return
            tool_results = [m for m in request["messages"] if m["role"] == "tool"]
            has_terminal = any(t.get("function", {}).get("name") == "terminal"
                               for t in request.get("tools", []))
            message = {"role": "assistant", "content": "Coordinator finished."}
            if has_terminal and not tool_results:
                message.update(content=None, tool_calls=[{
                    "id": "call_review", "type": "function", "function": {
                        "name": "terminal", "arguments": json.dumps({
                            "command": command, "background": True, "notify": True,
                        }),
                    },
                }])
            elif tool_results:
                observed.extend(json.loads(m["content"]) for m in tool_results)
                release.touch()
            response = {
                "id": "chatcmpl-local", "object": "chat.completion", "created": 1,
                "model": "test-model", "choices": [{
                    "index": 0, "message": message,
                    "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
                }], "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
            content_type = "application/json"
            if request.get("stream"):
                response["object"] = "chat.completion.chunk"
                response["choices"][0]["delta"] = response["choices"][0].pop("message")
                for index, tool in enumerate(message.get("tool_calls", [])):
                    tool["index"] = index
                raw = ("data: " + json.dumps(response) + "\n\ndata: [DONE]\n\n").encode()
                content_type = "text/event-stream"
            else:
                raw = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    env = {**os.environ, "HERMES_HOME": str(home), "HOME": str(tmp_path),
           "USERPROFILE": str(tmp_path), "TERMINAL_CWD": str(tmp_path),
           "OPENAI_BASE_URL": url, "OPENAI_API_KEY": "local-test-only",
           "PYTHONPATH": str(REPO_ROOT)}
    try:
        producer = subprocess.run([
            sys.executable, "-c",
            "import cli; cli.main(query='Run the background review', quiet=True, "
            "oneshot=True, provider='custom', model='test-model', api_key='local-test-only', "
            f"base_url={url!r}, toolsets='terminal', max_turns=3, ignore_rules=True)",
        ], cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, encoding="utf-8", timeout=60)
    finally:
        release.touch()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert producer.returncode == 0, producer.stdout + producer.stderr
    assert "Coordinator finished." in producer.stdout
    assert len(observed) == 1, (observed, producer.stdout, producer.stderr)
    process_id = observed[0]["session_id"]
    assert observed[0].get("notify_on_complete") is True, observed

    consumer = textwrap.dedent('''
        import json, sys
        import tools.process_registry as pr
        from tools.registry import registry
        result = registry.get_entry("process_manage").handler(
            {"action": "log", "session_id": sys.argv[1]})
        status = registry.get_entry("process_manage").handler(
            {"action": "poll", "session_id": sys.argv[1]})
        print(json.dumps({"result": json.loads(result), "status": json.loads(status),
                          "replayed": not pr.process_registry.completion_queue.empty()}))
    ''')
    def read_result(profile):
        result = subprocess.run([sys.executable, "-c", consumer, process_id],
                                cwd=tmp_path, env={**env, "HERMES_HOME": str(profile)},
                                check=True, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, encoding="utf-8", timeout=30)
        return json.loads(result.stdout)

    receipt = json.loads((home / "logs" / "process-results" / f"{process_id}.json").read_text(encoding="utf-8"))
    assert receipt["parent_session_id"]  # CLI owner must be stamped before its reader starts.
    env["HERMES_SESSION_ID"] = receipt["parent_session_id"]
    recovered = read_result(home)
    assert recovered["result"]["status"] == "exited", recovered
    assert recovered["status"]["exit_code"] == 7, recovered
    assert "SYNTHETIC_REVIEW_COMPLETE" in recovered["result"]["output"]
    assert "review stderr" in recovered["result"]["output"]
    assert recovered["replayed"] is False
    assert read_result(tmp_path / "other-profile")["result"]["status"] == "not_found"


def test_receipts_are_bounded_redacted_and_session_scoped(tmp_path, monkeypatch):
    import time
    from tools import process_registry_results as receipts
    from tools.process_registry import MAX_OUTPUT_CHARS, ProcessRegistry, ProcessSession

    from agent import redact
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    monkeypatch.setattr(receipts, "MAX_RETAINED_RESULTS", 2)
    secret = "sk-" + "aB2cD3eF4gH5iJ6kL7mN8pQ9rS0tU1vW2xY3zA4bC5dE6fG7"
    from gateway.session_context import scoped_current_session_id
    from tools.process_registry_results import load_completed_results
    monkeypatch.setenv("HERMES_SESSION_ID", "owner-session")
    sessions = []
    registry = ProcessRegistry()
    for index in range(3):
        session = ProcessSession(
            id=f"proc_{index:012x}", command=f"echo {secret}", task_id=f"owner-{index}",
            owner_task_id=f"owner-{index}", session_key=f"chat-{index}",
            parent_session_id="owner-session",
            started_at=time.time() - receipts.RESULT_RETENTION_SECONDS * 2,
            output_buffer="x" * MAX_OUTPUT_CHARS + "\n" + secret,
            exited=True, exit_code=index,
        )
        registry._running[session.id] = session
        registry._move_to_finished(session)
        sessions.append(session)
    from hermes_constants import get_hermes_home
    paths = list((get_hermes_home() / "logs" / "process-results").glob("*.json"))
    assert len(paths) == 2
    assert all(secret not in path.read_text(encoding="utf-8") for path in paths)
    fresh = ProcessRegistry()
    assert fresh.get(sessions[0].id) is None
    recovered = fresh.get(sessions[-1].id)
    assert recovered.owner_task_id == sessions[-1].owner_task_id
    assert len(recovered.output_buffer) <= MAX_OUTPUT_CHARS
    assert fresh.list_sessions() == []  # Status/liveness scans stay in memory.
    assert [s["session_id"] for s in fresh.list_sessions(
        task_id="owner-2", include_retained=True)] == [recovered.id]
    assert fresh.list_sessions(task_id="unrelated", session_key="unrelated", include_retained=True) == []
    assert fresh.get("proc_0000") is None  # Ambiguous across durable results.
    with scoped_current_session_id("unrelated-session"):
        assert load_completed_results(recovered.id) == {}
        assert fresh.get(recovered.id) is None
    from hermes_state import SessionDB
    db = SessionDB()
    try:
        db.create_session("owner-session", "cli")
        db.create_session("delegated-child", "subagent", parent_session_id="owner-session")
        with scoped_current_session_id("delegated-child"):
            assert fresh.get(recovered.id) is None
        db.end_session("owner-session", end_reason="compression")
        db.create_session("owner-tip", "cli", parent_session_id="owner-session")
        with scoped_current_session_id("owner-tip"):
            assert fresh.get(recovered.id).output_buffer == recovered.output_buffer
    finally:
        db.close()
    assert fresh.completion_queue.empty()
    for path in paths:
        expired = time.time() - receipts.RESULT_RETENTION_SECONDS - 1
        os.utime(path, (expired, expired))
    assert fresh.get(recovered.id) is None

    # Multiplex readers must keep the producer's profile on native threads.
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    profile = tmp_path / "thread-profile"
    token = set_hermes_home_override(profile)
    try:
        with scoped_current_session_id("thread-owner"):
            child = registry.spawn_local(
                shlex.join([Path(sys.executable).as_posix(), "-c", "print('SCOPED_RESULT')"]),
                cwd=str(tmp_path), task_id="thread-task")
            child._reader_thread.join(timeout=20)
            assert not child._reader_thread.is_alive()
            assert (profile / "logs" / "process-results" / f"{child.id}.json").exists()
            assert "SCOPED_RESULT" in ProcessRegistry().read_log(child.id)["output"]
    finally:
        reset_hermes_home_override(token)
    assert not (get_hermes_home() / "logs" / "process-results" / f"{child.id}.json").exists()


# ---------------------------------------------------------------------------
# Durable completion outbox (receipt schema 2)
#
# A finite API/Wizard run ends with its HTTP response, so a notify-on-complete
# worker it spawned reports to nobody. The receipt is promoted to an outbox row
# that a LATER API process claims and replays into the exact original session.
# The incident this reproduces: a /v1/runs turn answered "Executing now", the
# detached finisher failed after the run ended, and no completion ever arrived.
# ---------------------------------------------------------------------------

WIZARD_SESSION = "wizard-session-42"
RUN_SESSION_KEY = "run_5f5f5f"


def _api_session_fields(**overrides):
    """What terminal(background=true, notify=true) pre-binds on a resumable API session."""
    return {
        "notify_on_complete": True, "completion_delivery": "api_session_wake",
        "delivery_adapter_id": "api_server", "watcher_platform": "api_server",
        "watcher_chat_id": WIZARD_SESSION, **overrides,
    }


def _wake_adapter():
    from types import SimpleNamespace

    return SimpleNamespace(supports_async_delivery=False, _host="127.0.0.1", _port=1,
                           _api_key="k", _model_name="m")


def _await_exit(session, timeout=20.0):
    assert session._completion_event.wait(timeout), "background process never completed"


def test_api_completion_replays_once_into_the_exact_original_session(tmp_path, monkeypatch):
    """Real event order: armed spawn -> parent run ends -> the API process is recreated ->
    the terminal receipt wakes the ORIGINAL session, exactly once."""
    import asyncio

    import gateway.wake as wake_mod
    from gateway.platforms.api_server_completions import replay_pending_completions
    from tools.process_registry import ProcessRegistry, process_registry
    from tools.process_registry_outbox import completion_block, read_receipt, receipt_path

    session = process_registry.spawn_local(
        command="echo finisher-done; exit 3", cwd=str(tmp_path),
        session_key=RUN_SESSION_KEY, task_id="t-run", session_fields=_api_session_fields())
    _await_exit(session)

    # The parent run is gone and so is the process that served it; a fresh registry stands in
    # for the recreated API process, which knows nothing about this worker.
    recreated = ProcessRegistry()
    assert recreated.get(session.id) is None
    assert recreated.completion_queue.empty()

    posts = []

    async def fake_self_post(adapter, *, text, session_id, idempotency_key=""):
        posts.append((session_id, text, idempotency_key))

    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)
    adapter = _wake_adapter()
    assert asyncio.run(replay_pending_completions(adapter)) == 1
    # A second consumer pass (the loop ticks every few seconds) must not wake the session again.
    assert asyncio.run(replay_pending_completions(adapter)) == 0

    assert len(posts) == 1
    target, text, key = posts[0]
    assert target == WIZARD_SESSION, "the wake must resume the Wizard session, not the run id"
    assert RUN_SESSION_KEY not in target
    assert "finisher-done" in text and "exit code 3" in text
    assert session.id in key and key == key.strip()
    assert completion_block(read_receipt(receipt_path(session.id)))["state"] == "delivered"


def test_route_metadata_is_in_the_first_checkpoint_and_receipt(tmp_path, monkeypatch):
    """An instantly-exiting child must not win the race against its own bookkeeping: the route
    is on disk (checkpoint + armed receipt) before the process can produce an exit."""
    import json as _json

    from tools.process_registry import _checkpoint_path, process_registry
    from tools.process_registry_outbox import completion_block, read_receipt, receipt_path

    running = process_registry.spawn_local(
        command="sleep 30", cwd=str(tmp_path), session_key=RUN_SESSION_KEY,
        session_fields=_api_session_fields())
    try:
        entry = next(e for e in _json.loads(_checkpoint_path().read_text(encoding="utf-8"))
                     if e["session_id"] == running.id)
        assert entry["completion_delivery"] == "api_session_wake"
        assert entry["delivery_adapter_id"] == "api_server"
        assert entry["watcher_chat_id"] == WIZARD_SESSION
        armed = completion_block(read_receipt(receipt_path(running.id)))
        assert armed["state"] == "armed" and armed["origin_session_id"] == WIZARD_SESSION
    finally:
        process_registry.kill_process(running.id, consume_output=False)

    instant = process_registry.spawn_local(
        command="exit 0", cwd=str(tmp_path), session_key=RUN_SESSION_KEY,
        session_fields=_api_session_fields())
    _await_exit(instant)
    block = completion_block(read_receipt(receipt_path(instant.id)))
    assert block["state"] == "pending"
    assert block["origin_session_id"] == WIZARD_SESSION
    assert block["event"]["origin_session_id"] == WIZARD_SESSION


def test_claim_is_exclusive_releasable_and_survives_pruning(tmp_path, monkeypatch):
    """One consumer at a time; a failed delivery is retryable; and a receipt that still owes a
    delivery is immune to the settled-result pruning that would otherwise erase it."""
    from tools import process_registry_results as receipts
    from tools.process_registry import process_registry
    from tools.process_registry_outbox import (
        claim, completion_block, new_claim_id, read_receipt, receipt_path, release)

    session = process_registry.spawn_local(
        command="exit 0", cwd=str(tmp_path), session_key=RUN_SESSION_KEY,
        session_fields=_api_session_fields())
    _await_exit(session)

    first = new_claim_id("api_server")
    assert claim(session.id, first, adapter_id="api_server") is not None
    assert claim(session.id, new_claim_id("api_server"), adapter_id="api_server") is None
    # A different adapter never claims another surface's completion.
    assert claim(session.id, new_claim_id("other"), adapter_id="other") is None

    # Pruning must not erase an unsettled completion under the retained-results count cap —
    # 64 finished processes per profile is ordinary pressure, losing the delivery is not.
    monkeypatch.setattr(receipts, "MAX_RETAINED_RESULTS", 0)
    path = receipt_path(session.id)
    receipts.retained_result_paths()
    assert path.exists(), "a claimed completion must survive settled-result pruning"

    assert release(session.id, first) is True
    assert completion_block(read_receipt(path))["state"] == "pending"
    retry = new_claim_id("api_server")
    claimed = claim(session.id, retry, adapter_id="api_server")
    assert claimed is not None
    assert completion_block(claimed)["attempts"] == 2

    # Retrying is not retrying forever: the attempt budget converges to a terminal drop even
    # when the consumer keeps dying mid-delivery and never releases its claim.
    from tools.process_registry_outbox import MAX_DELIVERY_ATTEMPTS
    for _ in range(MAX_DELIVERY_ATTEMPTS + 2):
        release(session.id, retry)
        retry = new_claim_id("api_server")
        claim(session.id, retry, adapter_id="api_server")
    assert completion_block(read_receipt(path))["state"] == "dropped"
    assert claim(session.id, new_claim_id("api_server"), adapter_id="api_server") is None


def test_consumed_and_legacy_receipts_never_replay(tmp_path, monkeypatch):
    """Two silences that are correct: a result the owning turn already read, and a receipt
    written before the outbox existed (schema 1 carries no delivery contract)."""
    import asyncio

    import gateway.wake as wake_mod
    from gateway.platforms.api_server_completions import replay_pending_completions
    from tools.process_registry import process_registry
    from tools.process_registry_outbox import completion_block, read_receipt, receipt_path, write_receipt

    session = process_registry.spawn_local(
        command="echo already-read", cwd=str(tmp_path), session_key=RUN_SESSION_KEY,
        session_fields=_api_session_fields())
    _await_exit(session)
    # The agent read the outcome inside its own turn.
    process_registry.wait(session.id, timeout=10)
    assert completion_block(read_receipt(receipt_path(session.id)))["state"] == "consumed"

    legacy = {"id": "proc_legacyaaaaaa", "command": "echo old", "cwd": str(tmp_path),
              "task_id": "t", "owner_task_id": "t", "session_key": RUN_SESSION_KEY,
              "parent_session_id": WIZARD_SESSION, "started_at": time.time(), "exit_code": 0,
              "completion_reason": "exited", "termination_source": "",
              "notify_on_complete": True, "output": "old output"}
    write_receipt(legacy)

    posts = []

    async def fake_self_post(adapter, *, text, session_id, idempotency_key=""):
        posts.append(session_id)

    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)
    assert asyncio.run(replay_pending_completions(_wake_adapter())) == 0
    assert posts == []


def test_dead_armed_worker_recovers_as_a_pending_unknown_outcome(tmp_path, monkeypatch):
    """A worker that died with its gateway still owes its session an answer: recovery turns the
    armed receipt into a pending completion with an honestly unknown outcome."""
    import json as _json

    from tools.process_registry import ProcessRegistry, process_registry
    from tools.process_registry_outbox import completion_block, read_receipt, receipt_path

    session = process_registry.spawn_local(
        command="sleep 30", cwd=str(tmp_path), session_key=RUN_SESSION_KEY,
        session_fields=_api_session_fields())
    from tools.process_registry import _checkpoint_path
    entries = _json.loads(_checkpoint_path().read_text(encoding="utf-8"))
    process_registry.kill_process(session.id, consume_output=False)
    # Rewrite the checkpoint as it looked while the worker was alive, but with a PID that is
    # provably not ours any more (the process died with the old gateway).
    for entry in entries:
        if entry["session_id"] == session.id:
            entry["host_start_time"] = (entry.get("host_start_time") or 1) + 999_999
    _checkpoint_path().write_text(_json.dumps(entries), encoding="utf-8")
    # Re-arm: the kill above already settled the receipt for the live process.
    armed = read_receipt(receipt_path(session.id))
    armed["completion"]["state"] = "armed"
    armed["exited"] = False
    from tools.process_registry_outbox import write_receipt
    write_receipt(armed)

    assert ProcessRegistry().recover_from_checkpoint() == 0
    block = completion_block(read_receipt(receipt_path(session.id)))
    assert block["state"] == "pending"
    assert block["event"]["exit_code"] is None
    assert block["event"]["completion_reason"] == "lost"
    assert block["origin_session_id"] == WIZARD_SESSION
