"""Completed work remains retrievable when its finite CLI owner exits."""

import contextlib
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

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _pristine_session_identity(monkeypatch):
    """Start every test here with NO bound session identity.

    Receipt ownership, completion routing and the outbox's resumable destination are all read
    from session state, and that state has two leak paths into this file. Gateway tests bind the
    ``gateway.session_context`` ContextVars and do not always unbind them (``set_session_vars``
    without a matching ``clear_session_vars``), and ``get_session_env`` falls back to the real
    ``HERMES_SESSION_*`` environment — which is populated for anyone running pytest from inside a
    Hermes session. Either one silently re-parents these receipts onto a session that is not the
    one under test, so a fresh registry "recovers" a worker it should never have owned.
    """
    import gateway.session_context as sc

    for name in sc._VAR_MAP:
        monkeypatch.delenv(name, raising=False)
    sc.reset_session_vars()
    yield
    sc.reset_session_vars()


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

    async def fake_self_post(adapter, *, text, session_id, idempotency_key="", profile="",
                             api_key=""):
        posts.append((session_id, text, idempotency_key, profile))

    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)
    adapter = _wake_adapter()
    assert asyncio.run(replay_pending_completions(adapter)) == 1
    # A second consumer pass (the loop ticks every few seconds) must not wake the session again.
    assert asyncio.run(replay_pending_completions(adapter)) == 0

    assert len(posts) == 1
    target, text, key, profile = posts[0]
    assert profile == "", "the default profile is served on the unprefixed route"
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

    async def fake_self_post(adapter, *, text, session_id, idempotency_key="", profile="",
                             api_key=""):
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


# ---------------------------------------------------------------------------
# Crash, contention and pruning: the outbox has to survive its consumers.
#
# Every test below drives the real cross-process receipt lock. The ones that
# need a genuinely dead claim holder spawn a real interpreter and kill it with
# os._exit() rather than simulating the state on disk — a consumer that dies
# mid-delivery is the case the TTL and the attempt budget exist for, and the
# only way to know they are reachable is to produce one.
# ---------------------------------------------------------------------------

_OUTBOX_CHILD = textwrap.dedent('''
    import os, sys
    from tools import process_registry_outbox as outbox
    ttl = os.environ.get("HERMES_TEST_CLAIM_TTL")
    if ttl:  # age every prior claim: this child stands in for a LATER gateway process
        outbox.CLAIM_TTL_SECONDS = float(ttl)
    got = outbox.claim(sys.argv[1], outbox.new_claim_id("api_server"), adapter_id="api_server")
    sys.stdout.write("claimed" if got else "missed")
    sys.stdout.flush()
    os._exit(0)  # die holding it: no release, no settle, no atexit
''')


def _child_env(**extra):
    return {**os.environ, "PYTHONPATH": str(REPO_ROOT), **extra}


def _spawn_durable(tmp_path, command="exit 0", **fields):
    from tools.process_registry import process_registry

    session = process_registry.spawn_local(
        command=command, cwd=str(tmp_path), session_key=RUN_SESSION_KEY,
        session_fields=_api_session_fields(**fields))
    _await_exit(session)
    return session


def test_a_consumer_that_crashes_mid_delivery_is_recovered_after_the_claim_ttl(tmp_path, monkeypatch):
    """A claim holder that dies without releasing must not wedge the completion forever.

    Before this, ``pending_completions`` only returned armed/pending rows, so a CLAIMED receipt
    was never looked at again: the claim TTL and the attempt budget were both unreachable and the
    Wizard session waited on a delivery no process would ever retry.
    """
    import asyncio

    import gateway.wake as wake_mod
    from gateway.platforms.api_server_completions import replay_pending_completions
    from tools import process_registry_outbox as outbox

    session = _spawn_durable(tmp_path, "echo crashed-mid-delivery")
    child = subprocess.run([sys.executable, "-c", _OUTBOX_CHILD, session.id], cwd=str(tmp_path),
                           env=_child_env(), capture_output=True, text=True, timeout=60)
    assert child.stdout == "claimed", (child.stdout, child.stderr)
    claimed = outbox.completion_block(outbox.read_receipt(outbox.receipt_path(session.id)))
    assert claimed["state"] == "claimed" and claimed["attempts"] == 1

    # A FRESH claim is still refused: exclusivity is not weakened by making stale claims visible.
    assert outbox.pending_completions("api_server") == []
    assert outbox.claim(session.id, outbox.new_claim_id("api_server"), adapter_id="api_server") is None

    posts = []

    async def fake_self_post(adapter, *, text, session_id, idempotency_key="", profile="",
                             api_key=""):
        posts.append(session_id)

    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)
    monkeypatch.setattr(outbox, "CLAIM_TTL_SECONDS", 0.0)  # the crash was 5 minutes ago
    assert [r["id"] for r in outbox.pending_completions("api_server")] == [session.id]
    assert asyncio.run(replay_pending_completions(_wake_adapter())) == 1
    assert posts == [WIZARD_SESSION]
    block = outbox.completion_block(outbox.read_receipt(outbox.receipt_path(session.id)))
    assert block["state"] == "delivered"
    assert block["attempts"] == 2, "the crashed attempt is counted, so the budget can run out"


def test_the_attempt_budget_terminates_a_completion_no_consumer_survives(tmp_path, monkeypatch):
    """Repeated crash-without-release converges to an observable terminal drop, not a loop."""
    from tools import process_registry_outbox as outbox

    session = _spawn_durable(tmp_path)
    monkeypatch.setattr(outbox, "CLAIM_TTL_SECONDS", 0.0)
    for _ in range(outbox.MAX_DELIVERY_ATTEMPTS + 1):
        subprocess.run([sys.executable, "-c", _OUTBOX_CHILD, session.id], cwd=str(tmp_path),
                       env=_child_env(HERMES_TEST_CLAIM_TTL="0"), capture_output=True, text=True,
                       timeout=60)
    block = outbox.completion_block(outbox.read_receipt(outbox.receipt_path(session.id)))
    assert block["state"] == "dropped"
    assert outbox.pending_completions("api_server") == []


def test_only_one_of_many_concurrent_processes_claims_a_completion(tmp_path):
    """Exclusivity is a cross-PROCESS property; two gateways must not both wake the session."""
    from tools import process_registry_outbox as outbox

    session = _spawn_durable(tmp_path)
    children = [
        subprocess.Popen([sys.executable, "-c", _OUTBOX_CHILD, session.id], cwd=str(tmp_path),
                         env=_child_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True)
        for _ in range(6)
    ]
    outcomes = []
    for child in children:
        stdout, stderr = child.communicate(timeout=90)
        outcomes.append(stdout or stderr)
    assert outcomes.count("claimed") == 1, outcomes
    block = outbox.completion_block(outbox.read_receipt(outbox.receipt_path(session.id)))
    assert block["state"] == "claimed" and block["attempts"] == 1


def test_a_wake_accepted_before_a_restart_never_runs_a_second_turn(tmp_path, monkeypatch):
    """Restart-stable duplicate suppression.

    The API server's ``_IdempotencyCache`` is in-memory with a 5-minute TTL, so the sequence
    "wake accepted -> gateway restarts -> claim goes stale -> wake re-posted" used to run the
    user's turn twice. The gate is now the receipt itself, which outlives the process.
    """
    import asyncio

    import gateway.wake as wake_mod
    from gateway.platforms.api_server_completions import replay_pending_completions
    from tools import process_registry_outbox as outbox

    session = _spawn_durable(tmp_path, "echo delivered-once")
    decisions = []

    async def fake_self_post(adapter, *, text, session_id, idempotency_key="", profile="",
                             api_key=""):
        # Stand in for the receiving API turn: it reserves BEFORE running the agent.
        decisions.append(outbox.reserve_wake_delivery(idempotency_key))

    monkeypatch.setattr(wake_mod, "_self_post_chat_completion", fake_self_post)
    # The delivering process died between "202 accepted" and settling its claim.
    real_settle = outbox.settle
    monkeypatch.setattr(outbox, "settle", lambda *a, **k: False)
    assert asyncio.run(replay_pending_completions(_wake_adapter())) == 0
    assert decisions == [outbox.WAKE_RUN]
    assert outbox.completion_block(
        outbox.read_receipt(outbox.receipt_path(session.id)))["state"] == "claimed"

    # A new process takes the stale claim over and re-posts the same key. The turn must NOT run.
    # (Restored by hand, not monkeypatch.undo(): undo() would also revert the session-scoped
    # HERMES_HOME redirect this whole suite is isolated by.)
    monkeypatch.setattr(outbox, "settle", real_settle)
    monkeypatch.setattr(outbox, "CLAIM_TTL_SECONDS", 0.0)
    assert asyncio.run(replay_pending_completions(_wake_adapter())) == 1
    assert decisions == [outbox.WAKE_RUN, outbox.WAKE_REPLAY]
    assert outbox.completion_block(
        outbox.read_receipt(outbox.receipt_path(session.id)))["state"] == "delivered"


def test_reserve_wake_delivery_ignores_foreign_keys_and_mismatched_incarnations(tmp_path):
    """The durable gate governs completion wakes only, and only the exact incarnation it names."""
    from tools import process_registry_outbox as outbox

    session = _spawn_durable(tmp_path)
    record = outbox.read_receipt(outbox.receipt_path(session.id))
    key = outbox.idempotency_key(record)

    assert outbox.reserve_wake_delivery("customer-supplied-key") == outbox.WAKE_RUN
    assert outbox.reserve_wake_delivery(f"{outbox.WAKE_KEY_PREFIX}proc_missing-123") == outbox.WAKE_RUN
    # Same process id, different spawn epoch: a recycled id is a distinct job, not a duplicate.
    stale = outbox.idempotency_key({**record, "started_at": record["started_at"] + 1000})
    assert outbox.reserve_wake_delivery(stale) == outbox.WAKE_RUN
    assert outbox.reserve_wake_delivery(key) == outbox.WAKE_RUN
    assert outbox.reserve_wake_delivery(key) == outbox.WAKE_REPLAY


def test_consume_never_steals_a_live_delivery_claim(tmp_path):
    """wait/log/kill inside the owning turn must not race a delivery that is already in flight.

    Marking a CLAIMED receipt consumed used to succeed, after which the delivering consumer's
    ``settle()`` silently failed — a wake the session had already answered, with no record that
    it happened. The intent is recorded instead, and whoever holds the claim converges it.
    """
    from tools import process_registry_outbox as outbox

    session = _spawn_durable(tmp_path)
    claim_id = outbox.new_claim_id("api_server")
    assert outbox.claim(session.id, claim_id, adapter_id="api_server") is not None

    assert outbox.consume(session.id) is False, "a claimed delivery is not ours to settle"
    block = outbox.completion_block(outbox.read_receipt(outbox.receipt_path(session.id)))
    assert block["state"] == "claimed" and block["consume_requested"] is True
    # Nobody else may start a competing delivery for a completion the turn has already read.
    assert outbox.claim(session.id, outbox.new_claim_id("api_server"), adapter_id="api_server") is None
    assert outbox.completion_block(
        outbox.read_receipt(outbox.receipt_path(session.id)))["state"] == "consumed"
    # The original holder's settle now honestly reports that the receipt is no longer its own.
    assert outbox.settle(session.id, claim_id, "delivered") is False


def test_release_after_a_consume_request_settles_consumed_not_pending(tmp_path):
    """The other order: the delivery FAILED while the turn read the result itself."""
    from tools import process_registry_outbox as outbox

    session = _spawn_durable(tmp_path)
    claim_id = outbox.new_claim_id("api_server")
    assert outbox.claim(session.id, claim_id, adapter_id="api_server") is not None
    assert outbox.consume(session.id) is False
    assert outbox.release(session.id, claim_id) is True
    assert outbox.completion_block(
        outbox.read_receipt(outbox.receipt_path(session.id)))["state"] == "consumed"


def test_wait_settles_an_unclaimed_completion_consumed(tmp_path):
    """The ordinary path still converges terminally, so no replay repeats a read result."""
    from tools import process_registry_outbox as outbox
    from tools.process_registry import process_registry

    session = _spawn_durable(tmp_path, "echo read-in-turn")
    process_registry.wait(session.id, timeout=10)
    assert outbox.completion_block(
        outbox.read_receipt(outbox.receipt_path(session.id)))["state"] == "consumed"
    assert outbox.pending_completions("api_server") == []


def test_pruning_never_drops_an_unresolved_obligation(tmp_path, monkeypatch):
    """Age-pruning an armed/pending/claimed receipt is the silent loss the outbox prevents.

    Seven days is shorter than "forever", and the completion's whole point is that nobody is
    watching it — so the old ``mtime < cutoff -> unlink`` deleted exactly the rows that still
    owed a delivery. Only a terminal receipt is collectable.
    """
    from tools import process_registry_results as receipts
    from tools import process_registry_outbox as outbox

    session = _spawn_durable(tmp_path, "echo owed")
    path = outbox.receipt_path(session.id)
    lock = path.with_suffix(".lock")
    expired = time.time() - receipts.RESULT_RETENTION_SECONDS - 1
    os.utime(path, (expired, expired))

    assert path in receipts.retained_result_paths()
    assert path.exists(), "a pending completion outlives the retention window"
    assert outbox.pending_completions("api_server"), "and stays replayable"

    # The count cap is not a loophole either.
    monkeypatch.setattr(receipts, "MAX_RETAINED_RESULTS", 0)
    receipts.retained_result_paths()
    assert path.exists()

    # Terminal only after the attempt budget ran out, with the warning that names it.
    claim_id = outbox.new_claim_id("api_server")
    outbox.claim(session.id, claim_id, adapter_id="api_server")
    outbox.settle(session.id, claim_id, "dropped")
    os.utime(path, (expired, expired))
    receipts.retained_result_paths()
    assert not path.exists() and not lock.exists(), "a settled receipt takes its lock with it"


def test_receipt_locks_stay_exclusive_while_the_pruner_collects_lock_files(tmp_path):
    """Mutual exclusion must survive concurrent pruning of the lock file itself.

    POSIX locks live on the INODE. Unlinking ``<id>.lock`` and letting the next caller create a
    fresh one at the same path produces two lock inodes and therefore two simultaneous
    "holders" — which for this outbox means two gateways delivering one completion twice. Two
    rules make that unreachable and this test drives both: the pruner unlinks only from inside
    the lock (``drop_receipt``), and an acquirer that locked an inode the path no longer names
    re-opens instead of believing it holds the lock.
    """
    from tools.process_registry_outbox import (
        _lock_path, _locked_the_live_inode, _try_lock, receipt_lock, receipt_path, results_dir)

    results_dir().mkdir(mode=0o700, parents=True, exist_ok=True)
    session_id = "proc_lockraceaaa"
    path = receipt_path(session_id)

    # The split, constructed exactly: a caller opens the lock file, the holder of that lock
    # prunes it, and only THEN does the caller flock its (now detached) handle. The flock
    # succeeds — nothing else can see that inode — so without the re-check the caller would
    # believe it holds a lock that excludes nobody.
    if sys.platform != "win32":
        detached = open(_lock_path(session_id), "a+b")
        try:
            _lock_path(session_id).unlink()
            _try_lock(detached)
            assert _locked_the_live_inode(_lock_path(session_id), detached) is False
            with receipt_lock(session_id, timeout=1.0) as held:
                assert held is True, "the live inode must still be lockable by someone else"
        finally:
            detached.close()
    inside, overlaps, errors = [], [], []
    stop = threading.Event()

    def holder():
        try:
            for _ in range(200):
                with receipt_lock(session_id, timeout=5.0) as held:
                    if not held:
                        continue
                    inside.append(1)
                    if len(inside) > 1:
                        overlaps.append(len(inside))
                    time.sleep(0.0005)
                    inside.pop()
        except Exception as exc:  # surfaced below rather than dying silently in a thread
            errors.append(exc)

    def pruner():
        from tools.process_registry_outbox import drop_receipt

        while not stop.is_set():
            # A settled receipt reappearing is exactly the churn a busy profile produces.
            path.write_text('{"id": "%s", "schema": 2}' % session_id, encoding="utf-8")
            drop_receipt(path)
            time.sleep(0.0005)

    holders = [threading.Thread(target=holder) for _ in range(3)]
    pruning = threading.Thread(target=pruner, daemon=True)
    pruning.start()
    for thread in holders:
        thread.start()
    for thread in holders:
        thread.join(timeout=180)
        assert not thread.is_alive()
    stop.set()
    pruning.join(timeout=10)
    assert errors == []
    assert overlaps == [], f"two callers held the same receipt lock at once: {overlaps}"


def test_drop_receipt_refuses_an_obligation_that_reappears_under_the_lock(tmp_path):
    """The unsettled check is re-read while holding the lock, not trusted from the scan."""
    from tools import process_registry_outbox as outbox

    session = _spawn_durable(tmp_path, "echo still-owed")
    path = outbox.receipt_path(session.id)
    assert outbox.drop_receipt(path) is False
    assert path.exists() and path.with_suffix(".lock").exists()


# ---------------------------------------------------------------------------
# Multiplex profiles: every profile's outbox, on its own route, with its own key
#
# Receipts are profile-LOCAL — they live under the spawning profile's
# HERMES_HOME — and one adapter serves /v1/... plus every /p/<profile>/v1/...
# mirror. Draining only the default home therefore stranded every named
# profile's completions, and waking one on the bare /v1/ route would resolve its
# session id against the OWNER's home.
# ---------------------------------------------------------------------------

NAMED_PROFILE = "wiz"
NAMED_SESSION = "named-wizard-session-7"


class _RecordingAPI(http.server.BaseHTTPRequestHandler):
    """Stands in for the in-pod API server the wake self-posts to."""

    requests = []

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).requests.append({
            "path": self.path,
            "authorization": self.headers.get("Authorization", ""),
            "session_id": self.headers.get("X-Hermes-Session-Id", ""),
            "idempotency_key": self.headers.get("Idempotency-Key", ""),
        })
        body = json.dumps({"id": "chatcmpl-wake", "object": "chat.completion", "created": 1,
                           "model": "m", "choices": [{"index": 0, "finish_reason": "stop",
                                                      "message": {"role": "assistant",
                                                                  "content": "ack"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _multiplex_adapter(port):
    """An adapter with the real multiplex hooks: per-profile scope + per-profile key."""
    import contextlib as _contextlib
    from types import SimpleNamespace

    from gateway.platforms.api_server import _api_request_profile

    def expected_api_key():
        profile = _api_request_profile.get()
        if not profile or profile == "default":
            return "default-key"
        return f"{profile}-key" if profile == NAMED_PROFILE else ""

    return SimpleNamespace(
        supports_async_delivery=False, _host="127.0.0.1", _port=port, _api_key="default-key",
        _model_name="m", _expected_api_key=expected_api_key,
        _profile_scope=lambda profile: _contextlib.nullcontext(),
        gateway_runner=SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True)))


def test_each_profiles_outbox_replays_on_its_own_route_with_its_own_credential(
        tmp_path, monkeypatch):
    """Default and named profile: both delivered, neither under the other's identity."""
    import asyncio

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from gateway.platforms.api_server_completions import replay_pending_completions
    from hermes_cli import profiles as profiles_mod
    from tools import process_registry_outbox as outbox
    from tools.process_registry import process_registry

    default_home = outbox.results_dir().parent.parent
    named_home = tmp_path / "profiles" / NAMED_PROFILE
    named_home.mkdir(parents=True)

    default_session = _spawn_durable(tmp_path, "echo default-work")
    token = set_hermes_home_override(str(named_home))
    try:
        named_session = process_registry.spawn_local(
            command="echo named-work", cwd=str(tmp_path), session_key=RUN_SESSION_KEY,
            session_fields=_api_session_fields(
                delivery_profile=NAMED_PROFILE, watcher_chat_id=NAMED_SESSION))
        _await_exit(named_session)
        # Profile identity is on the record, not inferred from where the scan happened.
        named_block = outbox.completion_block(
            outbox.read_receipt(outbox.receipt_path(named_session.id)))
        assert named_block["profile"] == NAMED_PROFILE
        assert named_block["origin_session_id"] == NAMED_SESSION
        # Isolation: the named receipt is invisible to the default profile's drain.
        assert outbox.pending_completions("api_server", profile="") == []
    finally:
        reset_hermes_home_override(token)
    assert [r["id"] for r in outbox.pending_completions("api_server")] == [default_session.id]

    monkeypatch.setattr(profiles_mod, "profiles_to_serve",
                        lambda multiplex: [("default", default_home), (NAMED_PROFILE, named_home)])
    _RecordingAPI.requests = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RecordingAPI)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert asyncio.run(replay_pending_completions(_multiplex_adapter(server.server_port))) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    by_session = {r["session_id"]: r for r in _RecordingAPI.requests}
    assert set(by_session) == {WIZARD_SESSION, NAMED_SESSION}
    assert by_session[WIZARD_SESSION]["path"] == "/v1/chat/completions"
    assert by_session[WIZARD_SESSION]["authorization"] == "Bearer default-key"
    assert by_session[NAMED_SESSION]["path"] == f"/p/{NAMED_PROFILE}/v1/chat/completions"
    assert by_session[NAMED_SESSION]["authorization"] == f"Bearer {NAMED_PROFILE}-key"
    assert all(r["idempotency_key"].startswith(outbox.WAKE_KEY_PREFIX)
               for r in _RecordingAPI.requests)

    # No credential is ever written into a durable record.
    for home, sid in ((default_home, default_session.id), (named_home, named_session.id)):
        text = (home / "logs" / "process-results" / f"{sid}.json").read_text(encoding="utf-8")
        assert "default-key" not in text and f"{NAMED_PROFILE}-key" not in text


def test_a_named_profile_without_its_own_key_is_held_not_delivered(tmp_path, monkeypatch):
    """A profile whose API_SERVER_KEY is unresolvable must never borrow the owner's.

    Waking it on the default route would authenticate as the owner and resolve the session id
    against the owner's home — a different conversation, or a brand-new one nobody reads.
    """
    import asyncio

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from gateway.platforms.api_server_completions import replay_pending_completions
    from hermes_cli import profiles as profiles_mod
    from tools import process_registry_outbox as outbox
    from tools.process_registry import process_registry

    default_home = outbox.results_dir().parent.parent
    keyless_home = tmp_path / "profiles" / "keyless"
    keyless_home.mkdir(parents=True)
    token = set_hermes_home_override(str(keyless_home))
    try:
        session = process_registry.spawn_local(
            command="echo keyless", cwd=str(tmp_path), session_key=RUN_SESSION_KEY,
            session_fields=_api_session_fields(delivery_profile="keyless"))
        _await_exit(session)
    finally:
        reset_hermes_home_override(token)

    monkeypatch.setattr(profiles_mod, "profiles_to_serve",
                        lambda multiplex: [("default", default_home), ("keyless", keyless_home)])
    _RecordingAPI.requests = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RecordingAPI)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert asyncio.run(replay_pending_completions(_multiplex_adapter(server.server_port))) == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert _RecordingAPI.requests == []

    token = set_hermes_home_override(str(keyless_home))
    try:
        # Held, not lost: it is still pending and still costs no delivery attempt.
        block = outbox.completion_block(outbox.read_receipt(outbox.receipt_path(session.id)))
        assert block["state"] == "pending" and block["attempts"] == 0
    finally:
        reset_hermes_home_override(token)


# ---------------------------------------------------------------------------
# Arming is a precondition, not a best effort
# ---------------------------------------------------------------------------


def test_a_spawn_whose_obligation_cannot_be_written_leaves_no_process_and_no_receipt(
        tmp_path, monkeypatch):
    """Fail closed after the spawn: terminate the child and fail the tool call.

    Suppressing the arming failure reported a PID and ``notify_on_complete: true`` to the model
    while nothing on disk could ever deliver the completion — and on a finite API run there is no
    later turn in which to discover that.
    """
    from tools import process_registry_outbox as outbox
    from tools.process_registry import ProcessRegistry, process_registry

    reaped = []
    real_reap = ProcessRegistry._reap_untracked

    def spy_reap(self, session, proc):
        reaped.append(proc.pid)
        return real_reap(self, session, proc)

    monkeypatch.setattr(ProcessRegistry, "_reap_untracked", spy_reap)

    @contextlib.contextmanager
    def wedged_lock(session_id, *, timeout=5.0):
        yield False  # every acquisition fails, as on a read-only or contended home

    monkeypatch.setattr(outbox, "receipt_lock", wedged_lock)
    before = set(process_registry._running)
    with pytest.raises(outbox.DurableArmingError):
        process_registry.spawn_local(
            command="sleep 120", cwd=str(tmp_path), session_key=RUN_SESSION_KEY,
            session_fields=_api_session_fields())

    assert set(process_registry._running) == before, "no process entry survives a failed arming"
    assert len(reaped) == 1, "the already-spawned child must be reaped, not orphaned"
    with pytest.raises(ProcessLookupError):
        os.kill(reaped[0], 0)
    assert list(outbox.results_dir().glob("proc_*.json")) == [], "and no success receipt exists"


def test_the_tool_reports_a_failed_arming_as_a_failed_spawn(tmp_path, monkeypatch):
    """What the model sees: exit_code -1, no session_id, no pid, and the corrective shape."""
    from tools import terminal_tool_background as bg

    def raise_arming(*args, **kwargs):
        from tools.process_registry_outbox import DurableArmingError

        raise DurableArmingError("receipt lock busy")

    monkeypatch.setattr(bg, "_spawn", raise_arming)
    monkeypatch.setattr(bg, "_resolve_completion_support",
                        lambda notify, watch, data: (_api_session_fields(), None))
    result = json.loads(bg.spawn_background_process(
        env=None, env_type="local", command="./deploy.sh", cwd=str(tmp_path),
        effective_task_id="t", task_id="t", session_key=RUN_SESSION_KEY, effective_pty=False,
        workdir=None, notify_on_complete=True, watch_patterns=None, approval_note=None,
        pty_disabled_reason=None))
    assert result["exit_code"] == -1
    assert "session_id" not in result and "pid" not in result
    assert "background=false" in result["error"]
