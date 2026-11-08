from gateway.run import GatewayRunner


def test_runtime_authority_blocks_gateway_self_restart(monkeypatch):
    monkeypatch.setenv("WIZARD_RUNTIME_AUTHORITY_REQUIRED", "1")
    runner = object.__new__(GatewayRunner)
    runner._restart_task_started = False

    assert runner.request_restart(detached=True, via_service=True) is False
    assert getattr(runner, "_restart_requested", False) is False
    assert getattr(runner, "_restart_task", None) is None
