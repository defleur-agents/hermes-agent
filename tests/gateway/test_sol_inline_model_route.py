"""Regression coverage for same-message Sol routing in the gateway."""

from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


@pytest.mark.parametrize(
    "text",
    [
        "Can you use your beefiest model for this? Design the architecture.",
        "Please use Sol to evaluate this contract.",
        "Sol: audit this plan.",
        "beefiest model, solve this one.",
        "Sol",
    ],
)
def test_sol_model_requested_recognizes_explicit_directives(text):
    assert gateway_run._sol_model_requested(text)


@pytest.mark.parametrize(
    "text",
    [
        "Sol is the name of a character in this draft.",
        "Should we use a beefiest model later?",
        "Compare Terra and Sol.",
        "The solar model is interesting.",
    ],
)
def test_sol_model_requested_ignores_non_directive_mentions(text):
    assert not gateway_run._sol_model_requested(text)


@pytest.mark.asyncio
async def test_sol_route_switches_session_only_and_preserves_original_prompt():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._handle_model_command = AsyncMock(return_value=None)
    event = MessageEvent(
        text="Use Sol for this: review the code.",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm"),
    )

    result = await runner._maybe_route_sol_request(event)

    assert result is None
    assert event.text == "Use Sol for this: review the code."
    runner._handle_model_command.assert_awaited_once_with(
        event,
        suppress_response=True,
        skip_cost_confirmation=True,
    )


@pytest.mark.asyncio
async def test_sol_route_returns_switch_failure_without_calling_an_agent():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._handle_model_command = AsyncMock(return_value="Model switch failed")
    event = MessageEvent(
        text="Sol: handle this.",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm"),
    )

    result = await runner._maybe_route_sol_request(event)

    assert result == "Model switch failed"
    assert event.text == "Sol: handle this."


@pytest.mark.asyncio
async def test_suppressed_session_switch_leaves_original_prompt_for_sol():
    from hermes_cli.model_switch import ModelSwitchResult

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    event = MessageEvent(
        text="/model gpt-5.6-sol --provider openai-codex --session",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm"),
    )

    result = ModelSwitchResult(
        success=True,
        new_model="gpt-5.6-sol",
        target_provider="openai-codex",
        provider_changed=False,
        api_key="test-key",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        provider_label="OpenAI Codex",
        is_global=False,
    )
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {
            "model": {"default": "gpt-5.6-terra", "provider": "openai-codex"}
        })
        monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_: result)
        switch_response = await runner._handle_model_command(
            event,
            suppress_response=True,
            skip_cost_confirmation=True,
        )
    finally:
        monkeypatch.undo()

    assert switch_response is None
    session_key = runner._session_key_for_source(event.source)
    assert runner._session_model_overrides[session_key]["model"] == "gpt-5.6-sol"
    assert runner._session_model_overrides[session_key]["provider"] == "openai-codex"
