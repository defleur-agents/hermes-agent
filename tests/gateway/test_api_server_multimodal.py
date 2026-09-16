"""End-to-end tests for inline image inputs on /v1/chat/completions and /v1/responses.

Covers the multimodal normalization path added to the API server.  Unlike the
adapter-level tests that patch ``_run_agent``, these tests patch
``AIAgent.run_conversation`` instead so the adapter's full request-handling
path (including the ``run_agent`` prologue that used to crash on list content)
executes against a real aiohttp app.
"""

import asyncio
import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _content_has_visible_payload,
    _normalize_multimodal_content,
    cors_middleware,
    security_headers_middleware,
)
from gateway.platforms.api_server_input_media import (
    _persisted_image_paths,
    durable_api_user_message,
    route_api_user_message,
)
from agent.session_persistence import _durable_content


# ---------------------------------------------------------------------------
# Pure-function tests for _normalize_multimodal_content
# ---------------------------------------------------------------------------


class TestNormalizeMultimodalContent:
    def test_string_passthrough(self):
        assert _normalize_multimodal_content("hello") == "hello"

    def test_none_returns_empty_string(self):
        assert _normalize_multimodal_content(None) == ""

    def test_text_only_list_collapses_to_string(self):
        content = [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]
        assert _normalize_multimodal_content(content) == "hi\nthere"

    def test_responses_input_text_canonicalized(self):
        content = [{"type": "input_text", "text": "hello"}]
        assert _normalize_multimodal_content(content) == "hello"


    def test_input_image_converted_to_canonical_shape(self):
        content = [
            {"type": "input_text", "text": "hi"},
            {"type": "input_image", "image_url": "https://example.com/cat.png"},
        ]
        out = _normalize_multimodal_content(content)
        assert out == [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ]

    def test_malformed_inline_image_is_rejected_during_normalization(self):
        with pytest.raises(ValueError, match="invalid_image_data"):
            _normalize_multimodal_content([
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ])

    def test_private_remote_image_is_rejected_before_network_io(self):
        with pytest.raises(ValueError, match="invalid_image_url"):
            _normalize_multimodal_content([
                {"type": "image_url", "image_url": {"url": "http://127.0.0.1/private.png"}},
            ])


class TestContentHasVisiblePayload:


    def test_list_with_image_only(self):
        assert _content_has_visible_payload([{"type": "image_url", "image_url": {"url": "x"}}])


def test_inline_image_is_persisted_as_a_content_addressed_replayable_reference(tmp_path, monkeypatch):
    hermes_home = tmp_path / "Hermes Home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    data_url = "data:image/png;base64," + base64.b64encode(png).decode()
    content = [
        {"type": "input_text", "text": "Describe this screenshot"},
        {"type": "input_image", "image_url": data_url},
    ]

    first = durable_api_user_message(content)
    second = durable_api_user_message(content)

    assert first is not None
    assert first == second
    persisted_text = _durable_content(first)
    assert "Describe this screenshot" in persisted_text
    assert "[screenshot]" not in persisted_text
    quoted_path = persisted_text.split("@image:`", 1)[1].split("`", 1)[0]
    path = Path(quoted_path)
    assert path.parent == hermes_home / "images"
    assert path.read_bytes() == png
    assert len(list(path.parent.iterdir())) == 1

    from hermes_state import SessionDB
    db_path = hermes_home / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("image-replay", source="api_server")
    db.append_message("image-replay", "user", persisted_text)
    db.close()
    reopened = SessionDB(db_path=db_path)
    assert reopened.get_messages("image-replay")[0]["content"] == persisted_text
    reopened.close()

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    from tools.image_source import ResolveContext, resolve_image_source
    resolved = asyncio.run(resolve_image_source(str(path), ResolveContext()))
    assert resolved.data == png
    tiff = b"II*\x00" + b"\x00" * 32
    tiff_content = _normalize_multimodal_content([
        {"type": "text", "text": "Inspect TIFF"},
        {"type": "image_url", "image_url": {"url": "data:image/tiff;base64," + base64.b64encode(tiff).decode()}},
    ])
    tiff_persisted = durable_api_user_message(tiff_content)
    assert tiff_persisted and ".tiff" in tiff_persisted[0]["text"]
    assert "[screenshot]" not in _durable_content(tiff_persisted)


def test_remote_image_is_materialized_for_durable_replay_and_text_routing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    content = [
        {"type": "text", "text": "Inspect remote"},
        {"type": "image_url", "image_url": {"url": "https://example.com/signed.png"}},
    ]
    agent = MagicMock(provider="zai", model="glm-5.3", _hermes_api_runtime={"provider": "zai", "model": "glm-5.3"})
    analysis = AsyncMock(return_value=json.dumps({"success": True, "analysis": "Remote image contents."}))

    from tools.image_source import ResolvedImage
    with (
        patch("tools.image_source.resolve_image_source", AsyncMock(return_value=ResolvedImage(png, "image/png", "http"))),
        patch("agent.image_routing.decide_image_input_mode", return_value="text"),
        patch("tools.vision_tools.vision_analyze_tool", analysis),
    ):
        routed, persisted = route_api_user_message(agent, content)

    assert "Remote image contents." in routed
    assert "Inspect remote" in routed
    assert persisted and "@image:" in persisted[0]["text"]
    durable_path = Path(_persisted_image_paths(persisted)[0])
    assert durable_path.read_bytes() == png


def test_text_only_model_gets_auxiliary_vision_description_without_native_parts(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    content = [
        {"type": "text", "text": "What changed?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
    ]
    agent = MagicMock(provider="zai", model="glm-5.3", _hermes_api_runtime={"provider": "zai", "model": "glm-5.3"})
    analysis = AsyncMock(return_value=json.dumps({"success": True, "analysis": "A pipeline diagram."}))

    with (
        patch("agent.image_routing.decide_image_input_mode", return_value="text"),
        patch("tools.vision_tools.vision_analyze_tool", analysis),
    ):
        routed, persisted = route_api_user_message(agent, content)

    assert isinstance(routed, str)
    assert "A pipeline diagram." in routed
    assert "What changed?" in routed
    assert "data:image" not in routed
    assert persisted and "@image:" in persisted[0]["text"]


# ---------------------------------------------------------------------------
# HTTP integration — real aiohttp client hitting the adapter handlers
# ---------------------------------------------------------------------------


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


class TestChatCompletionsMultimodalHTTP:
    @pytest.mark.asyncio
    async def test_inline_image_preserved_to_run_agent(self, adapter):
        """Multimodal user content reaches _run_agent as a list of parts."""
        image_payload = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_run_agent",
                new=MagicMock(),
            ) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "A cat.", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": image_payload}],
                    },
                )

            assert resp.status == 200, await resp.text()
            assert mock_run.captured["user_message"] == image_payload


class TestResponsesMultimodalHTTP:
    @pytest.mark.asyncio
    async def test_input_image_canonicalized_and_forwarded(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "ok", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "Describe."},
                                    {
                                        "type": "input_image",
                                        "image_url": "https://example.com/cat.png",
                                    },
                                ],
                            }
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            expected = [
                {"type": "text", "text": "Describe."},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ]
            assert mock_run.captured["user_message"] == expected

