"""Regression tests for holographic memory auto-extraction.

Auto extraction should only persist durable preferences/standing rules. It must
not mirror one-off user tasks or Command Center request-runner boilerplate into
FactStore as ``user_pref`` facts.
"""

from plugins.memory.holographic import HolographicMemoryProvider


def _make_provider(tmp_path):
    provider = HolographicMemoryProvider(
        config={"db_path": str(tmp_path / "memory_store.db"), "auto_extract": True, "hrr_dim": 64}
    )
    provider.initialize(session_id="test-session")
    return provider


def _facts(provider):
    assert provider._store is not None
    rows = provider._store._conn.execute(
        "SELECT content, category, tags FROM facts ORDER BY fact_id"
    ).fetchall()
    return [dict(r) for r in rows]


def test_auto_extract_rejects_command_center_request_boilerplate(tmp_path):
    provider = _make_provider(tmp_path)
    try:
        provider.on_session_end(
            [
                {
                    "role": "user",
                    "content": (
                        "James gave Archimedes a direct task/request from Command Center floating request. "
                        "Execute it with normal Hermes tools when safe, then update the Command Center report JSON in place.\n\n"
                        "Run id: request-example-20260710\n"
                        "Report id: request-example\n"
                        "Report path: /home/telos/.hermes/data/apps/command-center/reports/request-example.json\n\n"
                        "User request: Archimedes, I need you to file a report for me about those Home rows."
                    ),
                }
            ]
        )
        assert _facts(provider) == []
    finally:
        provider.shutdown()


def test_auto_extract_rejects_one_off_task_instructions(tmp_path):
    provider = _make_provider(tmp_path)
    try:
        provider.on_session_end(
            [
                {
                    "role": "user",
                    "content": (
                        "Please extract audio from the homosexuality.mp4 file on the mac mini. "
                        "Then I want you to consult with claude opus and have it perform a fact check."
                    ),
                },
                {"role": "user", "content": "Tell me what I need to do to re-auth Claude."},
            ]
        )
        assert _facts(provider) == []
    finally:
        provider.shutdown()


def test_auto_extract_rejects_frozen_slice_delivery_and_compaction_prompts(tmp_path):
    provider = _make_provider(tmp_path)
    try:
        provider.on_session_end(
            [
                {
                    "role": "user",
                    "content": (
                        "You are the fresh implementation agent for Bitcoin Flow-Balance MVP "
                        "Slice S10 only. Repository: /home/telos/projects/bitcoin-flow-model "
                        "Read AGENTS.md and the architecture/S10 plan. The project requires the "
                        "default monotone logistic backend and must never hide the bound width."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "You are the fresh repair implementer for Bitcoin Flow-Balance MVP "
                        "Slice S9 only. Repository: /home/telos/projects/bitcoin-flow-model "
                        "Invalid Python dates must always return HTTP 422, never ValueError/500."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted. "
                        "S8 secondary observations must never alter the primary dependency hash."
                    ),
                },
            ]
        )
        assert _facts(provider) == []
    finally:
        provider.shutdown()


def test_auto_extract_keeps_long_preferences_and_multiline_project_decisions(tmp_path):
    provider = _make_provider(tmp_path)
    try:
        provider.on_session_end(
            [
                {
                    "role": "user",
                    "content": "I prefer " + ("compact evidence receipts " * 20),
                },
                {
                    "role": "user",
                    "content": (
                        "Status update before the durable choice.\n"
                        "We decided to use one canonical project store "
                        + ("with deterministic provenance " * 20)
                    ),
                },
            ]
        )
        facts = _facts(provider)
        assert [fact["category"] for fact in facts] == ["user_pref", "project"]
        assert all(len(fact["content"]) <= 320 for fact in facts)
        assert facts[0]["content"].startswith("User prefers compact evidence receipts")
        assert facts[1]["content"].startswith("We decided to use one canonical project store")
        assert all("\n" not in fact["content"] for fact in facts)
    finally:
        provider.shutdown()


def test_auto_extract_keeps_standing_preferences(tmp_path):
    provider = _make_provider(tmp_path)
    try:
        provider.on_session_end(
            [
                {
                    "role": "user",
                    "content": (
                        "/queue also, I should have received an alert in Command Center that Claude auth was failing. "
                        "Anytime any part of our setup breaks or is not working, alert me in Command Center."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "James does not want hidden swipe-to-open menu gestures added to apps by default; "
                        "prefer explicit controls like hamburger and back buttons."
                    ),
                },
            ]
        )
        facts = _facts(provider)
        contents = [f["content"] for f in facts]
        assert len(contents) == 2
        assert any("alert" in c.lower() and "setup breaks" in c.lower() for c in contents)
        assert any("swipe-to-open" in c.lower() and "by default" in c.lower() for c in contents)
        assert all(f["category"] == "user_pref" for f in facts)
    finally:
        provider.shutdown()
