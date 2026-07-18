"""Regression tests for FactStore's distilled-statement write boundary."""

import pytest

from plugins.memory.holographic.store import MemoryStore


PROMPT_FRAGMENTS = (
    "Tags: dreaming, factstore-memory-semantics, approval-needed",
    "Requirements: (1) verify the packet; (2) return only a PASS verdict.",
    "**Risk:** approval-needed because memory semantics may change.",
    "User standing preference: Requirements: route reports to sources, never active context.",
    "You are a fresh independent reviewer. Return only a PASS or REVISE verdict.",
)


def test_duplicate_fact_rolls_back_failed_insert_and_releases_writer_lock(tmp_path):
    database = tmp_path / "memory_store.db"
    with MemoryStore(database, hrr_dim=64) as first:
        fact_id = first.add_fact("James prefers concise, distilled FactStore entries.")
        assert first.add_fact("James prefers concise, distilled FactStore entries.") == fact_id
        assert first._conn.in_transaction is False

        with MemoryStore(database, hrr_dim=64) as second:
            second.add_fact("The project uses one canonical repository.", category="project")
            assert second._conn.in_transaction is False


def test_factstore_rejects_raw_handoffs_and_oversized_quotes(tmp_path):
    with MemoryStore(tmp_path / "memory_store.db", hrr_dim=64) as store:
        with pytest.raises(ValueError, match="compaction or handoff"):
            store.add_fact("[CONTEXT COMPACTION — REFERENCE ONLY] raw session handoff")

        with pytest.raises(ValueError, match="single distilled statement"):
            store.add_fact("Okay, " + ("raw quote " * 50))

        fact_id = store.add_fact("James prefers concise, distilled FactStore entries.")
        with pytest.raises(ValueError, match="single distilled statement"):
            store.update_fact(fact_id, content="x" * 321)

        row = store._conn.execute(
            "SELECT content FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        assert row["content"] == "James prefers concise, distilled FactStore entries."


@pytest.mark.parametrize("content", PROMPT_FRAGMENTS)
def test_factstore_rejects_prompt_fragments_on_add(tmp_path, content):
    with MemoryStore(tmp_path / "memory_store.db", hrr_dim=64) as store:
        with pytest.raises(ValueError, match="prompt fragments or instructions"):
            store.add_fact(content)
        assert store.list_facts(min_trust=0.0) == []


def test_factstore_rejects_prompt_fragments_on_update_without_losing_fact(tmp_path):
    with MemoryStore(tmp_path / "memory_store.db", hrr_dim=64) as store:
        durable = "James requires FactStore facts to be distilled declarative statements."
        fact_id = store.add_fact(durable)

        with pytest.raises(ValueError, match="prompt fragments or instructions"):
            store.update_fact(fact_id, content="Return only: Verdict PASS or REVISE.")

        row = store._conn.execute(
            "SELECT content FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        assert row["content"] == durable


def test_factstore_allows_declarative_standing_rules(tmp_path):
    with MemoryStore(tmp_path / "memory_store.db", hrr_dim=64) as store:
        content = (
            "User standing preference: Never retain credentials in notes or summaries; "
            "redact them before storage."
        )
        fact_id = store.add_fact(content, category="user_pref")
        row = store._conn.execute(
            "SELECT content FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        assert row["content"] == content
