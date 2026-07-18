"""hermes-memory-store — holographic memory plugin using MemoryProvider interface.

Registers as a MemoryProvider plugin, giving the agent structured fact storage
with entity resolution, trust scoring, and HRR-based compositional retrieval.

Original plugin by dusterbloom (PR #2351), adapted to the MemoryProvider ABC.

Config in $HERMES_HOME/config.yaml (profile-scoped):
  plugins:
    hermes-memory-store:
      db_path: $HERMES_HOME/memory_store.db   # omit to use the default
      auto_extract: false
      default_trust: 0.5
      min_trust_threshold: 0.3
      temporal_decay_half_life: 0
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from .store import MemoryStore, _MAX_FACT_CONTENT_CHARS
from .retrieval import FactRetriever
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas (unchanged from original PR)
# ---------------------------------------------------------------------------

FACT_STORE_SCHEMA = {
    "name": "fact_store",
    "description": (
        "Deep structured memory with algebraic reasoning. "
        "Use alongside the memory tool — memory for always-on context, "
        "fact_store for deep recall and compositional queries.\n\n"
        "ACTIONS (simple → powerful):\n"
        "• add — Store a fact the user would expect you to remember.\n"
        "• search — Keyword lookup ('editor config', 'deploy process').\n"
        "• probe — Entity recall: ALL facts about a person/thing.\n"
        "• related — What connects to an entity? Structural adjacency.\n"
        "• reason — Compositional: facts connected to MULTIPLE entities simultaneously.\n"
        "• contradict — Memory hygiene: find facts making conflicting claims.\n"
        "• update/remove/list — CRUD operations.\n\n"
        "IMPORTANT: Before answering questions about the user, ALWAYS probe or reason first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "probe", "related", "reason", "contradict", "update", "remove", "list"],
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search')."},
            "entity": {"type": "string", "description": "Entity name for 'probe'/'related'."},
            "entities": {"type": "array", "items": {"type": "string"}, "description": "Entity names for 'reason'."},
            "fact_id": {"type": "integer", "description": "Fact ID for 'update'/'remove'."},
            "category": {"type": "string", "enum": ["user_pref", "project", "tool", "general"]},
            "tags": {"type": "string", "description": "Comma-separated tags."},
            "trust_delta": {"type": "number", "description": "Trust adjustment for 'update'."},
            "min_trust": {"type": "number", "description": "Minimum trust filter (default: 0.3)."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["action"],
    },
}

FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": (
        "Rate a fact after using it. Mark 'helpful' if accurate, 'unhelpful' if outdated. "
        "This trains the memory — good facts rise, bad facts sink."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["helpful", "unhelpful"]},
            "fact_id": {"type": "integer", "description": "The fact ID to rate."},
        },
        "required": ["action", "fact_id"],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    from hermes_constants import get_hermes_home
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8-sig") as f:
            all_config = yaml.safe_load(f) or {}
        return cfg_get(all_config, "plugins", "hermes-memory-store", default={}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Auto-extraction guards
# ---------------------------------------------------------------------------

_COMMAND_CENTER_REQUEST_RE = re.compile(
    r"James gave Archimedes a direct task/request from Command Center floating request|"
    r"\bRun id:\s*|\bReport id:\s*|\bReport path:\s*|"
    r"/home/[^\s]+/\.hermes/data/apps/command-center/reports/",
    re.IGNORECASE,
)

# These are orchestration messages written by Archimedes for a fresh delivery
# worker, not statements authored by the user. They commonly contain words such
# as "always" and "never" because they freeze a repository slice; treating
# those clauses as user preferences pollutes user-level recall.
_FROZEN_SLICE_DELIVERY_PROMPT_RE = re.compile(
    r"^\s*you\s+are\b"
    r"(?=[\s\S]{0,700}\brepository:\s*/\S+)"
    r"(?=[\s\S]{0,700}\b(?:mvp\s+)?slice\s+s\d+\b)",
    re.IGNORECASE,
)

_NON_AUTHORIAL_CONTEXT_RE = re.compile(
    r"^\s*\[(?:context compaction|conversation compaction|session handoff)\b",
    re.IGNORECASE,
)

_DIRECT_TASK_RE = re.compile(
    r"\b(?:i\s+(?:need|want)\s+you\s+to|what\s+i\s+need\s+to\s+do|"
    r"tell\s+me|can\s+you|could\s+you|go\s+ahead\s+and|please)\b|"
    r"^\s*(?:/\w+\s+)?(?:please\s+)?(?:extract|consult|fact[- ]?check|audit|build|fix|"
    r"update|create|make|run|send|file|queue|execute|restart|re-auth|reauth|load)\b",
    re.IGNORECASE,
)

_DURABLE_PREF_RE = re.compile(
    r"\b(?:prefer|prefers|preferred|favorite|default|by default|always|never|usually|"
    r"does\s+not\s+want|do\s+not\s+want|should\s+always|should\s+never|"
    r"must\s+always|must\s+never|part\s+of\s+the\s+process|anytime)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_candidate_sentences(content: str) -> list[str]:
    sentences = [s.strip(" \t\r\n\"'") for s in _SENTENCE_SPLIT_RE.split(content)]
    return [s for s in sentences if s]


def _looks_like_one_off_task(sentence: str) -> bool:
    """Return True for imperative/session-specific requests, not durable prefs."""
    if not _DIRECT_TASK_RE.search(sentence):
        return False
    # Explicit recurring alert rules often use imperative wording
    # ("you need to alert me") but are still durable preferences.
    if re.search(r"\banytime\b", sentence, re.IGNORECASE) and re.search(r"\b(?:alert|notify|notification)\b", sentence, re.IGNORECASE):
        return False
    # "Please always ..." and similar can be a genuine standing preference.
    # Direct "you do X" requests are still task/procedure captures, even when
    # they contain want/need language elsewhere in the sentence.
    if re.search(r"\b(?:i\s+(?:need|want)\s+you\s+to|tell\s+me|what\s+i\s+need\s+to\s+do)\b", sentence, re.IGNORECASE):
        return True
    return not _DURABLE_PREF_RE.search(sentence)


def _skip_auto_extraction(content: str) -> bool:
    """Return True when a user-role message is generated orchestration context."""
    return bool(
        _COMMAND_CENTER_REQUEST_RE.search(content)
        or _FROZEN_SLICE_DELIVERY_PROMPT_RE.search(content)
        or _NON_AUTHORIAL_CONTEXT_RE.search(content)
    )


def _clip_auto_fact(text: str) -> str:
    """Normalize an extracted fact to the FactStore write-boundary limit."""
    normalized = re.sub(r"\s+", " ", text).strip(" -;:.,")
    return normalized[:_MAX_FACT_CONTENT_CHARS].rstrip(" -;:.,")


def _normalize_auto_pref_fact(sentence: str) -> str:
    text = re.sub(r"\s+", " ", sentence).strip(" -;:.,")
    text = re.sub(r"^/\w+\s*(?:also,?\s*)?", "", text, flags=re.IGNORECASE).strip()

    m = re.search(r"\bI\s+prefer\s+(.+)", text, re.IGNORECASE)
    if m:
        return _clip_auto_fact(f"User prefers {m.group(1).strip()}")
    m = re.search(r"\bI\s+(like|love|use)\s+(.+)", text, re.IGNORECASE)
    if m:
        verb = {"like": "likes", "love": "loves", "use": "uses"}[m.group(1).lower()]
        return _clip_auto_fact(f"User {verb} {m.group(2).strip()}")
    m = re.search(r"\bI\s+(always|never|usually)\s+(.+)", text, re.IGNORECASE)
    if m:
        return _clip_auto_fact(f"User {m.group(1).lower()} {m.group(2).strip()}")
    m = re.search(r"\bmy\s+((?:favorite|preferred|default)\s+[^.]+?)\s+is\s+(.+)", text, re.IGNORECASE)
    if m:
        return _clip_auto_fact(f"User's {m.group(1).strip()} is {m.group(2).strip()}")

    if re.match(r"^(?:james|user)\b", text, re.IGNORECASE):
        return _clip_auto_fact(text)
    return _clip_auto_fact(f"User standing preference: {text}")


def _extract_auto_user_pref_facts(content: str) -> list[str]:
    """Extract durable user preferences from a user message.

    This intentionally refuses Command Center request-runner prompts,
    Archimedes-authored frozen-slice delivery prompts, non-authorial context
    handoffs, and imperative one-off tasks. Auto-extraction is a safety net;
    nuanced or procedural memories should be curated explicitly by the
    agent/skills.
    """
    if _skip_auto_extraction(content):
        return []

    facts: list[str] = []
    seen: set[str] = set()
    for sentence in _split_candidate_sentences(content):
        if len(sentence) < 10:
            continue
        if not _DURABLE_PREF_RE.search(sentence):
            continue
        if _looks_like_one_off_task(sentence):
            continue
        fact = _normalize_auto_pref_fact(sentence)
        norm = fact.casefold()
        if norm not in seen:
            seen.add(norm)
            facts.append(fact)
    return facts


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class HolographicMemoryProvider(MemoryProvider):
    """Holographic memory with structured facts, entity resolution, and HRR retrieval."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store = None
        self._retriever = None
        self._min_trust = float(self._config.get("min_trust_threshold", 0.3))

    @property
    def name(self) -> str:
        return "holographic"

    def is_available(self) -> bool:
        return True  # SQLite is always available, numpy is optional

    def save_config(self, values, hermes_home):
        """Write config to config.yaml under plugins.hermes-memory-store."""
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["hermes-memory-store"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    def get_config_schema(self):
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/memory_store.db"
        return [
            {"key": "db_path", "description": "SQLite database path", "default": _default_db},
            {"key": "auto_extract", "description": "Auto-extract facts at session end", "default": "false", "choices": ["true", "false"]},
            {"key": "default_trust", "description": "Default trust score for new facts", "default": "0.5"},
            {"key": "hrr_dim", "description": "HRR vector dimensions", "default": "1024"},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        _hermes_home = str(get_hermes_home())
        _default_db = _hermes_home + "/memory_store.db"
        db_path = self._config.get("db_path", _default_db)
        # Expand $HERMES_HOME in user-supplied paths so config values like
        # "$HERMES_HOME/memory_store.db" or "~/.hermes/memory_store.db" both
        # resolve to the active profile's directory.
        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", _hermes_home)
            db_path = db_path.replace("${HERMES_HOME}", _hermes_home)
        default_trust = float(self._config.get("default_trust", 0.5))
        hrr_dim = int(self._config.get("hrr_dim", 1024))
        hrr_weight = float(self._config.get("hrr_weight", 0.3))
        temporal_decay = int(self._config.get("temporal_decay_half_life", 0))

        self._store = MemoryStore(db_path=db_path, default_trust=default_trust, hrr_dim=hrr_dim)
        self._retriever = FactRetriever(
            store=self._store,
            temporal_decay_half_life=temporal_decay,
            hrr_weight=hrr_weight,
            hrr_dim=hrr_dim,
        )
        self._session_id = session_id

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        try:
            total = self._store._conn.execute(
                "SELECT COUNT(*) FROM facts"
            ).fetchone()[0]
        except Exception:
            total = 0
        if total == 0:
            return (
                "# Holographic Memory\n"
                "Active. Empty fact store — proactively add facts the user would expect you to remember.\n"
                "Use fact_store(action='add') to store durable structured facts about people, projects, preferences, decisions.\n"
                "Use fact_feedback to rate facts after using them (trains trust scores)."
            )
        return (
            f"# Holographic Memory\n"
            f"Active. {total} facts stored with entity resolution and trust scoring.\n"
            f"Use fact_store to search, probe entities, reason across entities, or add facts.\n"
            f"Use fact_feedback to rate facts after using them (trains trust scores)."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._retriever or not query:
            return ""
        try:
            results = self._retriever.search(query, min_trust=self._min_trust, limit=5)
            if not results:
                return ""
            lines = []
            for r in results:
                trust = r.get("trust_score", r.get("trust", 0))
                lines.append(f"- [{trust:.1f}] {r.get('content', '')}")
            return "## Holographic Memory\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("Holographic prefetch failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Holographic memory stores explicit facts via tools, not auto-sync.
        # The on_session_end hook handles auto-extraction if configured.
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "fact_store":
            return self._handle_fact_store(args)
        elif tool_name == "fact_feedback":
            return self._handle_fact_feedback(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._config.get("auto_extract", False):
            return
        if not self._store or not messages:
            return
        self._auto_extract_facts(messages)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes as facts."""
        if action == "add" and self._store and content:
            try:
                category = "user_pref" if target == "user" else "general"
                self._store.add_fact(content, category=category)
            except Exception as e:
                logger.debug("Holographic memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        # Close the SQLite connection deterministically instead of leaking it
        # to GC. MemoryStore opens its connection with check_same_thread=False
        # (store.py), so without an explicit close() the sqlite3.Connection's
        # fd is released by refcount/GC at a non-deterministic time on a
        # non-deterministic thread, churning a DB fd through the kernel's free
        # pool on every session teardown. close() already exists and is cheap.
        if self._store is not None:
            try:
                self._store.close()
            except Exception as e:
                logger.debug("Holographic shutdown close() failed: %s", e)
        self._store = None
        self._retriever = None

    # -- Tool handlers -------------------------------------------------------

    def _handle_fact_store(self, args: dict) -> str:
        try:
            action = args["action"]
            store = self._store
            retriever = self._retriever

            if action == "add":
                fact_id = store.add_fact(
                    args["content"],
                    category=args.get("category", "general"),
                    tags=args.get("tags", ""),
                )
                return json.dumps({"fact_id": fact_id, "status": "added"})

            elif action == "search":
                results = retriever.search(
                    args["query"],
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", self._min_trust)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "probe":
                results = retriever.probe(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "related":
                results = retriever.related(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "reason":
                entities = args.get("entities", [])
                if not entities:
                    return tool_error("reason requires 'entities' list")
                results = retriever.reason(
                    entities,
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "contradict":
                results = retriever.contradict(
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "update":
                updated = store.update_fact(
                    int(args["fact_id"]),
                    content=args.get("content"),
                    trust_delta=float(args["trust_delta"]) if "trust_delta" in args else None,
                    tags=args.get("tags"),
                    category=args.get("category"),
                )
                return json.dumps({"updated": updated})

            elif action == "remove":
                removed = store.remove_fact(int(args["fact_id"]))
                return json.dumps({"removed": removed})

            elif action == "list":
                facts = store.list_facts(
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", 0.0)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"facts": facts, "count": len(facts)})

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_fact_feedback(self, args: dict) -> str:
        try:
            fact_id = int(args["fact_id"])
            helpful = args["action"] == "helpful"
            result = self._store.record_feedback(fact_id, helpful=helpful)
            return json.dumps(result)
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    # -- Auto-extraction (on_session_end) ------------------------------------

    def _auto_extract_facts(self, messages: list) -> None:
        store = self._store
        if store is None:
            return

        _DECISION_PATTERNS = [
            re.compile(r'\bwe\s+(?:decided|agreed|chose)\s+(?:to\s+)?(.+)', re.IGNORECASE),
            re.compile(r'\bthe\s+project\s+(?:uses|needs|requires)\s+(.+)', re.IGNORECASE),
        ]

        extracted = 0
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 10:
                continue
            if _skip_auto_extraction(content):
                continue

            for fact in _extract_auto_user_pref_facts(content):
                try:
                    store.add_fact(fact, category="user_pref")
                    extracted += 1
                except Exception:
                    pass

            project_fact: str | None = None
            for sentence in _split_candidate_sentences(content):
                if any(pattern.search(sentence) for pattern in _DECISION_PATTERNS):
                    project_fact = _clip_auto_fact(sentence)
                    break
            if project_fact:
                try:
                    store.add_fact(project_fact, category="project")
                    extracted += 1
                except Exception:
                    pass

        if extracted:
            logger.info("Auto-extracted %d facts from conversation", extracted)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the holographic memory provider with the plugin system."""
    config = _load_plugin_config()
    provider = HolographicMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
