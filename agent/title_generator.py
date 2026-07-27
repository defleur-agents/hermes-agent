"""Auto-generate short session titles from the user's opening message.

Two stages, both off the critical path: an **instant** deterministic title (written before the model
is called, cannot fail), then an **upgrade** from one small-model call (cheap tier, thinking off,
JSON-constrained). Storage enforces provenance ``derived < llm < user``: stage 2 only replaces stage 1
and neither replaces a name the user typed."""

import json
import logging
import re
import threading
from contextlib import suppress
from contextvars import copy_context
from typing import Any, Callable, Optional

from agent.auxiliary_client import call_llm
from agent.context_compressor import LEGACY_SUMMARY_PREFIX
from agent.message_content import flatten_message_text

logger = logging.getLogger(__name__)

# (task_name, exception) -> None; surfaces auxiliary failures so silent drops don't pile up as NULL titles.
FailureCallback = Callable[[str, BaseException], None]
# (title, source) -> None; source is the persisted provenance (``derived`` / ``llm``). Consumers paying a
# rate-limited remote rename per title (Discord thread, Telegram topic) should act on ``llm`` only.
TitleCallback = Callable[[str, str], None]
# () -> bool, called right before the LLM request; False skips (e.g. the user switched models and
# the request would reload one the runtime already evicted).
# Validation callback: () -> bool. See #19027.
RuntimeValidator = Callable[[], bool]

# Text budget handed to the model (Claude Code / OpenClaw converged on 1000).
MAX_TITLE_INPUT_CHARS = 1000
# Cap on the instant derived title; a raw fragment reads worse the longer it runs.
MAX_DERIVED_TITLE_CHARS = 48
# Answer-shaped guard: a tiny model sometimes answers instead of titling; longer is rejected, not truncated.
# Upper bound on accepted title word count. Titling is a 3-7 word task; a small tiny-model sometimes ignores
# the task and answers the user's message instead — that answer must never become the session title (see the
# answer-shaped output guard in generate_title; port of can1357/oh-my-pi#7306). 12 leaves headroom for
# legitimate wordy titles while excluding full-sentence answers.
_MAX_TITLE_WORDS = 12

_TITLE_PROMPT_TEMPLATE = (
    "You name chat sessions. Given the user's opening message, write a title "
    "that lets them find this conversation again in a list.\n\n"
    "Rules:\n"
    "__LENGTH_RULE__\n"
    "- Name what the user wants DONE, not that they asked a question.\n"
    "- Prefer an explicitly named project, person, product, or other proper name.\n"
    "- Avoid generic leading labels such as Fixing, Update, or Analysis when a specific subject is available.\n"
    "- Keep technical terms, filenames, numbers, and error codes exact.\n"
    "__ALIAS_RULE__"
    "- Drop filler words: the, this, my, a, an.\n"
    "- Do not include emoji.\n"
    "- No trailing punctuation, no quotes, no tool names, no 'Title:' prefix.\n"
    "- Never answer the message. Name it.\n"
    "- Always produce something, even for a bare greeting.\n"
    "__LANGUAGE_RULE__\n"
    'Good: {"title": "Fix login button on mobile"}\n'
    'Good: {"title": "Postgres connection pool exhaustion"}\n'
    'Good: {"title": "Friendly greeting"}\n'
    'Too vague: {"title": "Code changes"}\n'
    'Too long: {"title": "Investigate and fix the issue where the login button '
    'does not respond on mobile devices"}\n\n'
    'Reply with JSON only: {"title": "..."}'
)

_LANGUAGE_RULE_MATCH_USER = "- Write the title in the same language as the user's message."
_LANGUAGE_RULE_PINNED = "- Write the title in {language}."

# Constrains the response to a single title field ("model answered instead of titling" failure class).
_TITLE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "session_title", "strict": True, "schema": {
        "type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"], "additionalProperties": False}},
}

# Control-tag wrappers around machine-authored content inside a nominal "user" message (Codex CLI's
# RECOGNIZED_CONTROL_WRAPPERS): stripped, titling continues on what remains.
_CONTROL_WRAPPERS = tuple(
    (f"<{tag}>", f"</{tag}>")
    for tag in ("command-message", "command-name", "command-args", "local-command-caveat", "local-command-stderr",
                "local-command-stdout", "task-notification", "system-reminder", "ide_opened_file", "ide_selection")
)

# Hermes' own machine-authored openers: a compaction handoff or resumed session must not be titled after them.
_MACHINE_PREFIXES = (
    "[CONTEXT COMPACTION", LEGACY_SUMMARY_PREFIX, "[Runtime note:", "[System note:", "[SYSTEM]",
    # tui_gateway.server._MODEL_SWITCH_MARKER_PREFIX (keep in sync); persisted as role="user" because
    # strict providers reject a non-first system message.
    # Model-switch marker from tui_gateway.server._append_model_switch_marker. It is persisted with
    # role="user" (strict OpenAI-compatible providers reject a system message that is not first — #48338),
    # so without this entry it looks like a real opening turn: switching models before the first real
    # message titled the session "[System: The active model for this chat has…" instead of the user's actual
    # question.
    "[System: The active model for this chat has changed to ",
)


def _title_config() -> dict:
    """``auxiliary.title_generation`` (lazy read-only import: no hermes_cli cycle, no migration writes)."""
    from hermes_cli.config import load_config_readonly
    return ((load_config_readonly() or {}).get("auxiliary") or {}).get("title_generation") or {}


def _title_language() -> str:
    """Configured title language, or "" to match the user."""
    try:
        return str(_title_config().get("language", "")).strip()
    except Exception:
        return ""


def _title_preferences() -> tuple[int, int, int, dict[str, str]]:
    """Return compact-title limits and user-defined canonical aliases."""
    defaults = 2, 3, 40
    try:
        title_config = _title_config()
        max_words = max(1, min(int(title_config.get("max_words", defaults[1])), 12))
        min_words = max(1, min(int(title_config.get("min_words", defaults[0])), max_words))
        max_characters = max(
            12, min(int(title_config.get("max_characters", defaults[2])), 120)
        )
        raw_aliases = title_config.get("name_aliases", {})
        aliases = (
            {
                str(alias).strip(): str(canonical).strip()
                for alias, canonical in raw_aliases.items()
                if str(alias).strip() and str(canonical).strip()
            }
            if isinstance(raw_aliases, dict)
            else {}
        )
        return min_words, max_words, max_characters, aliases
    except Exception:
        return defaults[0], defaults[1], defaults[2], {}


def _canonical_name_for_message(
    user_message: str, name_aliases: dict[str, str]
) -> Optional[str]:
    """Return the canonical name for the longest whole-term alias match."""
    for alias, canonical in sorted(
        name_aliases.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", user_message, re.IGNORECASE):
            return canonical
    return None


def _build_title_prompt(
    *,
    language: str,
    min_words: int,
    max_words: int,
    max_characters: int,
    name_aliases: dict[str, str],
) -> str:
    """Build the structured-output title prompt from user preferences."""
    language_rule = (
        _LANGUAGE_RULE_PINNED.format(language=language)
        if language
        else _LANGUAGE_RULE_MATCH_USER
    )
    if min_words == max_words:
        word_preference = f"exactly {max_words} " + ("word" if max_words == 1 else "words")
    else:
        word_preference = f"{min_words}-{max_words} words"
    length_rule = (
        f"- Prefer {word_preference} and at most {max_characters} characters; "
        "use sentence case (capitalize only the first word and proper nouns)."
    )
    alias_rule = ""
    if name_aliases:
        aliases_json = json.dumps(name_aliases, ensure_ascii=False, sort_keys=True)
        alias_rule = (
            "- Use these case-insensitive canonical name replacements when the "
            f"opening message contains an alias: {aliases_json}.\n"
        )
    # Placeholder substitution, not str.format: the prompt embeds literal JSON braces.
    return (
        _TITLE_PROMPT_TEMPLATE.replace("__LANGUAGE_RULE__", language_rule)
        .replace("__LENGTH_RULE__", length_rule)
        .replace("__ALIAS_RULE__", alias_rule)
    )


def _auto_title_enabled() -> bool:
    try:
        from utils import is_truthy_value
        return is_truthy_value(_title_config().get("enabled"), default=True)
    except Exception:
        logger.debug("Failed to read title_generation.enabled", exc_info=True)
        return True


def strip_control_wrappers(text: str) -> str:
    """Remove leading control wrappers (nested too) so a slash-command turn reduces to the prose the user typed."""
    current = (text or "").strip()
    for _ in range(len(_CONTROL_WRAPPERS) * 2):  # bounded: each pass must remove a wrapper or we stop
        stripped = _strip_one_wrapper(current)
        if stripped == current:
            break
        current = stripped
    return current


def _strip_one_wrapper(text: str) -> str:
    lowered = text.lower()
    for open_tag, close_tag in _CONTROL_WRAPPERS:
        if not lowered.startswith(open_tag):
            continue
        end = lowered.find(close_tag)
        if end == -1:  # unterminated wrapper: drop the opening tag and keep the body
            return text[len(open_tag):].strip()
        # Prefer trailing prose; otherwise the wrapper body is all we have.
        return (text[end + len(close_tag):].strip() or text[len(open_tag):end].strip()).strip()
    return text


def _summarize_user_message(user_message: str) -> str:
    """Text worth titling: describe a ``/skill`` invocation (it embeds the whole skill body), then strip wrappers."""
    if not user_message:
        return ""
    described = None
    try:
        from agent.skill_commands import describe_skill_invocation
        described = describe_skill_invocation(user_message)
    except Exception:
        logger.debug("Skill-scaffolding summary failed; titling raw", exc_info=True)
    return strip_control_wrappers(user_message if described is None else described)


def is_titleable_user_message(user_message: str) -> bool:
    """False for machine-authored openers and turns that reduce to nothing once scaffolding is stripped."""
    return (isinstance(user_message, str) and bool(user_message.strip()) and not user_message.lstrip().startswith(_MACHINE_PREFIXES)
            and bool(_summarize_user_message(user_message).strip()))


def derive_title(user_message: str) -> Optional[str]:
    """Instant title: first meaningful line trimmed to a word boundary. No model, never fails."""
    line = " ".join(_first_line(_summarize_user_message(user_message)).split())
    if len(line) > MAX_DERIVED_TITLE_CHARS:
        cut = line[:MAX_DERIVED_TITLE_CHARS]
        space = cut.rfind(" ")
        line = (cut[:space] if space > MAX_DERIVED_TITLE_CHARS // 2 else cut).rstrip(" ,.;:—-") + "…"
    return line or None


def _strip_title_prefix(text: str) -> str:
    return text[6:].strip() if text.lower().startswith("title:") else text


def _first_line(text: str) -> str:
    return next((ln.strip() for ln in text.splitlines() if ln.strip()), "")


def _extract_title_text(content: str) -> str:
    """Strict JSON, then a loose JSON scan, then first-line prose (a provider ignoring ``response_format`` still titles)."""
    if not content:
        return ""
    raw = content.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("title"), str):
            return parsed["title"].strip()
    except (ValueError, TypeError):
        pass
    match = re.search(r'"title\"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if match:
        with suppress(ValueError):
            return json.loads(f'"{match.group(1)}"').strip()
        return match.group(1).strip()
    # Prose fallback: scrub <think> blocks so reasoning can't leak into a title.
    try:
        from agent.agent_runtime_helpers import strip_think_blocks
        raw = strip_think_blocks(None, raw).strip()
    except Exception:
        logger.debug("strip_think_blocks unavailable for title output", exc_info=True)
    return _strip_title_prefix(_first_line(raw)).strip("\"'").strip()


def _clean_title(text: str, max_characters: int = 80) -> Optional[str]:
    """Normalize a model-produced title, or None when nothing usable remains."""
    title = _strip_title_prefix(" ".join((text or "").split()).strip("\"'").strip()).rstrip(".!,;:")
    if len(title) > max_characters:
        title = title[: max_characters - 3].rstrip() + "..."
    return title or None


def _enforce_max_words(title: Optional[str], max_words: int) -> Optional[str]:
    """Deterministically enforce the configured title word maximum."""
    if not title:
        return None
    words = title.split()
    return title if len(words) <= max_words else " ".join(words[:max_words]).rstrip(".!,;:")


def _safe_callback(callback: Optional[Callable], args: tuple, log_fmt: str, label: str) -> None:
    """Invoke an optional consumer callback, never raising."""
    try:
        if callback is not None:
            callback(*args)
    except Exception:
        logger.debug(log_fmt, label, exc_info=True)


def _report_failure(failure_callback: Optional[FailureCallback], exc: BaseException, label: str) -> None:
    _safe_callback(failure_callback, ("title generation", exc), "%s failure_callback raised", label)


def _notify_title(title_callback: Optional[TitleCallback], title: str, source: str, label: str) -> None:
    _safe_callback(title_callback, (title, source), "%s callback failed", label)


def generate_title(
    user_message: str,
    timeout: Optional[float] = None,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> Optional[str]:
    """Title from the opening message alone (waiting for the assistant made this slow and bought
    nothing). ``runtime_validator`` runs right before the request; False skips silently.

    If it returns False (e.g. the user's model was switched since the background thread captured its runtime
    snapshot), the call is skipped silently — no request is sent, so a stale title request can't reload a
    model the runtime already unloaded (#19027).
    """
    if not _auto_title_enabled():
        logger.debug("Auto-title skipped: auxiliary.title_generation.enabled=false")
        return None
    try:
        if runtime_validator is not None and not runtime_validator():
            logger.debug("Title generation skipped: runtime validator returned False")
            return None
    except Exception:  # fail open: a broken validator must not disable titling
        logger.debug("Title runtime validator raised; proceeding", exc_info=True)
    summarized_user_message = _summarize_user_message(user_message)
    user_snippet = summarized_user_message[:MAX_TITLE_INPUT_CHARS]
    if not user_snippet.strip():
        return None
    language = _title_language()
    min_words, max_words, max_characters, name_aliases = _title_preferences()
    prompt = _build_title_prompt(
        language=language,
        min_words=min_words,
        max_words=max_words,
        max_characters=max_characters,
        name_aliases=name_aliases,
    )
    try:
        response = call_llm(
            task="title_generation",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_snippet}],
            # A title is a handful of tokens; a larger ceiling let chatty models burn seconds.
            max_tokens=64, temperature=0.3, timeout=timeout, main_runtime=main_runtime,
            extra_body={"response_format": _TITLE_RESPONSE_FORMAT},
        )
        title = _clean_title(
            _extract_title_text(response.choices[0].message.content or ""), max_characters
        )
        canonical_name = _canonical_name_for_message(summarized_user_message, name_aliases)
        # Answer-shaped output guard: titling is a 3-7 word task, so a title with many words is a model that
        # ignored the task and answered the user's message instead ("I don't have context on X — that's not
        # something I recognize..."). Truncating would store half an assistant blob as the session title,
        # which is still an assistant blob — reject instead so the caller retries on the next exchange
        # (maybe_auto_title fires for the first two exchanges). Port of can1357/oh-my-pi#7306.
        if canonical_name is None and title is not None and len(title.split()) > _MAX_TITLE_WORDS:
            # Answer-shaped output: reject (not truncate) so the caller retries next exchange.
            logger.debug("Rejecting answer-shaped title output (%d words > %d)", len(title.split()), _MAX_TITLE_WORDS)
            return None
        return _enforce_max_words(canonical_name or title, max_words)
    except Exception as e:
        # WARNING so it shows in agent.log without debug mode; stack at debug.
        logger.warning("Title generation failed: %s", e)
        logger.debug("Title generation traceback", exc_info=True)
        _report_failure(failure_callback, e, "Title generation")
        return None


def choose_topic_icon(
    title: str,
    user_message: str,
    allowed_emojis: list[str],
    timeout: float = 30.0,
    *,
    recent_emojis: Optional[list[str]] = None,
) -> Optional[str]:
    """Choose a varied semantic Telegram topic emoji from a live allowlist.

    The auxiliary model ranks several valid candidates so the caller can avoid
    recently used icons without sacrificing semantic fit. The returned value is
    a normal emoji key; the Telegram adapter resolves it to the allowed
    sticker's ``custom_emoji_id``.
    """
    allowed = list(
        dict.fromkeys(str(emoji).strip() for emoji in allowed_emojis if str(emoji).strip())
    )
    if not allowed:
        return None

    def _emoji_key(value: str) -> str:
        return value.replace("\ufe0e", "").replace("\ufe0f", "")

    allowed_by_key = {_emoji_key(emoji): emoji for emoji in allowed}
    recent = list(
        dict.fromkeys(
            allowed_by_key[key]
            for emoji in (recent_emojis or [])
            if (key := _emoji_key(str(emoji).strip())) in allowed_by_key
        )
    )
    recent_keys = {_emoji_key(emoji) for emoji in recent}
    fresh_allowed = [emoji for emoji in allowed if _emoji_key(emoji) not in recent_keys]
    # When there are enough fresh choices, omit recent icons from the model's
    # candidate pool entirely. Prompt wording and temperature are advisory;
    # allowlist exclusion is what guarantees real rotation across the live set.
    exclude_recent = bool(recent) and len(fresh_allowed) >= min(4, len(allowed))
    selection_pool = fresh_allowed if exclude_recent else allowed
    candidate_count = min(4, len(selection_pool))
    prompt = (
        f"Choose {candidate_count} distinct ranked emoji candidates for a conversation topic, "
        "from best fit to least preferred. Select only from this allowed list: "
        + json.dumps(selection_pool, ensure_ascii=False)
        + ". Prefer specific, playful visual metaphors over generic computer, robot, or rocket "
        "icons when a more distinctive allowed icon fits. Keep every candidate clearly related "
        "to the topic; novelty must not override meaning. Return only the ranked emoji characters "
        "separated by spaces, with no words or explanation."
    )
    if recent:
        if exclude_recent:
            prompt += (
                " Icons used recently in this chat were removed from the allowed list: "
                + json.dumps(recent, ensure_ascii=False)
                + ". Do not return them."
            )
        else:
            prompt += (
                " These icons were used recently in this chat: "
                + json.dumps(recent, ensure_ascii=False)
                + ". Avoid repeating them unless one is unmistakably the best semantic fit."
            )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                f"Title: {str(title or '')[:120]}\n"
                f"Opening request: {str(user_message or '')[:500]}"
            ),
        },
    ]

    try:
        response = call_llm(
            task="title_generation",
            messages=messages,
            max_tokens=64,
            temperature=0.7,
            timeout=timeout,
        )
        content = response.choices[0].message.content or ""
        from agent.agent_runtime_helpers import strip_think_blocks

        choice = strip_think_blocks(None, content).strip().strip("\"'")
        normalized_choice = _emoji_key(choice)

        # Extract allowed emojis in the model's ranked response order. Matching
        # the variation-selector-free form accepts e.g. ``⚡`` for allowed
        # ``⚡️`` while overlap filtering keeps a compound emoji from also
        # matching one of its component glyphs.
        hits: list[tuple[int, int, int, str]] = []
        for allowed_index, emoji in enumerate(selection_pool):
            key = _emoji_key(emoji)
            start = normalized_choice.find(key)
            while start >= 0:
                hits.append((start, start + len(key), allowed_index, emoji))
                start = normalized_choice.find(key, start + max(1, len(key)))
        hits.sort(key=lambda hit: (hit[0], -(hit[1] - hit[0]), hit[2]))

        candidates: list[str] = []
        occupied: list[tuple[int, int]] = []
        for start, end, _, emoji in hits:
            if any(start < used_end and end > used_start for used_start, used_end in occupied):
                continue
            occupied.append((start, end))
            if emoji not in candidates:
                candidates.append(emoji)
        if not candidates:
            return None

        return next(
            (emoji for emoji in candidates if _emoji_key(emoji) not in recent_keys),
            candidates[0],
        )
    except Exception:
        logger.debug("Telegram topic icon selection failed", exc_info=True)
        return None


def _has_upgraded_title(session_db, session_id: str) -> bool:
    """True when the session already carries an ``llm``/``user`` title (or the check fails)."""
    try:
        source_fn = getattr(session_db, "get_session_title_source", None)
        if source_fn is not None:
            return source_fn(session_id) not in (None, "derived")
        return bool(session_db.get_session_title(session_id))
    except Exception:
        return True


def _persist_session_title(session_db, session_id, title, *, source, dedupe=True):
    """Persist at *source* authority via ``set_auto_title`` (precedence check + write in one
    transaction, so a manual ``/title`` is never overwritten); None when a higher authority held the row.
    ``ValueError`` = unique-title index collision → append ``#N`` via ``get_next_title_in_lineage``;
    ``dedupe=False`` re-raises instead (the derived title is on the critical path, collides constantly
    on "hi", and the model replaces it a second later anyway).

    ``ValueError`` means the name is taken by an unrelated session (the unique-title index); rather than
    leave the session untitled (#50537), append a ``#N`` suffix via ``get_next_title_in_lineage``.
    """
    auto_fn = getattr(session_db, "set_auto_title", None)

    def _set(candidate):
        if auto_fn is not None:
            if auto_fn(session_id, candidate, source=source):
                return candidate
            logger.debug("Skipping %s title: a higher-authority title already holds session %s", source, session_id)
            return None
        legacy_fn = getattr(session_db, "set_auto_title_if_empty", None)  # older store without provenance
        if legacy_fn is not None:
            return candidate if legacy_fn(session_id, candidate) else None
        if session_db.set_session_title(session_id, candidate) is False:
            raise RuntimeError(f"session {session_id} not found when storing title")
        return candidate

    try:
        return _set(title)
    except ValueError:
        next_title_fn = getattr(session_db, "get_next_title_in_lineage", None)
        deduped = next_title_fn(title) if dedupe and next_title_fn is not None else None
        if not deduped or deduped == title:
            raise
        return _set(deduped)


def apply_instant_title(session_db, session_id: str, user_message: str, title_callback: Optional[TitleCallback] = None) -> Optional[str]:
    """Write the derived title inline. Returns it, or None (no usable text, or a ``derived``+ title exists). Never raises."""
    if not session_db or not session_id:
        return None
    try:
        title = derive_title(user_message) if is_titleable_user_message(user_message) else None
        persisted = _persist_session_title(session_db, session_id, title, source="derived", dedupe=False) if title else None
        if persisted:
            _notify_title(title_callback, persisted, "derived", "Instant-title")
        return persisted
    except Exception:
        logger.debug("Instant title failed", exc_info=True)
        return None


def auto_title_session(
    session_db,
    session_id: str,
    user_message: str,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> None:
    """Generate and store the model title (daemon-thread target); skips sessions already carrying an
    ``llm``/``user`` title (a ``derived`` one is expected — upgrading it is the point). Never lets an
    exception escape (the threading excepthook would spray a traceback into the terminal); the canonical
    trigger is the post-``hermes update`` window where lazy imports read NEW source against OLD modules."""
    try:
        if not session_db or not session_id or _has_upgraded_title(session_db, session_id):
            return
        # This thread starts AFTER the turn's ambient context was reset; republish it so the call carries
        # the same Portal ``conversation=`` tag (root-of-lineage) and bills usage to this session.
        from agent.aux_accounting import set_accounting_context
        from agent.portal_tags import set_conversation_context
        conversation_id = session_id
        with suppress(Exception):
            conversation_id = session_db.get_conversation_root(session_id) or session_id
        set_conversation_context(conversation_id)
        # Same for the accounting context, so the title call's token usage is recorded against this session
        # (task='title_generation', #23270).
        set_accounting_context(session_db, session_id)
        title, source = generate_title(
            user_message, failure_callback=failure_callback, main_runtime=main_runtime, runtime_validator=runtime_validator,
        ), "llm"
        if not title:  # the inline attempt declined collisions; off the critical path the lineage scan is affordable
            title, source = derive_title(user_message), "derived"
        if not title:
            return
        try:
            persisted = _persist_session_title(session_db, session_id, title, source=source)
        except Exception as e:
            logger.debug("Failed to set auto-generated title: %s", e)
            return
        if persisted is not None:
            logger.debug("Auto-generated session title: %s", persisted)
            _notify_title(title_callback, persisted, source, "Auto-title")
    except Exception as e:
        # WARNING so operators see it in agent.log; names the likely cause.
        logger.warning("Auto-title failed (harmless; if this started after an update, restart the running Hermes process): %s", e)
        logger.debug("Auto-title traceback", exc_info=True)
        _report_failure(failure_callback, e, "Auto-title")


def _is_real_user_turn(message: Any) -> bool:
    """A question a person actually asked (Hermes persists machinery under ``role="user"``)."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content = message.get("content")
    return is_titleable_user_message(content if isinstance(content, str) else flatten_message_text(content))


def _session_is_untitled(session_db, session_id: str) -> bool:
    """No title of any provenance; False when it can't tell (no model call per turn for an unreadable title)."""
    getter = getattr(session_db, "get_session_title", None)
    try:
        return callable(getter) and not str(getter(session_id) or "").strip()
    except Exception:
        logger.debug("Untitled check failed for %s", session_id, exc_info=True)
        return False


def maybe_auto_title(
    session_db,
    session_id: str,
    user_message: str,
    conversation_history: Optional[list] = None,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> None:
    """Instant inline title, then a daemon-thread upgrade. Call at the START of a turn, before the model."""
    if not session_db or not session_id or not user_message:
        return
    # History may be pre- or post-message. Skip only when BOTH past the opening turn AND named: count alone
    # left a machinery-opened session nameless; title alone never titles on an old store.
    user_msg_count = sum(1 for m in (conversation_history or []) if _is_real_user_turn(m))
    if (user_msg_count > 1 and not _session_is_untitled(session_db, session_id)) or not is_titleable_user_message(user_message):
        return
    if not _auto_title_enabled():  # config read after the cheap guards so the file isn't touched every turn
        logger.debug("Auto-title skipped: auxiliary.title_generation.enabled=false")
        return
    apply_instant_title(session_db, session_id, user_message, title_callback)
    # Preserve the active profile/runtime context for the background request.
    context = copy_context()
    threading.Thread(
        target=context.run,
        args=(auto_title_session, session_db, session_id, user_message),
        kwargs=dict(failure_callback=failure_callback, main_runtime=main_runtime, title_callback=title_callback, runtime_validator=runtime_validator),
        daemon=True,
        name="auto-title",
    ).start()
