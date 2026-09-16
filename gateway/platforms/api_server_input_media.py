"""Durable transcript projection for multimodal API inputs.

API clients commonly submit images as data URLs.  Providers can consume those bytes for the
live turn, but the text-only transcript projection used to replace each image with
``[screenshot]``.  Materialize inline images in Hermes's durable media directory before the
turn starts and put replayable ``@image:`` references in the persistence override.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


_DATA_IMAGE_RE = re.compile(
    r"^data:(image/[A-Za-z0-9.+-]+)(?:;[^,;]+)*;base64,([A-Za-z0-9+/=\r\n]+)$",
    re.IGNORECASE,
)
_IMAGE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/avif": ".avif",
    "image/heic": ".heic",
    "image/x-icon": ".ico",
    "image/svg+xml": ".svg",
}


def _image_url(part: dict[str, Any]) -> str:
    value = part.get("image_url")
    if isinstance(value, dict):
        value = value.get("url")
    return str(value or "").strip()


def _decode_inline_image(url: str, *, strict: bool = False) -> tuple[bytes, str] | None:
    match = _DATA_IMAGE_RE.fullmatch(url)
    if match is None:
        if strict:
            raise ValueError("invalid_image_data:Image data URLs must contain valid base64 image bytes.")
        return None
    try:
        payload = base64.b64decode(re.sub(r"\s+", "", match.group(2)), validate=True)
    except (binascii.Error, ValueError) as exc:
        if strict:
            raise ValueError("invalid_image_data:Image data URLs must contain valid base64 image bytes.") from exc
        return None

    from agent.image_routing import _sniff_mime_from_bytes
    from gateway.platforms.base import validate_inbound_media_size

    try:
        validate_inbound_media_size(len(payload), media_type="image")
    except ValueError as exc:
        if strict:
            raise ValueError(f"invalid_image_data:{exc}") from exc
        return None
    actual_mime = _sniff_mime_from_bytes(payload)
    if actual_mime not in _IMAGE_EXTENSIONS:
        if strict:
            raise ValueError("invalid_image_data:Image data URL bytes are not a supported image format.")
        return None
    return payload, actual_mime


def validate_inline_image_data_url(url: str) -> None:
    """Raise a request-normalization error for malformed or unsupported inline image bytes."""
    _decode_inline_image(url, strict=True)


def _store_image_bytes(payload: bytes, actual_mime: str) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    # This is the canonical host-side media root.  The authenticated dashboard media
    # endpoint, remote-backend vision resolver, TUI and Desktop all already allow it.
    media_dir = Path(get_hermes_home()) / "images"
    media_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(media_dir, 0o700)
    target = media_dir / f"api_{digest}{_IMAGE_EXTENSIONS[actual_mime]}"
    if not target.exists():
        temporary = media_dir / f".{target.name}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_bytes(payload)
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    os.chmod(target, 0o600)
    return str(target)


def _materialize_image(url: str) -> str | None:
    decoded = _decode_inline_image(url)
    if decoded is not None:
        return _store_image_bytes(*decoded)
    if not url.lower().startswith(("http://", "https://")):
        return None

    from gateway.platforms.base import validate_inbound_media_size
    from model_tools import _run_async
    from tools.image_source import ResolveContext, resolve_image_source

    try:
        resolved = _run_async(resolve_image_source(url, ResolveContext()))
        validate_inbound_media_size(len(resolved.data), media_type="image")
    except Exception as exc:
        raise ValueError(f"invalid_image_url:Unable to retrieve a safe image URL: {exc}") from exc
    if resolved.mime not in _IMAGE_EXTENSIONS:
        raise ValueError("invalid_image_url:Remote URL did not resolve to a supported image format.")
    return _store_image_bytes(resolved.data, resolved.mime)


def validate_remote_image_url(url: str) -> None:
    """Reject private or policy-blocked remote image URLs before the request is accepted."""
    from tools.image_source import _http_block_reason

    reason = _http_block_reason(url)
    if reason:
        raise ValueError(f"invalid_image_url:{reason}")


def durable_api_user_message(content: Any) -> Any | None:
    """Return a persistence override with durable refs for native image parts.

    The live provider payload is deliberately unchanged.  The returned value keeps a list shape
    so the existing persistence guard accepts it as a replacement for list content; its sole text
    part carries paths that survive the SQLite projection and process restarts.
    """
    if not isinstance(content, list):
        return None

    text_parts: list[str] = []
    references: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"text", "input_text", "output_text"}:
            text_parts.append(str(part.get("text") or ""))
            continue
        if part_type not in {"image", "image_url", "input_image"}:
            continue
        url = _image_url(part)
        if not url:
            continue
        path = _materialize_image(url)
        if path:
            from agent.context_references import format_reference_value
            references.append(f"@image:{format_reference_value(path)}")

    if not references:
        return None
    visible_text = "\n".join(text_parts).strip()
    durable_text = "\n\n".join(value for value in (visible_text, "\n".join(references)) if value)
    # Keep the list shape so the multimodal persist-override guard accepts it, but omit the
    # image parts: the live user message still carries them while SQLite receives only the
    # durable references (and therefore never regenerates a visible ``[screenshot]`` token).
    return [{"type": "text", "text": durable_text}]


def _persisted_image_paths(persist_user_message: Any) -> list[str]:
    """Local paths from the text-only persistence projection."""
    if not isinstance(persist_user_message, list) or not persist_user_message:
        return []
    text = persist_user_message[0].get("text") if isinstance(persist_user_message[0], dict) else ""
    if not isinstance(text, str):
        return []
    paths: list[str] = []
    for match in re.finditer(r"(?m)^\s*@image:(`[^`\n]+`|\"[^\"\n]+\"|'[^'\n]+'|\S+)\s*$", text):
        value = match.group(1)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "`\"'":
            value = value[1:-1]
        if Path(value).is_file() and value not in paths:
            paths.append(value)
    return paths


def _vision_descriptions(paths: list[str]) -> list[str]:
    """Pre-describe local images for a text-only main model; failure stays actionable."""
    from model_tools import _run_async
    from tools.vision_tools import vision_analyze_tool

    prompt = (
        "Describe everything visible in this user-attached image in thorough detail. "
        "Include text, code, data, objects, layout, and any consequential visual evidence."
    )
    descriptions: list[str] = []
    for path in paths:
        try:
            payload = json.loads(_run_async(vision_analyze_tool(image_url=path, user_prompt=prompt)))
            analysis = str(payload.get("analysis") or "").strip() if payload.get("success") else ""
        except Exception:
            analysis = ""
        if analysis:
            descriptions.append(
                f"[The user attached an image. Its visual contents are:\n{analysis}]\n"
                f"[Durable image path for a closer look: {path}]"
            )
        else:
            descriptions.append(
                "[The user attached an image. Automatic visual analysis was unavailable. "
                f"Use vision_analyze with image_url: {path} before answering.]"
            )
    return descriptions


def route_api_user_message(agent: Any, content: Any) -> tuple[Any, Any | None]:
    """Route inline API images for the active model and return its durable transcript override.

    Vision-capable models keep the original native parts. Text-only or unknown models receive an
    auxiliary-vision description plus the durable local path, avoiding a provider-level multimodal
    rejection while preserving the actual image for reinspection and history.
    """
    persist_user_message = durable_api_user_message(content)
    paths = _persisted_image_paths(persist_user_message)
    if not paths:
        return content, persist_user_message
    runtime = getattr(agent, "_hermes_api_runtime", {}) or {}
    try:
        from agent.image_routing import decide_image_input_mode
        from hermes_cli.config import load_config_readonly

        mode = decide_image_input_mode(
            str(getattr(agent, "provider", "") or runtime.get("provider") or ""),
            str(getattr(agent, "model", "") or runtime.get("model") or ""),
            load_config_readonly(),
            requested_provider=str(runtime.get("requested_provider") or ""),
        )
    except Exception:
        mode = "text"
    native_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    if mode == "native" and all(Path(path).suffix.lower() in native_suffixes for path in paths):
        return content, persist_user_message

    text = "\n".join(
        str(part.get("text") or "")
        for part in content
        if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}
    ).strip()
    from agent.auxiliary_client import scoped_runtime_main
    provider = str(getattr(agent, "provider", "") or runtime.get("provider") or "")
    model = str(getattr(agent, "model", "") or runtime.get("model") or "")
    runtime_scope = {
        "provider": provider,
        "model": model,
        "requested_provider": str(runtime.get("requested_provider") or getattr(agent, "requested_provider", "") or ""),
        "base_url": str(getattr(agent, "base_url", "") or ""),
        "api_key": str(getattr(agent, "api_key", "") or ""),
        "api_mode": str(getattr(agent, "api_mode", "") or ""),
        "auth_mode": str(getattr(agent, "auth_mode", "") or ""),
    }
    with scoped_runtime_main(runtime_scope):
        descriptions = _vision_descriptions(paths)
    routed = "\n\n".join([*descriptions, text] if text else descriptions)
    return routed or text, persist_user_message
