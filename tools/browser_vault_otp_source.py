"""Opt-in server-side OTP command sources, bound to saved login metadata.

The command is operator configuration, never a model-supplied tool argument.
Its stdout is a secret channel; only the native vault fill consumes the result.
"""
from __future__ import annotations

import json
import subprocess
import time


class OTPSourceError(RuntimeError):
    """Sanitized source failure; never include command output or exceptions."""


def resolve_configured_otp(meta, origin: str, *, config=None, runner=None) -> str | None:
    if meta is None or meta.kind != "login" or meta.origin != origin:
        return None
    if config is None:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    sources = (config.get("vault") or {}).get("otp_sources") or []
    matched = [s for s in sources if isinstance(s, dict)
               and s.get("origin") == origin
               and s.get("identifier") == meta.identifier
               and s.get("enabled") is True]
    if not matched:
        return None
    if len(matched) != 1:
        raise OTPSourceError("Multiple OTP sources match this login; select one in vault configuration.")
    command = matched[0].get("command")
    if (not isinstance(command, list) or not command or
            not all(isinstance(a, str) and a for a in command) or
            not command[0].startswith("/")):
        raise OTPSourceError("OTP source requires an absolute executable and an argument list.")
    if runner is None:
        runner = subprocess.run
    try:
        from hermes_constants import get_hermes_home
        from tools.environments.local import served_profile_child_env
        proc = runner(command, capture_output=True, text=True, timeout=30,
                      stdin=subprocess.DEVNULL,
                      env=served_profile_child_env(target_home=get_hermes_home(), inherit_credentials=False))
        if proc.returncode != 0 or len(proc.stdout) > 2048:
            raise ValueError("source_failed")
        data = json.loads(proc.stdout)
        code = data["code"]
        issued = data["issued_at"]
        if (not isinstance(code, str) or not code.isascii() or not code.isdigit()
                or not 4 <= len(code) <= 10 or isinstance(issued, bool)
                or not isinstance(issued, (float, int))
                or not 0 <= time.time() - issued <= 180):
            raise ValueError("invalid_or_expired")
        return code
    except Exception:
        raise OTPSourceError("The configured OTP source failed or returned an expired/invalid code.") from None
