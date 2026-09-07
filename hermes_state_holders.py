"""Process and descriptor authority for state.db structural maintenance.

This module owns the proof that no foreign process still holds the active or
an unlinked SQLite DB/WAL/SHM generation.  ``hermes_state`` supplies only the
SQLite connection factory needed by the final lock probe.
"""

from __future__ import annotations

import errno
import logging
import os
import sqlite3
import stat
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Set, Tuple

try:  # Hard dependency, but tolerate scaffold-phase imports before pip install.
    import psutil
except ImportError:  # pragma: no cover - stripped/scaffold installs only
    psutil = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"
_HERMES_EXECUTABLES = frozenset({"hermes", "hermes-agent", "hermes-acp"})
_HERMES_PYTHON_MODULES = frozenset({"acp_adapter", "hermes_cli.main"})
_HERMES_PYTHON_SCRIPTS = frozenset({"hermes_cli/main.py", "run_agent.py"})
_PYTHON_SHORT_OPTIONS_WITH_OPERANDS = frozenset({"Q", "W", "X"})
_PYTHON_LONG_OPTIONS_WITH_OPERANDS = frozenset(
    {"--check-hash-based-pycs", "--jit"}
)


def _read_proc_argv(pid: int) -> Optional[List[str]]:
    """Read /proc/<pid>/cmdline without losing argv boundaries."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read()
        if not raw:
            return None
        argv = raw.decode("utf-8", "replace").split("\x00")
        if argv[-1] == "":
            argv.pop()
        return argv or None
    except OSError:
        return None


def _looks_like_python_executable(program: str) -> bool:
    name = os.path.basename(program).lower().removesuffix(".exe")
    for prefix in ("python", "pypy"):
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            return not suffix or all(char.isdigit() or char == "." for char in suffix)
    return False


def _python_execution_target(argv: Sequence[str]) -> Optional[Tuple[str, str]]:
    """Return the Python module or script selected by interpreter options."""
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--":
            index += 1
            return ("script", argv[index]) if index < len(argv) else None
        if arg in _PYTHON_LONG_OPTIONS_WITH_OPERANDS:
            index += 2
            continue
        if arg.startswith("--check-hash-based-pycs=") or arg.startswith("--jit="):
            index += 1
            continue
        if arg.startswith("--"):
            index += 1
            continue
        if arg.startswith("-") and arg != "-":
            options = arg[1:]
            option_index = 0
            consumed_next = False
            while option_index < len(options):
                option = options[option_index]
                attached = options[option_index + 1 :]
                if option == "c":
                    return None
                if option == "m":
                    if attached:
                        return "module", attached
                    index += 1
                    return ("module", argv[index]) if index < len(argv) else None
                if option in _PYTHON_SHORT_OPTIONS_WITH_OPERANDS:
                    consumed_next = not attached
                    break
                option_index += 1
            index += 2 if consumed_next else 1
            continue
        return "script", arg
    return None


def _looks_like_hermes(argv: Sequence[str]) -> bool:
    """Return whether argv identifies a supported Hermes execution target."""
    if not argv:
        return False
    program = os.path.basename(argv[0]).lower().removesuffix(".exe")
    if program in _HERMES_EXECUTABLES:
        return True
    if not _looks_like_python_executable(program):
        return False
    target = _python_execution_target(argv)
    if target is None:
        return False
    kind, value = target
    normalized = value.lower().replace("\\", "/")
    if kind == "module":
        return normalized in _HERMES_PYTHON_MODULES
    return any(
        normalized == script or normalized.endswith(f"/{script}")
        for script in _HERMES_PYTHON_SCRIPTS
    )


def canonical_sqlite_path(path: str) -> str:
    """Normalize a /proc fd target, stripping the Linux `` (deleted)`` suffix."""
    return os.path.normcase(os.path.abspath(path.removesuffix(" (deleted)")))


def _read_proc_credentials(pid: int) -> Optional[Tuple[int, Set[int]]]:
    """Return a Linux process's effective uid and supplementary groups."""
    try:
        uid: Optional[int] = None
        groups: Set[int] = set()
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Uid:"):
                    values = line.split()[1:]
                    if len(values) >= 2:
                        uid = int(values[1])
                elif line.startswith("Groups:"):
                    groups = {int(value) for value in line.split()[1:]}
        return (uid, groups) if uid is not None else None
    except (OSError, ValueError):
        return None


def _mode_allows(
    mode: int,
    *,
    owner_uid: int,
    owner_gid: int,
    process_uid: int,
    process_groups: Set[int],
    directory: bool,
) -> bool:
    """Evaluate ordinary POSIX access bits for a foreign process identity."""
    if process_uid == owner_uid:
        bits = (mode >> 6) & 0b111
    elif owner_gid in process_groups:
        bits = (mode >> 3) & 0b111
    else:
        bits = mode & 0b111
    required = 0b001 if directory else 0b110
    return bool(bits & required)


def _process_may_access_watched_paths(pid: int, paths: Sequence[str]) -> bool:
    """Fail closed unless a different uid is fenced by current path permissions.

    On ``hidepid`` hosts another user's ``/proc/<pid>/fd`` can be unreadable.
    Treating that process as a holder of every database on the machine makes
    structural maintenance impossible.  A foreign non-root uid can be ignored
    only when an ordinary POSIX permission check proves it cannot traverse or
    open any existing watched path.  Same-uid, root, ACL-bearing, and otherwise
    unprovable cases still block.
    """
    credentials = _read_proc_credentials(pid)
    if credentials is None:
        return True
    process_uid, process_groups = credentials
    if process_uid in (0, os.geteuid()):
        return True

    for raw_path in paths:
        path = Path(raw_path)
        try:
            path.stat()
        except FileNotFoundError:
            continue
        except OSError:
            return True

        chain = [*reversed(path.parents), path]
        accessible = True
        for node in chain:
            try:
                node_stat = node.stat()
                if hasattr(os, "listxattr") and "system.posix_acl_access" in os.listxattr(node):
                    return True
            except OSError:
                return True
            if not _mode_allows(
                node_stat.st_mode,
                owner_uid=node_stat.st_uid,
                owner_gid=node_stat.st_gid,
                process_uid=process_uid,
                process_groups=process_groups,
                directory=stat.S_ISDIR(node_stat.st_mode),
            ):
                accessible = False
                break
        if accessible:
            return True
    return False


def foreign_state_db_holders(db_path: Path) -> List[Tuple[int, str]]:
    """Return foreign holders of the DB or one of its WAL sidecars.

    A scan failure is represented as an unknown holder. Structural maintenance
    must not assume quiescence when an old, unlinked SQLite generation may
    still be open by another process.
    """
    if _IS_WINDOWS:
        return []

    db_path_str = os.path.abspath(os.fspath(db_path))
    watched = {
        canonical_sqlite_path(db_path_str),
        canonical_sqlite_path(db_path_str + "-wal"),
        canonical_sqlite_path(db_path_str + "-shm"),
    }
    holders: List[Tuple[int, str]] = []
    watched_ids: Set[Tuple[int, int]] = set()
    db_dev: Optional[int] = None
    for candidate in (db_path_str, db_path_str + "-wal", db_path_str + "-shm"):
        try:
            stat_result = os.stat(candidate)
        except OSError as exc:
            if exc.errno not in (errno.ENOENT, errno.ESRCH):
                holders.append(
                    (-1, f"watched-file stat failed: {candidate}: {exc}")
                )
            continue
        watched_ids.add((stat_result.st_dev, stat_result.st_ino))
        if candidate == db_path_str:
            db_dev = stat_result.st_dev

    if sys.platform.startswith("linux"):
        try:
            own_pid = os.getpid()
            for pid_str in os.listdir("/proc"):
                if not pid_str.isdigit():
                    continue
                pid = int(pid_str)
                if pid == own_pid:
                    continue
                fd_dir = f"/proc/{pid}/fd"
                try:
                    fds = os.listdir(fd_dir)
                except OSError:
                    argv = _read_proc_argv(pid)
                    if (
                        argv is not None
                        and _looks_like_hermes(argv)
                        and _process_may_access_watched_paths(pid, tuple(watched))
                    ):
                        cmdline = " ".join(argv)
                        holders.append((pid, f"uninspectable holder: {cmdline[:80]}"))
                    continue
                for fd in fds:
                    fd_path = f"{fd_dir}/{fd}"
                    try:
                        target = os.readlink(fd_path)
                    except OSError as exc:
                        if exc.errno in (errno.ENOENT, errno.ESRCH):
                            continue
                        argv = _read_proc_argv(pid)
                        if (
                            argv is not None
                            and _looks_like_hermes(argv)
                            and _process_may_access_watched_paths(pid, tuple(watched))
                        ):
                            holders.append(
                                (
                                    pid,
                                    f"uninspectable descriptor: {fd_path}: {exc}",
                                )
                            )
                        continue
                    target_is_watched = canonical_sqlite_path(target) in watched
                    try:
                        fd_stat = os.stat(fd_path)
                    except OSError as exc:
                        if exc.errno in (errno.ENOENT, errno.ESRCH):
                            continue
                        if target_is_watched:
                            holders.append(
                                (pid, f"uninspectable descriptor: {target}: {exc}")
                            )
                        else:
                            argv = _read_proc_argv(pid)
                            if (
                                argv is not None
                                and _looks_like_hermes(argv)
                                and _process_may_access_watched_paths(pid, tuple(watched))
                            ):
                                holders.append(
                                    (
                                        pid,
                                        "uninspectable descriptor: "
                                        f"{target}: {exc}",
                                    )
                                )
                        continue
                    if (fd_stat.st_dev, fd_stat.st_ino) in watched_ids or (
                        target_is_watched
                        and target.endswith(" (deleted)")
                        and db_dev is not None
                        and fd_stat.st_dev == db_dev
                    ):
                        holders.append((pid, target))
        except Exception as exc:
            logger.warning(
                "Could not prove state.db has no foreign holders; "
                "deferring structural maintenance: %s",
                exc,
            )
            holders.append((-1, f"open-file scan failed: {exc}"))
        return holders

    if psutil is None:
        return [(-1, "open-file scan unavailable")]
    try:
        for process in psutil.process_iter(["pid", "open_files"]):
            info = process.info
            pid = int(info["pid"])
            if pid == os.getpid():
                continue
            for opened in info.get("open_files") or ():
                path = getattr(opened, "path", "")
                if path and canonical_sqlite_path(path) in watched:
                    holders.append((pid, path))
    except Exception as exc:
        logger.warning(
            "Could not prove state.db has no foreign holders; "
            "deferring structural maintenance: %s",
            exc,
        )
        holders.append((-1, f"open-file scan failed: {exc}"))
    return holders


def live_writer_holds_db(
    db_path: Path,
    *,
    connect_repair_durable: Callable[..., sqlite3.Connection],
) -> bool:
    """Return whether repair lacks proven exclusive ownership of ``db_path``."""
    foreign_holders = foreign_state_db_holders(db_path)
    if any(
        pid < 0
        or path.startswith("uninspectable holder:")
        or path.startswith("uninspectable descriptor:")
        or path.endswith(" (deleted)")
        for pid, path in foreign_holders
    ):
        return True

    probe = None
    try:
        probe = connect_repair_durable(db_path, timeout=0.0)
        probe.execute("PRAGMA locking_mode=EXCLUSIVE")
        probe.execute("BEGIN IMMEDIATE")
        probe.execute("ROLLBACK")
        return False
    except sqlite3.OperationalError as exc:
        lowered = str(exc).lower()
        return "locked" in lowered or "busy" in lowered
    except sqlite3.DatabaseError:
        return False
    except Exception:
        return False
    finally:
        if probe is not None:
            try:
                probe.execute("PRAGMA locking_mode=NORMAL")
            except Exception:
                pass
            try:
                probe.close()
            except Exception:
                pass
