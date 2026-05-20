"""Tool registry: schemas (for Claude tool_use) + Python handlers.

Each tool has:
  - `SCHEMA`: the JSON Schema dict Claude sees
  - a handler function `handle_<name>(**input) -> dict` returning {ok, output, ...}

`TOOL_SCHEMAS` is the list passed to `client.messages.create(tools=...)`.
`HANDLERS` maps tool name -> handler.
`REQUIRES_APPROVAL` marks tools that the agent loop must pause on before executing.
"""
from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path
from typing import Any

from safety import (
    WORKSPACE_ROOT,
    WorkspaceViolation,
    classify_command,
    resolve_in_workspace,
    short_path,
)

MAX_READ_BYTES = 200_000
MAX_GREP_MATCHES = 200
MAX_LIST_ENTRIES = 500
MAX_BASH_OUTPUT_BYTES = 64_000
BASH_TIMEOUT_SEC = 60


# ---------- read_file ----------

def handle_read_file(path: str, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
    try:
        p = resolve_in_workspace(path)
    except WorkspaceViolation as e:
        return {"ok": False, "error": str(e)}
    if not p.exists():
        return {"ok": False, "error": f"no such file: {short_path(p)}"}
    if not p.is_file():
        return {"ok": False, "error": f"not a file: {short_path(p)}"}
    try:
        data = p.read_bytes()
    except Exception as e:
        return {"ok": False, "error": f"read failed: {e}"}
    truncated = False
    if len(data) > MAX_READ_BYTES:
        data = data[:MAX_READ_BYTES]
        truncated = True
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"ok": False, "error": f"file is not utf-8 text: {short_path(p)}"}
    lines = text.splitlines()
    if limit is not None:
        end = offset + int(limit)
        slice_ = lines[int(offset):end]
    else:
        slice_ = lines[int(offset):]
    out = "\n".join(slice_)
    return {
        "ok": True,
        "output": out,
        "path": short_path(p),
        "total_lines": len(lines),
        "truncated_bytes": truncated,
    }


# ---------- write_file ----------

def handle_write_file(path: str, content: str) -> dict[str, Any]:
    try:
        p = resolve_in_workspace(path)
    except WorkspaceViolation as e:
        return {"ok": False, "error": str(e)}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except Exception as e:
        return {"ok": False, "error": f"write failed: {e}"}
    return {
        "ok": True,
        "output": f"wrote {len(content)} bytes to {short_path(p)}",
        "path": short_path(p),
        "bytes": len(content.encode("utf-8")),
    }


# ---------- list_dir ----------

def handle_list_dir(path: str = ".") -> dict[str, Any]:
    try:
        p = resolve_in_workspace(path)
    except WorkspaceViolation as e:
        return {"ok": False, "error": str(e)}
    if not p.exists():
        return {"ok": False, "error": f"no such directory: {short_path(p)}"}
    if not p.is_dir():
        return {"ok": False, "error": f"not a directory: {short_path(p)}"}
    entries = []
    try:
        for child in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            kind = "dir" if child.is_dir() else "file"
            try:
                size = child.stat().st_size if kind == "file" else None
            except OSError:
                size = None
            entries.append({"name": child.name, "kind": kind, "size": size})
            if len(entries) >= MAX_LIST_ENTRIES:
                break
    except PermissionError as e:
        return {"ok": False, "error": f"permission denied: {e}"}
    lines = [
        f"{e['kind']:>4}  {e['size'] if e['size'] is not None else '':>10}  {e['name']}"
        for e in entries
    ]
    return {"ok": True, "output": "\n".join(lines), "path": short_path(p), "count": len(entries)}


# ---------- glob ----------

def handle_glob(pattern: str, path: str = ".") -> dict[str, Any]:
    try:
        base = resolve_in_workspace(path)
    except WorkspaceViolation as e:
        return {"ok": False, "error": str(e)}
    if not base.is_dir():
        return {"ok": False, "error": f"not a directory: {short_path(base)}"}
    matches: list[str] = []
    for root, _dirs, files in _walk(base):
        for name in files:
            full = Path(root) / name
            try:
                rel = full.relative_to(base)
            except ValueError:
                continue
            if fnmatch.fnmatch(str(rel), pattern) or fnmatch.fnmatch(name, pattern):
                matches.append(short_path(full))
                if len(matches) >= MAX_GREP_MATCHES:
                    break
        if len(matches) >= MAX_GREP_MATCHES:
            break
    return {"ok": True, "output": "\n".join(matches), "count": len(matches)}


# ---------- grep ----------

def handle_grep(pattern: str, path: str = ".", glob: str | None = None) -> dict[str, Any]:
    try:
        base = resolve_in_workspace(path)
    except WorkspaceViolation as e:
        return {"ok": False, "error": str(e)}
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return {"ok": False, "error": f"bad regex: {e}"}
    hits: list[str] = []
    targets: list[Path]
    if base.is_file():
        targets = [base]
    else:
        targets = []
        for root, _dirs, files in _walk(base):
            for name in files:
                full = Path(root) / name
                if glob and not (fnmatch.fnmatch(name, glob) or fnmatch.fnmatch(str(full.relative_to(base)), glob)):
                    continue
                targets.append(full)
    for f in targets:
        try:
            with f.open("r", encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    if regex.search(line):
                        hits.append(f"{short_path(f)}:{i}: {line.rstrip()}")
                        if len(hits) >= MAX_GREP_MATCHES:
                            return {"ok": True, "output": "\n".join(hits), "count": len(hits), "truncated": True}
        except (OSError, UnicodeDecodeError):
            continue
    return {"ok": True, "output": "\n".join(hits), "count": len(hits), "truncated": False}


# ---------- bash ----------

def handle_bash(command: str, cwd: str | None = None) -> dict[str, Any]:
    if cwd:
        try:
            workdir = resolve_in_workspace(cwd)
        except WorkspaceViolation as e:
            return {"ok": False, "error": str(e)}
    else:
        workdir = WORKSPACE_ROOT
    try:
        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {BASH_TIMEOUT_SEC}s"}
    except Exception as e:
        return {"ok": False, "error": f"exec failed: {e}"}

    stdout = (proc.stdout or "")[:MAX_BASH_OUTPUT_BYTES]
    stderr = (proc.stderr or "")[:MAX_BASH_OUTPUT_BYTES]
    return {
        "ok": proc.returncode == 0,
        "output": stdout,
        "stderr": stderr,
        "exit_code": proc.returncode,
        "cwd": short_path(workdir),
    }


# ---------- ask_user ----------
# Interactive: handled by the agent loop, not by this module. The schema is here
# so Claude can call it; the loop intercepts the tool_use and routes it to the UI.

def handle_ask_user(question: str, options: list[str] | None = None) -> dict[str, Any]:
    return {
        "ok": True,
        "output": "ask_user is handled by the agent loop; this is a no-op shim.",
        "question": question,
        "options": options or [],
    }


# ---------- internal helpers ----------

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"}


def _walk(base: Path):
    """Like os.walk but prunes vendor / cache directories."""
    import os
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        yield root, dirs, files


# ---------- tool schema list (Claude tool_use format) ----------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read a UTF-8 text file from the workspace. Use this before editing any file. "
            "Returns the file contents with line numbers implied by position."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root, or absolute under it."},
                "offset": {"type": "integer", "description": "0-based line offset to start reading from.", "default": 0},
                "limit": {"type": "integer", "description": "Max number of lines to return."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_dir",
        "description": "List the immediate contents of a directory under the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: workspace root).", "default": "."},
            },
        },
    },
    {
        "name": "glob",
        "description": "Find files whose path matches a glob pattern (e.g. '**/*.py'). Skips vendor dirs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern."},
                "path": {"type": "string", "description": "Base directory.", "default": "."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": "Search for a regex across files in the workspace. Returns up to 200 hits.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regex pattern."},
                "path": {"type": "string", "description": "File or directory to search.", "default": "."},
                "glob": {"type": "string", "description": "Optional filename glob filter, e.g. '*.py'."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write a text file in the workspace. REQUIRES USER APPROVAL. "
            "Always read_file first if the file exists, to avoid clobbering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "bash",
        "description": (
            "Run a bash command from the workspace. REQUIRES USER APPROVAL for anything "
            "that writes, networks, or has side effects. Timeout: 60s. Output truncated to 64KB."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "cwd": {"type": "string", "description": "Working directory under the workspace.", "default": "."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "ask_user",
        "description": (
            "Pause and ask the user a single question. Use this when the skill says to call "
            "AskUserQuestion: present 2-4 options as a multiple choice."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}, "description": "2-4 option labels."},
            },
            "required": ["question"],
        },
    },
]

HANDLERS = {
    "read_file": handle_read_file,
    "list_dir": handle_list_dir,
    "glob": handle_glob,
    "grep": handle_grep,
    "write_file": handle_write_file,
    "bash": handle_bash,
    "ask_user": handle_ask_user,
}

# Tools that require explicit user approval per call.
REQUIRES_APPROVAL = {"write_file", "bash"}

# Tools whose result must come from the user, not from the handler.
INTERACTIVE = {"ask_user"}
