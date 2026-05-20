"""Path scoping + command policy for the tool layer.

Every tool that touches the filesystem goes through `resolve_in_workspace`,
which rejects paths that escape WORKSPACE_ROOT. Bash commands go through
`classify_command` so the frontend can show a clear approval prompt.
"""
from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

WORKSPACE_ROOT = Path(
    os.environ.get("BFS_WORKSPACE", "/home/yashb98/Downloads/BuildFromScratch")
).resolve()


class WorkspaceViolation(Exception):
    pass


def resolve_in_workspace(path: str) -> Path:
    """Resolve `path` (absolute or relative to WORKSPACE_ROOT) and ensure it stays inside."""
    raw = Path(path)
    candidate = (raw if raw.is_absolute() else WORKSPACE_ROOT / raw).resolve()
    try:
        candidate.relative_to(WORKSPACE_ROOT)
    except ValueError as e:
        raise WorkspaceViolation(
            f"path {candidate} is outside workspace {WORKSPACE_ROOT}"
        ) from e
    return candidate


def short_path(p: Path) -> str:
    try:
        return str(p.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(p)


@dataclass
class CommandClassification:
    kind: str           # "read" | "write" | "exec" | "network" | "unknown"
    summary: str        # one-line human-readable description
    requires_approval: bool


_READ_ONLY_HEADS = {
    "ls", "cat", "head", "tail", "wc", "file", "stat",
    "pwd", "echo", "which", "whoami", "uname", "date",
    "find", "grep", "rg", "tree", "du", "df",
}
_WRITE_HEADS = {"mkdir", "touch", "cp", "mv", "tee", "dd", "chmod", "chown", "ln"}
_DANGER_HEADS = {"rm", "rmdir", "shred", "kill", "pkill", "killall", "reboot", "shutdown"}
_NETWORK_HEADS = {"curl", "wget", "pip", "uv", "npm", "yarn", "pnpm", "apt", "apt-get", "ssh", "scp", "rsync", "git"}


def classify_command(command: str) -> CommandClassification:
    """Inspect a command string and decide whether it needs approval."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return CommandClassification("unknown", command, requires_approval=True)
    if not tokens:
        return CommandClassification("unknown", command, requires_approval=True)
    head = os.path.basename(tokens[0])
    if head in _DANGER_HEADS:
        return CommandClassification("exec", f"destructive: {head}", True)
    if head in _NETWORK_HEADS:
        return CommandClassification("network", f"network: {head}", True)
    if head in _WRITE_HEADS:
        return CommandClassification("write", f"writes filesystem: {head}", True)
    if head in _READ_ONLY_HEADS:
        return CommandClassification("read", f"read-only: {head}", False)
    return CommandClassification("unknown", head, requires_approval=True)
