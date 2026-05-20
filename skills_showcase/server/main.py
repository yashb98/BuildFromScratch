"""HTTP surface for the tool execution bridge.

Two modes:

1. Direct tool invocation (no API key needed):
     POST /tools/<name>   with JSON body matching the tool's input_schema
   Useful from curl, Claude CLI, or any external driver.

2. Agent loop (requires ANTHROPIC_API_KEY):
     POST /agent/run      starts a session against a skill's SKILL.md as system prompt
     GET  /agent/stream/<session_id>   SSE stream of events
     POST /agent/respond  send user reply / approval / ask_user answer back

Static files (the showcase page) are served from the parent directory.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from safety import WORKSPACE_ROOT, classify_command
from tools import HANDLERS, INTERACTIVE, REQUIRES_APPROVAL, TOOL_SCHEMAS

SHOWCASE_DIR = Path(__file__).resolve().parent.parent
SKILL_FILES = {
    "from-scratch-build": SHOWCASE_DIR / "from-scratch-build.md",
    "finance-research-loop": SHOWCASE_DIR / "finance-research-loop.md",
}

app = FastAPI(title="BuildFromScratch · Tool Bridge", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- meta ----------

@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "workspace": str(WORKSPACE_ROOT),
        "tools": [t["name"] for t in TOOL_SCHEMAS],
        "anthropic_key_present": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "skills": list(SKILL_FILES.keys()),
    }


@app.get("/api/tools")
def list_tools() -> dict[str, Any]:
    return {
        "tools": TOOL_SCHEMAS,
        "requires_approval": sorted(REQUIRES_APPROVAL),
        "interactive": sorted(INTERACTIVE),
    }


# ---------- direct tool invocation ----------

class ToolCallRequest(BaseModel):
    input: dict[str, Any]


@app.post("/api/tools/{name}")
def run_tool(name: str, req: ToolCallRequest) -> dict[str, Any]:
    handler = HANDLERS.get(name)
    if handler is None:
        raise HTTPException(404, f"unknown tool: {name}")
    started = time.monotonic()
    classification = None
    if name == "bash" and "command" in req.input:
        c = classify_command(req.input["command"])
        classification = {"kind": c.kind, "summary": c.summary, "requires_approval": c.requires_approval}
    try:
        result = handler(**req.input)
    except TypeError as e:
        return {"ok": False, "error": f"bad input: {e}"}
    return {
        "tool": name,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
        "classification": classification,
        **result,
    }


# ---------- agent loop ----------

class AgentRunRequest(BaseModel):
    skill: str
    message: str
    model: str = "claude-opus-4-7"


class AgentRespondRequest(BaseModel):
    session_id: str
    # Exactly one of these is set:
    user_message: str | None = None
    approve_tool_use_id: str | None = None       # approval for write_file/bash
    deny_tool_use_id: str | None = None
    deny_reason: str | None = None
    ask_user_tool_use_id: str | None = None      # response to an ask_user pause
    ask_user_answer: str | None = None


# Session state lives in-process. One process, one user — fine for localhost.
SESSIONS: dict[str, "AgentSession"] = {}


class AgentSession:
    def __init__(self, session_id: str, skill_md: str, model: str):
        self.id = session_id
        self.skill_md = skill_md
        self.model = model
        self.messages: list[dict[str, Any]] = []
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.pending_tool_use: dict[str, dict[str, Any]] = {}   # id -> tool_use block
        self.pending_results: list[dict[str, Any]] = []         # tool_result blocks ready to send back
        self.waiting_event = asyncio.Event()
        self.closed = False
        self._lock = asyncio.Lock()

    async def emit(self, event: dict[str, Any]) -> None:
        await self.queue.put(event)

    async def close(self) -> None:
        self.closed = True
        await self.queue.put({"type": "session_end"})


def _load_skill(slug: str) -> str:
    path = SKILL_FILES.get(slug)
    if path is None or not path.exists():
        raise HTTPException(404, f"unknown skill: {slug}")
    text = path.read_text(encoding="utf-8")
    # Strip YAML frontmatter so the system prompt is pure instructions.
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2].lstrip()
    return text


@app.post("/api/agent/run")
async def agent_run(req: AgentRunRequest) -> dict[str, Any]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            503,
            "ANTHROPIC_API_KEY is not set. Use the direct /api/tools/<name> endpoints "
            "to drive the tool layer from Claude CLI, or export the key and retry.",
        )
    skill_md = _load_skill(req.skill)
    session_id = f"sess_{int(time.time() * 1000)}"
    sess = AgentSession(session_id, skill_md, req.model)
    sess.messages.append({"role": "user", "content": req.message})
    SESSIONS[session_id] = sess
    asyncio.create_task(_drive_agent(sess))
    return {"session_id": session_id}


@app.get("/api/agent/stream/{session_id}")
async def agent_stream(session_id: str) -> StreamingResponse:
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(404, "no such session")

    async def gen():
        while True:
            event = await sess.queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") == "session_end":
                break

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/agent/respond")
async def agent_respond(req: AgentRespondRequest) -> dict[str, Any]:
    sess = SESSIONS.get(req.session_id)
    if sess is None:
        raise HTTPException(404, "no such session")
    async with sess._lock:
        if req.approve_tool_use_id:
            tu = sess.pending_tool_use.pop(req.approve_tool_use_id, None)
            if tu is None:
                raise HTTPException(400, "no pending tool_use with that id")
            handler = HANDLERS[tu["name"]]
            try:
                result = handler(**tu["input"])
            except TypeError as e:
                result = {"ok": False, "error": f"bad input: {e}"}
            await sess.emit({"type": "tool_result", "tool_use_id": tu["id"], "name": tu["name"], "result": result})
            sess.pending_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": json.dumps(result),
                "is_error": not result.get("ok", False),
            })
        elif req.deny_tool_use_id:
            tu = sess.pending_tool_use.pop(req.deny_tool_use_id, None)
            if tu is None:
                raise HTTPException(400, "no pending tool_use with that id")
            reason = req.deny_reason or "denied by user"
            await sess.emit({"type": "tool_denied", "tool_use_id": tu["id"], "reason": reason})
            sess.pending_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": f"User denied this tool call: {reason}",
                "is_error": True,
            })
        elif req.ask_user_tool_use_id:
            tu = sess.pending_tool_use.pop(req.ask_user_tool_use_id, None)
            if tu is None:
                raise HTTPException(400, "no pending ask_user with that id")
            answer = req.ask_user_answer or ""
            await sess.emit({"type": "ask_user_answered", "tool_use_id": tu["id"], "answer": answer})
            sess.pending_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": answer,
                "is_error": False,
            })
        elif req.user_message:
            sess.messages.append({"role": "user", "content": req.user_message})
        else:
            raise HTTPException(400, "no actionable field in body")
        sess.waiting_event.set()
    return {"ok": True}


async def _drive_agent(sess: AgentSession) -> None:
    """The agent loop. Runs until end_turn or session closed."""
    try:
        from anthropic import Anthropic       # noqa: WPS433 (deferred so health route works without the dep)
    except ImportError:
        await sess.emit({"type": "error", "error": "anthropic SDK not installed; pip install -r server/requirements.txt"})
        await sess.close()
        return

    client = Anthropic()
    await sess.emit({"type": "agent_started", "session_id": sess.id, "model": sess.model})

    try:
        while not sess.closed:
            # Drain any tool results the UI sent us into the messages array.
            if sess.pending_results:
                sess.messages.append({"role": "user", "content": list(sess.pending_results)})
                sess.pending_results.clear()

            await sess.emit({"type": "model_request_started"})
            try:
                response = client.messages.create(
                    model=sess.model,
                    max_tokens=8192,
                    system=[{"type": "text", "text": sess.skill_md, "cache_control": {"type": "ephemeral"}}],
                    tools=TOOL_SCHEMAS,
                    messages=sess.messages,
                )
            except Exception as e:
                await sess.emit({"type": "error", "error": f"messages.create failed: {e}"})
                break

            await sess.emit({
                "type": "model_response",
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_read": getattr(response.usage, "cache_read_input_tokens", 0),
                    "cache_write": getattr(response.usage, "cache_creation_input_tokens", 0),
                },
            })

            sess.messages.append({"role": "assistant", "content": [_block_to_dict(b) for b in response.content]})

            tool_uses: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "text":
                    await sess.emit({"type": "assistant_text", "text": block.text})
                elif block.type == "tool_use":
                    tu = {"id": block.id, "name": block.name, "input": dict(block.input)}
                    tool_uses.append(tu)
                    await sess.emit({"type": "tool_use", **tu})

            if response.stop_reason != "tool_use" or not tool_uses:
                await sess.emit({"type": "turn_complete"})
                break  # waits for next user message via /api/agent/respond loop below

            # Execute / gate each tool_use.
            for tu in tool_uses:
                name = tu["name"]
                if name in INTERACTIVE:
                    sess.pending_tool_use[tu["id"]] = tu
                    await sess.emit({"type": "awaiting_user", "tool_use_id": tu["id"], "name": name, "input": tu["input"]})
                    await _wait_until_resolved(sess, tu["id"])
                elif name in REQUIRES_APPROVAL:
                    classification = None
                    if name == "bash":
                        c = classify_command(tu["input"].get("command", ""))
                        classification = {"kind": c.kind, "summary": c.summary, "auto_approvable": not c.requires_approval}
                    sess.pending_tool_use[tu["id"]] = tu
                    await sess.emit({
                        "type": "awaiting_approval",
                        "tool_use_id": tu["id"],
                        "name": name,
                        "input": tu["input"],
                        "classification": classification,
                    })
                    await _wait_until_resolved(sess, tu["id"])
                else:
                    handler = HANDLERS[name]
                    try:
                        result = handler(**tu["input"])
                    except TypeError as e:
                        result = {"ok": False, "error": f"bad input: {e}"}
                    await sess.emit({"type": "tool_result", "tool_use_id": tu["id"], "name": name, "result": result})
                    sess.pending_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": json.dumps(result),
                        "is_error": not result.get("ok", False),
                    })
    finally:
        await sess.close()


async def _wait_until_resolved(sess: AgentSession, tool_use_id: str) -> None:
    while tool_use_id in sess.pending_tool_use and not sess.closed:
        sess.waiting_event.clear()
        await sess.waiting_event.wait()


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Convert an SDK content block to the dict shape `messages.create` accepts."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    # Fall back to model_dump for anything else (thinking blocks, etc.).
    return block.model_dump()


# ---------- static files ----------
# Serve the showcase page + skill markdown from the parent directory.

@app.get("/")
def root() -> FileResponse:
    return FileResponse(SHOWCASE_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(SHOWCASE_DIR)), name="static")


@app.get("/{filename}")
def serve_static(filename: str) -> FileResponse:
    target = (SHOWCASE_DIR / filename).resolve()
    try:
        target.relative_to(SHOWCASE_DIR)
    except ValueError:
        raise HTTPException(404)
    if not target.is_file():
        raise HTTPException(404)
    return FileResponse(target)
