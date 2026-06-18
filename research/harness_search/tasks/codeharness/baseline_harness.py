"""The INCUMBENT coding-agent harness — the control the search must beat. A
'harness' here is the PROMPT + CONTEXT + POST-PROCESS strategy that drives the
agent to write code (NOT the model). The search evolves THIS: it might add
few-shot examples, retrieval of similar solved tasks, a self-check/repair loop,
better output parsing, etc.

The agent call is INJECTED (`agent_fn(prompt) -> text`): in production it is the
real model; in tests it is a fake. That is what lets the deterministic reward
machinery (runner.py) be unit-tested NOW, while the agent-in-the-loop SEARCH waits
for the box/model.
"""

INCUMBENT = {
    "name": "baseline-zero-shot",
    "system": ("You are an expert Python programmer. Implement the requested "
               "function. Return ONLY the function definition — no prose, no "
               "markdown fences, no examples."),
    "context": "none",   # a searched harness may switch this to few-shot / retrieval / self-repair
}


def build_prompt(task, harness=INCUMBENT):
    """Render the agent prompt from a harness config + a task. The search may
    rewrite this entirely (the harness IS the prompt-construction code)."""
    return f"{harness['system']}\n\nComplete this function:\n{task['prompt']}\n"


def strip_fences(text):
    """Post-process the agent's raw text into runnable code (a harness concern —
    the search can improve this too). Strips ``` / ```python fences."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t[3:]
        if t.lower().startswith("python"):
            t = t[6:]
        if "```" in t:
            t = t[:t.index("```")]
    return t.strip()


def make_solve(agent_fn, harness=INCUMBENT):
    """solve(task) -> code, using `harness` and an injected agent_fn(prompt)->text.
    In production agent_fn calls the model; the search evolves `harness` /
    build_prompt / strip_fences to raise the held-out pass rate."""
    def solve(task):
        return strip_fences(agent_fn(build_prompt(task, harness)))
    return solve
