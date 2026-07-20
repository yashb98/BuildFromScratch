#!/usr/bin/env python3
"""Render every SKILL.md under .claude/skills/ into one self-contained HTML page.

Re-run any time a skill changes:  python3 research/build_skills_page.py
Output: research/skills_overview.html (open in any browser; works offline —
marked.js is inlined from research/.marked.min.js).
"""
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # BuildFromScratch/
SKILLS_DIR = ROOT / ".claude" / "skills"
OUT = ROOT / "research" / "skills_overview.html"
MARKED_JS = ROOT / "research" / ".marked.min.js"

GROUPS = [
    ("Orchestrator", ["research-loop"]),
    ("Scouting & research", ["community-pulse", "model-radar", "ml-research"]),
    ("Data & experiments", ["dataset-forge", "ablation-runner", "eval-harness"]),
    ("Memory, safety & reporting", ["experiment-ledger", "resource-sentinel", "weekly-retro"]),
    ("Pre-existing (interactive, approval-gated)", ["from-scratch-build", "finance-research-loop"]),
]
NEW_SKILLS = {n for g, names in GROUPS[:4] for n in names}
OWNED_SCRIPTS = {
    "experiment-ledger": [ROOT / "research" / "ledger" / "ledger.py"],
    "resource-sentinel": [ROOT / "sentinel.py"],
}


def parse_frontmatter(text):
    meta, body = {}, text
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if m:
        body = m.group(2)
        for line in m.group(1).splitlines():
            kv = re.match(r"^([\w-]+):\s*(.*)$", line)
            if kv:
                meta[kv.group(1)] = kv.group(2).strip().strip('"')
    return meta, body


def md_template(dom_id, markdown_text):
    """Embed raw markdown for client-side rendering by marked.js."""
    safe = markdown_text.replace("</script", "<\\/script")
    return f'<script type="text/template" id="{dom_id}">{safe}</script>\n<div class="md" data-src="{dom_id}"></div>'


def code_block(path):
    try:
        src = path.read_text()
    except OSError as e:
        src = f"(unreadable: {e})"
    return f"<pre class='codefile'><code>{html.escape(src)}</code></pre>"


sections, toc = [], []
missing = []
for group, names in GROUPS:
    toc.append(f"<div class='toc-group'>{html.escape(group)}</div>")
    for name in names:
        skill_md = SKILLS_DIR / name / "SKILL.md"
        badge = "<span class='badge new'>new</span>" if name in NEW_SKILLS else "<span class='badge old'>existing</span>"
        if not skill_md.exists():
            missing.append(name)
            toc.append(f"<a class='toc-link pending' href='#s-{name}'>{name}<span class='dot pend'></span></a>")
            sections.append(
                f"<section class='card pending' id='s-{name}'><div class='cardhead'><h2>/{name}</h2>{badge}"
                f"<span class='badge pend'>not written yet</span></div>"
                f"<p class='desc'>The authoring workflow has not finished this skill — regenerate the page once it lands.</p></section>"
            )
            continue

        meta, body = parse_frontmatter(skill_md.read_text())
        lines = skill_md.read_text().count("\n") + 1
        toc.append(f"<a class='toc-link' href='#s-{name}'>{name}<span class='dot ok'></span></a>")

        chips = [f"<span class='chip'>{lines} lines</span>",
                 f"<span class='chip path'>{skill_md.relative_to(ROOT)}</span>"]
        if meta.get("argument-hint"):
            chips.insert(0, f"<span class='chip args'>args: {html.escape(meta['argument-hint'])}</span>")

        extras = []
        refs_dir = SKILLS_DIR / name / "references"
        if refs_dir.is_dir():
            for ref in sorted(refs_dir.glob("*.md")):
                extras.append(
                    f"<details><summary>reference: {ref.name} ({ref.stat().st_size // 1024} KB)</summary>"
                    f"{md_template(f'ref-{name}-{ref.stem}', ref.read_text())}</details>"
                )
        scripts_dir = SKILLS_DIR / name / "scripts"
        if scripts_dir.is_dir():
            for s in sorted(scripts_dir.iterdir()):
                extras.append(f"<details><summary>script: scripts/{s.name}</summary>{code_block(s)}</details>")
        for s in OWNED_SCRIPTS.get(name, []):
            if s.exists():
                extras.append(f"<details><summary>owned script: {s.relative_to(ROOT)}</summary>{code_block(s)}</details>")

        desc = html.escape(meta.get("description", ""))
        sections.append(
            f"<section class='card' id='s-{name}'>"
            f"<div class='cardhead'><h2>/{name}</h2>{badge}</div>"
            f"<div class='chips'>{''.join(chips)}</div>"
            f"<p class='desc'>{desc}</p>"
            f"{md_template(f'md-{name}', body)}"
            f"{''.join(extras)}"
            f"</section>"
        )

banner = ""
if missing:
    banner = (f"<div class='banner'>⏳ The authoring workflow is still writing {len(missing)} skill(s): "
              f"{', '.join(missing)}. Regenerate with <code>python3 research/build_skills_page.py</code>.</div>")

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BuildFromScratch — research-loop skills</title>
<style>
:root {{
  --bg:#FAF6EF; --card:#FFFDF8; --peach:#FFE3CE; --peach-soft:#FFF0E4; --peach-deep:#E8A87C;
  --mint:#DDF2E4; --mint-soft:#ECF8F0; --mint-deep:#7FC8A0;
  --ink:#3D3A34; --muted:#8A8378; --line:#EDE5D8;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.65 -apple-system,'Segoe UI',Roboto,'Noto Sans',sans-serif; }}
.layout {{ display:flex; max-width:1480px; margin:0 auto; }}
nav {{ width:280px; flex-shrink:0; position:sticky; top:0; height:100vh; overflow-y:auto;
  padding:28px 18px; background:var(--mint-soft); border-right:1px solid var(--line); }}
nav h1 {{ font-size:18px; margin:0 0 4px; }}
nav .sub {{ font-size:12px; color:var(--muted); margin-bottom:14px; }}
#filter {{ width:100%; padding:8px 12px; border:1px solid var(--mint-deep); border-radius:10px;
  background:#fff; font-size:14px; margin-bottom:14px; outline:none; }}
.toc-group {{ font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);
  margin:16px 0 6px; }}
.toc-link {{ display:flex; justify-content:space-between; align-items:center; padding:6px 10px;
  border-radius:8px; color:var(--ink); text-decoration:none; font-size:14px; }}
.toc-link:hover {{ background:var(--mint); }}
.toc-link.pending {{ color:var(--muted); }}
.dot {{ width:8px; height:8px; border-radius:50%; }}
.dot.ok {{ background:var(--mint-deep); }} .dot.pend {{ background:var(--peach-deep); }}
main {{ flex:1; padding:34px 44px; min-width:0; }}
.banner {{ background:var(--peach); border:1px solid var(--peach-deep); border-radius:12px;
  padding:12px 18px; margin-bottom:24px; font-size:14px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:16px;
  padding:26px 32px; margin-bottom:34px; box-shadow:0 1px 3px rgba(120,100,80,.06); }}
.card.pending {{ opacity:.7; border-style:dashed; }}
.cardhead {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap;
  background:var(--peach-soft); margin:-26px -32px 16px; padding:18px 32px;
  border-bottom:1px solid var(--line); border-radius:16px 16px 0 0; }}
.cardhead h2 {{ margin:0; font-size:24px; }}
.badge {{ font-size:11px; padding:3px 10px; border-radius:999px; font-weight:600; }}
.badge.new {{ background:var(--mint); color:#2E6647; }}
.badge.old {{ background:var(--peach); color:#8A4B22; }}
.badge.pend {{ background:#fff; color:var(--peach-deep); border:1px dashed var(--peach-deep); }}
.chips {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }}
.chip {{ font-size:12px; background:var(--mint-soft); border:1px solid var(--mint);
  border-radius:8px; padding:2px 10px; color:var(--muted); }}
.chip.args {{ background:var(--peach-soft); border-color:var(--peach); color:#8A4B22; }}
.desc {{ font-size:14px; color:var(--muted); border-left:3px solid var(--mint-deep);
  padding-left:12px; margin:0 0 18px; }}
.md h1 {{ font-size:22px; border-bottom:2px solid var(--peach); padding-bottom:6px; }}
.md h2 {{ font-size:19px; margin-top:28px; border-bottom:1px solid var(--line); padding-bottom:4px; }}
.md h3 {{ font-size:16px; margin-top:22px; }}
.md code {{ background:var(--mint-soft); border:1px solid var(--mint); border-radius:5px;
  padding:1px 5px; font-size:13.5px; }}
.md pre, .codefile {{ background:#F4F0E6; border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; overflow-x:auto; font-size:13px; line-height:1.5; }}
.md pre code {{ background:none; border:none; padding:0; }}
.md table {{ border-collapse:collapse; width:100%; font-size:14px; margin:14px 0; display:block; overflow-x:auto; }}
.md th {{ background:var(--peach); text-align:left; }}
.md th, .md td {{ border:1px solid var(--line); padding:7px 11px; vertical-align:top; }}
.md tr:nth-child(even) td {{ background:var(--mint-soft); }}
.md blockquote {{ margin:14px 0; padding:8px 16px; background:var(--peach-soft);
  border-left:4px solid var(--peach-deep); border-radius:0 8px 8px 0; }}
details {{ margin:14px 0; background:var(--mint-soft); border:1px solid var(--mint);
  border-radius:10px; padding:10px 16px; }}
summary {{ cursor:pointer; font-weight:600; font-size:14px; color:#2E6647; }}
a {{ color:#B06A3B; }}
</style></head><body>
<div class="layout">
<nav>
  <h1>research-loop skills</h1>
  <div class="sub">BuildFromScratch · generated from .claude/skills/</div>
  <input id="filter" placeholder="filter skills…">
  {''.join(toc)}
</nav>
<main>
{banner}
{''.join(sections)}
</main>
</div>
<script>{MARKED_JS.read_text()}</script>
<script>
marked.use({{ gfm: true, breaks: false }});
document.querySelectorAll('.md[data-src]').forEach(el => {{
  const t = document.getElementById(el.dataset.src);
  if (t) el.innerHTML = marked.parse(t.textContent.replaceAll('<\\\\/script', '</script'));
}});
document.getElementById('filter').addEventListener('input', e => {{
  const q = e.target.value.toLowerCase();
  document.querySelectorAll('main .card').forEach(c => {{
    c.style.display = c.id.includes(q) || !q ? '' : 'none';
  }});
  document.querySelectorAll('.toc-link').forEach(l => {{
    l.style.display = l.getAttribute('href').includes(q) || !q ? '' : 'none';
  }});
}});
</script>
</body></html>"""

OUT.write_text(page)
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB), {len(missing)} skill(s) still pending: {missing or 'none'}")
