#!/usr/bin/env python3
"""Render a Markdown file to a styled, self-contained HTML page.

Handles: fenced/inline code (Pygments), GFM tables, a sticky TOC sidebar, and
LaTeX math via MathJax -- WITHOUT mis-rendering prose dollar amounts ($30M etc.).
Only genuine math spans are converted to \\(...\\) / \\[...\\] delimiters; MathJax
is configured to NOT scan bare `$`, so "$30M" stays literal.

Usage: render_md.py <input.md> <output.html> "<title>" "<subtitle>"
"""
import re
import sys
import html
from pygments.formatters import HtmlFormatter

import markdown


def protect(text, pattern, store, tag, flags=0):
    """Replace each match of `pattern` with an inert token, stashing the raw text."""
    def repl(m):
        store.append(m.group(0))
        return f"zz{tag}{len(store)-1}zz"
    return re.sub(pattern, repl, text, flags=flags)


def is_math(content):
    """Heuristic: is this $...$ span genuine LaTeX math vs a prose dollar amount?"""
    c = content.strip()
    if not c:
        return False
    # currency / numeric amount like 30M, 1.5B, 2-4/GPU-hr  -> NOT math
    if re.fullmatch(r"[\d.,]+\s*[-–]?\s*[\d.,]*\s*(?:[MBKk]|million|billion|trillion)?(?:/[\w-]+)?", c):
        return False
    # genuine math indicators, or a short symbol token
    if re.search(r"[\\^_{}]", c):
        return True
    if len(c) <= 12:
        return True
    return False


def extract_math(text):
    """Pull display ($$..$$) then inline ($..$) math into \\[..\\] / \\(..\\) tokens."""
    disp, inl = [], []

    def disp_repl(m):
        disp.append(m.group(1).strip())
        return f"zzDMATHzz{len(disp)-1}zz"
    text = re.sub(r"\$\$(.+?)\$\$", disp_repl, text, flags=re.DOTALL)

    # paired inline $...$ with no leading/trailing space and no $ or newline inside
    def inl_repl(m):
        content = m.group(1)
        if not is_math(content):
            return m.group(0)  # leave prose dollars untouched
        inl.append(content)
        return f"zzIMATHzz{len(inl)-1}zz"
    text = re.sub(r"\$(?=\S)([^$\n]{1,160}?)(?<=\S)\$", inl_repl, text)
    return text, disp, inl


def main():
    src, out, title, subtitle = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    raw = open(src, encoding="utf-8").read()

    # 1. protect code so dollars inside it are never treated as math
    code_store = []
    raw = protect(raw, r"```.*?```", code_store, "CODE", flags=re.DOTALL)
    raw = protect(raw, r"`[^`\n]+`", code_store, "ICODE")

    # 2. extract genuine math
    raw, disp, inl = extract_math(raw)

    # 3. restore code blocks so markdown renders them
    for i, block in enumerate(code_store):
        raw = raw.replace(f"zzCODE{i}zz", block).replace(f"zzICODE{i}zz", block)

    md = markdown.Markdown(extensions=["extra", "codehilite", "toc", "sane_lists", "smarty"],
                           extension_configs={"codehilite": {"guess_lang": False},
                                              "toc": {"permalink": "#"}})
    body = md.convert(raw)
    toc = md.toc

    # 4. restore math as MathJax delimiters
    for i, m in enumerate(disp):
        body = body.replace(f"zzDMATHzz{i}zz", "\\[" + m + "\\]")
    for i, m in enumerate(inl):
        body = body.replace(f"zzIMATHzz{i}zz", "\\(" + m + "\\)")

    pyg = HtmlFormatter(style="monokai").get_style_defs(".codehilite")
    page = TEMPLATE.format(title=html.escape(title), subtitle=html.escape(subtitle),
                           toc=toc, body=body, pygments=pyg)
    open(out, "w", encoding="utf-8").write(page)
    # sanity: math delimiter balance
    print(f"wrote {out}  | display-math={len(disp)} inline-math={len(inl)} "
          f"| balance \\[={body.count(chr(92)+'[')} \\]={body.count(chr(92)+']')} "
          f"\\(={body.count(chr(92)+'(')} \\)={body.count(chr(92)+')')}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script>
MathJax = {{
  tex: {{ inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']], processEscapes: true }},
  options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }}
}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
:root {{
  --bg:#0e1116; --panel:#161b22; --fg:#d7dde5; --muted:#8b949e; --line:#262d36;
  --accent:#58a6ff; --accent2:#ffb454; --good:#3fb950; --bad:#f85149; --maxw:880px;
}}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:16px/1.72 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif; }}
.wrap {{ display:flex; align-items:flex-start; gap:0; }}
nav.toc {{ position:sticky; top:0; height:100vh; overflow-y:auto; width:320px; flex:0 0 320px;
  background:var(--panel); border-right:1px solid var(--line); padding:22px 18px 60px; }}
nav.toc h2 {{ font-size:12px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted);
  margin:0 0 12px; }}
nav.toc ul {{ list-style:none; margin:0; padding:0; }}
nav.toc li {{ margin:1px 0; }}
nav.toc a {{ color:var(--muted); text-decoration:none; font-size:13px; display:block;
  padding:3px 8px; border-radius:6px; border-left:2px solid transparent; line-height:1.4; }}
nav.toc a:hover {{ color:var(--fg); background:#1f2630; }}
nav.toc > ul > li > ul {{ padding-left:12px; }}
nav.toc > ul > li > ul > li > ul {{ display:none; }}  /* hide h3+ to keep sidebar tight */
main {{ flex:1 1 auto; min-width:0; padding:46px 7vw 140px; }}
.inner {{ max-width:var(--maxw); margin:0 auto; }}
header.doc {{ border-bottom:1px solid var(--line); padding-bottom:22px; margin-bottom:34px; }}
header.doc h1 {{ font-size:30px; line-height:1.2; margin:0 0 10px; letter-spacing:-.01em; }}
header.doc .sub {{ color:var(--muted); font-size:15px; }}
h1,h2,h3,h4 {{ scroll-margin-top:24px; }}
h2 {{ font-size:24px; margin:46px 0 14px; padding-top:10px; border-top:1px solid var(--line); letter-spacing:-.01em; }}
h3 {{ font-size:19px; margin:30px 0 10px; color:#e8edf3; }}
h4 {{ font-size:16px; margin:22px 0 8px; color:var(--accent2); }}
a {{ color:var(--accent); }}
a.headerlink {{ opacity:0; margin-left:.4em; text-decoration:none; font-weight:400; }}
h2:hover a.headerlink, h3:hover a.headerlink, h4:hover a.headerlink {{ opacity:.5; }}
p {{ margin:0 0 15px; }}
ul,ol {{ margin:0 0 15px; padding-left:24px; }}
li {{ margin:4px 0; }}
strong {{ color:#fff; }}
blockquote {{ margin:18px 0; padding:12px 18px; border-left:3px solid var(--accent2);
  background:#15191f; border-radius:0 8px 8px 0; color:#c8d1da; }}
blockquote p:last-child {{ margin:0; }}
code {{ font-family:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace; font-size:.88em;
  background:#1d232c; padding:.12em .4em; border-radius:5px; color:#e6b673; }}
pre {{ background:#11151b !important; border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; overflow-x:auto; margin:0 0 18px; }}
pre code {{ background:none; padding:0; color:#d7dde5; font-size:13.5px; line-height:1.55; }}
table {{ border-collapse:collapse; width:100%; margin:18px 0; font-size:14.5px; display:block; overflow-x:auto; }}
th,td {{ border:1px solid var(--line); padding:8px 11px; text-align:left; vertical-align:top; }}
th {{ background:#1b212a; color:#fff; }}
tr:nth-child(even) td {{ background:#13181f; }}
hr {{ border:none; border-top:1px solid var(--line); margin:38px 0; }}
.codehilite {{ background:#11151b; border:1px solid var(--line); border-radius:10px; margin:0 0 18px; }}
.codehilite pre {{ border:none; margin:0; background:none !important; }}
{pygments}
@media (max-width:980px) {{ nav.toc {{ display:none; }} main {{ padding:30px 6vw 100px; }} }}
.topbar {{ position:fixed; top:0; right:0; left:0; height:0; }}
</style>
</head>
<body>
<div class="wrap">
  <nav class="toc"><h2>Contents</h2>{toc}</nav>
  <main><div class="inner">
    <header class="doc"><h1>{title}</h1><div class="sub">{subtitle}</div></header>
    {body}
  </div></main>
</div>
</body>
</html>"""


if __name__ == "__main__":
    main()
