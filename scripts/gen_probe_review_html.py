#!/usr/bin/env python3
"""Generate PROBE_REVIEW.html next to compare3/ (self-contained paths)."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/recon_pairs_random"
doc = json.load(open(OUT / "PROBE_EDIT.json"))

sections = []
for it in doc["items"]:
    i = it["i"]
    c3 = Path(it.get("compare3", ""))
    c3_name = c3.name if c3.name else ""
    img_src = f"compare3/{c3_name}" if c3_name else ""
    rows = []
    for q in it.get("questions") or []:
        rows.append(
            f"<tr><td><code>{html.escape(q['id'])}</code></td>"
            f"<td>{html.escape(q['question'])}</td>"
            f"<td><strong>{html.escape(q['answer'])}</strong></td>"
            f"<td><input type='checkbox'></td></tr>"
        )
    sections.append(
        f"""
<section id="img{i:02d}">
  <h2>#{i:02d} — {html.escape(it['category'])}</h2>
  <p class="hint">左栏=原图 GT，中=SD15，右=SD35</p>
  <img src="{html.escape(img_src)}" alt="compare3 #{i:02d}">
  <table>
    <thead><tr><th>id</th><th>问题</th><th>原图答案（请核对）</th><th>错？</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>"""
    )

(OUT / "PROBE_REVIEW.html").write_text(
    f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>Probe 核对</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:16px;background:#1a1a1a;color:#eee}}
h1{{font-size:1.3rem}} .top{{background:#252525;padding:12px;border-radius:8px;line-height:1.7;margin-bottom:16px}}
nav{{position:sticky;top:0;background:#1a1a1a;padding:8px 0;border-bottom:1px solid #444}}
nav a{{color:#7dbfff;margin-right:8px;text-decoration:none}}
section{{padding:20px 0;border-bottom:1px solid #333}}
.hint{{color:#aaa;font-size:0.9rem}} img{{max-width:100%;border:1px solid #555;margin:8px 0}}
table{{width:100%;border-collapse:collapse;font-size:0.95rem}}
th,td{{border:1px solid #444;padding:8px;vertical-align:top}}
th{{background:#252525}}
</style></head><body>
<h1>Probe 题目核对（24 图 × 164 题）</h1>
<div class="top">
<p><b>怎么看：</b>每张图下方表格是「问题 + 原图标准答案」。请对照图片<b>最左栏（GT）</b>核对答案是否正确。</p>
<p><b>怎么改：</b>编辑 <code>outputs/recon_pairs_random/PROBE_EDIT.json</code></p>
</div>
<nav>{''.join(f'<a href="#img{i:02d}">{i:02d}</a>' for i in range(24))}</nav>
{''.join(sections)}
</body></html>""",
    encoding="utf-8",
)
print("=> wrote", OUT / "PROBE_REVIEW.html")
