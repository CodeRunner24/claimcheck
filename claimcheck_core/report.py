"""The proof page: one self-contained HTML file a seller can keep or show a buyer."""
from __future__ import annotations

import base64
import html
from datetime import datetime, timezone
from pathlib import Path

from . import rules

MAX_EMBED_VIDEO = 14 * 1024 * 1024
LABEL = {rules.VERIFIED: "Verified on camera", rules.CONTRADICTED: "Contradicted", rules.NOT_VISIBLE: "Not settled", "open": "Not checked"}
COLOR = {rules.VERIFIED: "#0f8a5f", rules.CONTRADICTED: "#d63b3f", rules.NOT_VISIBLE: "#b36b00", "open": "#6b7280"}


def _image(session, ref: str) -> str:
    if not ref:
        return ""
    if ref.startswith("data:"):
        return ref
    path = Path(session.out_dir or ".") / ref
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}" if seconds >= 0 else ""


def build(session, approved_text: str, reviewer_note: str = "") -> str:
    e = html.escape
    rows = []
    for c in session.claims:
        if c.source == "plan" and c.verdict != rules.VERIFIED:
            continue
        picture = _image(session, c.flat or c.evidence)
        rows.append(f"""<article class="claim">
  <div class="thumb">{f'<img alt="Evidence for {e(c.text)}" src="{picture}">' if picture else '<span>No frame</span>'}</div>
  <div class="body">
    <div class="tag" style="color:{COLOR.get(c.verdict, '#6b7280')}">{LABEL.get(c.verdict, c.verdict)}</div>
    <h3>{e(c.text)}</h3>
    {f'<p>Seen: <b>{e(c.observed)}</b></p>' if c.observed else ''}
    <p class="small">{e(c.reason)}</p>
    <p class="mono">rule {e(c.rule or 'none')}{f' · at {_clock(c.at)}' if c.at >= 0 else ''}{f' · reviewer {e(c.human)}' if c.human else ''}{f'<br>frame sha256 {c.frame_sha256}' if c.frame_sha256 else ''}</p>
  </div>
</article>""")
    hero = _image(session, session.hero)
    hero_note = ""
    if hero and session.hero_info:
        check = "unchanged" if session.hero_info.get("pixel_exact") else "NOT verified as unchanged"
        hero_note = (f"<p class='small'>Product pixels are from the camera; {session.hero_info.get('product_pixels', 0):,} "
                     f"interior pixels checked as {check} before encoding. Only the background was generated, by Muse Image.</p>")
    reel = ""
    if session.reel and session.out_dir:
        path = Path(session.out_dir) / session.reel
        if path.exists() and path.stat().st_size <= MAX_EMBED_VIDEO:
            reel = ("<h2>The take</h2><video controls playsinline src='data:video/mp4;base64,"
                    + base64.b64encode(path.read_bytes()).decode() + "'></video><p class='small'>Masks tracked by SAM 3.1.</p>")
    calls: dict[str, list[float]] = {}
    for t in session.trace:
        slot = calls.setdefault(t.model, [0, 0.0])
        slot[0] += 1
        slot[1] += t.usd
    usage = " · ".join(f"{e(m)}: {int(n)} calls, ${usd:.3f}" for m, (n, usd) in calls.items())
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = session.title or "Listing proof"
    approved_heading = "The listing"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ClaimCheck {title}</title>
<style>
:root{{--ink:#15171c;--mute:#667085;--line:#e6e7eb;--bg:#fbfaf7;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--ink:#eceef2;--mute:#9aa3b2;--line:#2a2e37;--bg:#0f1115;--card:#171a20}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,'Segoe UI',sans-serif}}
main{{max-width:860px;margin:auto;padding:40px 16px 72px}} h1{{font-size:34px;margin:0 0 4px;letter-spacing:-.02em}}
h2{{margin:40px 0 12px;font-size:20px}} h3{{margin:2px 0 4px;font-size:17px}} p{{margin:4px 0}}
.small{{color:var(--mute);font-size:14px}} .mono{{color:var(--mute);font:12px/1.5 ui-monospace,Consolas,monospace;overflow-wrap:anywhere}}
.tag{{font:700 12px/1.4 system-ui;text-transform:uppercase;letter-spacing:.06em}}
.quote{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;white-space:pre-wrap}}
.claim{{display:grid;grid-template-columns:150px 1fr;gap:16px;padding:14px;background:var(--card);border:1px solid var(--line);border-radius:12px;margin-bottom:10px}}
.thumb{{background:#0002;border-radius:8px;overflow:hidden;min-height:90px;display:flex;align-items:center;justify-content:center;color:var(--mute);font-size:13px}}
.thumb img{{width:100%;height:100%;object-fit:cover;display:block}} img.hero,video{{width:100%;border-radius:12px;display:block}}
@media(max-width:560px){{.claim{{grid-template-columns:1fr}}}}
</style></head><body><main>
<p class="mono">ClaimCheck · {stamp}</p><h1>{title}</h1>
<p class="small">Checked on camera by an AI agent, approved by a person. A match is evidence of what the camera saw, not a certification.</p>
<h2>{approved_heading}</h2><div class="quote">{e(approved_text)}</div>
{f'<h2>Listing photo</h2><img class="hero" alt="Product photo" src="{hero}">{hero_note}' if hero else ''}
<h2>Each claim</h2>{''.join(rows)}
{reel}
<h2>What the seller originally said</h2><div class="quote">{e(session.listing)}</div>
{f'<p class="small">{e(reviewer_note)}</p>' if reviewer_note else ''}
<h2>How it was made</h2><p class="mono">{usage}<br>total model cost ${session.cost():.3f}</p>
</main></body></html>"""
