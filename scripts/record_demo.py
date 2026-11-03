"""Record the picture for the demo video from the running app in sample mode: real UI, live Meta API calls, no text added.

Start the app first, then:  python -u scripts/record_demo.py
Output: demo/claimcheck_demo.mp4, plus demo/tmp/timeline.json with when each part starts in the cut video, for laying a voice-over
on top. Waiting is cut out and nothing else is changed. The strong version of this video is filmed with a real product; see the README.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from claimcheck_core.take import FFMPEG

OUT = ROOT / "demo"
TMP = OUT / "tmp"
TMP.mkdir(parents=True, exist_ok=True)
QUESTIONS = ["Is it really 100% cotton?", "Does it have a pocket?", "Does it have a zipper?", "Is it comfortable?"]


def kept_time(moment: float, cuts: list[tuple[float, float]]) -> float:
    """Where a moment of the raw recording lands once the waiting is cut out."""
    removed = 0.0
    position = 0.0
    for start, end in sorted(cuts):
        start = max(start, position)
        if end <= start:
            continue
        if moment >= end:
            removed += end - start
        elif moment > start:
            removed += moment - start
        position = end
    return moment - removed


def main():
    with sync_playwright() as p:
        # Microsoft Edge rather than bundled Chromium: the bundled build can't play the H.264 clips the app makes.
        browser = p.chromium.launch(channel="msedge", args=["--autoplay-policy=no-user-gesture-required", "--headless=new"])
        ctx = browser.new_context(viewport={"width": 1440, "height": 810}, record_video_dir=str(TMP), record_video_size={"width": 1440, "height": 810})
        pg = ctx.new_page()
        t0 = time.time()
        clock = lambda: time.time() - t0  # noqa: E731
        cuts: list[tuple[float, float]] = []  # stretches of waiting to remove
        marks: dict[str, float] = {}

        def mark(name: str):
            marks[name] = clock()
            print(name, round(marks[name], 1), flush=True)

        pg.goto("http://localhost:3000", wait_until="networkidle", timeout=180000)
        pg.wait_for_selector("#go-sample")
        cuts.append((0.0, clock()))
        mark("1_start_screen")
        pg.wait_for_timeout(13000)
        pg.click("#go-sample")
        pg.wait_for_selector("#cc-tag", timeout=30000)
        mark("2_filming")
        struck = shown = 0
        while pg.locator("#finish").count() and clock() < 220:
            count = pg.locator(".seg.contradicted").count()
            if count != struck:
                struck = count
                shown += 1
                mark(f"filming_strike_{shown}")
            pg.wait_for_timeout(200)
        cutting_from = clock() + 1.0
        pg.wait_for_selector("#approve", timeout=240000)
        cuts.append((cutting_from, clock() - 0.5))
        mark("5_listing")
        pg.wait_for_timeout(8000)
        pg.evaluate("document.querySelector('#cc-noticed, .facts')?.scrollIntoView({behavior:'smooth',block:'center'})")
        pg.wait_for_timeout(6000)
        pg.evaluate("document.getElementById('cc-ask-band').scrollIntoView({behavior:'smooth',block:'start'})")
        mark("6_ask")
        pg.wait_for_timeout(4000)
        for count, question in enumerate(QUESTIONS, 1):
            pg.locator("#cc-question").press_sequentially(question, delay=45)
            pg.wait_for_timeout(300)
            mark(f"question_{count}")
            pg.press("#cc-question", "Enter")
            pg.wait_for_selector(".strip.scanning", timeout=8000)
            pg.wait_for_timeout(2600)  # enough of the search to see it working
            waiting = clock()
            pg.wait_for_function(f"document.querySelectorAll('.answer').length === {count} && !document.querySelector('.strip.scanning')", timeout=150000)
            if clock() - waiting > 1.5:
                cuts.append((waiting, clock() - 1.0))
            mark(f"answer_{count}")
            pg.evaluate("document.getElementById('cc-answers').scrollIntoView({behavior:'smooth',block:'start'})")
            print(question, "->", pg.locator(".a-head").first.inner_text(), flush=True)
            pg.wait_for_timeout(6000)
            pg.evaluate("document.getElementById('cc-ask-band').scrollIntoView({behavior:'smooth',block:'start'})")
            pg.wait_for_timeout(500)
        # End on the listing itself.
        pg.evaluate("window.scrollTo({top:0,behavior:'smooth'})")
        mark("7_end_on_listing")
        pg.wait_for_timeout(14000)
        total = clock()
        video = pg.video.path()
        ctx.close()
        browser.close()

    keep, position = [], 0.0
    for start, end in sorted(cuts):
        if start > position:
            keep.append(f"between(t,{position:.2f},{start:.2f})")
        position = max(position, end)
    keep.append(f"between(t,{position:.2f},{total:.2f})")
    with open(OUT / "ffmpeg.log", "w") as log:
        subprocess.run([FFMPEG, "-y", "-i", str(video), "-vf", f"select='{'+'.join(keep)}',setpts=N/FRAME_RATE/TB", "-r", "25",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-an",
                        str(OUT / "claimcheck_demo.mp4")], check=True, stdout=log, stderr=subprocess.STDOUT)
    timeline = {name: round(kept_time(at, cuts), 2) for name, at in marks.items()}
    timeline["total"] = round(kept_time(total, cuts), 2)
    (TMP / "timeline.json").write_text(json.dumps(timeline, indent=1), encoding="utf-8")
    print("timeline", timeline, flush=True)


if __name__ == "__main__":
    sys.exit(main())
