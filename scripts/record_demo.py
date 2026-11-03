"""Record the silent backup demo from the running app in sample mode: real UI, live Meta API calls.

Start `reflex run` first, then:  python -u scripts/record_demo.py
Output: demo/claimcheck_demo.mp4. The strong version of this video is filmed with a real product; see the README.
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
ev = json.loads((ROOT / "eval" / "results_holdout.json").read_text(encoding="utf-8"))["summary"]
st = json.loads((ROOT / "eval" / "results_stress.json").read_text(encoding="utf-8"))["summary"]
right = round(ev["accuracy"] * ev["claims"])
FONT = "https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..900&display=swap"


def card(big: str, small: str = "") -> str:
    return f"""<html><head><link href="{FONT}" rel="stylesheet"></head><body style="margin:0;height:100vh;box-sizing:border-box;background:#ffdd00;color:#111;
    font-family:Archivo,Arial;padding:56px;display:flex;flex-direction:column;justify-content:space-between">
    <div style="font-size:112px;line-height:.92;font-weight:900;font-stretch:75%;text-transform:uppercase;letter-spacing:-.02em">{big}</div>
    <div style="font-size:30px;line-height:1.3;font-weight:500;max-width:1150px">{small}</div></body></html>"""


INTRO = card("Listings are written from memory.", "Buyers find out what's true when they open the parcel. Sellers should find out first.")
HOW = card("Models observe. Rules decide.",
           "SAM 3.1 finds and tracks the evidence. Muse Spark reads it. Muse Voice hears the seller. Muse Image draws only the background. "
           "Plain Python decides counts, percentages and colors. If the evidence can't be seen, the answer is &ldquo;not settled&rdquo;, never &ldquo;wrong&rdquo;.")
SCORE = card(f"{right} of {ev['claims']} right. {ev['false_contradictions']} false alarms.",
             f"Held-out test on AI-generated images. {st['claims']} damaged photos: {st['wrong']} wrong, all left open. "
             "This clip and its seller are AI-generated and scripted, and real phone photos are not measured yet. Built during the hackathon.")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        ctx = browser.new_context(viewport={"width": 1440, "height": 810}, record_video_dir=str(TMP),
                                  record_video_size={"width": 1440, "height": 810})
        pg = ctx.new_page()
        t0 = time.time()
        pg.goto("http://localhost:3000", wait_until="networkidle", timeout=180000)
        pg.wait_for_selector("#go-sample")
        start_clip = time.time() - t0
        pg.set_content(INTRO)
        pg.wait_for_timeout(4500)
        pg.goto("http://localhost:3000", wait_until="networkidle")
        pg.wait_for_selector("#go-sample")
        pg.wait_for_timeout(3500)
        pg.click("#go-sample")
        pg.wait_for_selector("#cc-tag", timeout=30000)
        print("live", round(time.time() - t0, 1), flush=True)
        while pg.locator("#finish").count() and time.time() - t0 < 200:
            pg.wait_for_timeout(300)
        cut_from = time.time() - t0 + 1.5
        print("cutting", round(cut_from, 1), flush=True)
        pg.wait_for_selector("#approve", timeout=240000)
        cut_to = time.time() - t0 - 1.0
        print("reveal", round(cut_to, 1), flush=True)
        pg.wait_for_timeout(5000)
        pg.locator("#cc-reel").scroll_into_view_if_needed()
        pg.wait_for_timeout(9000)
        pg.locator("summary").click()
        pg.locator("summary").scroll_into_view_if_needed()
        pg.mouse.wheel(0, 1200)
        pg.wait_for_timeout(5000)
        pg.set_content(HOW)
        pg.wait_for_timeout(6500)
        pg.set_content(SCORE)
        pg.wait_for_timeout(6500)
        total = time.time() - t0
        video = pg.video.path()
        ctx.close()
        browser.close()

    keep = f"between(t,{start_clip:.2f},{cut_from:.2f})+between(t,{cut_to:.2f},{total:.2f})"
    with open(OUT / "ffmpeg.log", "w") as log:
        subprocess.run([FFMPEG, "-y", "-i", str(video), "-vf", f"select='{keep}',setpts=N/FRAME_RATE/TB", "-r", "25",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-an",
                        str(OUT / "claimcheck_demo.mp4")], check=True, stdout=log, stderr=subprocess.STDOUT)
    print("video", OUT / "claimcheck_demo.mp4", flush=True)


if __name__ == "__main__":
    sys.exit(main())
