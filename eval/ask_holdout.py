"""Asking the footage on products it was not tuned on. Run once and reported as it came out. Costs about $0.70.

    python -u eval/condition_eval.py   # creates the images on first run
    python -u eval/ask_holdout.py

Each generated photo of a damaged product is moved like a handheld camera to make a six-second clip. No prompt or threshold was
changed after this ran. eval/ask_eval.py is different: its questions were used while building the feature.
"""
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from claimcheck_core import agent
from claimcheck_core import take as takes
from claimcheck_core.take import FFMPEG

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IMAGES = ROOT / "eval" / "images_condition"
OUT = ROOT / "eval" / "ask_out" / "holdout"
CLIPS = {
    "bag_rip": [("Does it have a zipper?", "found"), ("Does it have shoulder straps?", "found"), ("Is it damaged?", "found"), ("What color is it?", "found"),
                ("Does it have wheels?", "unseen"), ("Is there a laptop inside?", "unseen"), ("Is it waterproof?", "nocamera"), ("Is it heavy?", "nocamera")],
    "shoe_tear": [("Does it have laces?", "found"), ("Is the sole in good shape?", "found"), ("Does it have a buckle?", "unseen"), ("Where is the matching left shoe?", "unseen"),
                  ("Are they comfortable?", "nocamera"), ("What size is it?", "unseen")],
    "mug_chip": [("Does it have a handle?", "found"), ("Is it chipped?", "found"), ("Does it come with a lid?", "unseen"), ("Is there a logo on it?", "unseen"),
                 ("Is it dishwasher safe?", "nocamera")],
    "jacket_stain": [("Does it have buttons?", "found"), ("Does it have pockets?", "found"), ("Are there any stains?", "found"), ("Does it have a hood?", "unseen"),
                     ("What is on the back?", "unseen"), ("Will it keep me warm?", "nocamera")],
}


def clip_from(photo: Path) -> bytes:
    """Six seconds of slow handheld movement over one photo."""
    image = cv2.imread(str(photo))
    h, w = image.shape[:2]
    with tempfile.TemporaryDirectory() as d:
        raw = Path(d) / "raw.avi"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), 10, (1280, 720))
        scale = max(1280 / w, 720 / h) * 1.08
        big = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        bh, bw = big.shape[:2]
        for i in range(60):
            t = i / 10
            x = int((bw - 1280) / 2 + (bw - 1280) / 2.4 * math.sin(t * 0.8))
            y = int((bh - 720) / 2 + (bh - 720) / 2.4 * math.cos(t * 0.6))
            writer.write(big[y:y + 720, x:x + 1280])
        writer.release()
        out = Path(d) / "clip.mp4"
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(out)], check=True)
        return out.read_bytes()


def outcome(card: dict) -> str:
    return {"nocamera": "nocamera", "unseen": "unseen", "open": "unseen"}.get(card["tone"], "found")


def main() -> None:
    rows, right = [], 0
    for name, questions in CLIPS.items():
        take = takes.load(clip_from(IMAGES / f"{name}.jpg"))
        session = agent.Session("", spoken=True, out_dir=OUT / name)
        session.take = take
        agent.discover(session, take.frames[len(take.frames) // 2])
        print(name, "product:", session.product, flush=True)
        for question, expected in questions:
            card = agent.ask(session, question)
            got = outcome(card)
            ok = got == expected
            right += ok
            rows.append({"clip": name, "question": question, "expected": expected, "got": got, "ok": ok, "headline": card["headline"], "sentence": card["sentence"],
                         "at": card["at"], "concept": card["concept"], "seconds": card.get("seconds"), "usd": card.get("usd")})
            print("PASS" if ok else "FAIL", repr(question), "expected", expected, "got", got, "|", card["headline"], "|", card["sentence"], "|", card["concept"], card.get("seconds"), flush=True)
    wrongly_found = sum(1 for r in rows if r["expected"] != "found" and r["got"] == "found")
    missed = sum(1 for r in rows if r["expected"] == "found" and r["got"] != "found")
    times = sorted(float(r["seconds"].rstrip("s")) for r in rows if r["seconds"])
    summary = {"questions": len(rows), "right": right, "claimed_found_wrongly": wrongly_found, "missed": missed, "median_seconds": times[len(times) // 2], "worst_seconds": times[-1]}
    (ROOT / "eval" / "results_ask_holdout.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
