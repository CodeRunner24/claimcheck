"""Verdicts on real photos: freely licensed Wikimedia Commons pictures, labelled by eye.

  python -u eval/real_eval.py

Photos are taken by other people on real cameras, so lighting, angle and clutter are not arranged for this system.
They were not filmed through the live flow, and the set is small. See real_photos.json for sources and licences.
"""
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claimcheck_core import agent, perception, rules, vision

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
CACHE = HERE / "images_real"
HEADERS = {"User-Agent": "ClaimCheckEval/0.1 (hackathon evaluation)"}


def fetch(manifest: dict) -> None:
    CACHE.mkdir(exist_ok=True)
    with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as client:
        for name, photo in manifest["photos"].items():
            path = CACHE / f"{name}.jpg"
            if not path.exists():
                for attempt in range(5):  # Wikimedia rate-limits politely with 403 and 429
                    response = client.get(photo["url"])
                    if response.status_code not in (403, 429):
                        break
                    time.sleep(5 * (attempt + 1))
                response.raise_for_status()
                path.write_bytes(vision.prepare_image(response.content))
                time.sleep(1.0)


def run_case(case: dict) -> dict:
    session = agent.Session()
    claim = agent.Claim(id="c1", text=case["claimed"], attribute="", check=case["check"], claimed=case["claimed"],
                        sam_concept=case["concept"], shot_request="", attempts=1)
    try:
        _, _, results = perception.inspect_concept(session, case["concept"], [claim], (CACHE / f"{case['image']}.jpg").read_bytes())
        r = results["c1"]
    except Exception as e:  # counted, never hidden
        r = {"verdict": "error", "rule": type(e).__name__, "detail": str(e)[:200], "observed": ""}
    return {**case, "got": r["verdict"], "rule": r["rule"], "observed": r.get("observed", ""), "usd": session.cost()}


def main():
    manifest = json.loads((HERE / "real_photos.json").read_text(encoding="utf-8"))
    fetch(manifest)
    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(run_case, manifest["cases"]))
    for r in results:
        mark = "ok " if r["got"] == r["expected"] else ("open" if r["got"] == rules.NOT_VISIBLE else "BAD")
        print(mark, r["image"], r["check"], repr(r["claimed"]), "exp", r["expected"], "got", r["got"], "|", r["rule"], "|", r["observed"][:60], flush=True)
    correct = sum(r["got"] == r["expected"] for r in results)
    abstained = sum(r["got"] == rules.NOT_VISIBLE and r["expected"] != rules.NOT_VISIBLE for r in results)
    wrong = [r for r in results if r["got"] != r["expected"] and r["got"] != rules.NOT_VISIBLE]
    summary = {
        "photos": len(manifest["photos"]), "claims": len(results), "correct": correct, "left_open": abstained, "wrong": len(wrong),
        "false_contradictions": sum(r["got"] == rules.CONTRADICTED and r["expected"] != rules.CONTRADICTED for r in results),
        "false_verifications": sum(r["got"] == rules.VERIFIED and r["expected"] != rules.VERIFIED for r in results),
        "by_check": {k: f"{sum(r['got'] == r['expected'] for r in results if r['check'] == k)}/{v}"
                     for k, v in Counter(r["check"] for r in results).items()},
        "usd_total": round(sum(r["usd"] for r in results), 3),
    }
    (HERE / "results_real.json").write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
