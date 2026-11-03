"""Does Muse Spark turn what a seller says into the right facts?

  python -u eval/listening_eval.py

Hand-written sentences in spoken style. Each lists the attributes that should be noted, and the check that should decide
them. Sales talk must produce nothing. A small set, written by the author: a sanity check, not a benchmark.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claimcheck_core import agent

HERE = Path(__file__).parent
CASES = [
    ("This is a set of two black t-shirts.", {"quantity": "count", "color": "color"}),
    ("It's a hundred percent cotton and made in Portugal.", {"material": "composition", "origin": "text"}),
    ("So this one is sixty percent wool, forty percent acrylic.", {"material": "composition"}),
    ("You get three mugs in the box, all white.", {"quantity": "count", "color": "color"}),
    ("It's size medium, it says so on the tag.", {"size": "text"}),
    ("The bag is about thirty eight centimeters long.", {"dimension": "measure"}),
    ("It has a full front zip and a hood.", {"feature": "visual"}),
    ("Made in Italy, genuine leather.", {"origin": "text", "material": "composition"}),
    ("This jacket is navy blue with two side pockets.", {"color": "color", "feature": "visual"}),
    ("The brand is Levi's, model 505.", {"brand": "text", "model": "text"}),
    ("A pair of red sneakers.", {"quantity": "count", "color": "color"}),
    ("Pure linen, made in Lithuania.", {"material": "composition", "origin": "text"}),
    ("It's green. Dark green really.", {"color": "color"}),
    ("Four piece set, stainless steel.", {"quantity": "count"}),
    ("This is our best seller and everyone loves it.", {}),
    ("Super comfortable, premium quality, fast shipping.", {}),
    ("Okay um let me just get this in the light.", {}),
    ("Hi everyone, thanks for watching.", {}),
    ("You're going to love how soft this feels.", {}),
    ("It's a great gift for your mom.", {}),
]


def run(case):
    said, expected = case
    session = agent.Session(spoken=True)
    agent.hear(session, said)
    got = {c.attribute: c.check for c in session.claims}
    return {"said": said, "expected": expected, "got": got, "usd": session.cost()}


def main():
    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(run, CASES))
    want = sum(len(r["expected"]) for r in rows)
    found = sum(sum(a in r["got"] for a in r["expected"]) for r in rows)
    extra = sum(sum(a not in r["expected"] for a in r["got"]) for r in rows)
    right_check = sum(sum(r["got"].get(a) == c for a, c in r["expected"].items()) for r in rows)
    silent = [r for r in rows if not r["expected"]]
    summary = {
        "sentences": len(rows), "facts_expected": want, "facts_found": found, "extra_facts": extra,
        "right_check": right_check, "sales_talk_sentences": len(silent),
        "sales_talk_correctly_ignored": sum(not r["got"] for r in silent), "usd_total": round(sum(r["usd"] for r in rows), 3),
    }
    for r in rows:
        print("ok " if r["got"] == r["expected"] else "DIFF", r["said"][:56], "|", r["got"], flush=True)
    (HERE / "results_listening.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
