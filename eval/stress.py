"""Stress test: do damaged photos cause wrong verdicts? Abstaining (not_visible) is acceptable; a wrong call is not."""
import io, json, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claimcheck_core import agent, rules

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).parent


def jpg(im):
    b = io.BytesIO(); im.save(b, "JPEG", quality=85); return b.getvalue()

DAMAGE = {
    "heavy_blur": lambda im: jpg(im.filter(ImageFilter.GaussianBlur(14))),
    "tiny_resize": lambda im: jpg(im.resize((im.width // 8, im.height // 8))),
    "very_dark": lambda im: jpg(ImageEnhance.Brightness(im).enhance(0.08)),
    "right_half_cut": lambda im: jpg(im.crop((0, 0, int(im.width * 0.52), im.height))),
}


def run(args):
    case, damage = args
    im = Image.open(HERE / "images" / f"{case['image']}.jpg").convert("RGB")
    s = agent.Session("")
    c = agent.Claim("c1", case["text"], "", case["check"], case["claimed"], "care label", "", attempts=1)
    try:
        _, _, r = agent.inspect_concept(s, "care label", [c], DAMAGE[damage](im))
        r = r["c1"]
    except Exception as e:
        r = {"verdict": "error", "rule": type(e).__name__, "observed": ""}
    wrong = r["verdict"] not in (case["truth"], rules.NOT_VISIBLE)
    print("WRONG" if wrong else "ok", damage, case["image"], case["claimed"], "->", r["verdict"], r["rule"], r.get("observed", "")[:60], flush=True)
    return {"damage": damage, "image": case["image"], "claimed": case["claimed"], "truth": case["truth"],
            "got": r["verdict"], "rule": r["rule"], "observed": r.get("observed", ""), "wrong": wrong, "usd": s.cost()}


def main():
    scenes = [s for s in json.loads((HERE / "scenes.json").read_text(encoding="utf-8")) if s["kind"] == "label"][::2]
    jobs = []
    for s in scenes:
        for d in DAMAGE:
            jobs.append(({"image": s["id"], "check": "composition", "claimed": s["composition"], "text": s["composition"], "truth": rules.VERIFIED}, d))
            fake = "80% acrylic 20% wool"
            jobs.append(({"image": s["id"], "check": "composition", "claimed": fake, "text": fake, "truth": rules.CONTRADICTED}, d))
    with ThreadPoolExecutor(6) as p:
        results = list(p.map(run, jobs))
    summary = {"claims": len(results), "wrong": sum(r["wrong"] for r in results),
               "abstained": sum(r["got"] == rules.NOT_VISIBLE for r in results),
               "correct_despite_damage": sum(r["got"] == r["truth"] for r in results),
               "errors": sum(r["got"] == "error" for r in results),
               "wrong_by_damage": dict(Counter(r["damage"] for r in results if r["wrong"])),
               "usd_total": round(sum(r["usd"] for r in results), 4)}
    (HERE / "results_stress.json").write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)

main()
