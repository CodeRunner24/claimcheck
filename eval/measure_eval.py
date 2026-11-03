"""How good is the camera ruler? Synthetic top-down scenes with a known true length.

  python -u eval/measure_eval.py

A flat A4 sheet and a flat object are rendered at a known scale, then measured through live SAM 3.1 masks.
Perfectly flat, no lens distortion: read this as an upper bound. Real photos have not been measured.
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claimcheck_core import meta_api as api
from claimcheck_core import rules, vision

HERE = Path(__file__).parent
W, H = 1280, 720


def texture(rng: np.random.Generator) -> np.ndarray:
    base = np.array([96, 134, 178], np.float32) * rng.uniform(0.75, 1.05)
    grain = cv2.GaussianBlur(rng.normal(0, 16, (H, W, 1)).astype(np.float32), (0, 0), sigmaX=30, sigmaY=1.5)
    return np.clip(base + grain[..., None] if grain.ndim == 2 else base + grain, 0, 255)


def place(canvas: np.ndarray, centre, size_px, angle, color) -> None:
    box = cv2.boxPoints((centre, size_px, angle)).astype(np.int32)
    shadow = box + 5
    overlay = canvas.copy()
    cv2.fillConvexPoly(overlay, shadow, (40, 40, 40), cv2.LINE_AA)
    canvas[:] = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
    cv2.fillConvexPoly(canvas, box, color, cv2.LINE_AA)


def scene(seed: int) -> tuple[bytes, float]:
    rng = np.random.default_rng(seed)
    r = random.Random(seed)
    scale = r.uniform(1.25, 1.8)  # px per mm
    length = r.uniform(120, 300)  # mm
    width = r.uniform(70, 160)
    canvas = np.array(texture(rng), np.uint8)
    place(canvas, (W * 0.27, H * 0.5), (297 * scale, 210 * scale), r.uniform(-8, 8), (240, 243, 244))
    place(canvas, (W * 0.74, H * 0.5), (length * scale, width * scale), r.uniform(-14, 14),
          r.choice([(120, 50, 30), (40, 40, 150), (40, 110, 40), (30, 30, 30)]))
    return cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(), length


def main(n: int = 12):
    rows = []
    for seed in range(n):
        image, truth = scene(seed)
        if seed == 0:
            (HERE / "measure_example.jpg").write_bytes(image)
        size = vision.image_size(image)
        items = vision.parse_sam(api.sam_raw("colored rectangle", image))
        sheets = vision.parse_sam(api.sam_raw("sheet of paper", image))
        row = {"seed": seed, "truth_mm": round(truth, 1), "items": len(items), "sheets": len(sheets), "measured_mm": None}
        if items and sheets:
            # SAM may also return the sheet as a rectangle; take the item on the right, where it was placed.
            right = [i for i in items if (i.box[0] + i.box[2]) / 2 > W / 2]
            left = [s for s in sheets if (s.box[0] + s.box[2]) / 2 < W / 2]
            if right and left:
                found = vision.measure(max(right, key=lambda i: i.short_side), max(left, key=lambda i: i.short_side), size)
                row.update(measured_mm=found["mm"] and round(found["mm"], 1), reason=found["reason"], ratio=found.get("ratio"))
        if row["measured_mm"]:
            row["error_pct"] = round(abs(row["measured_mm"] - truth) / truth * 100, 1)
            row["true_claim"] = rules.decide_measure(f"{truth / 10:.1f} cm", row["measured_mm"])["verdict"]
            row["false_claim"] = rules.decide_measure(f"{truth * 1.35 / 10:.1f} cm", row["measured_mm"])["verdict"]
        rows.append(row)
        print(row, flush=True)
    done = [r for r in rows if r["measured_mm"]]
    errors = sorted(r["error_pct"] for r in done)
    summary = {
        "scenes": n, "measured": len(done),
        "median_error_pct": errors[len(errors) // 2] if errors else None, "max_error_pct": errors[-1] if errors else None,
        "true_claims_verified": sum(r["true_claim"] == rules.VERIFIED for r in done),
        "true_claims_wrongly_contradicted": sum(r["true_claim"] == rules.CONTRADICTED for r in done),
        "false_claims_contradicted": sum(r["false_claim"] == rules.CONTRADICTED for r in done),
        "false_claims_wrongly_verified": sum(r["false_claim"] == rules.VERIFIED for r in done),
    }
    (HERE / "results_measure.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
