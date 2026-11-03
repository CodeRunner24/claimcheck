"""Generate the disclosed synthetic test images with Muse Image. Logs progress; skips existing files."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claimcheck_core import meta_api as api, vision

HERE = Path(__file__).parent
OUT = HERE / "images"
STYLE = "Realistic amateur smartphone photo, natural indoor light, plain home table, sharp focus, no watermark, no brand logos. "


def main():
    OUT.mkdir(exist_ok=True)
    scenes = json.loads((HERE / "scenes.json").read_text(encoding="utf-8"))
    def make(scene):
        path = OUT / f"{scene['id']}.jpg"
        if path.exists():
            return
        raw = api.generate_image(STYLE + scene["prompt"])
        path.write_bytes(vision.prepare_image(raw))
        print("done", scene["id"], flush=True)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(make, scenes))
    print("spend", api.LEDGER_INSTANCE.total, flush=True)


if __name__ == "__main__":
    main()
