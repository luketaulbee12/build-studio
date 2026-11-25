"""Picture -> Minecraft build, every stage in one command (each stage in its own process to free GPU memory).

usage: python scripts/pipeline.py <picture> <name> [--width 16] [--at X Z]
       python scripts/pipeline.py --all <folder> [--width 16]      (every picture in a folder, report table)

Stages: Hunyuan shape (out/<name>_hunyuan.glb) -> Hunyuan Paint (out/<name>_painted.glb)
        -> blocks (out/dbg/<name>_compare.png, and a blueprint when --at is given).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
OUT = ROOT / "out"


def run(args):
    t = time.time()
    r = subprocess.run([PY, "-W", "ignore"] + args, cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise RuntimeError(f"stage failed: {args[0]}")
    return r.stdout, time.time() - t


def one(picture, name, width, at=None):
    safe = name.replace(" ", "_")
    shape = OUT / f"{safe}_hunyuan.glb"
    painted = OUT / f"{safe}_painted.glb"
    times = {}
    if not shape.exists():
        _, times["shape"] = run(["scripts/run_models.py", str(picture), safe, "hunyuan"])
    if not painted.exists():
        _, times["paint"] = run(["scripts/paint.py", str(shape), str(picture), str(painted)])
    args = ["scripts/build_from_model.py", str(painted), str(picture), name, "--width", str(width)]
    if at:
        args += ["--at", str(at[0]), str(at[1])]
    log, times["blocks"] = run(args)
    print(log)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picture", nargs="?")
    ap.add_argument("name", nargs="?")
    ap.add_argument("--all", default=None)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--at", type=int, nargs=2, default=None)
    a = ap.parse_args()
    if a.all:
        rows = []
        for f in sorted(Path(a.all).iterdir()):
            if f.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            print(f"=== {f.name}")
            try:
                t = one(f, f.stem, a.width)
                rows.append((f.stem, "ok", t))
            except Exception as e:  # keep going: one bad picture shouldn't stop the test set
                rows.append((f.stem, f"FAILED: {e}", {}))
        print("\npicture                          result   seconds (shape / paint / blocks)")
        for n, r, t in rows:
            print(f"{n:32s} {r:8s} {t.get('shape', 0):5.0f} {t.get('paint', 0):5.0f} {t.get('blocks', 0):5.0f}")
    else:
        print(json.dumps(one(a.picture, a.name, a.width, a.at)))


if __name__ == "__main__":
    main()
