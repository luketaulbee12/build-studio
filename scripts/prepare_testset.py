"""Make sure every test-set picture has its shape + painted model cached (out/<name>_painted.glb)."""
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
for t in json.load(open(ROOT / "data" / "testset.json")):
    shape, painted = ROOT / "out" / f"{t['name']}_hunyuan.glb", ROOT / "out" / f"{t['name']}_painted.glb"
    if not shape.exists():
        print("shape", t["name"], flush=True)
        subprocess.run([PY, "-W", "ignore", "scripts/run_models.py", t["picture"], t["name"], "hunyuan"], cwd=ROOT, check=True, capture_output=True)
    if not painted.exists():
        print("paint", t["name"], flush=True)
        subprocess.run([PY, "-W", "ignore", "scripts/paint.py", str(shape), t["picture"], str(painted)], cwd=ROOT, check=True, capture_output=True)
print("all ready")
