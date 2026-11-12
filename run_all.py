"""Run every analysis and save outputs to results/. Usage: python run_all.py"""
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
STEPS = ["theory_fold_check", "theory_feller_gamma_check", "e1_leverage_curvature",
         "e1_news_1997_2022", "e1_news_1928_2022", "e1_curvature_no_leverage",
         "btc_liquidation_events", "cojumps"]
(ROOT / "results").mkdir(exist_ok=True)
for s in STEPS:
    print(f"== {s}", flush=True)
    out = subprocess.run([sys.executable, f"{s}.py"], cwd=ROOT / "src", capture_output=True, text=True)
    (ROOT / "results" / f"{s}.txt").write_text(out.stdout + out.stderr)
    print(out.stdout[-1500:] if out.returncode == 0 else out.stderr[-1500:])
