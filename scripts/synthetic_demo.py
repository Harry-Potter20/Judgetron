"""Run the full evaluate pipeline on synthetic scores (no GPU, no data).

  python scripts/synthetic_demo.py && python scripts/evaluate.py --scores results_synthetic/scores.csv --out results_synthetic
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judgecal.report import write_scores  # noqa: E402
from judgecal.synthetic import make_scores  # noqa: E402

out = Path("results_synthetic/scores.csv")
if out.exists():
    out.unlink()
write_scores(out, make_scores())
print(f"wrote {out}")
