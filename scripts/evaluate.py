"""Fit the confidence layer on cal, evaluate on every test split, write the report.

  python scripts/evaluate.py --scores results/scores.csv --out results/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judgecal.report import evaluate, load_scores, plot, render_markdown  # noqa: E402


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/scores.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--target-risk", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    scores = load_scores(a.scores)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    has_ood_cal = all("cal_ood" in s for s in scores.values())
    modes = ["cal"] + (["cal_ood"] if has_ood_cal else [])
    sections, dump = [], {}
    for mode in modes:
        res = evaluate(scores, alpha=a.alpha, target_risk=a.target_risk, n_boot=a.n_boot, cal_split=mode)
        sections.append(render_markdown(res, a.alpha, a.target_risk, cal_split=mode))
        plot(res, out / "plots", tag=f"_{mode}")
        dump[mode] = _clean(res)
    (out / "report.md").write_text("\n\n---\n\n".join(sections))
    (out / "results.json").write_text(json.dumps(dump, indent=1))
    print(f"wrote {out / 'report.md'}, {out / 'results.json'}, {out / 'plots'}/")


if __name__ == "__main__":
    main()
