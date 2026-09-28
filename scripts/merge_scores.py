"""Merge scores CSVs from separate scoring runs, then report.

The full grid is split across two Kaggle kernels (zero-shot; LoRA) because together they overrun a
single session. Each writes only its own model's rows, so the combined report is built from the
union. evaluate.py is CPU-only, so this step runs anywhere.

  python scripts/merge_scores.py --out results/scores.csv \
      runs/zeroshot/scores.csv runs/lora/scores.csv
  python scripts/evaluate.py --scores results/scores.csv --out results/
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

FIELDS = ["uid", "model", "split", "domain", "task", "label", "logit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows, seen = [], set()
    for p in a.inputs:
        with open(p) as f:
            for r in csv.DictReader(f):
                # (model, split, uid) is the identity of a score. A duplicate means the same episode
                # was scored twice by the same model -- usually a kernel rerun appending to an
                # existing file -- and silently keeping both would double-weight it in every metric.
                key = (r["model"], r["split"], r["uid"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(r)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r[k] for k in FIELDS} for r in rows)

    per = Counter((r["model"], r["split"]) for r in rows)
    print(f"wrote {a.out}: {len(rows)} rows, {len({r['model'] for r in rows})} models")
    for (m, s), n in sorted(per.items()):
        print(f"  {m:20s} {s:10s} n={n}")

    # Every model must cover the same splits, or a cross-model comparison is reading different data.
    models = sorted({r["model"] for r in rows})
    splits = {m: {s for (mm, s) in per if mm == m} for m in models}
    if len(models) > 1 and len({frozenset(v) for v in splits.values()}) > 1:
        print("WARNING: models do not cover the same splits:",
              {m: sorted(v) for m, v in splits.items()})


if __name__ == "__main__":
    main()
