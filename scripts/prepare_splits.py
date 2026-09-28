"""Build leakage-safe manifests.

In-domain sources are split by task into train / cal / test_id.
The OOD source is never trained or calibrated on, except an optional small
task-disjoint slice (cal_ood) used only for the 'customer corrections' refit.

Example:
  python scripts/prepare_splits.py \
    --source rlbench=data/rlbench_fail/metadata_execution.jsonl \
    --source bridge=data/bridge_fail/metadata_execution.jsonl \
    --ood ur5=data/ur5_fail/metadata_execution.jsonl \
    --ood-cal-frac 0.2 --out manifests/
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judgecal.data import from_guardian_jsonl, grouped_split, save_manifest  # noqa: E402


def parse(spec):
    domain, path = spec.split("=", 1)
    return domain, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="domain=path/to/metadata.jsonl")
    ap.add_argument("--ood", required=True, help="domain=path/to/metadata.jsonl (held-out domain)")
    ap.add_argument("--ood-cal-frac", type=float, default=0.2)
    ap.add_argument("--out", default="manifests")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ind = []
    for spec in a.source:
        ind += from_guardian_jsonl(parse(spec)[1], domain=parse(spec)[0])
    splits = grouped_split(ind, {"train": 0.7, "cal": 0.15, "test_id": 0.15}, seed=a.seed)

    ood_domain, ood_path = parse(a.ood)
    ood = from_guardian_jsonl(ood_path, domain=ood_domain)
    if a.ood_cal_frac > 0:
        o = grouped_split(ood, {"cal_ood": a.ood_cal_frac, "test_ood": 1 - a.ood_cal_frac}, seed=a.seed)
        splits.update(o)
    else:
        splits["test_ood"] = ood

    for name, eps in splits.items():
        save_manifest(eps, Path(a.out) / f"{name}.jsonl")
        c = Counter(e.label for e in eps)
        print(f"{name:9s} n={len(eps):6d} success={c[1]:6d} failure={c[0]:6d} "
              f"tasks={len({e.task for e in eps}):4d} domains={sorted({e.domain for e in eps})}")

    # Leakage check: no task in more than one split.
    seen = {}
    for name, eps in splits.items():
        for t in {(e.domain, e.task) for e in eps}:
            assert t not in seen, f"task {t} in both {seen[t]} and {name}"
            seen[t] = name
    print("leakage check passed: splits are task-disjoint")


if __name__ == "__main__":
    main()
