"""Build the evaluation report from a scores table.

Scores table columns: uid, model, split, domain, task, label, logit.
Every confidence-layer component is fitted on split == "cal" only, per model,
then applied unchanged to every test split. Nothing is tuned on test data.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .confidence import CALIBRATORS, Conformal, SelectivePolicy, aurc, risk_coverage
from .metrics import METRICS, cluster_bootstrap_ci, reliability_curve


def load_scores(path: str | Path) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """-> scores[model][split] = {"logit", "label", "task", "domain", "uid"} arrays."""
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            d = rows[r["model"]][r["split"]]
            d["uid"].append(r["uid"])
            d["domain"].append(r["domain"])
            d["task"].append(r["task"])
            d["label"].append(int(r["label"]))
            d["logit"].append(float(r["logit"]))
    return {
        m: {s: {k: np.array(v) for k, v in d.items()} for s, d in splits.items()}
        for m, splits in rows.items()
    }


def write_scores(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["uid", "model", "split", "domain", "task", "label", "logit"])
        if new:
            w.writeheader()
        w.writerows(rows)


NON_TEST = ("train", "cal", "cal_ood")


def evaluate(scores, alpha: float = 0.1, target_risk: float = 0.05, n_boot: int = 2000, seed: int = 0,
             cal_split: str = "cal") -> dict:
    """Returns results[model][calibrator]["splits"][split] = {metrics..., conformal..., selective...}.

    cal_split="cal" fits the confidence layer on in-domain data only (the deployment-shift test).
    cal_split="cal_ood" refits it on a small labelled slice of the target domain, which models a
    customer sending back a handful of corrections.
    """
    results = {}
    for model, splits in scores.items():
        if cal_split not in splits:
            raise ValueError(f"{model}: no '{cal_split}' split to fit the confidence layer on")
        cal = splits[cal_split]
        test_splits = [s for s in splits if s not in NON_TEST]
        results[model] = {}
        for cname, C in CALIBRATORS.items():
            calib = C().fit(cal["logit"], cal["label"])
            p_cal = calib(cal["logit"])
            conf = Conformal(alpha=alpha, mondrian=True).fit(p_cal, cal["label"])
            sel = SelectivePolicy(target_risk=target_risk).fit(p_cal, cal["label"])
            per_split = {}
            for s in test_splits:
                d = splits[s]
                p, y, g = calib(d["logit"]), d["label"], d["task"]
                row = {"n": len(y), "n_fail": int((y == 0).sum()), "n_tasks": len(np.unique(g))}
                for mname, fn in METRICS.items():
                    row[mname] = cluster_bootstrap_ci(fn, p, y, g, n_boot=n_boot, seed=seed)
                row["aurc"] = aurc(p, y)
                row["conformal"] = conf.evaluate(p, y)
                row["selective"] = sel.evaluate(p, y)
                row["_p"], row["_y"] = p, y
                per_split[s] = row
            results[model][cname] = {"params": vars(calib).copy(), "splits": per_split}
    return results


def _fmt(t) -> str:
    point, lo, hi, _ = t
    if np.isnan(lo):
        return f"{point:.3f} (CI n/a)"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def render_markdown(results: dict, alpha: float, target_risk: float, cal_split: str = "cal") -> str:
    L = [f"# Judge evaluation report (confidence layer fitted on `{cal_split}`)", ""]
    L.append(
        f"Confidence layer fitted on `{cal_split}` only. 95% CIs from a bootstrap that resamples whole tasks. "
        f"Conformal: Mondrian (per-class), alpha = {alpha}, so the target is {1 - alpha:.0%} coverage for "
        f"failures and successes separately. Selective policy: threshold chosen on `{cal_split}` for "
        f"{target_risk:.0%} error on auto-accepted verdicts.",
        )
    L.append("")
    for model, by_cal in results.items():
        L += [f"## {model}", ""]
        splits = list(next(iter(by_cal.values()))["splits"])
        for s in splits:
            r0 = by_cal["raw"]["splits"][s]
            L += [f"### split: `{s}` (n = {r0['n']}, failures = {r0['n_fail']}, tasks = {r0['n_tasks']})", ""]
            L.append("| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |")
            L.append("|---|---|---|---|---|---|---|")
            for cname, block in by_cal.items():
                r = block["splits"][s]
                L.append(
                    f"| {cname} | {_fmt(r['auroc'])} | {_fmt(r['bal_acc'])} | {_fmt(r['fail_recall'])} | "
                    f"{_fmt(r['ece'])} | {_fmt(r['brier'])} | {r['aurc']:.3f} |"
                )
            L += ["", "| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | "
                      "auto-accept rate | auto-accept error | failures auto-passed |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for cname, block in by_cal.items():
                c, sl = block["splits"][s]["conformal"], block["splits"][s]["selective"]
                L.append(
                    f"| {cname} | {c['coverage']:.3f} | {c['cov_failures']:.3f} | {c['cov_successes']:.3f} | "
                    f"{c['singleton_rate']:.3f} | {c['ambiguous_rate']:.3f} | {c['empty_rate']:.3f} | {sl['auto_rate']:.3f} | "
                    f"{sl['auto_risk']:.3f} | {sl['missed_failures_auto']}/{sl['n_failures']} |"
                )
            L.append("")
        params = {k: v["params"] for k, v in by_cal.items() if v["params"]}
        L += [f"Fitted calibrator parameters: `{params}`", ""]
    return "\n".join(L)


def plot(results: dict, out_dir: str | Path, tag: str = "") -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for model, by_cal in results.items():
        splits = list(next(iter(by_cal.values()))["splits"])
        fig, axes = plt.subplots(2, len(splits), figsize=(4.2 * len(splits), 8), squeeze=False)
        for j, s in enumerate(splits):
            ax_rel, ax_rc = axes[0, j], axes[1, j]
            ax_rel.plot([0, 1], [0, 1], "k:", lw=1)
            for cname, block in by_cal.items():
                r = block["splits"][s]
                rc = reliability_curve(r["_p"], r["_y"])
                if len(rc):
                    ax_rel.plot(rc[:, 0], rc[:, 1], "o-", ms=3, label=cname)
                cov, risk = risk_coverage(r["_p"], r["_y"])
                ax_rc.plot(cov, risk, label=cname)
            ax_rel.set(title=f"{s}: reliability", xlabel="predicted P(success)", ylabel="observed success rate",
                       xlim=(0, 1), ylim=(0, 1))
            ax_rc.set(title=f"{s}: risk vs coverage", xlabel="coverage (auto-accept rate)", ylabel="error rate")
            ax_rel.legend(fontsize=8)
            ax_rc.legend(fontsize=8)
        fig.suptitle(model)
        fig.tight_layout()
        p = out_dir / f"{model.replace('/', '_')}{tag}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths
