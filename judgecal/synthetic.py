"""Synthetic scores for testing the pipeline without a GPU or real data.

Simulates a judge that is overconfident in-domain and, under domain shift,
loses discrimination and sees a different failure base rate. Not a result.
"""
from __future__ import annotations

import numpy as np

SHIFT = {
    #            n_tasks, eps/task, latent mean, latent sd, judge signal, overconfidence, judge bias
    "cal":      (30, 20, 0.0, 2.0, 1.0, 2.0, 0.0),
    "test_id":  (30, 20, 0.0, 2.0, 1.0, 2.0, 0.0),
    "cal_ood":  (8, 20, -0.8, 2.0, 0.5, 2.0, 1.0),
    "test_ood": (30, 20, -0.8, 2.0, 0.5, 2.0, 1.0),
}


def make_scores(model: str = "synthetic-judge", seed: int = 0) -> list[dict]:
    """z is the true success log-odds; y ~ Bernoulli(sigmoid(z)).
    The judge sees a degraded, scaled, biased version of z."""
    rng = np.random.default_rng(seed)
    rows = []
    for split, (n_tasks, per, mu, sd, signal, over, bias) in SHIFT.items():
        domain = "ood" if "ood" in split else "sim"
        for t in range(n_tasks):
            task_mu = mu + rng.normal(0, 0.7)  # task-level correlation
            for i in range(per):
                z = rng.normal(task_mu, sd)
                y = int(rng.random() < 1 / (1 + np.exp(-z)))
                seen = signal * z + (1 - signal) * rng.normal(0, sd)
                logit = over * seen + bias
                rows.append({"uid": f"{split}:{t}:{i}", "model": model, "split": split, "domain": domain,
                             "task": f"{split}_task{t}", "label": y, "logit": float(logit)})
    return rows
