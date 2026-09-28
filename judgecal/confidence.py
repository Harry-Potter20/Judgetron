"""The confidence layer: turn raw judge logits into signals a customer can act on.

Three tools, each answering a different customer question:

1. Recalibration (temperature, Platt): "When the judge says 90%, is it right 90% of the time?"
2. Conformal prediction sets: "Give me a verdict with a guaranteed error rate, or tell me it can't."
3. Selective prediction: "Auto-accept verdicts above a threshold, route the rest to a human.
   What error rate and what automation rate do I get?"

All three are fitted on a calibration split and evaluated on held-out splits.
Their guarantees assume the test data is exchangeable with calibration data.
The point of this repo is to measure what happens when it isn't (sim to real).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .metrics import nll, sigmoid


# ---------------------------------------------------------------- recalibration

@dataclass
class Temperature:
    T: float = 1.0

    def fit(self, z: np.ndarray, y: np.ndarray) -> "Temperature":
        res = minimize_scalar(
            lambda logT: nll(sigmoid(z / np.exp(logT)), y), bounds=(-4, 4), method="bounded"
        )
        self.T = float(np.exp(res.x))
        return self

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return sigmoid(np.asarray(z) / self.T)


@dataclass
class Platt:
    """Affine recalibration a*z + b. Unlike temperature, the bias can absorb a
    shift in success base rate between domains, at the cost of one more parameter."""
    a: float = 1.0
    b: float = 0.0

    def fit(self, z: np.ndarray, y: np.ndarray) -> "Platt":
        res = minimize(lambda w: nll(sigmoid(w[0] * z + w[1]), y), x0=[1.0, 0.0], method="L-BFGS-B")
        self.a, self.b = map(float, res.x)
        return self

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return sigmoid(self.a * np.asarray(z) + self.b)


@dataclass
class Identity:
    def fit(self, z, y):
        return self

    def __call__(self, z):
        return sigmoid(np.asarray(z))


CALIBRATORS = {"raw": Identity, "temperature": Temperature, "platt": Platt}


# ---------------------------------------------------------------- conformal

@dataclass
class Conformal:
    """Split conformal classification with the LAC score s = 1 - p(true class).

    mondrian=True computes a separate threshold per true class, so coverage holds
    for failures and successes separately. Failures are usually the minority, and a
    marginal guarantee can be met while failures are under-covered. That is the
    wrong way round for an eval product.
    """
    alpha: float = 0.1
    mondrian: bool = True
    qhat: dict | None = None

    @staticmethod
    def _q(scores: np.ndarray, alpha: float) -> float:
        n = len(scores)
        if n == 0:
            return 1.0
        level = np.ceil((n + 1) * (1 - alpha)) / n
        if level > 1:
            return 1.0  # too few calibration points: include everything
        return float(np.quantile(scores, level, method="higher"))

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Conformal":
        p, y = np.asarray(p, float), np.asarray(y, int)
        s = np.where(y == 1, 1 - p, p)
        if self.mondrian:
            self.qhat = {c: self._q(s[y == c], self.alpha) for c in (0, 1)}
        else:
            q = self._q(s, self.alpha)
            self.qhat = {0: q, 1: q}
        return self

    def predict_sets(self, p: np.ndarray) -> np.ndarray:
        """Boolean array (n, 2): column 0 = 'failure' in set, column 1 = 'success' in set."""
        p = np.asarray(p, float)
        return np.stack([p <= self.qhat[0], (1 - p) <= self.qhat[1]], axis=1)

    def evaluate(self, p: np.ndarray, y: np.ndarray) -> dict:
        y = np.asarray(y, int)
        sets = self.predict_sets(p)
        covered = sets[np.arange(len(y)), y]
        size = sets.sum(1)
        out = {
            "coverage": float(covered.mean()),
            "cov_failures": float(covered[y == 0].mean()) if (y == 0).any() else float("nan"),
            "cov_successes": float(covered[y == 1].mean()) if (y == 1).any() else float("nan"),
            "singleton_rate": float((size == 1).mean()),
            "ambiguous_rate": float((size == 2).mean()),
            "empty_rate": float((size == 0).mean()),
        }
        # Among singletons, how often is the committed verdict right?
        single = size == 1
        out["singleton_acc"] = float(covered[single].mean()) if single.any() else float("nan")
        return out


# ---------------------------------------------------------------- selective prediction

def risk_coverage(p: np.ndarray, y: np.ndarray):
    """Error rate among the k most confident predictions, for every k."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    conf = np.maximum(p, 1 - p)
    order = np.argsort(-conf, kind="stable")
    err = (((p >= 0.5).astype(int)) != y)[order].astype(float)
    k = np.arange(1, len(y) + 1)
    return k / len(y), np.cumsum(err) / k


def aurc(p: np.ndarray, y: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    _, risk = risk_coverage(p, y)
    return float(risk.mean())


@dataclass
class SelectivePolicy:
    """Pick a confidence threshold on calibration data to hit a target error rate
    on auto-accepted verdicts; everything below it goes to human review."""
    target_risk: float = 0.05
    min_accept: int = 20
    threshold: float = 1.01

    def fit(self, p: np.ndarray, y: np.ndarray) -> "SelectivePolicy":
        p, y = np.asarray(p, float), np.asarray(y, int)
        conf = np.maximum(p, 1 - p)
        order = np.argsort(-conf, kind="stable")
        err = (((p >= 0.5).astype(int)) != y)[order].astype(float)
        k = np.arange(1, len(y) + 1)
        risk = np.cumsum(err) / k
        ok = np.flatnonzero((risk <= self.target_risk) & (k >= self.min_accept))
        self.threshold = float(conf[order][ok.max()]) if len(ok) else 1.01
        return self

    def evaluate(self, p: np.ndarray, y: np.ndarray) -> dict:
        p, y = np.asarray(p, float), np.asarray(y, int)
        acc = np.maximum(p, 1 - p) >= self.threshold
        err = ((p >= 0.5).astype(int) != y)
        # Failures that got auto-labelled "success": the costly mistake.
        missed_fail = acc & (y == 0) & (p >= 0.5)
        return {
            "threshold": self.threshold,
            "auto_rate": float(acc.mean()),
            "auto_risk": float(err[acc].mean()) if acc.any() else float("nan"),
            "missed_failures_auto": int(missed_fail.sum()),
            "n_failures": int((y == 0).sum()),
        }
