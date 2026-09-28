import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from judgecal.confidence import Conformal, Platt, SelectivePolicy, Temperature, aurc  # noqa: E402
from judgecal.data import Episode, grouped_split, load_manifest, save_manifest  # noqa: E402
from judgecal.judge import DummyJudge  # noqa: E402
from judgecal.metrics import auroc, cluster_bootstrap_ci, ece, sigmoid  # noqa: E402
from judgecal.report import evaluate, load_scores, render_markdown, write_scores  # noqa: E402
from judgecal.synthetic import make_scores  # noqa: E402


def test_auroc_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.round(rng.random(500) * 0.5 + 0.3 * y, 2)  # includes ties
    assert abs(auroc(p, y) - sk.roc_auc_score(y, p)) < 1e-12


def test_ece_zero_when_perfectly_calibrated():
    rng = np.random.default_rng(1)
    p = rng.random(200000)
    y = (rng.random(200000) < p).astype(int)
    assert ece(p, y) < 0.01


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(2)
    true = rng.normal(0, 1.5, 20000)
    y = (rng.random(20000) < sigmoid(true)).astype(int)
    t = Temperature().fit(3.0 * true, y)
    assert abs(t.T - 3.0) < 0.2
    assert ece(t(3.0 * true), y) < ece(sigmoid(3.0 * true), y)


def test_platt_absorbs_bias():
    rng = np.random.default_rng(3)
    true = rng.normal(0, 1.5, 20000)
    y = (rng.random(20000) < sigmoid(true)).astype(int)
    pl = Platt().fit(true + 2.0, y)
    assert abs(pl.a - 1) < 0.1 and abs(pl.b + 2.0) < 0.15


def test_conformal_coverage_exchangeable():
    """Mondrian coverage should be >= 1 - alpha per class on exchangeable data."""
    rng = np.random.default_rng(4)
    covs = []
    for _ in range(200):
        true = rng.normal(0, 1.5, 600)
        y = (rng.random(600) < sigmoid(true)).astype(int)
        p = sigmoid(true)
        c = Conformal(alpha=0.1).fit(p[:300], y[:300])
        r = c.evaluate(p[300:], y[300:])
        covs.append((r["cov_failures"], r["cov_successes"]))
    covs = np.array(covs).mean(0)
    assert covs[0] >= 0.89 and covs[1] >= 0.89


def test_selective_policy_hits_target_in_distribution():
    rng = np.random.default_rng(5)
    true = rng.normal(0, 2.0, 20000)
    y = (rng.random(20000) < sigmoid(true)).astype(int)
    p = sigmoid(true)
    s = SelectivePolicy(target_risk=0.05).fit(p[:10000], y[:10000])
    r = s.evaluate(p[10000:], y[10000:])
    assert r["auto_risk"] < 0.065 and 0 < r["auto_rate"] < 1


def test_aurc_better_for_informative_scores():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 2000)
    good = sigmoid(3 * (2 * y - 1) + rng.normal(0, 2, 2000))
    bad = rng.random(2000)
    assert aurc(good, y) < aurc(bad, y)


def test_cluster_bootstrap_wider_than_iid():
    """With strong task correlation, clustered CIs must be wider than iid ones."""
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(20), 50)
    task_rate = rng.random(20)[groups]
    y = (rng.random(1000) < task_rate).astype(int)
    p = np.clip(task_rate + rng.normal(0, 0.1, 1000), 0, 1)
    _, lo_c, hi_c, _ = cluster_bootstrap_ci(ece, p, y, groups, n_boot=500)
    _, lo_i, hi_i, _ = cluster_bootstrap_ci(ece, p, y, np.arange(1000), n_boot=500)
    assert (hi_c - lo_c) > (hi_i - lo_i)


def test_grouped_split_is_task_disjoint(tmp_path):
    eps = [Episode(uid=f"{t}:{i}", images=[], instruction="x", label=i % 2, domain="d", task=f"t{t}")
           for t in range(40) for i in range(10)]
    sp = grouped_split(eps, {"train": 0.7, "cal": 0.15, "test_id": 0.15}, seed=0)
    tasks = [{e.task for e in v} for v in sp.values()]
    assert not (tasks[0] & tasks[1]) and not (tasks[0] & tasks[2]) and not (tasks[1] & tasks[2])
    assert sum(len(v) for v in sp.values()) == 400
    save_manifest(sp["cal"], tmp_path / "cal.jsonl")
    assert load_manifest(tmp_path / "cal.jsonl") == sp["cal"]


def test_dummy_judge_roundtrip():
    eps = [Episode(uid=str(i), images=[], instruction="", label=1, domain="d", task="t", meta={"latent": i})
           for i in range(3)]
    assert DummyJudge().logits(eps).tolist() == [0, 1, 2]


def test_end_to_end_synthetic(tmp_path):
    path = tmp_path / "scores.csv"
    write_scores(path, make_scores(seed=0))
    scores = load_scores(path)
    res = evaluate(scores, n_boot=200, cal_split="cal")
    raw = res["synthetic-judge"]["raw"]["splits"]
    temp = res["synthetic-judge"]["temperature"]["splits"]
    # Recalibration fixes in-domain overconfidence.
    assert temp["test_id"]["ece"][0] < raw["test_id"]["ece"][0]
    # Under shift, discrimination drops.
    assert raw["test_ood"]["auroc"][0] < raw["test_id"]["auroc"][0]
    # In-domain conformal guarantee roughly holds.
    assert temp["test_id"]["conformal"]["coverage"] > 0.85
    md = render_markdown(res, 0.1, 0.05)
    assert "test_ood" in md and "—" not in md
    # Refit on a small OOD slice.
    res2 = evaluate(scores, n_boot=200, cal_split="cal_ood")
    assert "test_ood" in res2["synthetic-judge"]["platt"]["splits"]
