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


# ----------------------------------------------------------- real Guardian dataset conventions
# Field names and path layouts below are taken from the actual val splits of
# paulpacaud/{rlbenchfail,bdv2fail,ur5fail}_val_dataset, not from the dataset cards.

def _rlbench_imgs():
    b = "records/open_drawer_long+2/wrong_sequence/ep_4/run_0/subtask_0"
    return [f"{b}/{t}_img_viewpoint_{v}.png"
            for t in ("start", "end") for v in ("left", "right", "wrist", "front")]


def _bridge_imgs():
    b = "data/failure_forge/data/bdv2fail_val_dataset/records/numpy_256_x_train/9"
    return [f"{b}/{t}_img_viewpoint_front.png" for t in ("start", "end")]


def _ur5_imgs():
    b = "records/real_put_fruit_in_box+4/0"
    return [f"{b}/{t}_img_viewpoint_{v}.png" for t in (1, 6) for v in (0, 1, 2)]


def test_bridge_image_paths_are_anchored_at_records():
    """bdv2fail stores absolute paths from the machine that built the dataset
    (`data/failure_forge/data/bdv2fail_val_dataset/records/...`). Left alone, every bridge image
    would resolve to a directory that does not exist locally."""
    from judgecal.data import normalise_image_path
    out = normalise_image_path(_bridge_imgs()[0])
    assert out.startswith("records/")
    assert "failure_forge" not in out
    # rlbench/ur5 paths are already relative and must be left alone
    assert normalise_image_path(_rlbench_imgs()[0]) == _rlbench_imgs()[0]


def test_every_domain_contributes_the_same_number_of_frames():
    """The raw datasets give 8 frames (rlbench), 6 (ur5) and 2 (bridge). Handing those straight to
    the judge makes view count vary WITH domain, so an in-domain vs OOD gap would partly measure
    how many views the judge saw rather than deployment shift -- which is the entire question."""
    from judgecal.data import normalise_image_path, select_start_end
    for imgs in (_rlbench_imgs(), _bridge_imgs(), _ur5_imgs()):
        sel = select_start_end([normalise_image_path(p) for p in imgs])
        assert len(sel) == 2, f"expected start+end, got {len(sel)}"


def test_selected_frames_are_start_and_end_of_one_viewpoint():
    from judgecal.data import normalise_image_path, select_start_end
    r = select_start_end(_rlbench_imgs())
    assert r[0].endswith("start_img_viewpoint_front.png")
    assert r[1].endswith("end_img_viewpoint_front.png")
    # ur5 uses numeric timesteps, so ordering must be numeric, not lexicographic
    u = select_start_end(_ur5_imgs())
    assert u[0].endswith("1_img_viewpoint_0.png") and u[1].endswith("6_img_viewpoint_0.png")
    assert len({p.split("_img_viewpoint_")[1] for p in u}) == 1, "frames must share one viewpoint"


def test_unparseable_frame_names_fall_back_to_the_raw_list():
    """An unfamiliar dataset must degrade to previous behaviour, not silently drop frames."""
    from judgecal.data import select_start_end
    odd = ["records/x/frame_000.png", "records/x/frame_009.png", "records/x/frame_017.png"]
    assert select_start_end(odd) == odd


def test_taskvar_is_the_grouping_key(tmp_path):
    """ur5 and rlbench carry no task/task_name/task_id/env_name, so the adapter previously grouped
    by INSTRUCTION. On ur5 that is 36 instruction groups over only 7 real taskvars, which puts
    episodes of one taskvar into different splits -- the leakage grouped_split exists to prevent."""
    import json
    from judgecal.data import from_guardian_jsonl
    p = tmp_path / "meta.jsonl"
    rows = [
        # same taskvar, different (perturbed) instructions -- must stay one group
        {"taskvar": "put_fruit+1", "task_instruction": "put the lemon in the box",
         "execution_reward": 0, "images": _ur5_imgs(), "failure_mode": "no_grasp"},
        {"taskvar": "put_fruit+1", "task_instruction": "place the lemon into the box",
         "execution_reward": 1, "images": _ur5_imgs(), "failure_mode": None},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    eps = from_guardian_jsonl(p, domain="ur5")
    assert {e.task for e in eps} == {"put_fruit+1"}, "differing instructions must not split a taskvar"


def test_taskvar_beats_bridge_task_name(tmp_path):
    """bridge has task_name, but it names the SCENE (13 values over 7,830 episodes). Grouping on it
    would make the clustered bootstrap absurdly coarse and merge unrelated tasks."""
    import json
    from judgecal.data import from_guardian_jsonl
    p = tmp_path / "meta.jsonl"
    rows = [{"taskvar": f"scene_drawer_pnp_0{i}_train", "task_name": "tabletop_dark_wood",
             "task_instruction": f"move object {i}", "execution_reward": i % 2,
             "images": _bridge_imgs(), "failure_mode": None} for i in range(4)]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    eps = from_guardian_jsonl(p, domain="bridge")
    assert len({e.task for e in eps}) == 4, "taskvar must win over the coarser task_name"
    assert all(e.task != "tabletop_dark_wood" for e in eps)
