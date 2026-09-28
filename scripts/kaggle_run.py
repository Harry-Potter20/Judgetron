"""Self-contained remote run: pulls code and data from the Hub, scores, evaluates.

Nothing is downloaded to a workstation. The Guardian tarballs total ~68 GB, and rlbench's test
tarball alone is 45.8 GB because the archives carry `global.avi` videos alongside the frames, so
only the tarballs a given stage actually needs are fetched, on the machine that will read them.

Stages, smallest first, so a failure costs minutes rather than an hour of GPU:
  --stage smoke   bridge + ur5, --limit episodes per split, zero-shot only   (~1.1 GB)
  --stage full    adds rlbench train, all splits, zero-shot + LoRA           (~9.9 GB)

Usage on Kaggle (T4, 4-bit):
  python kaggle_run.py --stage smoke --limit 24
  python kaggle_run.py --stage full
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

CODE_REPO = "Chucks90/judgetron"

# domain -> list of HF dataset repos to pool. prepare_splits does its own task-disjoint split, so
# a source's upstream train/val/test division is not reused; pooling only widens the task pool.
#
# ur5 pools all three of its splits deliberately. It is the OOD domain, so it carries the headline
# numbers, and the bootstrap resamples whole TASKS: ur5 train alone has just 7 taskvars, which after
# the cal_ood/test_ood split leaves 5 task clusters in test_ood and CIs too wide to read. Pooling
# gives 34 taskvars over 570 episodes (train/test taskvars are disjoint upstream) for 0.3 GB total.
UR5_ALL = [("ur5", "paulpacaud/ur5fail_train_dataset"),
           ("ur5", "paulpacaud/ur5fail_val_dataset"),
           ("ur5", "paulpacaud/ur5fail_test_dataset")]

SOURCES = {
    "smoke": {
        "in_domain": [("bridge", "paulpacaud/bdv2fail_train_dataset")],
        "ood": UR5_ALL,
    },
    "full": {
        # rlbench train is 8.78 GB; its val/test tarballs (11.9 / 45.8 GB) are deliberately not used
        # -- they are mostly .avi video, and one source split already gives 70 taskvars.
        "in_domain": [("rlbench", "paulpacaud/rlbenchfail_train_dataset"),
                      ("bridge", "paulpacaud/bdv2fail_train_dataset")],
        "ood": UR5_ALL,
    },
}


def sh(cmd: list[str], **kw) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def fetch_source(domain: str, repo: str, root: Path) -> Path:
    """Download one source's metadata + image tarball and extract it. Returns the metadata path."""
    from huggingface_hub import hf_hub_download

    out = root / repo.split("/")[-1]      # one dir per repo: pooled repos must not overwrite
    out.mkdir(parents=True, exist_ok=True)
    meta = hf_hub_download(repo, "metadata_execution.jsonl", repo_type="dataset",
                           local_dir=out)
    marker = out / ".extracted"
    if not marker.exists():
        t0 = time.time()
        tar = hf_hub_download(repo, "records.tar.gz", repo_type="dataset", local_dir=out)
        print(f"[{domain}] downloaded records.tar.gz in {time.time()-t0:.0f}s", flush=True)
        t0 = time.time()
        with tarfile.open(tar) as tf:
            # Only the frames are needed; the archives also carry .avi videos, which is most of
            # their size. Filtering here keeps peak disk well under Kaggle's working-dir limit.
            members = [m for m in tf.getmembers()
                       if m.isfile() and m.name.lower().endswith((".png", ".jpg", ".jpeg"))]
            tf.extractall(out, members=members)
        print(f"[{domain}] extracted {len(members)} frames in {time.time()-t0:.0f}s", flush=True)
        os.remove(tar)                      # reclaim the archive immediately
        marker.write_text("ok")
    return Path(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--limit", type=int, default=None, help="episodes per split (smoke)")
    ap.add_argument("--data", default="/kaggle/tmp/data")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--code", default=None, help="local judgecal checkout; otherwise pulled from HF")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--train-steps", type=int, default=1500)
    ap.add_argument("--skip-lora", action="store_true")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data); data.mkdir(parents=True, exist_ok=True)

    # ---- code
    if a.code:
        code = Path(a.code)
    else:
        from huggingface_hub import snapshot_download
        code = Path(snapshot_download(CODE_REPO, repo_type="model",
                                      local_dir=str(out / "judgetron_code")))
    sys.path.insert(0, str(code))
    print(f"[code] {code}", flush=True)

    # ---- data
    spec = SOURCES[a.stage]
    src_args = []
    for domain, repo in spec["in_domain"]:
        m = fetch_source(domain, repo, data)
        src_args += ["--source", f"{domain}={m}"]
    ood_args = []
    for domain, repo in spec["ood"]:
        m = fetch_source(domain, repo, data)
        ood_args += ["--ood", f"{domain}={m}"]

    # ---- splits
    manifests = out / "manifests"
    sh([sys.executable, str(code / "scripts" / "prepare_splits.py"), *src_args,
        *ood_args, "--out", str(manifests)])

    # ---- zero-shot
    scores = out / "results" / "scores.csv"
    scores.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(code / "scripts" / "score.py"),
           "--manifests", str(manifests), "--scores", str(scores),
           "--model-id", a.model_id, "--load-in-4bit",
           "--batch-size", str(a.batch_size), "--name", "qwen3b-zeroshot"]
    if a.limit:
        cmd += ["--limit", str(a.limit)]
    sh(cmd)

    # ---- LoRA
    if not a.skip_lora:
        adapter = out / "adapters" / "qwen3b-lora"
        sh([sys.executable, str(code / "scripts" / "train.py"),
            "--manifests", str(manifests), "--out", str(adapter),
            "--model-id", a.model_id, "--load-in-4bit", "--steps", str(a.train_steps)])
        cmd = [sys.executable, str(code / "scripts" / "score.py"),
               "--manifests", str(manifests), "--scores", str(scores),
               "--model-id", a.model_id, "--adapter", str(adapter), "--load-in-4bit",
               "--batch-size", str(a.batch_size), "--name", "qwen3b-lora"]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        sh(cmd)

    # ---- report
    sh([sys.executable, str(code / "scripts" / "evaluate.py"),
        "--scores", str(scores), "--out", str(out / "results")])
    print(json.dumps({"stage": a.stage, "scores": str(scores),
                      "report": str(out / "results" / "report.md")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
