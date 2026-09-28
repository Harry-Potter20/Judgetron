"""Kaggle T4 entry point: LoRA fine-tune + score the adapter.

Separate kernel from the zero-shot scoring. Together they overrun a single Kaggle session; split,
each fits and neither loses its work if the other dies. Splits are regenerated here rather than
carried over -- prepare_splits is deterministic (seed 0, sorted task keys), so the manifests are
identical to the zero-shot kernel's.

Step budget is sized from MEASURED cost, not guessed. The zero-shot run scored 6,742 episodes in
8,303 s = 1.23 s/episode at batch 2. A training pass costs roughly 2-3x a forward pass, so the
originally-specified 1,500 steps x 4 accum = 6,000 passes works out at 8-12 h of training before
any scoring -- against a 12 h wall, with the adapter saved only at the end. That first attempt was
killed ~4 h in rather than gambling the whole run on finishing.

600 steps x 4 accum = 2,400 passes lands near 3.3 h of training plus 2.3 h of scoring. The budget is
still FIXED in advance and nothing selects on cal or test, which is the property Section "no
checkpoint selection" actually protects; only the total compute changed.

The scores.csv this produces holds only the LoRA rows; merge it with the zero-shot kernel's and
rerun evaluate.py (CPU-only) to get the combined report.
"""
import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.51", "accelerate>=0.33", "peft>=0.12",
                "bitsandbytes>=0.43", "huggingface_hub>=0.26"], check=True)

from huggingface_hub import snapshot_download  # noqa: E402

code = snapshot_download("Chucks90/judgetron", repo_type="model",
                         local_dir="/kaggle/working/judgetron_code")

subprocess.run([sys.executable, f"{code}/scripts/kaggle_run.py",
                "--stage", os.environ.get("JUDGETRON_STAGE", "full"),
                "--code", code, "--data", "/kaggle/tmp/data", "--out", "/kaggle/working",
                "--train-steps", os.environ.get("JUDGETRON_STEPS", "600"),
                "--train-batch-size", "2", "--grad-accum", "4"], check=True)
