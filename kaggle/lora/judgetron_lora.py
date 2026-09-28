"""Kaggle T4 entry point: LoRA fine-tune + score the adapter.

Separate kernel from the zero-shot scoring. Together they are ~7-10 h of GPU, which overruns a
single Kaggle session; split, each fits and neither loses its work if the other dies. Splits are
regenerated here rather than carried over -- prepare_splits is deterministic (seed 0, sorted task
keys), so the manifests are identical to the zero-shot kernel's.

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
                "--train-steps", os.environ.get("JUDGETRON_STEPS", "1500"),
                "--train-batch-size", "2", "--grad-accum", "4"], check=True)
