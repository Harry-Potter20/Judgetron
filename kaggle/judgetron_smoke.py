"""Kaggle T4 entry point for judgecal. See scripts/kaggle_run.py for the pipeline.

Stage is read from an env var so the same kernel serves the smoke and full runs:
    JUDGETRON_STAGE=smoke JUDGETRON_LIMIT=24   (default; ~1.1 GB, zero-shot only)
    JUDGETRON_STAGE=full                        (~9.9 GB, zero-shot + LoRA)
"""
import os
import subprocess
import sys

STAGE = os.environ.get("JUDGETRON_STAGE", "full")
LIMIT = os.environ.get("JUDGETRON_LIMIT", "24")

# Qwen2.5-VL needs transformers>=4.49; Kaggle's image is usually older.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.51", "accelerate>=0.33", "peft>=0.12",
                "bitsandbytes>=0.43", "huggingface_hub>=0.26"], check=True)

from huggingface_hub import snapshot_download  # noqa: E402

code = snapshot_download("Chucks90/judgetron", repo_type="model",
                         local_dir="/kaggle/working/judgetron_code")

cmd = [sys.executable, f"{code}/scripts/kaggle_run.py", "--stage", STAGE,
       "--code", code, "--data", "/kaggle/tmp/data", "--out", "/kaggle/working"]
if STAGE == "smoke":
    cmd += ["--limit", LIMIT, "--skip-lora"]
if os.environ.get("JUDGETRON_SKIP_LORA", "1") == "1":
    # Zero-shot scoring of the full grid is ~2.4 h; LoRA (1500 steps x 8 accum = 12,000 passes)
    # is another 4-7 h. Together they overrun a single Kaggle session, so the two run as separate
    # kernels rather than as one job that dies at the 12 h wall with nothing saved.
    cmd += ["--skip-lora"]
subprocess.run(cmd, check=True)
