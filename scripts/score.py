"""Score every split with a judge and append logits to a scores CSV.

Run once zero-shot and once per adapter:
  python scripts/score.py --manifests manifests/ --scores results/scores.csv
  python scripts/score.py --manifests manifests/ --scores results/scores.csv --adapter adapters/qwen3b-lora
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judgecal.data import load_manifest  # noqa: E402
from judgecal.report import write_scores  # noqa: E402

SPLITS = ("cal", "test_id", "cal_ood", "test_ood")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--scores", default="results/scores.csv")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", default=None, help="model label in the report")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="cap episodes per split (smoke tests)")
    a = ap.parse_args()

    from judgecal.judge import QwenVLJudge

    judge = QwenVLJudge(a.model_id, adapter=a.adapter, load_in_4bit=a.load_in_4bit)
    name = a.name or judge.name
    for split in SPLITS:
        path = Path(a.manifests) / f"{split}.jsonl"
        if not path.exists():
            continue
        eps = load_manifest(path)[: a.limit]
        z = judge.logits(eps, batch_size=a.batch_size)
        write_scores(a.scores, [
            {"uid": e.uid, "model": name, "split": split, "domain": e.domain,
             "task": e.task, "label": e.label, "logit": float(v)}
            for e, v in zip(eps, z)
        ])
        print(f"{name}: scored {split} (n={len(eps)})", flush=True)


if __name__ == "__main__":
    main()
