"""LoRA fine-tune a VLM judge on the train split.

No checkpoint selection on cal or test: the final adapter after a fixed budget
is the one that gets evaluated. Selecting on the calibration split would
contaminate the confidence layer; selecting on test would make the numbers
meaningless.

Example (Kaggle T4, 16 GB):
  python scripts/train.py --manifests manifests/ --out adapters/qwen3b-lora \
    --load-in-4bit --steps 1500 --grad-accum 8
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judgecal.data import load_manifest  # noqa: E402
from judgecal.judge import QwenVLJudge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500, help="optimizer steps")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=100,
                    help="save the adapter every N steps; 0 disables")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    train = load_manifest(Path(a.manifests) / "train.jsonl")
    pos = [e for e in train if e.label == 1]
    neg = [e for e in train if e.label == 0]
    print(f"train: {len(pos)} success, {len(neg)} failure")

    judge = QwenVLJudge(a.model_id, max_pixels=a.max_pixels, load_in_4bit=a.load_in_4bit)
    if a.load_in_4bit:
        from peft import prepare_model_for_kbit_training
        judge.model = prepare_model_for_kbit_training(judge.model, use_gradient_checkpointing=True)
    else:
        judge.model.gradient_checkpointing_enable()
        judge.model.enable_input_require_grads()
    # q/k/v/o_proj names match the language model only; the vision tower is left frozen.
    cfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    judge.model = get_peft_model(judge.model, cfg)
    judge.model.print_trainable_parameters()
    judge.model.train()

    opt = torch.optim.AdamW([p for p in judge.model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)

    def sample_batch():
        # Class-balanced sampling: failures are often the minority.
        return [rng.choice(pos if rng.random() < 0.5 else neg) for _ in range(a.batch_size)]

    t0, running = time.time(), None
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.grad_accum):
            batch = sample_batch()
            z = judge.forward_logits(batch)
            y = torch.tensor([e.label for e in batch], dtype=z.dtype, device=z.device)
            loss = F.binary_cross_entropy_with_logits(z, y) / a.grad_accum
            loss.backward()
            running = loss.item() * a.grad_accum if running is None else 0.98 * running + 0.02 * loss.item() * a.grad_accum
        torch.nn.utils.clip_grad_norm_(judge.model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0 or step == a.steps - 1:
            print(f"step {step:5d}  loss(ema) {running:.4f}  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s",
                  flush=True)
        # Save periodically. Saving only at the end means a session that is killed at its wall --
        # which a long fine-tune on a fixed-length runtime can genuinely hit -- yields nothing at
        # all, discarding every GPU-hour spent. This is NOT checkpoint selection: the adapter that
        # gets evaluated is still whichever one the fixed budget ends on, and nothing here consults
        # cal or test.
        if a.checkpoint_every and (step + 1) % a.checkpoint_every == 0:
            judge.model.save_pretrained(a.out)
            Path(a.out).joinpath("PROGRESS.json").write_text(
                json.dumps({"step": step + 1, "of": a.steps, "loss_ema": running}))
            print(f"  checkpointed at step {step + 1}", flush=True)

    judge.model.save_pretrained(a.out)
    print(f"saved adapter to {a.out}")


if __name__ == "__main__":
    main()
