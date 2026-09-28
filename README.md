# judgecal

**Does a VLM robot-failure judge's confidence survive deployment shift?**

VLMs are being used to judge whether a robot policy succeeded. Recent benchmarks measure how often they're right. This repo measures something a customer needs more: when the judge says it's confident, can you act on that, and does that still hold when you move from simulation to a real robot?

The pipeline:

1. **Judge.** Qwen2.5-VL asked "did the robot complete the task?" over before/after frames. The success logit is read directly from the `Yes` vs `No` token logits, so every judge (zero-shot or fine-tuned) produces a proper probability through the same code path.
2. **Fine-tune.** LoRA on the language model with BCE on that same logit, class-balanced sampling, fixed step budget. No checkpoint selection on calibration or test data.
3. **Confidence layer**, fitted on a calibration split only:
   - Temperature and Platt recalibration: is "90% sure" right 90% of the time?
   - Mondrian split conformal: verdict sets with a coverage target **per class**, so failures (usually the minority, always the costly case) are covered on their own terms, not averaged away.
   - Selective prediction: a confidence threshold that auto-accepts verdicts at a target error rate and routes the rest to a human. Reported as automation rate, realised error, and failures wrongly auto-passed.
4. **Evaluation under shift.** Train and calibrate on sim + one real robot, test on a held-out real robot. Then refit the confidence layer on a small labelled slice of the new robot, which models a customer sending back corrections.

Every metric has a 95% CI from a bootstrap that resamples **whole tasks**, because episodes of the same task are correlated and episode-level resampling understates uncertainty. Splits are **task-disjoint** and checked for leakage.

## Status

- Pipeline, confidence layer, metrics and report: implemented, 11 tests passing, verified end to end on synthetic scores (`results_synthetic/`). Synthetic numbers are a pipeline check, not a result.
- Real-data results: **not yet run.** Pending GPU time.

## Data

Guardian failure-detection datasets (Pacaud et al.) on Hugging Face: RLBench-Fail (sim, Franka), BridgeDataV2-Fail (real, WidowX), UR5-Fail (real, UR5, failures from a deployed policy). Metadata is jsonl with `images`, `execution_reward` (1 success, 0 failure) and `failure_mode`. Check each dataset card and confirm field names before building splits; the adapter tries several candidate keys for instruction and task and warns when it falls back.

Default protocol: in-domain = RLBench-Fail + BridgeDataV2-Fail (split by task into train / cal / test_id), OOD = UR5-Fail (split by task into cal_ood 20% / test_ood 80%).

## Run

```bash
pip install -r requirements.txt
python -m pytest -q tests

# 1. data (repo ids from the Guardian collection on HF)
huggingface-cli download paulpacaud/ur5fail_val_dataset --repo-type dataset --local-dir data/ur5_fail
# ...same for RLBench-Fail and BridgeDataV2-Fail

# 2. task-disjoint splits
python scripts/prepare_splits.py \
  --source rlbench=data/rlbench_fail/metadata_execution.jsonl \
  --source bridge=data/bridge_fail/metadata_execution.jsonl \
  --ood ur5=data/ur5_fail/metadata_execution.jsonl --out manifests/

# 3. zero-shot baseline
python scripts/score.py --manifests manifests/ --scores results/scores.csv --name qwen3b-zeroshot

# 4. LoRA fine-tune (Kaggle T4: --load-in-4bit), then score
python scripts/train.py --manifests manifests/ --out adapters/qwen3b-lora --load-in-4bit --steps 1500
python scripts/score.py --manifests manifests/ --scores results/scores.csv \
  --adapter adapters/qwen3b-lora --name qwen3b-lora --load-in-4bit

# 5. report
python scripts/evaluate.py --scores results/scores.csv --out results/
```

Smoke-test scoring first with `--limit 20`.

No GPU? `python scripts/synthetic_demo.py && python scripts/evaluate.py --scores results_synthetic/scores.csv --out results_synthetic`.

## Questions the report answers

- Zero-shot vs fine-tuned: does LoRA improve discrimination in-domain, and does it survive the move to a new robot, or does it overfit the training domain?
- Is the fine-tuned judge more or less calibrated than zero-shot, before and after recalibration?
- Does per-class conformal coverage hold on the new robot when calibrated only in-domain? How many failures fall outside their set?
- At a 5% target error on auto-accepted verdicts, what error do you actually get on the new robot, and how many real failures get auto-passed?
- How much of that is recovered by refitting on a small labelled slice of the new robot?

## Design notes and known limits

- Recalibration is monotone, so it does not change AUROC, conformal sets or which episodes are most confident. It changes the probability a customer sees. Conformal and selective results are therefore close to identical across raw and temperature; that is expected, not a bug.
- Conformal guarantees assume calibration and test episodes are exchangeable. Task-disjoint splits already violate this mildly; cross-robot splits violate it strongly. Measuring that violation is the point.
- The selective threshold is the plain empirical choice on the calibration split, with no finite-sample guarantee. A Learn-then-Test style bound is the obvious next step.
- Bootstrap ECE is biased upward, so its point estimate can sit near the bottom of its CI.
- Judge sees start and end frames only. Contact-rich failures that are invisible in the final frame are out of reach by construction and should show up as a failure mode, not be hidden.

## Author

Chukwudi Eke. Physician and independent ML researcher working on evaluation and calibration for medical imaging models. Same problem, different body.
