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

- Pipeline, confidence layer, metrics and report: implemented, 17 tests passing, verified end to end on synthetic scores (`results_synthetic/`). Synthetic numbers are a pipeline check, not a result.
- Adapter verified against the **real** Guardian datasets (field names, path layouts, frame counts, task cardinality), which corrected three defects that would have produced plausible but meaningless numbers — see Data below.
- **Zero-shot real-data results: in** (`results_zeroshot/`). 6,742 episodes on a Kaggle T4.
  Headline: the auto-accept rate is **0.000 in every condition** — no confidence threshold lets this
  judge's verdicts be auto-accepted at a 5% error target. In-domain AUROC 0.588 [0.562, 0.620] over
  96 held-out tasks. Per-class conformal coverage holds in-domain but the success class falls to
  0.800 on the new robot (target 0.90), and refitting on a small labelled slice of it repairs that.
- **LoRA results: in** (). Fine-tuning lifts in-domain AUROC 0.588 -> 0.775 with
  non-overlapping CIs, but only 0.650 -> 0.690 on the new robot. It buys real automation in-domain
  (14.5% of verdicts auto-accepted at 1.8% realised error against a 5% target) -- and that same
  threshold auto-accepts 7.9% at **26.5% error** on the new robot, five times its target, with
  nothing in the in-domain numbers to warn you. Refitting on a small labelled slice of the new robot
  correctly drops automation to zero.

## Data

Guardian failure-detection datasets (Pacaud et al.) on Hugging Face: RLBench-Fail (sim, Franka), BridgeDataV2-Fail (real, WidowX), UR5-Fail (real, UR5, failures from a deployed policy). Metadata is jsonl with `images`, `execution_reward` (1 success, 0 failure) and `failure_mode`.

Measured from the real val splits, not the cards:

| source | repo stem | train / val / test | taskvars (train) | frames per episode |
|---|---|---|---|---|
| RLBench-Fail | `paulpacaud/rlbenchfail_*` | 12,358 / 1,000 / 1,000 | 70 | 8 (4 viewpoints x start/end) |
| BridgeDataV2-Fail | `paulpacaud/bdv2fail_*` | 7,830 / 1,000 / 1,000 | 830 | 2 |
| UR5-Fail | `paulpacaud/ur5fail_*` | 400 / 30 / 140 | 7 / 7 / 23 | 6 (3 viewpoints x 2 times) |

All three are close to label-balanced. Three things about this data break the obvious adapter, and all three are handled:

- **Group by `taskvar`.** RLBench and UR5 carry no `task`/`task_name`/`task_id`/`env_name`, so a candidate-key adapter falls back to grouping by instruction — on UR5 that is 36 instruction groups spanning only 7 real taskvars, putting episodes of one taskvar into different splits. Bridge *does* have `task_name`, but it names the scene (13 values over 7,830 episodes) while its instructions are perturbed per episode (4,320 distinct), so both candidates are wrong in opposite directions.
- **Use start and end of one viewpoint.** Frame count varies with domain (8 / 6 / 2). Feeding all of them makes view count a domain confound, so an in-domain vs OOD gap would partly measure how many views the judge saw rather than deployment shift.
- **Anchor image paths at `records/`.** Bridge stores paths from the machine that built the dataset (`data/failure_forge/...`).

Default protocol: in-domain = RLBench-Fail + BridgeDataV2-Fail (split by task into train / cal / test_id), OOD = UR5-Fail (split by task into cal_ood 20% / test_ood 80%). `--ood` is repeatable, and UR5 pools all three of its upstream splits: the bootstrap resamples whole tasks, and UR5 train alone leaves 5 task clusters in `test_ood`, which makes the headline OOD intervals unreadable. Pooling gives 24 clusters over 435 episodes.

The image tarballs total ~68 GB (RLBench's test archive alone is 45.8 GB, mostly `.avi` video). `scripts/kaggle_run.py` runs the pipeline where the data is, fetching only the tarballs a stage needs and extracting only frames.

## Run

```bash
pip install -r requirements.txt
python -m pytest -q tests

# 1. data (repo ids from the Guardian collection on HF)
huggingface-cli download paulpacaud/ur5fail_train_dataset --repo-type dataset --local-dir data/ur5_fail
# ...same for RLBench-Fail and BridgeDataV2-Fail; see the size warning above

# 2. task-disjoint splits (--ood repeatable; pool UR5's splits for enough task clusters)
python scripts/prepare_splits.py \
  --source rlbench=data/rlbench_fail/metadata_execution.jsonl \
  --source bridge=data/bridge_fail/metadata_execution.jsonl \
  --ood ur5=data/ur5_fail_train/metadata_execution.jsonl \
  --ood ur5=data/ur5_fail_test/metadata_execution.jsonl --out manifests/

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

Or run the whole thing remotely, without staging 68 GB locally:

```bash
python scripts/kaggle_run.py --stage smoke --limit 24 --skip-lora   # ~1.1 GB, zero-shot
python scripts/kaggle_run.py --stage full                           # ~9.9 GB, + LoRA
```

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
