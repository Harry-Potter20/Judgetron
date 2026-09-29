# Zero-shot vs LoRA, on the real Guardian data

Kaggle T4, Qwen2.5-VL-3B in 4-bit. Zero-shot scored 6,742 episodes in 2.3 h; LoRA trained 600 steps
(effective batch 8, loss 0.852 -> 0.456) then scored the same grid. In-domain = RLBench-Fail (sim) +
BridgeDataV2-Fail (real WidowX); OOD = UR5-Fail. Task-disjoint splits, CIs resample whole tasks.

## Fine-tuning works in-domain, and mostly does not transfer

| | zero-shot | LoRA | delta |
|---|---|---|---|
| `test_id` AUROC (96 tasks) | 0.588 [0.562, 0.620] | **0.775 [0.727, 0.837]** | **+0.187** |
| `test_ood` AUROC (24 tasks) | 0.650 [0.573, 0.722] | 0.690 [0.640, 0.757] | +0.040 |
| `test_id` failure recall | 0.097 | 0.574 | +0.477 |
| `test_id` AURC | 0.406 | 0.166 | -0.240 |

In-domain the CIs do not overlap (0.620 vs 0.727), so the in-domain gain is real. On the new robot
the gain is a fifth of the size and the CIs overlap heavily. LoRA learned the training domains, not
the task of judging robot failure.

## The headline: a threshold that looks safe in-domain is 5x over target on the new robot

Selective policy, threshold chosen on `cal` for 5% error on auto-accepted verdicts.

| fitted on | split | auto-accept rate | realised error | failures auto-passed |
|---|---|---|---|---|
| `cal` | `test_id` | 0.145 | **0.018** | 5 / 1511 |
| `cal` | `test_ood` | 0.079 | **0.265** | 0 / 206 |
| `cal_ood` (refit) | `test_ood` | 0.000 | — | 0 / 206 |

Zero-shot could not automate anything anywhere (auto-accept 0.000 in every condition). Fine-tuning
buys real automation in-domain: 14.5% of verdicts auto-accepted at 1.8% realised error, comfortably
inside the 5% target.

Carry that same threshold to the new robot and it auto-accepts 7.9% of verdicts at **26.5% error** --
five times the target it was built to hold. Nothing about the in-domain numbers warns you: the
policy was validated at 1.8% error and silently degrades by an order of magnitude on deployment.
This is the failure the repo exists to measure, and it only becomes visible because the judge was
good enough to automate anything at all.

Refitting the confidence layer on a small labelled slice of the new robot drops automation back to
0.000 -- the refit correctly concludes that nothing there is safe to auto-accept. That is the right
answer, and it is what a customer sending back corrections would buy.

## Fine-tuning degrades per-class conformal coverage

Mondrian conformal, alpha = 0.1, target 0.90 per class.

| fitted on | split | model | coverage, failures | coverage, successes | singleton rate |
|---|---|---|---|---|---|
| `cal` | `test_id` | zero-shot | 0.927 | 0.950 | 0.181 |
| `cal` | `test_id` | LoRA | **0.860** | 0.934 | 0.521 |
| `cal` | `test_ood` | zero-shot | 0.976 | 0.803 | 0.350 |
| `cal` | `test_ood` | LoRA | 0.937 | **0.682** | 0.594 |
| `cal_ood` | `test_ood` | LoRA | 0.937 | 0.848 | 0.352 |

The fine-tuned judge commits far more often (singleton rate 0.181 -> 0.521 in-domain), and pays for
it in coverage: failures fall to 0.860 in-domain, under the 0.90 target, and successes to 0.682 on
the new robot. Higher AUROC did not buy better-behaved verdict sets.

## Caveats

- **Six episodes were lost to a uid collision.** `--ood` is repeatable and UR5 pools three jsonls;
  uids were built from domain + episode_id + line index, all of which restart per file, so six
  `test_ood` and two `cal_ood` episodes collided and the merge deduplicated them away. `test_ood` is
  therefore n=429 / 206 failures rather than 435 / 210. Fixed (uid now carries the source file) and
  regression-tested, but these numbers predate the fix. The retained rows are correct; the sample is
  1.4% smaller than intended.
- **OOD AUROC still exceeds in-domain for the zero-shot arm** (0.650 vs 0.588). UR5 differs from the
  in-domain pool in task difficulty as well as robot, so "OOD" here is not a pure deployment shift.
- **24 task clusters** in `test_ood` against 96 in-domain, so OOD intervals are wide.
- **600 steps, not the 1,500 originally specified.** Resized from measured throughput to fit the
  session wall; the budget was fixed in advance and nothing selects on cal or test.
