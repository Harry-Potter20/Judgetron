# Zero-shot Qwen2.5-VL-3B on the real Guardian data

Kaggle T4, 4-bit, 6,742 episodes scored in 2.3 h. In-domain = RLBench-Fail (sim) +
BridgeDataV2-Fail (real WidowX), task-disjoint; OOD = UR5-Fail (real UR5), never trained or
calibrated on except the small `cal_ood` slice used for the refit. CIs resample whole tasks.

## The judge is weak, and the confidence layer says so rather than hiding it

| split | n | tasks | AUROC | bal. acc | failure recall (raw) |
|---|---|---|---|---|---|
| `test_id` | 3,064 | 96 | 0.588 [0.562, 0.620] | 0.523 [0.512, 0.536] | 0.097 [0.066, 0.143] |
| `test_ood` | 435 | 24 | 0.649 [0.573, 0.722] | 0.600 [0.531, 0.648] | 0.400 [0.207, 0.560] |

An AUROC of 0.588 on 96 held-out tasks is barely above chance, and raw failure recall of 0.097 means
the zero-shot judge waves through roughly nine of every ten failures at a threshold of 0.

## The headline: no automation is safe at the 5% target

**The auto-accept rate is 0.000 in every condition** — both splits, all three calibrators, whether
the confidence layer is fitted in-domain or refitted on the new robot. There is no confidence
threshold at which this judge's verdicts can be auto-accepted at a 5% error target. Not "automation
is low": it is zero.

This is a real finding and not a stuck value. The same code on the n=24 smoke run produced
`auto_rate` of 0.958 and 1.0, so the threshold search does return non-zero when the scores support
it. At full n it correctly finds that nothing does.

## Per-class conformal coverage breaks on the new robot, in the success class

Mondrian conformal, alpha = 0.1, so the target is 0.90 per class.

| fitted on | split | coverage, failures | coverage, successes |
|---|---|---|---|
| `cal` (in-domain) | `test_id` | 0.927 | 0.950 |
| `cal` (in-domain) | `test_ood` | 0.976 | **0.800** |
| `cal_ood` (refit) | `test_ood` | 0.943 | 0.960 |

Calibrated in-domain, both classes hold in-domain. Moved to UR5, failures are *over*-covered (0.976)
while successes fall to 0.800, well under target — the exchangeability assumption breaking in
exactly the way a cross-robot split predicts. Refitting on a small labelled slice of the new robot
repairs it, which is the "customer sends back corrections" path the protocol was built to measure.

Set sizes make the same point: 0.82 of verdicts are ambiguous (both labels) on both splits. The
judge rarely commits.

## Two things not to over-read

- **OOD AUROC (0.649) is higher than in-domain (0.588).** That inverts the expected direction, but
  the CIs overlap substantially, `test_ood` has 24 task clusters against `test_id`'s 96, and "OOD"
  here differs from in-domain in task difficulty as well as robot. This is not evidence of
  robustness to deployment shift; it is a reason to be careful about what the in-domain pool
  contains.
- **Recalibration does not move AUROC, conformal sets, or the ranking.** It is monotone, exactly as
  the design notes say. What Platt does move is the operating point: raw failure recall 0.097 → 0.499
  on `test_id` and 0.400 → 0.781 on `test_ood`, at a cost in balanced accuracy. ECE improves from
  0.070 to 0.014 in-domain.

## Not yet run

LoRA fine-tuning, and therefore the zero-shot vs fine-tuned comparison that the repo's first
question asks. This file covers the zero-shot arm only.
