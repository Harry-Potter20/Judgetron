# Real-data smoke run — a plumbing check, not a result

First end-to-end run on the real Guardian data: Kaggle T4, Qwen2.5-VL-3B-Instruct in 4-bit,
zero-shot only, `--limit 24` episodes per split.

**Do not read these numbers as findings.** With 24 episodes per split, `test_id` has 5 task
clusters and `test_ood` has 2, and the bootstrap resamples whole tasks — so the intervals are as
wide as the range itself (`test_ood` AUROC 0.799 [0.375, 0.875]). That the CIs are that wide is the
machinery working, not failing.

What this run establishes, which is all it was for:

- The adapter reads the real metadata correctly. Splits came out
  train 5,483 / cal 1,181 / test_id 1,166 (bridge) and cal_ood 135 / test_ood 435 (ur5), and the
  task-disjointness assertion passed on real data.
- Pooling UR5's upstream splits gives 24 task clusters in `test_ood` rather than 5.
- Qwen2.5-VL-3B loads and scores in 4-bit on a 16 GB T4 at roughly 1.3 s/episode at batch 2.
- The confidence layer, conformal sets, selective policy and report all run on real logits.

One behaviour worth carrying forward into the full run rather than concluding from here: zero-shot
failure recall is 0.000 on `test_id` at a threshold of 0, with balanced accuracy exactly 0.500 —
the judge answers "Yes" to essentially everything. If that holds at full scale it is the whole
motivation for the confidence layer, but at n=24 it is not yet evidence of anything.
