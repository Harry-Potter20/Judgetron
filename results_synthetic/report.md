# Judge evaluation report (confidence layer fitted on `cal`)

Confidence layer fitted on `cal` only. 95% CIs from a bootstrap that resamples whole tasks. Conformal: Mondrian (per-class), alpha = 0.1, so the target is 90% coverage for failures and successes separately. Selective policy: threshold chosen on `cal` for 5% error on auto-accepted verdicts.

## synthetic-judge

### split: `test_id` (n = 600, failures = 302, tasks = 30)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.863 [0.832, 0.891] | 0.785 [0.750, 0.818] | 0.795 [0.749, 0.837] | 0.120 [0.100, 0.157] | 0.166 [0.143, 0.191] | 0.100 |
| temperature | 0.863 [0.832, 0.891] | 0.785 [0.750, 0.818] | 0.795 [0.749, 0.837] | 0.053 [0.049, 0.097] | 0.151 [0.135, 0.168] | 0.100 |
| platt | 0.863 [0.832, 0.891] | 0.778 [0.745, 0.811] | 0.762 [0.713, 0.807] | 0.046 [0.039, 0.092] | 0.151 [0.135, 0.169] | 0.100 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.913 | 0.921 | 0.906 | 0.652 | 0.348 | 0.000 | 0.308 | 0.043 | 3/302 |
| temperature | 0.913 | 0.921 | 0.906 | 0.652 | 0.348 | 0.000 | 0.308 | 0.043 | 3/302 |
| platt | 0.913 | 0.921 | 0.906 | 0.652 | 0.348 | 0.000 | 0.358 | 0.056 | 7/302 |

### split: `test_ood` (n = 600, failures = 375, tasks = 30)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.759 [0.725, 0.794] | 0.709 [0.679, 0.740] | 0.632 [0.576, 0.683] | 0.171 [0.148, 0.209] | 0.240 [0.217, 0.262] | 0.239 |
| temperature | 0.759 [0.725, 0.794] | 0.709 [0.679, 0.740] | 0.632 [0.576, 0.683] | 0.070 [0.062, 0.114] | 0.213 [0.198, 0.227] | 0.239 |
| platt | 0.759 [0.725, 0.794] | 0.698 [0.669, 0.728] | 0.579 [0.519, 0.634] | 0.083 [0.065, 0.133] | 0.224 [0.209, 0.239] | 0.266 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.860 | 0.811 | 0.942 | 0.515 | 0.485 | 0.000 | 0.122 | 0.123 | 9/375 |
| temperature | 0.860 | 0.811 | 0.942 | 0.515 | 0.485 | 0.000 | 0.122 | 0.123 | 9/375 |
| platt | 0.860 | 0.811 | 0.942 | 0.515 | 0.485 | 0.000 | 0.172 | 0.175 | 18/375 |

Fitted calibrator parameters: `{'temperature': {'T': 2.242893025531473}, 'platt': {'a': 0.450192066880014, 'b': 0.18257342499208248}}`


---

# Judge evaluation report (confidence layer fitted on `cal_ood`)

Confidence layer fitted on `cal_ood` only. 95% CIs from a bootstrap that resamples whole tasks. Conformal: Mondrian (per-class), alpha = 0.1, so the target is 90% coverage for failures and successes separately. Selective policy: threshold chosen on `cal_ood` for 5% error on auto-accepted verdicts.

## synthetic-judge

### split: `test_id` (n = 600, failures = 302, tasks = 30)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.863 [0.832, 0.891] | 0.785 [0.750, 0.818] | 0.795 [0.749, 0.837] | 0.120 [0.100, 0.157] | 0.166 [0.143, 0.191] | 0.100 |
| temperature | 0.863 [0.832, 0.891] | 0.785 [0.750, 0.818] | 0.795 [0.749, 0.837] | 0.064 [0.051, 0.104] | 0.152 [0.137, 0.168] | 0.100 |
| platt | 0.863 [0.832, 0.891] | 0.750 [0.703, 0.791] | 0.930 [0.898, 0.959] | 0.093 [0.077, 0.134] | 0.176 [0.156, 0.197] | 0.132 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.895 | 0.957 | 0.832 | 0.667 | 0.333 | 0.000 | 0.000 | nan | 0/302 |
| temperature | 0.895 | 0.957 | 0.832 | 0.667 | 0.333 | 0.000 | 0.000 | nan | 0/302 |
| platt | 0.895 | 0.957 | 0.832 | 0.667 | 0.333 | 0.000 | 0.377 | 0.102 | 0/302 |

### split: `test_ood` (n = 600, failures = 375, tasks = 30)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.759 [0.725, 0.794] | 0.709 [0.679, 0.740] | 0.632 [0.576, 0.683] | 0.171 [0.148, 0.209] | 0.240 [0.217, 0.262] | 0.239 |
| temperature | 0.759 [0.725, 0.794] | 0.709 [0.679, 0.740] | 0.632 [0.576, 0.683] | 0.053 [0.049, 0.099] | 0.211 [0.197, 0.223] | 0.239 |
| platt | 0.759 [0.725, 0.794] | 0.665 [0.627, 0.697] | 0.832 [0.789, 0.872] | 0.059 [0.053, 0.111] | 0.193 [0.176, 0.209] | 0.175 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.907 | 0.931 | 0.867 | 0.508 | 0.492 | 0.000 | 0.000 | nan | 0/375 |
| temperature | 0.907 | 0.931 | 0.867 | 0.508 | 0.492 | 0.000 | 0.000 | nan | 0/375 |
| platt | 0.907 | 0.931 | 0.867 | 0.508 | 0.492 | 0.000 | 0.190 | 0.114 | 4/375 |

Fitted calibrator parameters: `{'temperature': {'T': 2.7137754527979423}, 'platt': {'a': 0.5030791366023857, 'b': -0.9138199700969122}}`
