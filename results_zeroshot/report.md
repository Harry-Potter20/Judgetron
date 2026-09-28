# Judge evaluation report (confidence layer fitted on `cal`)

Confidence layer fitted on `cal` only. 95% CIs from a bootstrap that resamples whole tasks. Conformal: Mondrian (per-class), alpha = 0.1, so the target is 90% coverage for failures and successes separately. Selective policy: threshold chosen on `cal` for 5% error on auto-accepted verdicts.

## qwen3b-zeroshot

### split: `test_id` (n = 3064, failures = 1511, tasks = 96)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.588 [0.562, 0.620] | 0.523 [0.512, 0.536] | 0.097 [0.066, 0.143] | 0.070 [0.049, 0.097] | 0.250 [0.244, 0.256] | 0.406 |
| temperature | 0.588 [0.562, 0.620] | 0.523 [0.512, 0.536] | 0.097 [0.066, 0.143] | 0.041 [0.023, 0.066] | 0.247 [0.245, 0.249] | 0.406 |
| platt | 0.588 [0.562, 0.620] | 0.558 [0.529, 0.589] | 0.499 [0.354, 0.644] | 0.014 [0.013, 0.048] | 0.244 [0.241, 0.248] | 0.388 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.939 | 0.927 | 0.950 | 0.181 | 0.819 | 0.000 | 0.000 | nan | 0/1511 |
| temperature | 0.939 | 0.927 | 0.950 | 0.181 | 0.819 | 0.000 | 0.000 | nan | 0/1511 |
| platt | 0.939 | 0.927 | 0.950 | 0.181 | 0.819 | 0.000 | 0.000 | nan | 0/1511 |

### split: `test_ood` (n = 435, failures = 210, tasks = 24)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.649 [0.573, 0.722] | 0.600 [0.531, 0.648] | 0.400 [0.207, 0.560] | 0.062 [0.038, 0.120] | 0.236 [0.222, 0.248] | 0.328 |
| temperature | 0.649 [0.573, 0.722] | 0.600 [0.531, 0.648] | 0.400 [0.207, 0.560] | 0.075 [0.029, 0.138] | 0.241 [0.234, 0.247] | 0.327 |
| platt | 0.649 [0.573, 0.722] | 0.584 [0.525, 0.647] | 0.781 [0.637, 0.890] | 0.049 [0.031, 0.125] | 0.239 [0.228, 0.248] | 0.350 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.885 | 0.976 | 0.800 | 0.352 | 0.648 | 0.000 | 0.000 | nan | 0/210 |
| temperature | 0.885 | 0.976 | 0.800 | 0.352 | 0.648 | 0.000 | 0.000 | nan | 0/210 |
| platt | 0.885 | 0.976 | 0.800 | 0.352 | 0.648 | 0.000 | 0.000 | nan | 0/210 |

Fitted calibrator parameters: `{'temperature': {'T': 2.302276181772167}, 'platt': {'a': 0.8318281581001736, 'b': -0.2557617121123847}}`


---

# Judge evaluation report (confidence layer fitted on `cal_ood`)

Confidence layer fitted on `cal_ood` only. 95% CIs from a bootstrap that resamples whole tasks. Conformal: Mondrian (per-class), alpha = 0.1, so the target is 90% coverage for failures and successes separately. Selective policy: threshold chosen on `cal_ood` for 5% error on auto-accepted verdicts.

## qwen3b-zeroshot

### split: `test_id` (n = 3064, failures = 1511, tasks = 96)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.588 [0.562, 0.620] | 0.523 [0.512, 0.536] | 0.097 [0.066, 0.143] | 0.070 [0.049, 0.097] | 0.250 [0.244, 0.256] | 0.406 |
| temperature | 0.588 [0.562, 0.620] | 0.523 [0.512, 0.536] | 0.097 [0.066, 0.143] | 0.049 [0.033, 0.076] | 0.248 [0.244, 0.251] | 0.405 |
| platt | 0.588 [0.562, 0.620] | 0.558 [0.529, 0.589] | 0.499 [0.354, 0.644] | 0.022 [0.016, 0.055] | 0.244 [0.239, 0.250] | 0.388 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.905 | 0.822 | 0.986 | 0.256 | 0.744 | 0.000 | 0.000 | nan | 0/1511 |
| temperature | 0.905 | 0.822 | 0.986 | 0.256 | 0.744 | 0.000 | 0.000 | nan | 0/1511 |
| platt | 0.905 | 0.822 | 0.986 | 0.256 | 0.744 | 0.000 | 0.000 | nan | 0/1511 |

### split: `test_ood` (n = 435, failures = 210, tasks = 24)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.649 [0.573, 0.722] | 0.600 [0.531, 0.648] | 0.400 [0.207, 0.560] | 0.062 [0.038, 0.120] | 0.236 [0.222, 0.248] | 0.328 |
| temperature | 0.649 [0.573, 0.722] | 0.600 [0.531, 0.648] | 0.400 [0.207, 0.560] | 0.052 [0.031, 0.116] | 0.238 [0.227, 0.247] | 0.327 |
| platt | 0.649 [0.573, 0.722] | 0.584 [0.525, 0.647] | 0.781 [0.637, 0.890] | 0.064 [0.051, 0.126] | 0.241 [0.223, 0.254] | 0.363 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.952 | 0.943 | 0.960 | 0.182 | 0.818 | 0.000 | 0.000 | nan | 0/210 |
| temperature | 0.952 | 0.943 | 0.960 | 0.182 | 0.818 | 0.000 | 0.000 | nan | 0/210 |
| platt | 0.952 | 0.943 | 0.960 | 0.182 | 0.818 | 0.000 | 0.000 | nan | 0/210 |

Fitted calibrator parameters: `{'temperature': {'T': 1.350585502551474}, 'platt': {'a': 1.3042553405376198, 'b': -0.4588611181412279}}`
