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

### split: `test_ood` (n = 429, failures = 206, tasks = 24)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.650 [0.573, 0.722] | 0.600 [0.532, 0.650] | 0.398 [0.201, 0.563] | 0.061 [0.038, 0.121] | 0.235 [0.221, 0.248] | 0.327 |
| temperature | 0.650 [0.573, 0.722] | 0.600 [0.532, 0.650] | 0.398 [0.201, 0.563] | 0.076 [0.029, 0.140] | 0.241 [0.234, 0.247] | 0.327 |
| platt | 0.650 [0.573, 0.722] | 0.584 [0.523, 0.647] | 0.782 [0.630, 0.898] | 0.045 [0.032, 0.125] | 0.239 [0.228, 0.248] | 0.350 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.886 | 0.976 | 0.803 | 0.350 | 0.650 | 0.000 | 0.000 | nan | 0/206 |
| temperature | 0.886 | 0.976 | 0.803 | 0.350 | 0.650 | 0.000 | 0.000 | nan | 0/206 |
| platt | 0.886 | 0.976 | 0.803 | 0.350 | 0.650 | 0.000 | 0.000 | nan | 0/206 |

Fitted calibrator parameters: `{'temperature': {'T': 2.302276181772167}, 'platt': {'a': 0.8318296343344633, 'b': -0.2557621647059706}}`

## qwen3b-lora

### split: `test_id` (n = 3064, failures = 1511, tasks = 96)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.775 [0.727, 0.837] | 0.682 [0.638, 0.742] | 0.574 [0.463, 0.689] | 0.059 [0.039, 0.088] | 0.194 [0.165, 0.214] | 0.166 |
| temperature | 0.775 [0.727, 0.837] | 0.682 [0.638, 0.742] | 0.574 [0.463, 0.689] | 0.025 [0.023, 0.054] | 0.190 [0.164, 0.208] | 0.167 |
| platt | 0.775 [0.727, 0.837] | 0.682 [0.636, 0.744] | 0.544 [0.434, 0.661] | 0.024 [0.022, 0.056] | 0.191 [0.164, 0.209] | 0.168 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.898 | 0.860 | 0.934 | 0.521 | 0.479 | 0.000 | 0.145 | 0.018 | 5/1511 |
| temperature | 0.898 | 0.860 | 0.934 | 0.521 | 0.479 | 0.000 | 0.147 | 0.018 | 5/1511 |
| platt | 0.898 | 0.860 | 0.934 | 0.521 | 0.479 | 0.000 | 0.140 | 0.019 | 5/1511 |

### split: `test_ood` (n = 429, failures = 206, tasks = 24)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.690 [0.640, 0.757] | 0.642 [0.585, 0.700] | 0.840 [0.708, 0.932] | 0.146 [0.127, 0.209] | 0.251 [0.227, 0.273] | 0.312 |
| temperature | 0.690 [0.640, 0.757] | 0.642 [0.585, 0.700] | 0.840 [0.708, 0.932] | 0.090 [0.079, 0.169] | 0.235 [0.216, 0.252] | 0.312 |
| platt | 0.690 [0.640, 0.757] | 0.655 [0.601, 0.717] | 0.835 [0.705, 0.930] | 0.079 [0.069, 0.143] | 0.232 [0.213, 0.247] | 0.303 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.804 | 0.937 | 0.682 | 0.594 | 0.406 | 0.000 | 0.079 | 0.265 | 0/206 |
| temperature | 0.804 | 0.937 | 0.682 | 0.594 | 0.406 | 0.000 | 0.091 | 0.333 | 0/206 |
| platt | 0.804 | 0.937 | 0.682 | 0.594 | 0.406 | 0.000 | 0.065 | 0.214 | 0/206 |

Fitted calibrator parameters: `{'temperature': {'T': 1.5868768061988183}, 'platt': {'a': 0.6268395360444199, 'b': 0.08513111765786475}}`


---

# Judge evaluation report (confidence layer fitted on `cal_ood`)

Confidence layer fitted on `cal_ood` only. 95% CIs from a bootstrap that resamples whole tasks. Conformal: Mondrian (per-class), alpha = 0.1, so the target is 90% coverage for failures and successes separately. Selective policy: threshold chosen on `cal_ood` for 5% error on auto-accepted verdicts.

## qwen3b-zeroshot

### split: `test_id` (n = 3064, failures = 1511, tasks = 96)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.588 [0.562, 0.620] | 0.523 [0.512, 0.536] | 0.097 [0.066, 0.143] | 0.070 [0.049, 0.097] | 0.250 [0.244, 0.256] | 0.406 |
| temperature | 0.588 [0.562, 0.620] | 0.523 [0.512, 0.536] | 0.097 [0.066, 0.143] | 0.046 [0.031, 0.073] | 0.247 [0.244, 0.251] | 0.407 |
| platt | 0.588 [0.562, 0.620] | 0.558 [0.529, 0.589] | 0.499 [0.354, 0.644] | 0.023 [0.018, 0.053] | 0.244 [0.239, 0.250] | 0.388 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.905 | 0.822 | 0.986 | 0.256 | 0.744 | 0.000 | 0.000 | nan | 0/1511 |
| temperature | 0.905 | 0.822 | 0.986 | 0.256 | 0.744 | 0.000 | 0.000 | nan | 0/1511 |
| platt | 0.905 | 0.822 | 0.986 | 0.256 | 0.744 | 0.000 | 0.000 | nan | 0/1511 |

### split: `test_ood` (n = 429, failures = 206, tasks = 24)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.650 [0.573, 0.722] | 0.600 [0.532, 0.650] | 0.398 [0.201, 0.563] | 0.061 [0.038, 0.121] | 0.235 [0.221, 0.248] | 0.327 |
| temperature | 0.650 [0.573, 0.722] | 0.600 [0.532, 0.650] | 0.398 [0.201, 0.563] | 0.057 [0.024, 0.126] | 0.238 [0.227, 0.247] | 0.326 |
| platt | 0.650 [0.573, 0.722] | 0.584 [0.523, 0.647] | 0.782 [0.630, 0.898] | 0.048 [0.039, 0.118] | 0.241 [0.224, 0.254] | 0.363 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.951 | 0.942 | 0.960 | 0.184 | 0.816 | 0.000 | 0.000 | nan | 0/206 |
| temperature | 0.951 | 0.942 | 0.960 | 0.184 | 0.816 | 0.000 | 0.000 | nan | 0/206 |
| platt | 0.951 | 0.942 | 0.960 | 0.184 | 0.816 | 0.000 | 0.000 | nan | 0/206 |

Fitted calibrator parameters: `{'temperature': {'T': 1.443683613905659}, 'platt': {'a': 1.233826566272845, 'b': -0.4489856360575144}}`

## qwen3b-lora

### split: `test_id` (n = 3064, failures = 1511, tasks = 96)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.775 [0.727, 0.837] | 0.682 [0.638, 0.742] | 0.574 [0.463, 0.689] | 0.059 [0.039, 0.088] | 0.194 [0.165, 0.214] | 0.166 |
| temperature | 0.775 [0.727, 0.837] | 0.682 [0.638, 0.742] | 0.574 [0.463, 0.689] | 0.033 [0.028, 0.063] | 0.191 [0.165, 0.208] | 0.166 |
| platt | 0.775 [0.727, 0.837] | 0.693 [0.655, 0.745] | 0.698 [0.593, 0.797] | 0.036 [0.027, 0.061] | 0.192 [0.167, 0.209] | 0.169 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.940 | 0.889 | 0.990 | 0.387 | 0.613 | 0.000 | 0.000 | nan | 0/1511 |
| temperature | 0.940 | 0.889 | 0.990 | 0.387 | 0.613 | 0.000 | 0.000 | nan | 0/1511 |
| platt | 0.940 | 0.889 | 0.990 | 0.387 | 0.613 | 0.000 | 0.000 | nan | 0/1511 |

### split: `test_ood` (n = 429, failures = 206, tasks = 24)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.690 [0.640, 0.757] | 0.642 [0.585, 0.700] | 0.840 [0.708, 0.932] | 0.146 [0.127, 0.209] | 0.251 [0.227, 0.273] | 0.312 |
| temperature | 0.690 [0.640, 0.757] | 0.642 [0.585, 0.700] | 0.840 [0.708, 0.932] | 0.085 [0.074, 0.163] | 0.233 [0.215, 0.248] | 0.312 |
| platt | 0.690 [0.640, 0.757] | 0.635 [0.593, 0.686] | 0.898 [0.810, 0.966] | 0.090 [0.074, 0.160] | 0.240 [0.221, 0.259] | 0.334 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.890 | 0.937 | 0.848 | 0.352 | 0.648 | 0.000 | 0.000 | nan | 0/206 |
| temperature | 0.890 | 0.937 | 0.848 | 0.352 | 0.648 | 0.000 | 0.000 | nan | 0/206 |
| platt | 0.890 | 0.937 | 0.848 | 0.352 | 0.648 | 0.000 | 0.000 | nan | 0/206 |

Fitted calibrator parameters: `{'temperature': {'T': 1.855838286581927}, 'platt': {'a': 0.5174994521312144, 'b': -0.16876120531030397}}`
