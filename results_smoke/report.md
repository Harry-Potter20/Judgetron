# Judge evaluation report (confidence layer fitted on `cal`)

Confidence layer fitted on `cal` only. 95% CIs from a bootstrap that resamples whole tasks. Conformal: Mondrian (per-class), alpha = 0.1, so the target is 90% coverage for failures and successes separately. Selective policy: threshold chosen on `cal` for 5% error on auto-accepted verdicts.

## qwen3b-zeroshot

### split: `test_id` (n = 24, failures = 8, tasks = 5)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.598 [0.438, 0.856] | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.118 [0.093, 0.353] | 0.224 [0.183, 0.273] | 0.250 |
| temperature | 0.598 [0.438, 0.856] | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.165 [0.002, 0.395] | 0.249 [0.249, 0.250] | 0.250 |
| platt | 0.598 [0.438, 0.856] | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.333 [0.103, 0.571] | 0.333 [0.103, 0.571] | 0.250 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | nan | 0/8 |
| temperature | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | nan | 0/8 |
| platt | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.333 | 8/8 |

### split: `test_ood` (n = 24, failures = 12, tasks = 2)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.799 [0.375, 0.875] | 0.583 [0.375, 0.637] | 0.333 [0.250, 0.375] | 0.234 [0.205, 0.402] | 0.214 [0.184, 0.305] | 0.271 |
| temperature | 0.799 [0.375, 0.875] | 0.583 [0.375, 0.637] | 0.333 [0.250, 0.375] | 0.081 [0.081, 0.169] | 0.249 [0.248, 0.251] | 0.266 |
| platt | 0.799 [0.375, 0.875] | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.500 [0.444, 0.667] | 0.500 [0.444, 0.667] | 0.253 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.958 | 1.000 | 0.917 | 0.042 | 0.958 | 0.000 | 0.000 | nan | 0/12 |
| temperature | 0.958 | 1.000 | 0.917 | 0.042 | 0.958 | 0.000 | 0.000 | nan | 0/12 |
| platt | 0.958 | 1.000 | 0.917 | 0.042 | 0.958 | 0.000 | 0.958 | 0.522 | 12/12 |

Fitted calibrator parameters: `{'temperature': {'T': 54.59788095835596}, 'platt': {'a': 0.04914978896730504, 'b': 11.99200762560835}}`


---

# Judge evaluation report (confidence layer fitted on `cal_ood`)

Confidence layer fitted on `cal_ood` only. 95% CIs from a bootstrap that resamples whole tasks. Conformal: Mondrian (per-class), alpha = 0.1, so the target is 90% coverage for failures and successes separately. Selective policy: threshold chosen on `cal_ood` for 5% error on auto-accepted verdicts.

## qwen3b-zeroshot

### split: `test_id` (n = 24, failures = 8, tasks = 5)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.598 [0.438, 0.856] | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.118 [0.093, 0.353] | 0.224 [0.183, 0.273] | 0.250 |
| temperature | 0.598 [0.438, 0.856] | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.165 [0.002, 0.395] | 0.249 [0.249, 0.250] | 0.250 |
| platt | 0.402 [0.144, 0.562] | 0.500 [0.500, 0.500] | 1.000 [1.000, 1.000] | 0.326 [0.110, 0.555] | 0.331 [0.254, 0.401] | 0.750 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.792 | 0.375 | 1.000 | 0.708 | 0.292 | 0.000 | 0.000 | nan | 0/8 |
| temperature | 0.792 | 0.375 | 1.000 | 0.708 | 0.292 | 0.000 | 0.000 | nan | 0/8 |
| platt | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | nan | 0/8 |

### split: `test_ood` (n = 24, failures = 12, tasks = 2)

| calibrator | AUROC | bal. acc | failure recall | ECE | Brier | AURC |
|---|---|---|---|---|---|---|
| raw | 0.799 [0.375, 0.875] | 0.583 [0.375, 0.637] | 0.333 [0.250, 0.375] | 0.234 [0.205, 0.402] | 0.214 [0.184, 0.305] | 0.271 |
| temperature | 0.799 [0.375, 0.875] | 0.583 [0.375, 0.637] | 0.333 [0.250, 0.375] | 0.081 [0.081, 0.169] | 0.249 [0.248, 0.251] | 0.269 |
| platt | 0.201 [0.125, 0.625] | 0.500 [0.500, 0.500] | 1.000 [1.000, 1.000] | 0.360 [0.222, 0.406] | 0.294 [0.210, 0.322] | 0.747 |

| calibrator | coverage | cov. failures | cov. successes | singleton | ambiguous | empty | auto-accept rate | auto-accept error | failures auto-passed |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.833 | 0.750 | 0.917 | 0.542 | 0.458 | 0.000 | 0.000 | nan | 0/12 |
| temperature | 0.833 | 0.750 | 0.917 | 0.542 | 0.458 | 0.000 | 0.000 | nan | 0/12 |
| platt | 0.750 | 0.917 | 0.583 | 0.292 | 0.708 | 0.000 | 0.000 | nan | 0/12 |

Fitted calibrator parameters: `{'temperature': {'T': 54.597859744747765}, 'platt': {'a': -0.36086849315434844, 'b': -0.5292765466205601}}`
