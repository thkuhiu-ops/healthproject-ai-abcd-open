# ABCD v0.2 Final Action Semantics Audit

## Questions

1. D labels that map to MEASURE_OK after fix: FINAL_NORMAL_CONFIDENT.
2. Degraded labels that previously mapped to MEASURE_OK: PPG_DEGRADED_ECG_OK.
3. `PPG_DEGRADED_ECG_OK` was treated too optimistically when it mapped to `MEASURE_OK`.
4. `ECG_DEGRADED_PPG_OK` current observed action before fix: `RETEST_CONTACT`; fixed action remains `RETEST_CONTACT`.
5. Previous final_reason often used internal rule/ML codes; fixed final_reason now states partial validity explicitly.

## Observed Problem Case

- `ppg_bad_ecg_good` before fix: D=`PPG_DEGRADED_ECG_OK`, action=`MEASURE_OK`, reason=`ml_model_after_hard_rule_check`.
- Corrected behavior: action=`FINAL_UNCERTAIN`; reason says ECG is usable but PPG/SpO2 is degraded.
