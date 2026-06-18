# ABCD v0.2 Smoke Test Report

- Input: `LOCAL_PATH_REMOVED
- Output: `LOCAL_PATH_REMOVED
- Rows: 10
- Passed: True
- Model C obeys B gate: True
- Model D directly consumes current Model C v0.4 fields: True
- Rule fallback enabled: True
- final_action legal: True
- No diagnosis-style outputs: True
- No schema mismatch: True
- No old C proxy dependency remains: True

| Window | A | B | C | D | final_action |
| --- | --- | --- | --- | --- | --- |
| all_normal | PPG_GOOD | ECG_GOOD | NORMAL | FINAL_NORMAL_CONFIDENT | MEASURE_OK |
| ppg_bad_ecg_good | PPG_BAD | ECG_GOOD | NORMAL | PPG_DEGRADED_ECG_OK | FINAL_UNCERTAIN |
| ecg_bad_ppg_good | PPG_GOOD | ECG_BAD | OTHER_OR_UNCERTAIN | ECG_DEGRADED_PPG_OK | RETEST_CONTACT |
| rhythm_suspect | PPG_GOOD | ECG_GOOD | RHYTHM_SUSPECT | RHYTHM_SUSPECT_RETEST | RHYTHM_RISK_RETEST |
| motion_degraded | PPG_UNCERTAIN | ECG_UNCERTAIN | OTHER_OR_UNCERTAIN | MOTION_DEGRADED | KEEP_STILL |
| contact_bad | PPG_BAD | ECG_BAD | OTHER_OR_UNCERTAIN | CONTACT_BAD_RETEST | RETEST_CONTACT |
| recovery_wait | PPG_UNCERTAIN | ECG_GOOD | NORMAL | RECOVERY_WAIT | RETEST_AFTER_RECOVERY |
| sensor_conflict | PPG_GOOD | ECG_GOOD | OTHER_OR_UNCERTAIN | SENSOR_CONFLICT | SENSOR_CONFLICT_RETEST |
| measure_failed | PPG_BAD | ECG_BAD | OTHER_OR_UNCERTAIN | MEASURE_FAILED | MEASURE_FAILED |
| all_uncertain | PPG_UNCERTAIN | ECG_UNCERTAIN | OTHER_OR_UNCERTAIN | FINAL_UNCERTAIN | FINAL_UNCERTAIN |

## Expected Action Checks

| Case | Expected | Observed | Result |
| --- | --- | --- | --- |
| all_normal | MEASURE_OK | MEASURE_OK | PASS |
| ppg_bad_ecg_good | FINAL_UNCERTAIN | FINAL_UNCERTAIN | PASS |
| ecg_bad_ppg_good | RETEST_CONTACT | RETEST_CONTACT | PASS |
| rhythm_suspect | RHYTHM_RISK_RETEST | RHYTHM_RISK_RETEST | PASS |
| motion_degraded | KEEP_STILL | KEEP_STILL | PASS |
| contact_bad | RETEST_CONTACT | RETEST_CONTACT | PASS |
| recovery_wait | RETEST_AFTER_RECOVERY | RETEST_AFTER_RECOVERY | PASS |
| sensor_conflict | SENSOR_CONFLICT_RETEST | SENSOR_CONFLICT_RETEST | PASS |
| measure_failed | MEASURE_FAILED | MEASURE_FAILED | PASS |
| all_uncertain | FINAL_UNCERTAIN | FINAL_UNCERTAIN | PASS |
