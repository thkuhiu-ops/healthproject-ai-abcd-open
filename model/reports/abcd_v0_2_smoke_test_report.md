# ABCD v0.2 Smoke Test Report

> Current interpretation note: this document preserves a historical ABCD smoke-test route. Current default public deployment is ABD integrated / ABD_SAFE; Model C rows below are experimental context and are not the current default V1 firmware chain.

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
| ppg_bad_ecg_good | PPG_BAD | ECG_GOOD | NORMAL | PPG_DEGRADED_ECG_OK | MEASURE_OK |
| ecg_bad_ppg_good | PPG_GOOD | ECG_BAD | OTHER_OR_UNCERTAIN | ECG_DEGRADED_PPG_OK | RETEST_CONTACT |
| rhythm_suspect | PPG_GOOD | ECG_GOOD | RHYTHM_SUSPECT | RHYTHM_SUSPECT_RETEST | RHYTHM_RISK_RETEST |
| motion_degraded | PPG_UNCERTAIN | ECG_UNCERTAIN | OTHER_OR_UNCERTAIN | MOTION_DEGRADED | KEEP_STILL |
| contact_bad | PPG_BAD | ECG_BAD | OTHER_OR_UNCERTAIN | CONTACT_BAD_RETEST | RETEST_CONTACT |
| recovery_wait | PPG_UNCERTAIN | ECG_GOOD | NORMAL | RECOVERY_WAIT | RETEST_AFTER_RECOVERY |
| sensor_conflict | PPG_GOOD | ECG_GOOD | OTHER_OR_UNCERTAIN | SENSOR_CONFLICT | SENSOR_CONFLICT_RETEST |
| measure_failed | PPG_BAD | ECG_BAD | OTHER_OR_UNCERTAIN | MEASURE_FAILED | MEASURE_FAILED |
| all_uncertain | PPG_UNCERTAIN | ECG_UNCERTAIN | OTHER_OR_UNCERTAIN | FINAL_UNCERTAIN | FINAL_UNCERTAIN |
