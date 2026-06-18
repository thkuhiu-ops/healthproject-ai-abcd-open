# ABCD v0.2 Smoke Test Dataset Report

> Current interpretation note: this dataset report preserves historical ABCD smoke-test cases, including experimental rhythm-risk cases. The current default public deployment candidate is ABD integrated / ABD_SAFE and does not use Model C as a default dependency.

- Dataset: `LOCAL_PATH_REMOVED
- Rows: 10
- Contains required cases: all_normal, ppg_bad_ecg_good, ecg_bad_ppg_good, rhythm_suspect, motion_degraded, contact_bad, recovery_wait, sensor_conflict, measure_failed, all_uncertain.

| Case | A | B | p_rhythm_suspect | IMU | TMP |
| --- | --- | --- | ---: | --- | --- |
| all_normal | PPG_GOOD | ECG_GOOD | 0.2 | STATIC | TMP_NORMAL |
| ppg_bad_ecg_good | PPG_BAD | ECG_GOOD | 0.2 | STATIC | TMP_NORMAL |
| ecg_bad_ppg_good | PPG_GOOD | ECG_BAD | 0.75 | STATIC | TMP_NORMAL |
| rhythm_suspect | PPG_GOOD | ECG_GOOD | 0.75 | STATIC | TMP_NORMAL |
| motion_degraded | PPG_UNCERTAIN | ECG_UNCERTAIN | 0.5 | MOTION | TMP_NORMAL |
| contact_bad | PPG_BAD | ECG_BAD | 0.5 | STATIC | TMP_NORMAL |
| recovery_wait | PPG_UNCERTAIN | ECG_GOOD | 0.2 | STATIC | COLD_SUSPECT |
| sensor_conflict | PPG_GOOD | ECG_GOOD | 0.5 | STATIC | TMP_NORMAL |
| measure_failed | PPG_BAD | ECG_BAD | 0.5 | STATIC | TMP_INVALID |
| all_uncertain | PPG_UNCERTAIN | ECG_UNCERTAIN | 0.5 | STATIC | TMP_NORMAL |
