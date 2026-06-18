# ABCD v0.2 Final Action Mapping

Only `FINAL_NORMAL_CONFIDENT` maps to `MEASURE_OK`.

| Model D Label | final_action | final_reason |
| --- | --- | --- |
| FINAL_NORMAL_CONFIDENT | MEASURE_OK | All core signals are usable; no major risk hint. |
| PPG_DEGRADED_ECG_OK | FINAL_UNCERTAIN | ECG is usable, but PPG/SpO2 is degraded; retest PPG contact if SpO2 or PPG-HR is required. |
| ECG_DEGRADED_PPG_OK | RETEST_CONTACT | PPG is usable, but ECG is degraded; fix ECG contact/lead condition before ECG rhythm interpretation. |
| RHYTHM_SUSPECT_RETEST | RHYTHM_RISK_RETEST | ECG rhythm-risk hint detected; retest under stable contact and stillness. Not a clinical conclusion. |
| MOTION_DEGRADED | KEEP_STILL | Motion may degrade signal reliability; keep still and retest. |
| CONTACT_BAD_RETEST | RETEST_CONTACT | Contact or lead condition is unreliable; fix sensor contact and retest. |
| RECOVERY_WAIT | RETEST_AFTER_RECOVERY | Sensor recovery or physiological recovery window; wait and retest. |
| SENSOR_CONFLICT | SENSOR_CONFLICT_RETEST | Sensor outputs conflict; retest under stable conditions. |
| MEASURE_FAILED | MEASURE_FAILED | Multiple core signals unavailable or invalid. |
| FINAL_UNCERTAIN | FINAL_UNCERTAIN | Information is insufficient for confident measurement. |
