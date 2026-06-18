# ABCD v0.2 Model Cards

## Model A - PPG quality gate

- Version: v0.4
- Source: `LOCAL_PATH_REMOVED
- Inputs: 58
- Outputs: `PPG_GOOD|PPG_BAD|PPG_UNCERTAIN`
- Status: FOUND
- Deploy status: PC_CANDIDATE_REFERENCE_ONLY
- Safety limitations: PPG quality gate, not diagnosis.

## Model B - ECG quality gate

- Version: v0.1
- Source: `LOCAL_PATH_REMOVED
- Inputs: 40
- Outputs: `ECG_GOOD|ECG_BAD|ECG_UNCERTAIN`
- Status: FOUND
- Deploy status: PC_CANDIDATE_REFERENCE_ONLY
- Safety limitations: ECG quality gate, not diagnosis.

## Model C - ECG rhythm-risk hint candidate

- Version: v0.4 threshold-calibrated compromise
- Source: `LOCAL_PATH_REMOVED
- Inputs: 6
- Outputs: `NORMAL|RHYTHM_SUSPECT|OTHER_OR_UNCERTAIN`
- Status: SMOKE_USABLE_DO_NOT_DEPLOY
- Deploy status: DO_NOT_DEPLOY
- Safety limitations: Rhythm risk hint candidate, not diagnosis, not deployable; gated by Model B ECG_GOOD.

## Model D - Fusion decision

- Version: v0.2
- Source: `LOCAL_PATH_REMOVED
- Inputs: 34
- Outputs: `FINAL_NORMAL_CONFIDENT|PPG_DEGRADED_ECG_OK|ECG_DEGRADED_PPG_OK|MOTION_DEGRADED|CONTACT_BAD_RETEST|RECOVERY_WAIT|RHYTHM_SUSPECT_RETEST|SENSOR_CONFLICT|FINAL_UNCERTAIN|MEASURE_FAILED`
- Status: FOUND_PC_CANDIDATE
- Deploy status: PC_ONLY_NOT_FIRMWARE_READY
- Safety limitations: Fusion decision PC candidate with rule fallback; not firmware-ready.
