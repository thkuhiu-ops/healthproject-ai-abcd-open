# ABCD v0.2 Model Inventory

| Model | Version | Status | Deploy Status | Safety Boundary |
| --- | --- | --- | --- | --- |
| Model A - PPG quality gate | v0.4 | FOUND | PC_CANDIDATE_REFERENCE_ONLY | PPG quality gate, not diagnosis. |
| Model B - ECG quality gate | v0.1 | FOUND | PC_CANDIDATE_REFERENCE_ONLY | ECG quality gate, not diagnosis. |
| Model C - ECG rhythm-risk hint candidate | v0.4 threshold-calibrated compromise | SMOKE_USABLE_DO_NOT_DEPLOY | DO_NOT_DEPLOY | Rhythm risk hint candidate, not diagnosis, not deployable; gated by Model B ECG_GOOD. |
| Model D - Fusion decision | v0.2 | FOUND_PC_CANDIDATE | PC_ONLY_NOT_FIRMWARE_READY | Fusion decision PC candidate with rule fallback; not firmware-ready. |

## Artifact Details

### Model A - PPG quality gate

- Source directory: `LOCAL_PATH_REMOVED
- Model file: `LOCAL_PATH_REMOVED
- Preprocessor/scaler: `none`
- Feature schema: `LOCAL_PATH_REMOVED
- Label map: `LOCAL_PATH_REMOVED
- Threshold file: `none`
- Final report: `LOCAL_PATH_REMOVED
- Summary JSON: `LOCAL_PATH_REMOVED
- Input features: 58
- Output labels: `PPG_GOOD|PPG_BAD|PPG_UNCERTAIN`

### Model B - ECG quality gate

- Source directory: `LOCAL_PATH_REMOVED
- Model file: `LOCAL_PATH_REMOVED
- Preprocessor/scaler: `none`
- Feature schema: `LOCAL_PATH_REMOVED
- Label map: `LOCAL_PATH_REMOVED
- Threshold file: `none`
- Final report: `LOCAL_PATH_REMOVED
- Summary JSON: `LOCAL_PATH_REMOVED
- Input features: 40
- Output labels: `ECG_GOOD|ECG_BAD|ECG_UNCERTAIN`

### Model C - ECG rhythm-risk hint candidate

- Source directory: `LOCAL_PATH_REMOVED
- Model file: `LOCAL_PATH_REMOVED
- Preprocessor/scaler: `LOCAL_PATH_REMOVED
- Feature schema: `LOCAL_PATH_REMOVED
- Label map: `LOCAL_PATH_REMOVED
- Threshold file: `LOCAL_PATH_REMOVED
- Final report: `LOCAL_PATH_REMOVED
- Summary JSON: `LOCAL_PATH_REMOVED
- Input features: 6
- Output labels: `NORMAL|RHYTHM_SUSPECT|OTHER_OR_UNCERTAIN`

### Model D - Fusion decision

- Source directory: `LOCAL_PATH_REMOVED
- Model file: `LOCAL_PATH_REMOVED
- Preprocessor/scaler: `LOCAL_PATH_REMOVED
- Feature schema: `LOCAL_PATH_REMOVED
- Label map: `LOCAL_PATH_REMOVED
- Threshold file: `none`
- Final report: `LOCAL_PATH_REMOVED
- Summary JSON: `LOCAL_PATH_REMOVED
- Input features: 34
- Output labels: `FINAL_NORMAL_CONFIDENT|PPG_DEGRADED_ECG_OK|ECG_DEGRADED_PPG_OK|MOTION_DEGRADED|CONTACT_BAD_RETEST|RECOVERY_WAIT|RHYTHM_SUSPECT_RETEST|SENSOR_CONFLICT|FINAL_UNCERTAIN|MEASURE_FAILED`
