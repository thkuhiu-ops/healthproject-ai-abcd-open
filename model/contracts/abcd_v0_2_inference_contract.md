# ABD/ABCD v0.2 Inference Contract Notes

This file contains historical ABCD experiment terminology.
The current default deployment chain should be interpreted as **ABD integrated / ABD_SAFE**, with Model C treated as an experimental/reserved module rather than a default dependency.

## Current Default Chain

The current paper/open-release mainline is:

1. Load/read Model A PPG quality output.
2. Load/read Model B ECG quality output.
3. Build rule flags from IMU, TMP/contact, recovery, data freshness, and signal-conflict fields.
4. Run/read Model D_SAFE / ABD Fusion-assisted state classification.
5. Let the firmware Fusion arbitration layer combine hard rules, sensor states, and model-assisted evidence.
6. Emit firmware-side trust and UI state, such as `trust_score`, `trust_level`, `final_state`, and `hr_source`.

The deployment-candidate engineering classes are represented by states such as `FINAL_OK`, `SIGNAL_BAD_OR_CONTACT_BAD`, and `UNCERTAIN_OR_MOTION`.

## Historical Model C Gate

The following Model C gate is retained only for historical ABCD research-context documentation.
It is not the current default ABD integrated / ABD_SAFE deployment path.

```text
if model_b_label != "ECG_GOOD":
    model_c_final_label = "OTHER_OR_UNCERTAIN"
    model_c_reason = "ECG_QUALITY_NOT_GOOD"
elif model_c_p_rhythm_suspect <= 0.42:
    model_c_final_label = "NORMAL"
elif model_c_p_rhythm_suspect >= 0.60:
    model_c_final_label = "RHYTHM_SUSPECT"
else:
    model_c_final_label = "OTHER_OR_UNCERTAIN"
    model_c_reason = "LOW_CONFIDENCE_GRAY_ZONE"
```

## Safety Boundary

Model A and Model B are signal-quality gates.
Model D_SAFE / ABD Fusion provides model-assisted engineering state evidence.
The firmware Fusion arbitration layer is responsible for the final trust/UI output.
No model in this repository produces medical diagnostic conclusions.
