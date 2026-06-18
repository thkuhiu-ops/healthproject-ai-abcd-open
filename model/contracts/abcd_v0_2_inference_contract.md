# ABCD v0.2 Inference Contract

## Scope

ABCD Integrated v0.2 is a PC-side inference and fusion candidate only. It does
not retrain models, export TFLite, enter GD32 Embedded AI, or modify MDK firmware.

## Pipeline Order

1. Load/read Model A PPG quality output.
2. Load/read Model B ECG quality output.
3. Run/read Model C only if Model B is `ECG_GOOD`.
4. Build rule flags from IMU/TMP/contact/recovery/conflict fields.
5. Run Model D v0.2 fusion.
6. Apply hard-rule fallback; hard safety/retest rules override conflicting ML.
7. Emit `final_action` and `final_reason`.

## Model C Gate

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

## Safety

Model C is a rhythm-risk hint candidate only, not diagnosis. Model D is a
fusion decision layer only. Forbidden output tokens are listed in
`schemas/abcd_v0_2_unified_output_schema.json`.
