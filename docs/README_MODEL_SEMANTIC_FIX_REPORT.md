# README Model Semantic Fix Report

## Modified Files

- `README.md`
- `docs/MODEL_CARD.md`
- `docs/DATASET_CARD.md`
- `model/contracts/abcd_v0_2_inference_contract.md`
- `model/contracts/abcd_v0_2_final_action_mapping.md`
- `model/schemas/abcd_v0_2_unified_input_schema.json`
- `model/schemas/abcd_v0_2_unified_output_schema.json`
- `model/schemas/abcd_v0_2_final_action_mapping.json`
- `model/reports/abcd_v0_2_final_report.md`
- `model/reports/abcd_v0_2_final_report_fixed.md`
- `model/reports/abcd_v0_2_smoke_test_report.md`
- `model/reports/abcd_v0_2_smoke_test_report_fixed.md`
- `model/reports/abcd_v0_2_smoke_test_dataset_report.md`
- `model/reports/abcd_v0_2_model_inventory.md`
- `model/reports/abcd_v0_2_model_cards.md`
- `model/reports/ABC_V0_2_MODEL_AND_SYSTEM_EVALUATION_FOR_PAPER.md`
- `model/reports/ABC_V0_2_PAPER_FIGURE_AND_NAMING_REPORT.md`

## Why The Changes Were Needed

The previous README described the repository as if the full A+B+C+D ABCD chain were the current default deployment route.
That wording could imply that Model C is part of the current competition/paper default model chain, that complete ABCD neural inference is already closed-loop in V1 firmware, or that AI directly generates final trust/medical conclusions.

The updated documentation aligns the public repository with the current paper, V1 firmware project, and GD Embedded AI Tool validation evidence.

## Current Default Model Chain

The current default public deployment candidate is **ABD integrated / ABD_SAFE**.
It is an engineering trust-fusion candidate, not a clinical diagnostic model.

The main chain is:

- Model A: PPG quality gate.
- Model B: ECG quality gate.
- Model D_SAFE / ABD Fusion: fusion-assisted engineering state classification.

Default engineering-state examples include:

- `FINAL_OK`
- `SIGNAL_BAD_OR_CONTACT_BAD`
- `UNCERTAIN_OR_MOTION`

## Model A, Model B, And Model D_SAFE

Model A evaluates whether a PPG window has sufficient quality to support heart-rate and SpO2-related output.
Model B evaluates whether an ECG segment has sufficient quality to support heart-rate output and signal-quality explanation.
Model D_SAFE / ABD Fusion combines A/B outputs with IMU motion, temperature/contact state, recovery-window state, data freshness, and abnormal-signal flags to provide model-assisted engineering state evidence.

## Model C Status

Model C is now described as an experimental or reserved ECG rhythm-risk hint module from the broader research path.
It is not part of the current default ABD integrated / ABD_SAFE deployment chain.
Historical Model C materials are retained for research context and historical traceability.

## GD Embedded AI Tool Validation

GD Embedded AI Tool evidence is described as edge-deployment validation, operator-compatibility validation, PC/MCU numerical-consistency validation, and latency evidence.
It is not described as a complete medical-model release or direct medical-output validation.

## Firmware Fusion Arbitration

The final trust/UI behavior is described as firmware-side Fusion arbitration.
The firmware rule layer handles hard constraints such as lead-off, contact abnormality, cold-finger/low-perfusion state, strong motion, stale data, recovery windows, and trend suppression.
AI models provide quality gates and auxiliary evidence only.

## Non-Medical-Diagnosis Boundary

The documentation now explicitly states that repository outputs are measurement-trust states, abnormal-condition prompts, and retest suggestions.
They are not medical diagnosis, disease judgment, medical risk grading, or a replacement for physicians or certified medical devices.

## Historical ABCD Material Retained

Historical ABCD reports, smoke tests, schemas, and action mappings are still retained.
Where the retained text could be confused with the current default route, notes were added to explain that the current default public deployment candidate is ABD integrated / ABD_SAFE and that Model C references are historical or experimental unless explicitly stated otherwise.

## Remaining Terms That May Need Manual Review

The following terms remain in historical reports, schemas, or artifact names for compatibility and traceability:

- `ABCD`
- `Model C`
- `RHYTHM_SUSPECT`
- `RHYTHM_RISK_RETEST`
- `MEASURE_OK`
- `KEEP_STILL`
- `RETEST_CONTACT`
- `final_action`
- `final_reason`

These retained terms are now framed as historical compatibility terms, experimental/reserved-module terms, or firmware-facing retest/action hints rather than current default medical or diagnostic outputs.
