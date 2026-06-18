# FINAL ABD From ABCD v0.2 Deploy Report

## Verdict

- Successfully built a true 107-dim single-input ABD integrated model containing Model A, Model B, and the v0_1 D_SAFE head.
- Model C is disabled and not loaded into the graph.
- No model retraining was performed.
- No FINAL_OK samples were synthesized.
- Recommended deployment file is the float32 `.tflite` / `.lite` pair listed below.

## Recommended Files

- TFLite: `LOCAL_PATH_REMOVED
- Lite copy: `LOCAL_PATH_REMOVED
- Real NPZ: `LOCAL_PATH_REMOVED
- Expected output CSV: `LOCAL_PATH_REMOVED

## Model Contract

- Input shape: `[1, 107]`
- Input order: `[A_input_58, B_input_32, Context_input_17]`
- Output classes: `0:FINAL_OK, 1:SIGNAL_BAD_OR_CONTACT_BAD, 2:UNCERTAIN_OR_MOTION`
- Model A source was extracted from the v0_4_1 PPG trust gate.
- Model B source was reconstructed from the v0_1 ECG quality gate deploy weights.
- D_SAFE source was loaded from the v0_1 validated SAFE head.
- Internal branches are present but not exposed as output tensors: A produces PPG good/bad/uncertain, B produces ECG good/bad/uncertain, and D_SAFE produces the final three-class decision.
- Internal A/B/D analysis: `LOCAL_PATH_REMOVED

## Operator Audit

- Recommended float32 operators: `FULLY_CONNECTED, FULLY_CONNECTED, FULLY_CONNECTED, SOFTMAX, FULLY_CONNECTED, FULLY_CONNECTED, FULLY_CONNECTED, SOFTMAX, CONCATENATION, FULLY_CONNECTED, SOFTMAX`
- Contains EQUAL: `False`
- Forbidden operators present: `NONE`
- `CONCATENATION` is present and expected for A/B/context fusion.
- Dynamic quant and sliced standard artifacts were removed during cleanup.

## PC TFLite Validation

- Real NPZ shape: `[20, 107]`
- Input finite: `True`
- Output finite: `True`
- Predicted distribution: `{"UNCERTAIN_OR_MOTION": 12, "SIGNAL_BAD_OR_CONTACT_BAD": 8}`
- Max abs diff vs folded-source reference: `7.450580597e-08`
- Mean abs diff vs folded-source reference: `1.101846792e-08`
- Validation report: `LOCAL_PATH_REMOVED

## GD32 Deploy Tool Benchmark

- Automatic benchmark completed: `YES`
- Verified model: `LOCAL_PATH_REMOVED
- Verified NPZ: `LOCAL_PATH_REMOVED
- MCU_READY appeared: `YES`
- Total inference frames: `20`
- Inference average latency: `0.09 ms`
- Average error: `2.1440306833634772e-08`
- Max error: `1.19DATE_REMOVED078125e-07`
- Benchmark success report: `LOCAL_PATH_REMOVED

## Data Site Verify Result

- Verified no-slice model: `LOCAL_PATH_REMOVED
- No-slice operators: `FULLY_CONNECTED, FULLY_CONNECTED, FULLY_CONNECTED, SOFTMAX, FULLY_CONNECTED, FULLY_CONNECTED, FULLY_CONNECTED, SOFTMAX, CONCATENATION, FULLY_CONNECTED, SOFTMAX`
- Workaround report: `LOCAL_PATH_REMOVED

## Prior v0_1 D_SAFE Board Validation Reference

- Previous v0_1 benchmark log: `LOCAL_PATH_REMOVED
- MCU_READY: `present`
- Receive complete: `present`
- Average error: `1.986821492513021e-09`
- Max error: `9.313225746154785e-09`

## Deployment Recommendation

- The no-slice 107-dim ABD integrated model is recommended for GD32 deployment because it passed Data Site Verify and full deploy-tool benchmark with real NPZ input.
- For the competition runtime, FINAL_OK should still be guarded by deterministic quality rules because current real data lacks verifiable FINAL_OK samples.
- PC/MCU benchmark consistency proves model numerical consistency only; it does not prove full sensor closed-loop behavior.
