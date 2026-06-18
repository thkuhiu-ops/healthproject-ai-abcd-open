# ABD From ABCD v0.2 Deploy Tool Benchmark Success

## Verdict

The GD32 AI deploy tool benchmark completed successfully with the real 107-dim ABD integrated NPZ.

## Files

- Model: `LOCAL_PATH_REMOVED
- NPZ: `LOCAL_PATH_REMOVED

## Success Markers

- `NPZ file Number: 1 sent success`
- `MCU Response: MCU_READY`
- `Receiving No.20 mcu invoke result, total:20`
- `Receive complete`
- `Step3. Evaluate results.`

## Benchmark Metrics

- Inference times: `20`
- Inference average latency: `0.09 ms`
- Average error: `2.1440306833634772e-08`
- Max error: `1.1920928955078125e-07`

## Interpretation

The benchmark proves that the exported 107-dim ABD integrated TFLite model can be executed by the GD32 NN library with real NPZ input. PC TFLite outputs and MCU outputs are numerically consistent within the accepted tolerance.

This validates the model deployment path for:

- real NPZ input transfer,
- MCU benchmark handshake,
- GD32 NN invocation,
- softmax output transfer,
- PC/MCU numerical comparison.

## Remaining Runtime Boundary

The model output still follows the deployed three-class schema:

- `0 FINAL_OK`
- `1 SIGNAL_BAD_OR_CONTACT_BAD`
- `2 UNCERTAIN_OR_MOTION`

Because current real training/window data lacks verifiable FINAL_OK samples, competition runtime should still use the FINAL_OK rule-gating strategy before trusting class `0`.
