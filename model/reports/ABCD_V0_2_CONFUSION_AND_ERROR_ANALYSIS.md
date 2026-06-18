# ABCD v0.2 Confusion Matrix and Error Analysis

## Scope

The real NPZ benchmark does not include manual ground-truth labels, so the confusion matrix is a deployment consistency matrix: PC TFLite predicted class vs MCU predicted class.

## Files

- Raw benchmark log: `LOCAL_PATH_REMOVED
- Frame-level CSV: `LOCAL_PATH_REMOVED
- Confusion matrix CSV: `LOCAL_PATH_REMOVED
- Confusion matrix figure: `LOCAL_PATH_REMOVED
- Error analysis figure: `LOCAL_PATH_REMOVED

## Benchmark Metrics

- Inference times: `20`
- Inference average latency: `0.09 ms`
- Average error from tool: `2.144030683363e-08`
- Max error from tool: `1.192092895508e-07`
- Recomputed mean abs error: `2.144030683363e-08`
- Recomputed max abs error: `1.192092895508e-07`
- Prediction match count: `20/20`

## PC-vs-MCU Confusion Matrix

| PC pred \ MCU pred | FINAL_OK | SIGNAL_BAD_OR_CONTACT_BAD | UNCERTAIN_OR_MOTION |
|---|---:|---:|---:|
| 0:FINAL_OK | 0 | 0 | 0 |
| 1:SIGNAL_BAD_OR_CONTACT_BAD | 0 | 8 | 0 |
| 2:UNCERTAIN_OR_MOTION | 0 | 0 | 12 |

## Interpretation

- PC and MCU predicted labels match for all 20 real NPZ frames.
- Current real benchmark windows produce no FINAL_OK predictions; this is consistent with the known lack of verifiable FINAL_OK samples in the current data.
- Numerical error is within the accepted deployment tolerance.
