# ABCD v0.2 Internal A/B/D Output Analysis

## Clarification

The GD32 benchmark confusion matrix is a final-output PC-vs-MCU consistency matrix, so it has only the final D_SAFE three labels. That does not mean the deployed model is D-only.

The deployed no-slice model internally contains:

- Model A PPG quality branch: `PPG_GOOD`, `PPG_BAD`, `PPG_UNCERTAIN`
- Model B ECG quality branch: `ECG_GOOD`, `ECG_BAD`, `ECG_UNCERTAIN`
- D_SAFE fusion output: `FINAL_OK`, `SIGNAL_BAD_OR_CONTACT_BAD`, `UNCERTAIN_OR_MOTION`

The competition-facing TFLite output intentionally exposes only the final D_SAFE three-class decision. A and B are internal intermediate branches.

## Generated Files

- Internal output CSV: `LOCAL_PATH_REMOVED
- A/B/D output analysis figure: `LOCAL_PATH_REMOVED
- A/B to D joint heatmaps: `LOCAL_PATH_REMOVED

## Real NPZ Internal Prediction Distribution

| Branch | Class 0 | Class 1 | Class 2 |
|---|---:|---:|---:|
| A PPG | 0 PPG_GOOD | 0 PPG_BAD | 20 PPG_UNCERTAIN |
| B ECG | 1 ECG_GOOD | 12 ECG_BAD | 7 ECG_UNCERTAIN |
| D final | 0 FINAL_OK | 8 SIGNAL_BAD_OR_CONTACT_BAD | 12 UNCERTAIN_OR_MOTION |

## Latency Interpretation

The measured `0.09 ms` latency belongs to the full 107-dim integrated noslice model, not to a standalone 3-output D-only head. The graph includes A and B Dense branches plus D fusion. The visible output tensor is three-dimensional because the final decision contract is three classes.
