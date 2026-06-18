# ABCD v0.2 Confusion Source Audit

## Core Finding

The previous `abcd_v0_2_pc_vs_mcu_confusion_matrix.png` was not a training/test confusion matrix. It was a deployment consistency matrix comparing PC final D output with MCU final D output on a 20-frame benchmark NPZ.

Therefore it should not be compared directly with Model A, Model B, or D_ABD_SAFE original validation/test confusion matrices.

## Correct Matrix Types

| Matrix | Meaning | Uses ground truth? | Rows/columns |
|---|---|---:|---|
| Model A confusion matrix | PPG quality classifier validation | Yes | GOOD/BAD/UNCERTAIN |
| Model B RandomForest confusion matrix | PC-side original ECG classifier test | Yes | ECG_GOOD/ECG_BAD/ECG_UNCERTAIN |
| Model B TFLite MLP confusion matrix | deployable ECG branch test used in ABD graph | Yes | ECG_GOOD/ECG_BAD/ECG_UNCERTAIN |
| D_ABD_SAFE confusion matrix | SAFE fusion head test | Yes | FINAL_OK/SIGNAL_BAD_OR_CONTACT_BAD/UNCERTAIN_OR_MOTION |
| ABCD PC-vs-MCU matrix | board deployment numerical consistency | No | final D labels only |

## Source Confusion Matrices

- Panel figure: `LOCAL_PATH_REMOVED
- Copied Model A figure: `LOCAL_PATH_REMOVED
- Copied Model B RandomForest figure: `LOCAL_PATH_REMOVED
- Copied D_ABD_SAFE figure: `LOCAL_PATH_REMOVED

## Metrics Snapshot

| Model / split | Confusion matrix | Accuracy | Present macro F1 | Total |
|---|---|---:|---:|---:|
| A PPG v0.4.1 validation | `[[12, 0, 0], [0, 9, 2], [0, 1, 17]]` | 0.9268 | 0.9254 | 41 |
| B ECG RandomForest test | `[[77, 0, 0], [0, 211, 0], [0, 1, 359]]` | 0.9985 | 0.9987 | 648 |
| B ECG TFLite MLP test | `[[42, 0, 35], [0, 194, 17], [16, 15, 329]]` | 0.8719 | 0.8113 | 648 |
| D_ABD_SAFE test | `[[0, 0, 0], [0, 43, 3], [0, 0, 47]]` | 0.9677 | 0.9677 | 93 |

## Why The Previous ABCD Matrix Looked Wrong

1. It used benchmark NPZ without manual ground-truth labels, so it could only compare PC predictions to MCU predictions.
2. It only had final D labels because the exported competition model exposes one output tensor: the final three-class decision.
3. A and B are internal branches. Their intermediate outputs are not separate TFLite output nodes in the deployed model.
4. Model B has two important variants: the original RandomForest report/image, and the deployable TFLite MLP branch used inside ABD. The ABD model cannot embed the RandomForest image result; it embeds the TFLite MLP weights.
5. FINAL_OK has zero support in the current D_ABD_SAFE test matrix and no verified real FINAL_OK windows, so the final D matrix naturally has an empty FINAL_OK row.

## Correct Reporting Recommendation

- Use the original A/B/D confusion matrices when reporting classifier validation performance.
- Use the ABCD PC-vs-MCU matrix only when reporting deployment consistency.
- Add the internal A/B/D output analysis figure to show that the ABD graph contains A, B, and D even though the final output tensor has only three classes.
