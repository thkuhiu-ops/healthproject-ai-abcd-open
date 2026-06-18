# ABC v0.2 Paper Figure and Naming Report

## Naming

- Model A: PPG quality branch.
- Model B: ECG quality branch.
- Model C: fusion decision branch, previously named D_ABD_SAFE in engineering artifacts.
- The paper-facing integrated model is named ABC rather than ABCD.

## Paper Figures

- ABC three-panel confusion matrices: `LOCAL_PATH_REMOVED
- ABC three-panel confusion matrices PDF: `LOCAL_PATH_REMOVED
- Model A single confusion matrix: `LOCAL_PATH_REMOVED
- Model B single confusion matrix: `LOCAL_PATH_REMOVED
- Model C single confusion matrix: `LOCAL_PATH_REMOVED
- ABC deployment consistency matrix: `LOCAL_PATH_REMOVED
- ABC internal branch distribution: `LOCAL_PATH_REMOVED
- ABC PC/MCU error analysis: `LOCAL_PATH_REMOVED
- ABC model and system evaluation summary: `LOCAL_PATH_REMOVED

## Paper-Facing Model Files

- TFLite: `LOCAL_PATH_REMOVED
- Lite: `LOCAL_PATH_REMOVED
- Real NPZ: `LOCAL_PATH_REMOVED
- Expected output CSV: `LOCAL_PATH_REMOVED
- PC/MCU frame table: `LOCAL_PATH_REMOVED
- Internal A/B/C output table: `LOCAL_PATH_REMOVED

## Metrics Summary

- Model A: PPG Quality: accuracy `0.9268`, present macro F1 `0.9254`
- Model B: ECG Quality: accuracy `0.9985`, present macro F1 `0.9987`
- Model C: Fusion Decision: accuracy `0.9677`, present macro F1 `0.9677`
- PC/MCU mean absolute error: `2.144030683363e-08`
- PC/MCU max absolute error: `1.192092895508e-07`
- GD32 deploy-tool average latency: `0.09 ms`

## Recommended Paper Usage

- Use the Model A/B/C confusion matrices for classifier performance because they are computed against ground-truth labels.
- Use the PC/MCU deployment consistency matrix only for embedded consistency validation.
- Use the PC/MCU error analysis figure to report numerical agreement between TensorFlow Lite on PC and GD32 NN runtime.
- Use the internal branch distribution figure to show that the integrated ABC graph contains PPG, ECG, and fusion-decision branches even though the deployed output tensor has only three final classes.
- Use the model and system evaluation summary figure when the paper needs one compact result graphic covering A/B/C metrics and embedded latency.

## Model C Note

- Model C is the paper-facing name of the fusion decision branch that was previously named D_ABD_SAFE in engineering artifacts.
- Model C FINAL_OK has zero support in the current C test set; therefore FINAL_OK should be described as rule-gated in deployment rather than learned from verified normal-stable samples.

## Important Note

The ABC deployment consistency matrix is not a ground-truth confusion matrix. It validates PC/MCU numerical consistency. The classifier-performance confusion matrices are the Model A, Model B, and Model C panels.
