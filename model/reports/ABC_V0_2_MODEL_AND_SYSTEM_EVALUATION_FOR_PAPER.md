# ABC v0.2 Model and System Evaluation

## Paper-Facing Naming

- Model A: PPG signal quality assessment.
- Model B: ECG signal quality assessment.
- Model C: fusion decision module, corresponding to the former engineering D_ABD_SAFE module.
- Overall ABC: the integrated deployment graph and runtime decision output.

## Model-Level Classification Results

| Module | Task | Accuracy | Macro F1 | Evaluation note |
|---|---|---:|---:|---|
| Model A | PPG quality | 0.9268 | 0.9254 | Ground-truth validation set |
| Model B | ECG quality | 0.9985 | 0.9987 | Ground-truth test set |
| Model C | Fusion decision | 0.968 | 0.968 | Ground-truth test set, FINAL_OK support is 0 |

## Embedded Deployment Results

| Target | Board/runtime | Average latency | Prediction match rate | Numerical error |
|---|---|---:|---:|---|
| Model A | GD32H759I-START | 0.04 ms | 1.0 | PC/MCU predictions matched |
| Overall ABC | GD32H759I-START | 0.09 ms | 1.0 | mean abs error 2.14e-08, max abs error 1.19e-07 |

## Paper Text

The proposed ABC framework consists of three modules. Model A performs PPG signal-quality assessment, Model B performs ECG signal-quality assessment, and Model C performs the final fusion decision. In the model-level evaluation, Model A achieved an accuracy of 0.9268 and a macro F1-score of 0.9254. Model B achieved an accuracy of 0.9985 and a macro F1-score of 0.9987. Model C achieved an accuracy of 0.968 and a macro F1-score of 0.968 on the available fusion-decision test set.

For embedded deployment, Model A was verified on the GD32H759I-START board with an average inference time of 0.04 ms and a prediction match rate of 1.0. The overall ABC deployment graph was also verified on GD32H759I-START, with an average inference latency of 0.09 ms, a prediction match rate of 1.0, a mean absolute PC/MCU output error in the 1e-8 range, and a maximum absolute error below 1.2e-7.

The overall ABC deployment result should be reported as an embedded consistency and latency result, not as an additional ground-truth classification confusion matrix. The classification performance is represented by the Model A, Model B, and Model C confusion matrices.

## Generated Figure

- Summary figure PNG: `LOCAL_PATH_REMOVED
- Summary figure PDF: `LOCAL_PATH_REMOVED
