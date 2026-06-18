# ABCD v0.2 Model Cleanup Report

## Goal

Keep only the latest GD32-verified ABCD deployment model and current benchmark/analysis results. Remove stale ABCD model artifacts that can confuse deployment.

## Latest Model Kept

- `LOCAL_PATH_REMOVED
- `LOCAL_PATH_REMOVED
- `LOCAL_PATH_REMOVED

## Data and Results Kept

- Real NPZ/NPY/CSV benchmark input under `deployment_bundle`.
- Expected output CSV under `deployment_bundle`.
- PC/MCU frame-level error CSV under `tables`.
- PC-vs-MCU confusion matrix CSV under `tables`.
- Confusion matrix and error analysis figures under `figures`.
- Final deploy and benchmark success reports under `reports`.

## Removed Old Model Artifacts

- `exports\abd_from_abcd_v0_2_single_input_deploy_dynamic_quant.tflite`
- `exports\abd_from_abcd_v0_2_single_input_deploy_dynamic_quant.lite`
- `exports\abd_from_abcd_v0_2_single_input_deploy_float32.tflite`
- `exports\abd_from_abcd_v0_2_single_input_deploy_float32.lite`
- `exports\新建 文本文档.txt`
- `deploy_unified_v0_1\models\model_abcd_unified_deploy_sim_v0_1_float32.keras`
- `deploy_unified_v0_1\models\model_abcd_unified_deploy_sim_v0_1_float32.tflite`
- `GD_Embedded_AI\User_model\cur_tflite\abd_from_abcd_v0_2_single_input_deploy_float32.tflite`

## Current Recommendation

Use the no-slice model for GD32 deployment and competition demonstration because it passed:

- Data Site Verify workaround path,
- real NPZ transfer,
- MCU benchmark invocation,
- 20-frame PC/MCU numerical comparison,
- average latency `0.09 ms`,
- average error `2.1440306833634772e-08`,
- max error `1.1920928955078125e-07`.
