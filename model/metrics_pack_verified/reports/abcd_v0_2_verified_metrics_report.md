# ABCD v0.2 verified metrics 报告

## 范围

该 verified pack 只使用可追溯的本地来源。本次未重新训练模型，未修改阈值，未导出 TFLite，未进入 GD32，也未修改 firmware。

## 验证状态

- Model A：已从 prediction table 验证，来源为 `model_a_v0_4_1_fixed_val_predictions.csv`。
- Model B：report-only，来源为 `model_b_metrics.json`；未找到 prediction-level test table。
- Model C：已从 threshold-calibrated predictions 和阈值 0.42 / 0.60 验证。
- Model D：report-only，来源为 `model_d_v0_2_metrics.json`；未找到 prediction-level table。
- ABCD smoke：已从 fixed smoke output 验证；不用于单模型准确率。

## 无效 / 过期 / 不支持的原始产物

- 原始 figures 普遍缺少嵌入式 source note，因此已重新生成到 `figures_verified`。
- `model_c_binary_confusion_matrix.png` 是 raw/uncalibrated 图，不能作为最终 Model C 指标。
- 引用 Model A v0.4 的历史/finalizer 产物已标记为 stale source。

## 部署门禁

PC_ABCD_PIPELINE_READY = True
FIRMWARE_READY = False
TFLITE_ALL_READY = False
GD32_READY = False

## 建议

PPT、论文或 demo 证据只使用 `metrics_pack_verified` 和 verified deliverable。避免声明诊断能力或跨受试者临床泛化。
