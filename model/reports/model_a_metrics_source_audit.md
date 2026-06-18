# Model A 指标来源审计

## 审计范围

本次审计覆盖 `LOCAL_PATH_REMOVED 内所有与 Model A 相关的图、指标、表格和汇总文件。
以下内容不作为 Model A 权威指标来源：`archive/`、`old/`、`superseded/`、`deprecated/`、`temp/`、ABCD smoke 输出、合成 replay 表。

## 权威来源

- SOURCE_TYPE = `STANDALONE_VALIDATION_PREDICTION_TABLE`
- 来源表：`LOCAL_PATH_REMOVED
- 对应 standalone 报告/摘要：`LOCAL_PATH_REMOVED 和 `LOCAL_PATH_REMOVED
- 验证后的混淆矩阵：`[[12, 0, 0], [0, 9, 2], [0, 1, 17]]`

## 已审计产物

| 产物 | 来源分类 | 说明 |
| --- | --- | --- |
| `metrics_pack/json/model_a_metrics.json` | standalone 验证集预测表 | 由 v0.4.1 prediction table 重新计算。 |
| `metrics_pack/tables/model_a_metrics.csv` | standalone 验证集预测表 | 与同一 prediction table 派生。 |
| `metrics_pack/reports/model_a_metrics_report.md` | standalone 验证集预测表 | 人类可读的 Model A 指标摘要。 |
| `metrics_pack/figures/model_a_confusion_matrix.png` | standalone 验证集预测表 | 基于 38/41 验证结果重新生成。 |
| `metrics_pack/figures/model_a_label_distribution.png` | standalone 验证集预测表 | 基于同一混淆矩阵 support 派生。 |
| `metrics_pack/errors/model_a_error_cases.csv` | standalone 验证集预测表 | 只包含非对角线错误样本。 |
| `metrics_pack/json/abcd_model_metrics_summary.json` | 修正后 Model A 指标派生的 integrated summary | 只更新 Model A 行，B/C/D 不变。 |
| `metrics_pack/tables/abcd_model_metrics_summary.csv` | 修正后 Model A 指标派生的 integrated summary | 只更新 Model A 行，B/C/D 不变。 |
| `metrics_pack/figures/abcd_v0_2_metrics_dashboard.png` | 修正后 summary 派生的 integrated dashboard | 只改变 Model A 面板。 |
| `metrics_pack/reports/abcd_v0_2_final_metrics_report.md` | 修正后 summary 派生的 integrated report | 更新 Model A 章节和来源审计说明。 |
| `metrics_pack/ppt_assets/model_a_confusion_matrix.png` | 修正后 Model A 图的展示副本 | PPT 资产与修正图一致。 |
| `metrics_pack/ppt_assets/model_a_label_distribution.png` | 修正后 Model A 图的展示副本 | PPT 资产与修正图一致。 |

## 结论

Model A 现在锚定 standalone v0.4.1 验证集预测表，而不是旧的 v0.4 summary JSON。
