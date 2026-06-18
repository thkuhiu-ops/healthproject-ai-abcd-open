# ABCD v0.2 全量来源追溯审计报告

## 1. 执行摘要

最终 verified metrics 和 verified figures 均可追溯到真实的本地来源。Model A 和 Model C 的已接受指标均从 prediction table 重新计算。Model B 和 Model D 因未找到 prediction-level test table，当前按 report-only 处理。ABCD smoke 验证来自 fixed smoke output，仅用于集成逻辑验证，不作为单模型准确率。

## 2. 权威来源注册表

详见 `provenance_audit/json/source_of_truth_registry.json` 和 `provenance_audit/reports/source_of_truth_registry.md`。

## 3. 图表来源审计

原始图表已记录在 `provenance_audit/tables/figure_provenance_audit.csv`。现有原始图普遍缺少嵌入式 source note；另外，raw Model C binary confusion matrix 不能作为最终 Model C 指标。已在 `provenance_audit/figures_verified` 生成 verified replacement figures。

## 4. 指标声明来源审计

核心指标声明已映射在 `provenance_audit/tables/metric_claim_provenance.csv`。所有 verified-pack claims 均可追溯到来源文件。

## 5. 重新计算指标

- Model A accuracy：0.9268；macro F1：0.9254；matrix：[[12, 0, 0], [0, 9, 2], [0, 1, 17]]。
- Model B accuracy：0.9985；macro F1：0.9987；report-only。
- Model C suspect recall：0.8125；suspect-as-normal：0.0625；normal-as-suspect：0.1136；other/uncertain：0.2167。
- Model D accuracy：0.9987；macro F1：0.9975；high-risk errors：0；report-only。
- ABCD smoke pass rate：1.0000。

## 6. 无效 / 过期 / 不支持的产物

无效或未验证的原始产物列在 `provenance_audit/reports/invalid_or_unverified_artifacts.md`，并复制到 `provenance_audit/invalid_or_unverified`。本次未确认存在 fabricated metrics；但历史/finalizer 引用和 raw/uncalibrated Model C figures 存在 stale-source 风险。

## 7. Verified metrics pack

Verified pack：`LOCAL_PATH_REMOVED
Verified dashboard：`metrics_pack_verified/figures/abcd_v0_2_verified_metrics_dashboard.png`。

## 8. 部署门禁

PC_ABCD_PIPELINE_READY = True
FIRMWARE_READY = False
TFLITE_ALL_READY = False
GD32_READY = False

## 9. 最终建议

PPT、论文和 demo 只使用 `metrics_pack_verified` 以及 `deliverable/abcd_integrated_v0_2_pc_candidate_verified.zip`。不要把 old/raw/uncalibrated figures 作为最终指标图使用，也不要声明诊断能力或跨受试者临床泛化。
