# Model A integrated 图不一致根因

## 问题是什么

ABCD v0.2 原先从 `LOCAL_PATH_REMOVED 读取 Model A 指标。
这个文件是较旧的 v0.4 验证摘要，其 41 样本混淆矩阵为 `[[8, 0, 4], [0, 7, 4], [0, 5, 13]]`，因此 integrated 图和报告反映的是错误来源。

## 现在的权威来源

正确来源是 `LOCAL_PATH_REMOVED
从该预测表重新计算得到预期的 v0.4.1 混淆矩阵：`[[12, 0, 0], [0, 9, 2], [0, 1, 17]]`。

## 为什么图会变化

Integrated dashboard 和 Model A confusion matrix 都由被选中的 Model A 来源文件生成。
当来源从 stale v0.4 JSON 切换为 standalone v0.4.1 validation prediction table 后，图中的结果更新为正确的 38/41。

## 修正后的 Model A 指标

- Accuracy：0.9268
- Macro F1：0.9254
- BAD recall：0.8182
- UNCERTAIN recall：0.9444
- BAD -> GOOD：0
- UNCERTAIN -> GOOD：0

## ABCD 结论

ABCD 最终结论不变。该包仍然只是 PC 侧 integrated candidate，仍然不是 firmware-ready。
