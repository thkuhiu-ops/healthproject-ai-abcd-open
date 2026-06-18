#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AI_ROOT = Path(r"LOCAL_PATH_REMOVED")
OUT_ROOT = AI_ROOT / "model_abcd_integrated_v0_2"
REPORTS = OUT_ROOT / "reports"
FIGURES = OUT_ROOT / "figures"
TABLES = OUT_ROOT / "tables"

A_METRICS = AI_ROOT / "model_a_ppg_trust_gate_v0_4_1" / "reports" / "model_a_v0_4_1_metrics.json"
B_RF_METRICS = AI_ROOT / "model_b_ecg_quality_v0_1" / "reports" / "model_b_metrics.json"
B_TFLITE_METRICS = AI_ROOT / "model_b_ecg_quality_v0_1" / "reports" / "model_b_tflite_metrics.json"
D_METRICS = AI_ROOT / "model_abd_integrated_v0_1" / "reports" / "d_abd_safe_metrics.json"

A_IMG = AI_ROOT / "model_a_ppg_trust_gate_v0_4_1" / "figures" / "confusion_matrix_v0_4_1.png"
B_IMG = AI_ROOT / "model_b_ecg_quality_v0_1" / "figures" / "model_b_confusion_matrix.png"
D_IMG = AI_ROOT / "model_abd_integrated_v0_1" / "reports" / "d_abd_safe_confusion_matrix.png"

REPORT = REPORTS / "ABCD_V0_2_CONFUSION_SOURCE_AUDIT.md"
PANEL = FIGURES / "abcd_v0_2_source_confusion_matrix_panel.png"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_from_cm(cm: np.ndarray) -> dict[str, float]:
    total = int(cm.sum())
    correct = int(np.trace(cm))
    recalls = []
    precisions = []
    f1s = []
    supports = cm.sum(axis=1)
    for i in range(cm.shape[0]):
        tp = float(cm[i, i])
        fn = float(cm[i].sum() - cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    present = supports > 0
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1_present": float(np.mean(np.asarray(f1s)[present])) if present.any() else 0.0,
        "total": total,
        "correct": correct,
    }


def plot_cm(ax, cm: np.ndarray, labels: list[str], title: str) -> None:
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=9)
    return im


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    a = load_json(A_METRICS)["selected"]["val"]
    b_rf = load_json(B_RF_METRICS)["metrics_by_split"]["test"]
    b_tflite = load_json(B_TFLITE_METRICS)["splits"]["test"]
    d = load_json(D_METRICS)["abd_integrated_safe_single_input_float32_tflite"]

    a_cm = np.asarray(a["confusion_matrix"], dtype=int)
    b_rf_cm = np.asarray(b_rf["confusion_matrix"], dtype=int)
    b_tflite_cm = np.asarray(b_tflite["confusion_matrix"], dtype=int)
    d_cm = np.asarray(d["confusion_matrix"], dtype=int)

    labels_a = ["GOOD", "BAD", "UNCERTAIN"]
    labels_b = ["ECG_GOOD", "ECG_BAD", "ECG_UNCERTAIN"]
    labels_d = ["FINAL_OK", "SIGNAL_BAD_OR_CONTACT_BAD", "UNCERTAIN_OR_MOTION"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=180, constrained_layout=True)
    plot_cm(axes[0, 0], a_cm, labels_a, "A PPG v0.4.1 validation")
    plot_cm(axes[0, 1], b_rf_cm, labels_b, "B ECG RandomForest test (original report image)")
    plot_cm(axes[1, 0], b_tflite_cm, labels_b, "B ECG TFLite MLP test (ABD-deployable B)")
    plot_cm(axes[1, 1], d_cm, labels_d, "D_ABD_SAFE test")
    fig.suptitle("ABCD v0.2 Source Confusion Matrix Audit", fontsize=14, fontweight="bold")
    fig.savefig(PANEL, bbox_inches="tight")
    plt.close(fig)

    shutil.copyfile(A_IMG, FIGURES / "source_model_a_confusion_matrix_v0_4_1.png")
    shutil.copyfile(B_IMG, FIGURES / "source_model_b_randomforest_confusion_matrix.png")
    shutil.copyfile(D_IMG, FIGURES / "source_d_abd_safe_confusion_matrix.png")

    a_m = metrics_from_cm(a_cm)
    b_rf_m = metrics_from_cm(b_rf_cm)
    b_tflite_m = metrics_from_cm(b_tflite_cm)
    d_m = metrics_from_cm(d_cm)

    lines = [
        "# ABCD v0.2 Confusion Source Audit",
        "",
        "## Core Finding",
        "",
        "The previous `abcd_v0_2_pc_vs_mcu_confusion_matrix.png` was not a training/test confusion matrix. It was a deployment consistency matrix comparing PC final D output with MCU final D output on a 20-frame benchmark NPZ.",
        "",
        "Therefore it should not be compared directly with Model A, Model B, or D_ABD_SAFE original validation/test confusion matrices.",
        "",
        "## Correct Matrix Types",
        "",
        "| Matrix | Meaning | Uses ground truth? | Rows/columns |",
        "|---|---|---:|---|",
        "| Model A confusion matrix | PPG quality classifier validation | Yes | GOOD/BAD/UNCERTAIN |",
        "| Model B RandomForest confusion matrix | PC-side original ECG classifier test | Yes | ECG_GOOD/ECG_BAD/ECG_UNCERTAIN |",
        "| Model B TFLite MLP confusion matrix | deployable ECG branch test used in ABD graph | Yes | ECG_GOOD/ECG_BAD/ECG_UNCERTAIN |",
        "| D_ABD_SAFE confusion matrix | SAFE fusion head test | Yes | FINAL_OK/SIGNAL_BAD_OR_CONTACT_BAD/UNCERTAIN_OR_MOTION |",
        "| ABCD PC-vs-MCU matrix | board deployment numerical consistency | No | final D labels only |",
        "",
        "## Source Confusion Matrices",
        "",
        f"- Panel figure: `{PANEL}`",
        f"- Copied Model A figure: `{FIGURES / 'source_model_a_confusion_matrix_v0_4_1.png'}`",
        f"- Copied Model B RandomForest figure: `{FIGURES / 'source_model_b_randomforest_confusion_matrix.png'}`",
        f"- Copied D_ABD_SAFE figure: `{FIGURES / 'source_d_abd_safe_confusion_matrix.png'}`",
        "",
        "## Metrics Snapshot",
        "",
        "| Model / split | Confusion matrix | Accuracy | Present macro F1 | Total |",
        "|---|---|---:|---:|---:|",
        f"| A PPG v0.4.1 validation | `{a_cm.tolist()}` | {a_m['accuracy']:.4f} | {a_m['macro_f1_present']:.4f} | {a_m['total']} |",
        f"| B ECG RandomForest test | `{b_rf_cm.tolist()}` | {b_rf_m['accuracy']:.4f} | {b_rf_m['macro_f1_present']:.4f} | {b_rf_m['total']} |",
        f"| B ECG TFLite MLP test | `{b_tflite_cm.tolist()}` | {b_tflite_m['accuracy']:.4f} | {b_tflite_m['macro_f1_present']:.4f} | {b_tflite_m['total']} |",
        f"| D_ABD_SAFE test | `{d_cm.tolist()}` | {d_m['accuracy']:.4f} | {d_m['macro_f1_present']:.4f} | {d_m['total']} |",
        "",
        "## Why The Previous ABCD Matrix Looked Wrong",
        "",
        "1. It used benchmark NPZ without manual ground-truth labels, so it could only compare PC predictions to MCU predictions.",
        "2. It only had final D labels because the exported competition model exposes one output tensor: the final three-class decision.",
        "3. A and B are internal branches. Their intermediate outputs are not separate TFLite output nodes in the deployed model.",
        "4. Model B has two important variants: the original RandomForest report/image, and the deployable TFLite MLP branch used inside ABD. The ABD model cannot embed the RandomForest image result; it embeds the TFLite MLP weights.",
        "5. FINAL_OK has zero support in the current D_ABD_SAFE test matrix and no verified real FINAL_OK windows, so the final D matrix naturally has an empty FINAL_OK row.",
        "",
        "## Correct Reporting Recommendation",
        "",
        "- Use the original A/B/D confusion matrices when reporting classifier validation performance.",
        "- Use the ABCD PC-vs-MCU matrix only when reporting deployment consistency.",
        "- Add the internal A/B/D output analysis figure to show that the ABD graph contains A, B, and D even though the final output tensor has only three classes.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ABCD_CONFUSION_SOURCE_AUDIT_DONE")
    print(f"report={REPORT}")
    print(f"panel={PANEL}")


if __name__ == "__main__":
    main()
