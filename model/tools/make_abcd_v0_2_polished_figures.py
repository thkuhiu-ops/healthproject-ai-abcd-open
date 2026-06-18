#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AI_ROOT = Path(r"LOCAL_PATH_REMOVED")
OUT_ROOT = AI_ROOT / "model_abcd_integrated_v0_2"
FIGURES = OUT_ROOT / "figures"
REPORTS = OUT_ROOT / "reports"
TABLES = OUT_ROOT / "tables"

A_METRICS = AI_ROOT / "model_a_ppg_trust_gate_v0_4_1" / "reports" / "model_a_v0_4_1_metrics.json"
B_RF_METRICS = AI_ROOT / "model_b_ecg_quality_v0_1" / "reports" / "model_b_metrics.json"
B_TFLITE_METRICS = AI_ROOT / "model_b_ecg_quality_v0_1" / "reports" / "model_b_tflite_metrics.json"
D_METRICS = AI_ROOT / "model_abd_integrated_v0_1" / "reports" / "d_abd_safe_metrics.json"
PC_MCU_CM = TABLES / "abcd_v0_2_pc_vs_mcu_confusion_matrix.csv"
INTERNAL_CSV = TABLES / "abcd_v0_2_internal_a_b_d_outputs.csv"

PANEL_POLISHED = FIGURES / "abcd_v0_2_source_confusion_matrix_panel_polished.png"
DEPLOY_POLISHED = FIGURES / "abcd_v0_2_pc_mcu_deployment_consistency_matrix_polished.png"
INTERNAL_POLISHED = FIGURES / "abcd_v0_2_internal_branch_distribution_polished.png"
REPORT = REPORTS / "ABCD_V0_2_POLISHED_FIGURE_NOTES.md"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_from_cm(cm: np.ndarray) -> tuple[float, float]:
    total = cm.sum()
    acc = float(np.trace(cm) / total) if total else 0.0
    f1s = []
    support = cm.sum(axis=1)
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    present = support > 0
    macro_f1 = float(np.mean(np.asarray(f1s)[present])) if present.any() else 0.0
    return acc, macro_f1


def draw_cm(ax, cm: np.ndarray, xlabels: list[str], ylabels: list[str], title: str, subtitle: str = ""):
    vmax = max(int(cm.max()), 1)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True" if ylabels == xlabels else "Reference", fontsize=12)
    ax.set_xticks(np.arange(len(xlabels)), xlabels, rotation=28, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(ylabels)), ylabels, fontsize=10)
    threshold = vmax * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", fontsize=14, fontweight="bold", color=color)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    return im


def make_source_panel() -> None:
    a = np.asarray(load_json(A_METRICS)["selected"]["val"]["confusion_matrix"], dtype=int)
    b_rf = np.asarray(load_json(B_RF_METRICS)["metrics_by_split"]["test"]["confusion_matrix"], dtype=int)
    b_tflite = np.asarray(load_json(B_TFLITE_METRICS)["splits"]["test"]["confusion_matrix"], dtype=int)
    d = np.asarray(load_json(D_METRICS)["abd_integrated_safe_single_input_float32_tflite"]["confusion_matrix"], dtype=int)

    configs = [
        (a, ["GOOD", "BAD", "UNCERTAIN"], "A: PPG Trust Gate", "v0.4.1 validation"),
        (b_rf, ["ECG_GOOD", "ECG_BAD", "ECG_UNCERTAIN"], "B: ECG Quality", "original RandomForest test"),
        (b_tflite, ["ECG_GOOD", "ECG_BAD", "ECG_UNCERTAIN"], "B: ECG Quality", "deployable TFLite MLP test"),
        (d, ["FINAL_OK", "SIGNAL_BAD_OR_CONTACT_BAD", "UNCERTAIN_OR_MOTION"], "D: ABD_SAFE Fusion", "test split"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=180, constrained_layout=True)
    for ax, (cm, labels, title, split) in zip(axes.ravel(), configs):
        acc, f1 = metrics_from_cm(cm)
        draw_cm(ax, cm, labels, labels, title, f"{split} | Acc {acc:.3f} | F1 {f1:.3f}")
    fig.suptitle("Source Model Confusion Matrices Used by ABCD v0.2", fontsize=20, fontweight="bold")
    fig.savefig(PANEL_POLISHED, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_deploy_matrix() -> None:
    df = pd.read_csv(PC_MCU_CM)
    cm = df.iloc[:, 1:].to_numpy(dtype=int)
    labels = ["FINAL_OK", "SIGNAL_BAD_OR_CONTACT_BAD", "UNCERTAIN_OR_MOTION"]
    fig, ax = plt.subplots(figsize=(10.5, 8.5), dpi=180, constrained_layout=True)
    im = draw_cm(
        ax,
        cm,
        labels,
        labels,
        "ABCD v0.2 Deployment Consistency Matrix",
        "PC TFLite prediction vs MCU prediction, not a ground-truth confusion matrix",
    )
    ax.set_xlabel("MCU predicted final label", fontsize=12)
    ax.set_ylabel("PC predicted final label", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label("Frame count", fontsize=11)
    fig.savefig(DEPLOY_POLISHED, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_internal_distribution() -> None:
    df = pd.read_csv(INTERNAL_CSV)
    branches = [
        ("a_pred_label", ["PPG_GOOD", "PPG_BAD", "PPG_UNCERTAIN"], "Internal A Branch: PPG Quality"),
        ("b_pred_label", ["ECG_GOOD", "ECG_BAD", "ECG_UNCERTAIN"], "Internal B Branch: ECG Quality"),
        ("d_pred_label", ["FINAL_OK", "SIGNAL_BAD_OR_CONTACT_BAD", "UNCERTAIN_OR_MOTION"], "Final D_SAFE Decision"),
    ]
    colors = ["#2F6FBB", "#D95F02", "#2CA25F"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=180, constrained_layout=True)
    for ax, (col, labels, title) in zip(axes, branches):
        counts = [int((df[col] == label).sum()) for label in labels]
        x = np.arange(len(labels))
        ax.bar(x, counts, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Frame count", fontsize=11)
        ax.set_xticks(x, [label.replace("_", "\n") for label in labels], fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(counts):
            ax.text(i, value + 0.25, str(value), ha="center", va="bottom", fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(counts + [1]) + 3)
    fig.suptitle("ABCD v0.2 Internal A/B/D Prediction Distributions on Real NPZ", fontsize=18, fontweight="bold")
    fig.savefig(INTERNAL_POLISHED, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report() -> None:
    lines = [
        "# ABCD v0.2 Polished Figure Notes",
        "",
        "The earlier generated figures were engineering diagnostics and were visually weaker than the original single-purpose confusion matrices. The updated figures separate the reporting purposes clearly:",
        "",
        f"- Source model validation/test confusion matrices: `{PANEL_POLISHED}`",
        f"- Deployment consistency matrix: `{DEPLOY_POLISHED}`",
        f"- Internal A/B/D branch distribution: `{INTERNAL_POLISHED}`",
        "",
        "Important distinction:",
        "",
        "- A/B/D source confusion matrices use ground-truth labels.",
        "- The ABCD deployment matrix uses PC output vs MCU output and has no ground-truth labels.",
        "- The deployed TFLite exposes only final D output, while A and B remain internal branches.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    make_source_panel()
    make_deploy_matrix()
    make_internal_distribution()
    write_report()
    print("POLISHED_ABCD_FIGURES_DONE")
    print(PANEL_POLISHED)
    print(DEPLOY_POLISHED)
    print(INTERNAL_POLISHED)


if __name__ == "__main__":
    main()
