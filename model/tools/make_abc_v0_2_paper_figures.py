#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd


AI_ROOT = Path(r"LOCAL_PATH_REMOVED")
ROOT = AI_ROOT / "model_abcd_integrated_v0_2"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"
TABLES = ROOT / "tables"
EXPORTS = ROOT / "exports"
DEPLOY = ROOT / "deployment_bundle"

A_METRICS = AI_ROOT / "model_a_ppg_trust_gate_v0_4_1" / "reports" / "model_a_v0_4_1_metrics.json"
B_METRICS = AI_ROOT / "model_b_ecg_quality_v0_1" / "reports" / "model_b_metrics.json"
C_METRICS = AI_ROOT / "model_abd_integrated_v0_1" / "reports" / "d_abd_safe_metrics.json"

SRC_MODEL = EXPORTS / "abd_from_abcd_v0_2_single_input_deploy_float32_noslice_tool_verify.tflite"
SRC_LITE = EXPORTS / "abd_from_abcd_v0_2_single_input_deploy_float32_noslice_tool_verify.lite"
ABC_MODEL = EXPORTS / "abc_v0_2_single_input_deploy_float32_noslice_tool_verify.tflite"
ABC_LITE = EXPORTS / "abc_v0_2_single_input_deploy_float32_noslice_tool_verify.lite"

SRC_NPZ = DEPLOY / "real_benchmark_abd_from_abcd_v0_2_input.npz"
ABC_NPZ = DEPLOY / "real_benchmark_abc_v0_2_input.npz"
SRC_EXPECTED = DEPLOY / "real_benchmark_abd_from_abcd_v0_2_expected_output.csv"
ABC_EXPECTED = DEPLOY / "real_benchmark_abc_v0_2_expected_output.csv"

PAPER_CM = FIGURES / "abc_v0_2_model_a_b_c_confusion_matrices_for_paper.png"
PAPER_CM_PDF = FIGURES / "abc_v0_2_model_a_b_c_confusion_matrices_for_paper.pdf"
PAPER_CM_A = FIGURES / "abc_v0_2_model_a_ppg_confusion_matrix_for_paper.png"
PAPER_CM_A_PDF = FIGURES / "abc_v0_2_model_a_ppg_confusion_matrix_for_paper.pdf"
PAPER_CM_B = FIGURES / "abc_v0_2_model_b_ecg_confusion_matrix_for_paper.png"
PAPER_CM_B_PDF = FIGURES / "abc_v0_2_model_b_ecg_confusion_matrix_for_paper.pdf"
PAPER_CM_C = FIGURES / "abc_v0_2_model_c_fusion_confusion_matrix_for_paper.png"
PAPER_CM_C_PDF = FIGURES / "abc_v0_2_model_c_fusion_confusion_matrix_for_paper.pdf"
PAPER_DEPLOY = FIGURES / "abc_v0_2_deployment_pc_mcu_consistency_for_paper.png"
PAPER_DEPLOY_PDF = FIGURES / "abc_v0_2_deployment_pc_mcu_consistency_for_paper.pdf"
PAPER_INTERNAL = FIGURES / "abc_v0_2_internal_branch_distribution_for_paper.png"
PAPER_INTERNAL_PDF = FIGURES / "abc_v0_2_internal_branch_distribution_for_paper.pdf"
PAPER_ERROR = FIGURES / "abc_v0_2_pc_mcu_error_analysis_for_paper.png"
PAPER_ERROR_PDF = FIGURES / "abc_v0_2_pc_mcu_error_analysis_for_paper.pdf"
PAPER_EVAL_SUMMARY = FIGURES / "abc_v0_2_model_and_system_evaluation_summary_for_paper.png"
PAPER_EVAL_SUMMARY_PDF = FIGURES / "abc_v0_2_model_and_system_evaluation_summary_for_paper.pdf"
PAPER_REPORT = REPORTS / "ABC_V0_2_PAPER_FIGURE_AND_NAMING_REPORT.md"
PAPER_EVAL_REPORT = REPORTS / "ABC_V0_2_MODEL_AND_SYSTEM_EVALUATION_FOR_PAPER.md"

PC_MCU_FRAME_CSV = TABLES / "abcd_v0_2_pc_mcu_frame_outputs_and_errors.csv"
INTERNAL_CSV = TABLES / "abcd_v0_2_internal_a_b_d_outputs.csv"
ABC_PC_MCU_FRAME_CSV = TABLES / "abc_v0_2_pc_mcu_frame_outputs_and_errors.csv"
ABC_INTERNAL_CSV = TABLES / "abc_v0_2_internal_a_b_c_outputs.csv"

A_LABELS = ["GOOD", "BAD", "UNCERTAIN"]
B_LABELS = ["ECG_GOOD", "ECG_BAD", "ECG_UNCERTAIN"]
C_LABELS = ["FINAL_OK", "SIGNAL_BAD_OR_CONTACT_BAD", "UNCERTAIN_OR_MOTION"]
PAPER_A_LABELS = ["Good", "Bad", "Uncertain"]
PAPER_B_LABELS = ["Good", "Bad", "Uncertain"]
PAPER_C_LABELS = ["OK", "Signal/contact\nbad", "Uncertain/\nmotion"]

MODEL_A_BOARD_LATENCY_MS = 0.04
MODEL_A_BOARD_MATCH_RATE = 1.0
ABC_BOARD_LATENCY_MS = 0.09
ABC_BOARD_MATCH_RATE = 1.0


def configure_chinese_style() -> None:
    font_candidates = [
        Path(r"LOCAL_PATH_REMOVED Sans SC (TrueType).otf"),
        Path(r"LOCAL_PATH_REMOVED"),
        Path(r"LOCAL_PATH_REMOVED"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            font_name = fm.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_from_cm(cm: np.ndarray) -> tuple[float, float]:
    total = int(cm.sum())
    acc = float(np.trace(cm) / total) if total else 0.0
    support = cm.sum(axis=1)
    f1s = []
    for i in range(cm.shape[0]):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    present = support > 0
    return acc, float(np.mean(np.asarray(f1s)[present])) if present.any() else 0.0


def draw_cm(ax, cm: np.ndarray, labels: list[str], title: str, note: str) -> None:
    vmax = max(int(cm.max()), 1)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_title(f"{title}\n{note}", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_xticks(range(len(labels)), [label.replace("_", "\n") for label in labels], fontsize=10)
    ax.set_yticks(range(len(labels)), [label.replace("_", "\n") for label in labels], fontsize=10)
    threshold = vmax * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="white" if value > threshold else "black",
            )
    return im


def draw_paper_cm(ax, cm: np.ndarray, labels: list[str], title: str, note: str) -> None:
    vmax = max(int(cm.max()), 1)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_title(f"{title}\n{note}", fontsize=10.5, fontweight="bold", pad=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.tick_params(axis="both", length=0)
    threshold = vmax * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white" if value > threshold else "black",
            )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def save_figure(fig, png_path: Path, pdf_path: Path | None = None) -> None:
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    if pdf_path is not None:
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")


def make_single_confusion(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    split: str,
    png_path: Path,
    pdf_path: Path,
) -> dict[str, object]:
    acc, f1 = metrics_from_cm(cm)
    total = int(cm.sum())
    fig, ax = plt.subplots(figsize=(4.1, 3.75), dpi=240, constrained_layout=True)
    draw_paper_cm(ax, cm, labels, title, f"{split}, n={total}, Acc={acc:.3f}, Macro F1={f1:.3f}")
    save_figure(fig, png_path, pdf_path)
    plt.close(fig)
    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": acc,
        "macro_f1_present": f1,
        "total": total,
    }


def make_confusion_for_paper() -> dict[str, object]:
    a_cm = np.asarray(load_json(A_METRICS)["selected"]["val"]["confusion_matrix"], dtype=int)
    b_cm = np.asarray(load_json(B_METRICS)["metrics_by_split"]["test"]["confusion_matrix"], dtype=int)
    c_cm = np.asarray(load_json(C_METRICS)["abd_integrated_safe_single_input_float32_tflite"]["confusion_matrix"], dtype=int)
    items = [
        ("Model A: PPG Quality", "Validation", a_cm, PAPER_A_LABELS, PAPER_CM_A, PAPER_CM_A_PDF),
        ("Model B: ECG Quality", "Test", b_cm, PAPER_B_LABELS, PAPER_CM_B, PAPER_CM_B_PDF),
        ("Model C: Fusion Decision", "Test", c_cm, PAPER_C_LABELS, PAPER_CM_C, PAPER_CM_C_PDF),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8), dpi=220, constrained_layout=True)
    summary = {}
    for ax, (title, split, cm, labels, png_path, pdf_path) in zip(axes, items):
        acc, f1 = metrics_from_cm(cm)
        total = int(cm.sum())
        draw_cm(ax, cm, labels, title, f"{split} | n={total} | Acc {acc:.3f} | Macro F1 {f1:.3f}")
        summary[title] = make_single_confusion(cm, labels, title, split, png_path, pdf_path)
    fig.suptitle("ABC Model Confusion Matrices", fontsize=22, fontweight="bold")
    save_figure(fig, PAPER_CM, PAPER_CM_PDF)
    plt.close(fig)
    return summary


def make_deploy_for_paper() -> None:
    df = pd.read_csv(ABC_PC_MCU_FRAME_CSV)
    pc_pred = df["pc_pred_id"].to_numpy(dtype=int)
    mcu_pred = df["mcu_pred_id"].to_numpy(dtype=int)
    cm = np.zeros((3, 3), dtype=int)
    for p, m in zip(pc_pred, mcu_pred):
        cm[p, m] += 1
    fig, ax = plt.subplots(figsize=(10, 8), dpi=220, constrained_layout=True)
    draw_cm(ax, cm, C_LABELS, "ABC Deployment Consistency", "PC TFLite vs MCU, 20 real NPZ frames")
    ax.set_xlabel("MCU predicted final label", fontsize=12)
    ax.set_ylabel("PC predicted final label", fontsize=12)
    save_figure(fig, PAPER_DEPLOY, PAPER_DEPLOY_PDF)
    plt.close(fig)


def make_internal_for_paper() -> None:
    df = pd.read_csv(ABC_INTERNAL_CSV)
    branches = [
        ("a_pred_label", ["PPG_GOOD", "PPG_BAD", "PPG_UNCERTAIN"], "Model A\nPPG Quality"),
        ("b_pred_label", ["ECG_GOOD", "ECG_BAD", "ECG_UNCERTAIN"], "Model B\nECG Quality"),
        ("c_pred_label", C_LABELS, "Model C\nFusion Decision"),
    ]
    colors = ["#3B73B9", "#D9762D", "#3D9A62"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=220, constrained_layout=True)
    for ax, (col, labels, title) in zip(axes, branches):
        counts = [int((df[col] == label).sum()) for label in labels]
        x = np.arange(3)
        ax.bar(x, counts, color=colors, edgecolor="black", linewidth=0.9)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_ylabel("Frame count", fontsize=11)
        ax.set_xticks(x, [label.replace("_", "\n") for label in labels], fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(counts):
            ax.text(i, value + 0.25, str(value), ha="center", va="bottom", fontsize=14, fontweight="bold")
        ax.set_ylim(0, max(counts + [1]) + 3)
    fig.suptitle("ABC Internal Branch Outputs on Real NPZ", fontsize=20, fontweight="bold")
    save_figure(fig, PAPER_INTERNAL, PAPER_INTERNAL_PDF)
    plt.close(fig)


def make_error_for_paper() -> tuple[float, float]:
    df = pd.read_csv(ABC_PC_MCU_FRAME_CSV)
    err = df[["abs_err_0", "abs_err_1", "abs_err_2"]].to_numpy(dtype=float)
    frame_max = df["frame_max_abs_err"].to_numpy(dtype=float)
    frames = df["frame_idx"].to_numpy(dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=220, constrained_layout=True)
    ax = axes[0]
    ax.plot(frames, frame_max, marker="o", linewidth=2.0, color="#3B73B9")
    ax.set_title("Per-frame maximum absolute error", fontsize=15, fontweight="bold")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Max |PC - MCU|")
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    im = ax.imshow(err.T, aspect="auto", cmap="viridis")
    ax.set_title("Per-class absolute error", fontsize=15, fontweight="bold")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Final class")
    ax.set_yticks(range(3), [label.replace("_", "\n") for label in C_LABELS], fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="absolute error")
    fig.suptitle("ABC PC/MCU Numerical Error Analysis", fontsize=20, fontweight="bold")
    save_figure(fig, PAPER_ERROR, PAPER_ERROR_PDF)
    plt.close(fig)
    return float(err.mean()), float(err.max())


def make_evaluation_summary_for_paper(summary: dict[str, object], mean_err: float, max_err: float) -> None:
    configure_chinese_style()
    model_names = ["模型A\nPPG", "模型B\nECG", "模型C\n融合"]
    acc = [
        summary["Model A: PPG Quality"]["accuracy"],
        summary["Model B: ECG Quality"]["accuracy"],
        summary["Model C: Fusion Decision"]["accuracy"],
    ]
    f1 = [
        summary["Model A: PPG Quality"]["macro_f1_present"],
        summary["Model B: ECG Quality"]["macro_f1_present"],
        summary["Model C: Fusion Decision"]["macro_f1_present"],
    ]

    fig = plt.figure(figsize=(9.2, 4.9), dpi=240, constrained_layout=True)
    fig.patch.set_facecolor("#F7FAFC")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], width_ratios=[1.22, 0.82], wspace=0.18, hspace=0.26)
    ax_metrics = fig.add_subplot(gs[:, 0])
    ax_deploy = fig.add_subplot(gs[0, 1])
    ax_note = fig.add_subplot(gs[1, 1])

    x = np.arange(len(model_names))
    width = 0.34
    colors = ["#4C97C9", "#5EAD76"]
    ax_metrics.set_facecolor("white")
    ax_metrics.bar(x - width / 2, acc, width, label="Accuracy", color=colors[0], edgecolor="#2F4F5F", linewidth=0.5)
    ax_metrics.bar(x + width / 2, f1, width, label="Macro F1", color=colors[1], edgecolor="#2F4F5F", linewidth=0.5)
    ax_metrics.set_ylim(0.88, 1.01)
    ax_metrics.set_ylabel("评分")
    ax_metrics.set_xticks(x, model_names)
    ax_metrics.set_title("模型级分类性能", fontsize=13, fontweight="bold", pad=10)
    ax_metrics.grid(axis="y", color="#B7C3CC", alpha=0.32, linewidth=0.8)
    ax_metrics.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=2,
        fontsize=9,
        borderaxespad=0.0,
    )
    for i, value in enumerate(acc):
        ax_metrics.text(i - width / 2, value + 0.002, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    for i, value in enumerate(f1):
        ax_metrics.text(i + width / 2, value + 0.002, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    for spine in ax_metrics.spines.values():
        spine.set_color("#C8D2DA")

    deploy_labels = ["模型A\nGD32", "整体ABC\nGD32"]
    latencies = [MODEL_A_BOARD_LATENCY_MS, ABC_BOARD_LATENCY_MS]
    ax_deploy.set_facecolor("white")
    ax_deploy.bar(deploy_labels, latencies, color=["#4C97C9", "#F39A27"], edgecolor="#2F4F5F", linewidth=0.5)
    ax_deploy.set_ylabel("延迟 / ms")
    ax_deploy.set_title("嵌入式推理延迟", fontsize=13, fontweight="bold", pad=10)
    ax_deploy.set_ylim(0, 0.11)
    ax_deploy.grid(axis="y", color="#B7C3CC", alpha=0.32, linewidth=0.8)
    for i, value in enumerate(latencies):
        ax_deploy.text(i, value + 0.003, f"{value:.2f} ms", ha="center", va="bottom", fontsize=8)
    for spine in ax_deploy.spines.values():
        spine.set_color("#C8D2DA")

    ax_note.axis("off")
    ax_note.set_facecolor("#F7FAFC")
    ax_note.text(0.0, 1.0, "板端验证摘要", ha="left", va="top", fontsize=12, fontweight="bold", color="#263238")
    cards = [
        ("模型A匹配率", f"{MODEL_A_BOARD_MATCH_RATE:.1f}", "PC/MCU预测一致"),
        ("整体ABC匹配率", f"{ABC_BOARD_MATCH_RATE:.1f}", "20帧真实NPZ"),
        ("平均绝对误差", f"{mean_err:.2e}", "PC/MCU输出"),
        ("最大绝对误差", f"{max_err:.2e}", "低于1.2e-7"),
    ]
    positions = [(0.0, 0.54), (0.52, 0.54), (0.0, 0.12), (0.52, 0.12)]
    for (title, value, subtitle), (x0, y0) in zip(cards, positions):
        rect = plt.Rectangle(
            (x0, y0),
            0.45,
            0.32,
            transform=ax_note.transAxes,
            facecolor="#FFFFFF",
            edgecolor="#D6E0E7",
            linewidth=0.8,
        )
        ax_note.add_patch(rect)
        ax_note.text(x0 + 0.04, y0 + 0.245, title, transform=ax_note.transAxes, fontsize=7.6, color="#60717C")
        ax_note.text(x0 + 0.04, y0 + 0.135, value, transform=ax_note.transAxes, fontsize=10.5, fontweight="bold", color="#263238")
        ax_note.text(x0 + 0.04, y0 + 0.045, subtitle, transform=ax_note.transAxes, fontsize=7.0, color="#7A8A95")
    fig.suptitle("ABC模型与嵌入式系统评价", fontsize=16, fontweight="bold", color="#1D2B34")
    save_figure(fig, PAPER_EVAL_SUMMARY, PAPER_EVAL_SUMMARY_PDF)
    plt.close(fig)


def copy_latest_model_names() -> None:
    shutil.copyfile(SRC_MODEL, ABC_MODEL)
    shutil.copyfile(SRC_LITE, ABC_LITE)
    if SRC_NPZ.exists():
        shutil.copyfile(SRC_NPZ, ABC_NPZ)
    if SRC_EXPECTED.exists():
        shutil.copyfile(SRC_EXPECTED, ABC_EXPECTED)


def copy_latest_table_names() -> None:
    pc_mcu = pd.read_csv(PC_MCU_FRAME_CSV)
    pc_mcu.to_csv(ABC_PC_MCU_FRAME_CSV, index=False)

    internal = pd.read_csv(INTERNAL_CSV)
    internal = internal.rename(
        columns={
            "d_prob_0_final_ok": "c_prob_0_final_ok",
            "d_prob_1_signal_bad_or_contact_bad": "c_prob_1_signal_bad_or_contact_bad",
            "d_prob_2_uncertain_or_motion": "c_prob_2_uncertain_or_motion",
            "d_pred_id": "c_pred_id",
            "d_pred_label": "c_pred_label",
        }
    )
    internal.to_csv(ABC_INTERNAL_CSV, index=False)


def write_report(summary: dict[str, object], mean_err: float, max_err: float) -> None:
    lines = [
        "# ABC v0.2 Paper Figure and Naming Report",
        "",
        "## Naming",
        "",
        "- Model A: PPG quality branch.",
        "- Model B: ECG quality branch.",
        "- Model C: fusion decision branch, previously named D_ABD_SAFE in engineering artifacts.",
        "- The paper-facing integrated model is named ABC rather than ABCD.",
        "",
        "## Paper Figures",
        "",
        f"- ABC three-panel confusion matrices: `{PAPER_CM}`",
        f"- ABC three-panel confusion matrices PDF: `{PAPER_CM_PDF}`",
        f"- Model A single confusion matrix: `{PAPER_CM_A}`",
        f"- Model B single confusion matrix: `{PAPER_CM_B}`",
        f"- Model C single confusion matrix: `{PAPER_CM_C}`",
        f"- ABC deployment consistency matrix: `{PAPER_DEPLOY}`",
        f"- ABC internal branch distribution: `{PAPER_INTERNAL}`",
        f"- ABC PC/MCU error analysis: `{PAPER_ERROR}`",
        f"- ABC model and system evaluation summary: `{PAPER_EVAL_SUMMARY}`",
        "",
        "## Paper-Facing Model Files",
        "",
        f"- TFLite: `{ABC_MODEL}`",
        f"- Lite: `{ABC_LITE}`",
        f"- Real NPZ: `{ABC_NPZ}`",
        f"- Expected output CSV: `{ABC_EXPECTED}`",
        f"- PC/MCU frame table: `{ABC_PC_MCU_FRAME_CSV}`",
        f"- Internal A/B/C output table: `{ABC_INTERNAL_CSV}`",
        "",
        "## Metrics Summary",
        "",
    ]
    for name, values in summary.items():
        lines.append(f"- {name}: accuracy `{values['accuracy']:.4f}`, present macro F1 `{values['macro_f1_present']:.4f}`")
    lines += [
        f"- PC/MCU mean absolute error: `{mean_err:.12e}`",
        f"- PC/MCU max absolute error: `{max_err:.12e}`",
        "- GD32 deploy-tool average latency: `0.09 ms`",
        "",
        "## Recommended Paper Usage",
        "",
        "- Use the Model A/B/C confusion matrices for classifier performance because they are computed against ground-truth labels.",
        "- Use the PC/MCU deployment consistency matrix only for embedded consistency validation.",
        "- Use the PC/MCU error analysis figure to report numerical agreement between TensorFlow Lite on PC and GD32 NN runtime.",
        "- Use the internal branch distribution figure to show that the integrated ABC graph contains PPG, ECG, and fusion-decision branches even though the deployed output tensor has only three final classes.",
        "- Use the model and system evaluation summary figure when the paper needs one compact result graphic covering A/B/C metrics and embedded latency.",
        "",
        "## Model C Note",
        "",
        "- Model C is the paper-facing name of the fusion decision branch that was previously named D_ABD_SAFE in engineering artifacts.",
        "- Model C FINAL_OK has zero support in the current C test set; therefore FINAL_OK should be described as rule-gated in deployment rather than learned from verified normal-stable samples.",
        "",
        "## Important Note",
        "",
        "The ABC deployment consistency matrix is not a ground-truth confusion matrix. It validates PC/MCU numerical consistency. The classifier-performance confusion matrices are the Model A, Model B, and Model C panels.",
    ]
    PAPER_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_evaluation_report(summary: dict[str, object], mean_err: float, max_err: float) -> None:
    a = summary["Model A: PPG Quality"]
    b = summary["Model B: ECG Quality"]
    c = summary["Model C: Fusion Decision"]
    lines = [
        "# ABC v0.2 Model and System Evaluation",
        "",
        "## Paper-Facing Naming",
        "",
        "- Model A: PPG signal quality assessment.",
        "- Model B: ECG signal quality assessment.",
        "- Model C: fusion decision module, corresponding to the former engineering D_ABD_SAFE module.",
        "- Overall ABC: the integrated deployment graph and runtime decision output.",
        "",
        "## Model-Level Classification Results",
        "",
        "| Module | Task | Accuracy | Macro F1 | Evaluation note |",
        "|---|---|---:|---:|---|",
        f"| Model A | PPG quality | {a['accuracy']:.4f} | {a['macro_f1_present']:.4f} | Ground-truth validation set |",
        f"| Model B | ECG quality | {b['accuracy']:.4f} | {b['macro_f1_present']:.4f} | Ground-truth test set |",
        f"| Model C | Fusion decision | {c['accuracy']:.3f} | {c['macro_f1_present']:.3f} | Ground-truth test set, FINAL_OK support is 0 |",
        "",
        "## Embedded Deployment Results",
        "",
        "| Target | Board/runtime | Average latency | Prediction match rate | Numerical error |",
        "|---|---|---:|---:|---|",
        f"| Model A | GD32H759I-START | {MODEL_A_BOARD_LATENCY_MS:.2f} ms | {MODEL_A_BOARD_MATCH_RATE:.1f} | PC/MCU predictions matched |",
        f"| Overall ABC | GD32H759I-START | {ABC_BOARD_LATENCY_MS:.2f} ms | {ABC_BOARD_MATCH_RATE:.1f} | mean abs error {mean_err:.2e}, max abs error {max_err:.2e} |",
        "",
        "## Paper Text",
        "",
        "The proposed ABC framework consists of three modules. Model A performs PPG signal-quality assessment, Model B performs ECG signal-quality assessment, and Model C performs the final fusion decision. In the model-level evaluation, Model A achieved an accuracy of 0.9268 and a macro F1-score of 0.9254. Model B achieved an accuracy of 0.9985 and a macro F1-score of 0.9987. Model C achieved an accuracy of 0.968 and a macro F1-score of 0.968 on the available fusion-decision test set.",
        "",
        "For embedded deployment, Model A was verified on the GD32H759I-START board with an average inference time of 0.04 ms and a prediction match rate of 1.0. The overall ABC deployment graph was also verified on GD32H759I-START, with an average inference latency of 0.09 ms, a prediction match rate of 1.0, a mean absolute PC/MCU output error in the 1e-8 range, and a maximum absolute error below 1.2e-7.",
        "",
        "The overall ABC deployment result should be reported as an embedded consistency and latency result, not as an additional ground-truth classification confusion matrix. The classification performance is represented by the Model A, Model B, and Model C confusion matrices.",
        "",
        "## Generated Figure",
        "",
        f"- Summary figure PNG: `{PAPER_EVAL_SUMMARY}`",
        f"- Summary figure PDF: `{PAPER_EVAL_SUMMARY_PDF}`",
    ]
    PAPER_EVAL_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in [FIGURES, REPORTS, TABLES, EXPORTS, DEPLOY]:
        path.mkdir(parents=True, exist_ok=True)
    copy_latest_model_names()
    copy_latest_table_names()
    summary = make_confusion_for_paper()
    make_deploy_for_paper()
    make_internal_for_paper()
    mean_err, max_err = make_error_for_paper()
    make_evaluation_summary_for_paper(summary, mean_err, max_err)
    write_report(summary, mean_err, max_err)
    write_evaluation_report(summary, mean_err, max_err)
    print("ABC_V0_2_PAPER_FIGURES_DONE")
    print(f"confusion={PAPER_CM}")
    print(f"deploy={PAPER_DEPLOY}")
    print(f"internal={PAPER_INTERNAL}")
    print(f"error={PAPER_ERROR}")
    print(f"report={PAPER_REPORT}")


if __name__ == "__main__":
    main()
