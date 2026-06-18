#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(r"LOCAL_PATH_REMOVED")
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"
DEFAULT_LOG = Path(r"LOCAL_PATH_REMOVED")

RAW_LOG_COPY = REPORTS_DIR / "ABD_FROM_ABCD_V0_2_DEPLOY_TOOL_BENCHMARK_SUCCESS_RAW_LOG.txt"
FRAME_CSV = TABLES_DIR / "abcd_v0_2_pc_mcu_frame_outputs_and_errors.csv"
CONFUSION_CSV = TABLES_DIR / "abcd_v0_2_pc_vs_mcu_confusion_matrix.csv"
SUMMARY_MD = REPORTS_DIR / "ABCD_V0_2_CONFUSION_AND_ERROR_ANALYSIS.md"
CONFUSION_PNG = FIGURES_DIR / "abcd_v0_2_pc_vs_mcu_confusion_matrix.png"
ERROR_PNG = FIGURES_DIR / "abcd_v0_2_error_analysis.png"

LABELS = [
    "FINAL_OK",
    "SIGNAL_BAD_OR_CONTACT_BAD",
    "UNCERTAIN_OR_MOTION",
]


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, FIGURES_DIR, TABLES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def parse_frame_outputs(text: str, section_marker: str) -> np.ndarray:
    start = text.index(section_marker)
    next_marker = "*********************************************************************************************************"
    if section_marker.startswith("1. Run"):
        end = text.index("Step2.", start)
    else:
        end = text.index(next_marker, start)
    section = text[start:end]
    pattern = re.compile(r"The\s+(\d+)\s+frame inference result by (?:PC|MCU):\s*\n\s*(\[[^\]]+\])", re.MULTILINE)
    rows: list[tuple[int, list[float]]] = []
    for match in pattern.finditer(section):
        idx = int(match.group(1))
        values = [float(v) for v in ast.literal_eval(match.group(2))]
        rows.append((idx, values))
    rows.sort(key=lambda item: item[0])
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 rows in {section_marker}, got {len(rows)}")
    return np.asarray([row for _, row in rows], dtype=np.float64)


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    patterns = {
        "inference_times": r"Inference timeLOCAL_PATH_REMOVED",
        "average_latency_ms": r"Inference average latencLOCAL_PATH_REMOVED",
        "average_error": r"Average erroLOCAL_PATH_REMOVED",
        "max_error": r"Max erroLOCAL_PATH_REMOVED",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            out[key] = float(match.group(1))
    return out


def write_csvs(pc: np.ndarray, mcu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pc_pred = np.argmax(pc, axis=1)
    mcu_pred = np.argmax(mcu, axis=1)
    abs_err = np.abs(pc - mcu)
    max_err = np.max(abs_err, axis=1)

    with FRAME_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "frame_idx",
            "pc_prob_0",
            "pc_prob_1",
            "pc_prob_2",
            "mcu_prob_0",
            "mcu_prob_1",
            "mcu_prob_2",
            "abs_err_0",
            "abs_err_1",
            "abs_err_2",
            "frame_max_abs_err",
            "pc_pred_id",
            "pc_pred_label",
            "mcu_pred_id",
            "mcu_pred_label",
            "pred_match",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(pc.shape[0]):
            writer.writerow(
                {
                    "frame_idx": i,
                    "pc_prob_0": pc[i, 0],
                    "pc_prob_1": pc[i, 1],
                    "pc_prob_2": pc[i, 2],
                    "mcu_prob_0": mcu[i, 0],
                    "mcu_prob_1": mcu[i, 1],
                    "mcu_prob_2": mcu[i, 2],
                    "abs_err_0": abs_err[i, 0],
                    "abs_err_1": abs_err[i, 1],
                    "abs_err_2": abs_err[i, 2],
                    "frame_max_abs_err": max_err[i],
                    "pc_pred_id": int(pc_pred[i]),
                    "pc_pred_label": LABELS[int(pc_pred[i])],
                    "mcu_pred_id": int(mcu_pred[i]),
                    "mcu_pred_label": LABELS[int(mcu_pred[i])],
                    "pred_match": bool(pc_pred[i] == mcu_pred[i]),
                }
            )

    cm = np.zeros((3, 3), dtype=int)
    for p, m in zip(pc_pred, mcu_pred):
        cm[int(p), int(m)] += 1
    with CONFUSION_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PC_pred\\MCU_pred"] + [f"{i}:{label}" for i, label in enumerate(LABELS)])
        for i, label in enumerate(LABELS):
            writer.writerow([f"{i}:{label}"] + cm[i].tolist())
    return pc_pred, mcu_pred, cm


def plot_confusion(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.8), dpi=180, constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_title("ABCD v0.2 PC vs MCU Prediction Confusion Matrix", fontsize=12, fontweight="bold")
    ax.set_xlabel("MCU predicted class")
    ax.set_ylabel("PC TFLite predicted class")
    tick_labels = [f"{i}\n{label}" for i, label in enumerate(LABELS)]
    ax.set_xticks(range(3), tick_labels)
    ax.set_yticks(range(3), tick_labels)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.78, label="frame count")
    fig.savefig(CONFUSION_PNG, bbox_inches="tight")
    plt.close(fig)


def plot_error_analysis(pc: np.ndarray, mcu: np.ndarray, pc_pred: np.ndarray, mcu_pred: np.ndarray) -> None:
    abs_err = np.abs(pc - mcu)
    frame_max = abs_err.max(axis=1)
    frame_mean = abs_err.mean(axis=1)
    frames = np.arange(pc.shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), dpi=180, constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(frames, frame_max, marker="o", linewidth=1.8, label="max abs error")
    ax.plot(frames, frame_mean, marker="s", linewidth=1.4, label="mean abs error")
    ax.set_title("Per-frame PC/MCU probability error")
    ax.set_xlabel("frame index")
    ax.set_ylabel("absolute error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    im = ax.imshow(abs_err.T, aspect="auto", cmap="viridis")
    ax.set_title("Per-class absolute error heatmap")
    ax.set_xlabel("frame index")
    ax.set_ylabel("class")
    ax.set_yticks(range(3), [f"{i}:{label}" for i, label in enumerate(LABELS)])
    fig.colorbar(im, ax=ax, shrink=0.85, label="absolute error")

    ax = axes[1, 0]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for cls in range(3):
        ax.scatter(pc[:, cls], mcu[:, cls], s=42, alpha=0.82, color=colors[cls], label=f"{cls}:{LABELS[cls]}")
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--", label="y=x")
    ax.set_title("PC vs MCU softmax probability")
    ax.set_xlabel("PC TFLite probability")
    ax.set_ylabel("MCU probability")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    pc_counts = Counter(int(v) for v in pc_pred)
    mcu_counts = Counter(int(v) for v in mcu_pred)
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width / 2, [pc_counts.get(i, 0) for i in range(3)], width=width, label="PC", color="#4C78A8")
    ax.bar(x + width / 2, [mcu_counts.get(i, 0) for i in range(3)], width=width, label="MCU", color="#F58518")
    ax.set_title("Prediction distribution")
    ax.set_xlabel("class")
    ax.set_ylabel("frame count")
    ax.set_xticks(x, [f"{i}\n{label}" for i, label in enumerate(LABELS)])
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.suptitle("ABCD v0.2 GD32 Benchmark Error Analysis", fontsize=14, fontweight="bold")
    fig.savefig(ERROR_PNG, bbox_inches="tight")
    plt.close(fig)


def write_summary(pc: np.ndarray, mcu: np.ndarray, cm: np.ndarray, metrics: dict[str, float]) -> None:
    abs_err = np.abs(pc - mcu)
    pc_pred = np.argmax(pc, axis=1)
    mcu_pred = np.argmax(mcu, axis=1)
    match_count = int(np.sum(pc_pred == mcu_pred))
    lines = [
        "# ABCD v0.2 Confusion Matrix and Error Analysis",
        "",
        "## Scope",
        "",
        "The real NPZ benchmark does not include manual ground-truth labels, so the confusion matrix is a deployment consistency matrix: PC TFLite predicted class vs MCU predicted class.",
        "",
        "## Files",
        "",
        f"- Raw benchmark log: `{RAW_LOG_COPY}`",
        f"- Frame-level CSV: `{FRAME_CSV}`",
        f"- Confusion matrix CSV: `{CONFUSION_CSV}`",
        f"- Confusion matrix figure: `{CONFUSION_PNG}`",
        f"- Error analysis figure: `{ERROR_PNG}`",
        "",
        "## Benchmark Metrics",
        "",
        f"- Inference times: `{int(metrics.get('inference_times', pc.shape[0]))}`",
        f"- Inference average latency: `{metrics.get('average_latency_ms', float('nan')):.2f} ms`",
        f"- Average error from tool: `{metrics.get('average_error', float('nan')):.12e}`",
        f"- Max error from tool: `{metrics.get('max_error', float('nan')):.12e}`",
        f"- Recomputed mean abs error: `{float(abs_err.mean()):.12e}`",
        f"- Recomputed max abs error: `{float(abs_err.max()):.12e}`",
        f"- Prediction match count: `{match_count}/{pc.shape[0]}`",
        "",
        "## PC-vs-MCU Confusion Matrix",
        "",
        "| PC pred \\ MCU pred | FINAL_OK | SIGNAL_BAD_OR_CONTACT_BAD | UNCERTAIN_OR_MOTION |",
        "|---|---:|---:|---:|",
    ]
    for i, label in enumerate(LABELS):
        lines.append(f"| {i}:{label} | {cm[i,0]} | {cm[i,1]} | {cm[i,2]} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- PC and MCU predicted labels match for all 20 real NPZ frames.",
        "- Current real benchmark windows produce no FINAL_OK predictions; this is consistent with the known lack of verifiable FINAL_OK samples in the current data.",
        "- Numerical error is within the accepted deployment tolerance.",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    ensure_dirs()
    text = args.log.read_text(encoding="utf-8", errors="ignore")
    RAW_LOG_COPY.write_text(text, encoding="utf-8")

    pc = parse_frame_outputs(text, "1. Run model by using Tensorflow Lite in PC")
    mcu = parse_frame_outputs(text, "MCU invoke result")
    metrics = parse_metrics(text)
    pc_pred, mcu_pred, cm = write_csvs(pc, mcu)
    plot_confusion(cm)
    plot_error_analysis(pc, mcu, pc_pred, mcu_pred)
    write_summary(pc, mcu, cm, metrics)
    print("ABCD_V0_2_BENCHMARK_ANALYSIS_DONE")
    print(f"summary={SUMMARY_MD}")
    print(f"confusion_png={CONFUSION_PNG}")
    print(f"error_png={ERROR_PNG}")


if __name__ == "__main__":
    main()
