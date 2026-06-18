#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(r"LOCAL_PATH_REMOVED")
BUILD_SCRIPT = PROJECT_ROOT / "tools" / "build_abd_from_abcd_v0_2.py"
NPZ_PATH = PROJECT_ROOT / "deployment_bundle" / "real_benchmark_abd_from_abcd_v0_2_input.npz"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"

OUT_CSV = TABLES_DIR / "abcd_v0_2_internal_a_b_d_outputs.csv"
REPORT_MD = REPORTS_DIR / "ABCD_V0_2_INTERNAL_A_B_D_OUTPUT_ANALYSIS.md"
CHAIN_FIG = FIGURES_DIR / "abcd_v0_2_internal_a_b_d_output_analysis.png"
JOINT_FIG = FIGURES_DIR / "abcd_v0_2_internal_ab_to_d_joint_heatmaps.png"

A_LABELS = ["PPG_GOOD", "PPG_BAD", "PPG_UNCERTAIN"]
B_LABELS = ["ECG_GOOD", "ECG_BAD", "ECG_UNCERTAIN"]
D_LABELS = ["FINAL_OK", "SIGNAL_BAD_OR_CONTACT_BAD", "UNCERTAIN_OR_MOTION"]


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, FIGURES_DIR, TABLES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_build_module():
    spec = importlib.util.spec_from_file_location("build_abd_from_abcd_v0_2", BUILD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {BUILD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def submodel_outputs(model: tf.keras.Model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_model = tf.keras.Model(model.input, model.get_layer("A_out_ppg_quality").output)
    b_model = tf.keras.Model(model.input, model.get_layer("B_out_ecg_quality").output)
    d_model = tf.keras.Model(model.input, model.output)
    a = a_model(x, training=False).numpy()
    b = b_model(x, training=False).numpy()
    d = d_model(x, training=False).numpy()
    return a, b, d


def write_csv(a: np.ndarray, b: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_pred = np.argmax(a, axis=1)
    b_pred = np.argmax(b, axis=1)
    d_pred = np.argmax(d, axis=1)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "frame_idx",
            "a_prob_0_ppg_good",
            "a_prob_1_ppg_bad",
            "a_prob_2_ppg_uncertain",
            "a_pred_id",
            "a_pred_label",
            "b_prob_0_ecg_good",
            "b_prob_1_ecg_bad",
            "b_prob_2_ecg_uncertain",
            "b_pred_id",
            "b_pred_label",
            "d_prob_0_final_ok",
            "d_prob_1_signal_bad_or_contact_bad",
            "d_prob_2_uncertain_or_motion",
            "d_pred_id",
            "d_pred_label",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(a.shape[0]):
            writer.writerow(
                {
                    "frame_idx": i,
                    "a_prob_0_ppg_good": float(a[i, 0]),
                    "a_prob_1_ppg_bad": float(a[i, 1]),
                    "a_prob_2_ppg_uncertain": float(a[i, 2]),
                    "a_pred_id": int(a_pred[i]),
                    "a_pred_label": A_LABELS[int(a_pred[i])],
                    "b_prob_0_ecg_good": float(b[i, 0]),
                    "b_prob_1_ecg_bad": float(b[i, 1]),
                    "b_prob_2_ecg_uncertain": float(b[i, 2]),
                    "b_pred_id": int(b_pred[i]),
                    "b_pred_label": B_LABELS[int(b_pred[i])],
                    "d_prob_0_final_ok": float(d[i, 0]),
                    "d_prob_1_signal_bad_or_contact_bad": float(d[i, 1]),
                    "d_prob_2_uncertain_or_motion": float(d[i, 2]),
                    "d_pred_id": int(d_pred[i]),
                    "d_pred_label": D_LABELS[int(d_pred[i])],
                }
            )
    return a_pred, b_pred, d_pred


def counts_for(pred: np.ndarray, n: int = 3) -> list[int]:
    c = Counter(int(v) for v in pred)
    return [c.get(i, 0) for i in range(n)]


def plot_chain(a: np.ndarray, b: np.ndarray, d: np.ndarray, a_pred: np.ndarray, b_pred: np.ndarray, d_pred: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), dpi=180, constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for ax, pred, labels, title in [
        (axes[0, 0], a_pred, A_LABELS, "Model A PPG output distribution"),
        (axes[0, 1], b_pred, B_LABELS, "Model B ECG output distribution"),
        (axes[1, 0], d_pred, D_LABELS, "Final D_SAFE output distribution"),
    ]:
        x = np.arange(3)
        values = counts_for(pred)
        ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_ylabel("frame count")
        ax.set_xticks(x, [f"{i}\n{label}" for i, label in enumerate(labels)])
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(values):
            ax.text(i, v + 0.15, str(v), ha="center", va="bottom", fontweight="bold")

    ax = axes[1, 1]
    frames = np.arange(d.shape[0])
    ax.plot(frames, np.max(a, axis=1), marker="o", label="A confidence", linewidth=1.5)
    ax.plot(frames, np.max(b, axis=1), marker="s", label="B confidence", linewidth=1.5)
    ax.plot(frames, np.max(d, axis=1), marker="^", label="D confidence", linewidth=1.5)
    ax.set_title("A/B/D max probability by frame")
    ax.set_xlabel("frame index")
    ax.set_ylabel("max probability")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("ABCD v0.2 Internal A/B/D Output Analysis", fontsize=14, fontweight="bold")
    fig.savefig(CHAIN_FIG, bbox_inches="tight")
    plt.close(fig)


def plot_joint(a_pred: np.ndarray, b_pred: np.ndarray, d_pred: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180, constrained_layout=True)
    for d_cls, ax in enumerate(axes):
        mat = np.zeros((3, 3), dtype=int)
        mask = d_pred == d_cls
        for a_cls, b_cls in zip(a_pred[mask], b_pred[mask]):
            mat[int(a_cls), int(b_cls)] += 1
        im = ax.imshow(mat, cmap="YlGnBu", vmin=0)
        ax.set_title(f"D={d_cls}\n{D_LABELS[d_cls]}")
        ax.set_xlabel("B ECG pred")
        ax.set_ylabel("A PPG pred")
        ax.set_xticks(range(3), [f"{i}\n{label}" for i, label in enumerate(B_LABELS)], fontsize=7)
        ax.set_yticks(range(3), [f"{i}\n{label}" for i, label in enumerate(A_LABELS)], fontsize=7)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("ABCD v0.2 Joint A/B Prediction Counts by Final D Output", fontsize=14, fontweight="bold")
    fig.savefig(JOINT_FIG, bbox_inches="tight")
    plt.close(fig)


def write_report(a: np.ndarray, b: np.ndarray, d: np.ndarray, a_pred: np.ndarray, b_pred: np.ndarray, d_pred: np.ndarray) -> None:
    a_counts = counts_for(a_pred)
    b_counts = counts_for(b_pred)
    d_counts = counts_for(d_pred)
    lines = [
        "# ABCD v0.2 Internal A/B/D Output Analysis",
        "",
        "## Clarification",
        "",
        "The GD32 benchmark confusion matrix is a final-output PC-vs-MCU consistency matrix, so it has only the final D_SAFE three labels. That does not mean the deployed model is D-only.",
        "",
        "The deployed no-slice model internally contains:",
        "",
        "- Model A PPG quality branch: `PPG_GOOD`, `PPG_BAD`, `PPG_UNCERTAIN`",
        "- Model B ECG quality branch: `ECG_GOOD`, `ECG_BAD`, `ECG_UNCERTAIN`",
        "- D_SAFE fusion output: `FINAL_OK`, `SIGNAL_BAD_OR_CONTACT_BAD`, `UNCERTAIN_OR_MOTION`",
        "",
        "The competition-facing TFLite output intentionally exposes only the final D_SAFE three-class decision. A and B are internal intermediate branches.",
        "",
        "## Generated Files",
        "",
        f"- Internal output CSV: `{OUT_CSV}`",
        f"- A/B/D output analysis figure: `{CHAIN_FIG}`",
        f"- A/B to D joint heatmaps: `{JOINT_FIG}`",
        "",
        "## Real NPZ Internal Prediction Distribution",
        "",
        "| Branch | Class 0 | Class 1 | Class 2 |",
        "|---|---:|---:|---:|",
        f"| A PPG | {a_counts[0]} PPG_GOOD | {a_counts[1]} PPG_BAD | {a_counts[2]} PPG_UNCERTAIN |",
        f"| B ECG | {b_counts[0]} ECG_GOOD | {b_counts[1]} ECG_BAD | {b_counts[2]} ECG_UNCERTAIN |",
        f"| D final | {d_counts[0]} FINAL_OK | {d_counts[1]} SIGNAL_BAD_OR_CONTACT_BAD | {d_counts[2]} UNCERTAIN_OR_MOTION |",
        "",
        "## Latency Interpretation",
        "",
        "The measured `0.09 ms` latency belongs to the full 107-dim integrated noslice model, not to a standalone 3-output D-only head. The graph includes A and B Dense branches plus D fusion. The visible output tensor is three-dimensional because the final decision contract is three classes.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    build = load_build_module()
    x = np.load(NPZ_PATH)["input"].astype(np.float32)
    model, _ = build.build_abd_model(use_slices=False)
    a, b, d = submodel_outputs(model, x)
    a_pred, b_pred, d_pred = write_csv(a, b, d)
    plot_chain(a, b, d, a_pred, b_pred, d_pred)
    plot_joint(a_pred, b_pred, d_pred)
    write_report(a, b, d, a_pred, b_pred, d_pred)
    print("ABCD_V0_2_INTERNAL_OUTPUT_ANALYSIS_DONE")
    print(f"report={REPORT_MD}")
    print(f"csv={OUT_CSV}")
    print(f"figure={CHAIN_FIG}")
    print(f"joint={JOINT_FIG}")


if __name__ == "__main__":
    main()
