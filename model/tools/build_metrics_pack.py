#!/usr/bin/env python3
"""Build ABCD v0.2 metrics and visualization package."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, f1_score


ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = ROOT.parents[0]
A_DIR = AI_ROOT / "model_a_ppg_trust_gate_v0_4_1"
B_DIR = AI_ROOT / "model_b_ecg_quality_v0_1"
C_DIR = AI_ROOT / "model_c_ecg_rhythm_binary_v0_4_compromise"
D_DIR = AI_ROOT / "model_d_fusion_decision_v0_2"
PACK = ROOT / "metrics_pack"
SUBDIRS = ["tables", "figures", "reports", "errors", "json", "ppt_assets"]

FORBIDDEN_OUTPUT_TOKENS = {
    "HEART_DISEASE",
    "AF_DIAGNOSIS",
    "PVC_DIAGNOSIS",
    "MYOCARDIAL_ISCHEMIA",
    "DISEASE_DETECTED",
}


def ensure_dirs() -> None:
    for sub in SUBDIRS:
        (PACK / sub).mkdir(parents=True, exist_ok=True)
    write_text(
        PACK / "reports" / "metrics_pack_init_report.md",
        "# Metrics Pack Init Report\n\n"
        f"- Root: `{PACK}`\n"
        "- Subdirectories: `tables`, `figures`, `reports`, `errors`, `json`, `ppt_assets`.\n"
        "- Scope: evaluation and visualization package only. No retraining, no export, no firmware changes.\n",
    )


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = ["arialbd.ttf" if bold else "arial.ttf", "C:/Windows/Fonts/arial.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, max_width: int, ft: ImageFont.ImageFont) -> List[str]:
    words = str(text).split()
    lines: List[str] = []
    current = ""
    for word in words:
        trial = (current + " " + word).strip()
        if draw.textbbox((0, 0), trial, font=ft)[2] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def draw_confusion_matrix(path: Path, labels: Sequence[str], matrix: Sequence[Sequence[int]], title: str) -> None:
    n = max(1, len(labels))
    cell = 86 if n <= 4 else 68
    left = 280
    top = 120
    width = left + n * cell + 80
    height = top + n * cell + 250
    img = Image.new("RGB", (width, height), "#f8fafc")
    d = ImageDraw.Draw(img)
    title_font = font(30, True)
    small = font(18)
    tiny = font(15)
    d.text((40, 35), title, fill="#111827", font=title_font)
    max_val = max([max(row) if row else 0 for row in matrix] + [1])
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            intensity = int(245 - 175 * (value / max_val))
            fill = (intensity, min(245, intensity + 25), 255)
            x = left + j * cell
            y = top + i * cell
            d.rectangle([x, y, x + cell, y + cell], fill=fill, outline="#334155")
            d.text((x + cell // 2 - 12, y + cell // 2 - 10), str(value), fill="#0f172a", font=small)
    for j, label in enumerate(labels):
        x = left + j * cell + 5
        d.text((x, top - 42), str(j + 1), fill="#111827", font=small)
        for k, line in enumerate(wrap(d, f"{j+1}. {label}", cell + 90, tiny)[:3]):
            d.text((left + j * cell - 4, top + n * cell + 14 + k * 18), line, fill="#334155", font=tiny)
    for i, label in enumerate(labels):
        y = top + i * cell + cell // 2 - 10
        d.text((20, y), f"{i+1}.", fill="#111827", font=small)
        for k, line in enumerate(wrap(d, label, 220, tiny)[:2]):
            d.text((55, y + k * 18), line, fill="#334155", font=tiny)
    d.text((left, top - 78), "Predicted", fill="#475569", font=small)
    d.text((40, top - 42), "True", fill="#475569", font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def draw_bar_chart(path: Path, counts: Dict[str, int], title: str) -> None:
    labels = list(counts.keys())
    values = [int(counts[k]) for k in labels]
    width = 1300
    height = max(560, 150 + len(labels) * 48)
    img = Image.new("RGB", (width, height), "#f8fafc")
    d = ImageDraw.Draw(img)
    title_font = font(32, True)
    small = font(20)
    tiny = font(16)
    d.text((45, 35), title, fill="#111827", font=title_font)
    max_val = max(values + [1])
    x0, y0 = 360, 115
    bar_h = 30
    for i, (label, value) in enumerate(zip(labels, values)):
        y = y0 + i * 48
        d.text((35, y), label, fill="#334155", font=tiny)
        w = int((width - x0 - 120) * value / max_val)
        d.rectangle([x0, y, x0 + w, y + bar_h], fill="#60a5fa")
        d.text((x0 + w + 12, y + 3), str(value), fill="#111827", font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def draw_pass_fail(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    width, height = 1350, 720
    img = Image.new("RGB", (width, height), "#f8fafc")
    d = ImageDraw.Draw(img)
    title_font = font(34, True)
    body = font(22)
    small = font(18)
    d.text((40, 34), "ABCD v0.2 Smoke Case Pass/Fail", fill="#111827", font=title_font)
    x0, y0 = 60, 115
    for i, row in enumerate(rows):
        x = x0 + (i % 2) * 630
        y = y0 + (i // 2) * 105
        passed = str(row["pass_fail"]).upper() == "PASS"
        fill = "#dcfce7" if passed else "#fee2e2"
        outline = "#16a34a" if passed else "#dc2626"
        d.rectangle([x, y, x + 590, y + 78], fill=fill, outline=outline, width=3)
        d.text((x + 18, y + 12), row["case_name"], fill="#111827", font=body)
        d.text((x + 18, y + 44), f"{row['actual_final_action']} / {row['pass_fail']}", fill="#334155", font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def draw_threshold_tradeoff(path: Path, summary: Dict[str, Any]) -> None:
    metrics = {
        "Rhythm suspect recall": summary.get("RHYTHM_SUSPECT_RECALL", 0),
        "Suspect as normal rate": summary.get("SUSPECT_AS_NORMAL_RATE", 0),
        "Normal as suspect rate": summary.get("NORMAL_AS_SUSPECT_RATE", 0),
        "Other/uncertain rate": summary.get("OTHER_OR_UNCERTAIN_RATE", 0),
    }
    width, height = 1200, 620
    img = Image.new("RGB", (width, height), "#f8fafc")
    d = ImageDraw.Draw(img)
    d.text((45, 35), "Model C Threshold-Calibrated Tradeoff", fill="#111827", font=font(34, True))
    d.text((45, 82), f"normal_threshold={summary.get('BEST_NORMAL_THRESHOLD', 0.42)}   suspect_threshold={summary.get('BEST_SUSPECT_THRESHOLD', 0.60)}", fill="#475569", font=font(20))
    x0, y0 = 390, 155
    for i, (label, value) in enumerate(metrics.items()):
        y = y0 + i * 85
        d.text((50, y), label, fill="#334155", font=font(22))
        d.rectangle([x0, y, x0 + 650, y + 34], outline="#94a3b8", width=2)
        d.rectangle([x0, y, x0 + int(650 * float(value)), y + 34], fill="#f59e0b")
        d.text((x0 + 670, y + 2), f"{float(value):.3f}", fill="#111827", font=font(22, True))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def metrics_from_conf(labels: Sequence[str], matrix: Sequence[Sequence[int]]) -> Dict[str, Any]:
    y_true: List[str] = []
    y_pred: List[str] = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            y_true.extend([labels[i]] * int(value))
            y_pred.extend([labels[j]] * int(value))
    if not y_true:
        return {"available": False}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(labels), zero_division=0)
    return {
        "available": True,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(sum(p) / len(p)),
        "macro_recall": float(sum(r) / len(r)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "per_class": {
            label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, label in enumerate(labels)
        },
        "confusion_matrix": [list(map(int, row)) for row in matrix],
        "confusion_matrix_labels": list(labels),
        "label_distribution": {label: int(sum(matrix[i])) for i, label in enumerate(labels)},
    }


def write_metrics_outputs(prefix: str, metrics: Dict[str, Any]) -> None:
    rows = []
    if metrics.get("available"):
        rows.append(
            {
                "scope": "overall",
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "support": sum(v["support"] for v in metrics["per_class"].values()),
            }
        )
        for label, row in metrics["per_class"].items():
            rows.append({"scope": label, **row})
    else:
        rows.append({"scope": "METRICS_NOT_AVAILABLE", "reason": metrics.get("reason", "")})
    fields = sorted({k for row in rows for k in row.keys()})
    write_csv(PACK / "tables" / f"{prefix}_metrics.csv", rows, fields)
    write_json(PACK / "json" / f"{prefix}_metrics.json", metrics)


def aggregate_error_cases(path: Path, labels: Sequence[str], matrix: Sequence[Sequence[int]]) -> None:
    rows = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            count = int(matrix[i][j])
            if i != j and count:
                rows.append({"true_label": true_label, "pred_label": pred_label, "count": count})
    write_csv(path, rows, ["true_label", "pred_label", "count"])


def metrics_from_prediction_table(path: Path, true_col: str, pred_col: str, labels: Sequence[str]) -> Dict[str, Any]:
    df = pd.read_csv(path)
    matrix = confusion_matrix(df[true_col], df[pred_col], labels=list(labels)).tolist()
    metrics = metrics_from_conf(labels, matrix)
    metrics["source_type"] = "STANDALONE_VALIDATION_PREDICTION_TABLE"
    metrics["source"] = str(path)
    return metrics


def model_a() -> Dict[str, Any]:
    labels = ["GOOD", "BAD", "UNCERTAIN"]
    pred_path = A_DIR / "reports" / "model_a_v0_4_1_fixed_val_predictions.csv"
    metrics = metrics_from_prediction_table(pred_path, "true_label", "pred_label", labels)
    metrics["source_report"] = str(A_DIR / "reports" / "model_a_v0_4_1_training_report.md")
    metrics["source_metrics_json"] = str(A_DIR / "reports" / "model_a_v0_4_1_metrics.json")
    matrix = metrics["confusion_matrix"]
    write_metrics_outputs("model_a", metrics)
    if metrics.get("available"):
        draw_confusion_matrix(PACK / "figures" / "model_a_confusion_matrix.png", labels, matrix, "Model A Confusion Matrix")
        draw_bar_chart(PACK / "figures" / "model_a_label_distribution.png", metrics["label_distribution"], "Model A Label Distribution")
        aggregate_error_cases(PACK / "errors" / "model_a_error_cases.csv", labels, matrix)
    write_text(PACK / "reports" / "model_a_metrics_report.md", report_for_model("Model A", "PPG quality gate", metrics, "Quality gate only; no clinical conclusion."))
    return metrics


def model_b() -> Dict[str, Any]:
    raw = read_json(B_DIR / "reports" / "model_b_metrics.json")
    labels = ["ECG_GOOD", "ECG_BAD", "ECG_UNCERTAIN"]
    test = raw.get("metrics_by_split", {}).get("test", {})
    matrix = test.get("confusion_matrix", [])
    metrics = metrics_from_conf(labels, matrix)
    metrics["source"] = str(B_DIR / "reports" / "model_b_metrics.json")
    write_metrics_outputs("model_b", metrics)
    if metrics.get("available"):
        draw_confusion_matrix(PACK / "figures" / "model_b_confusion_matrix.png", labels, matrix, "Model B Confusion Matrix")
        draw_bar_chart(PACK / "figures" / "model_b_label_distribution.png", metrics["label_distribution"], "Model B Label Distribution")
        aggregate_error_cases(PACK / "errors" / "model_b_error_cases.csv", labels, matrix)
    write_text(PACK / "reports" / "model_b_metrics_report.md", report_for_model("Model B", "ECG quality gate", metrics, "Quality gate only; no clinical conclusion."))
    return metrics


def model_c() -> Dict[str, Any]:
    pred_path = C_DIR / "reports" / "model_c_after_s05_s06_prediction_table.csv"
    summary = read_json(C_DIR / "reports" / "model_c_after_s05_s06_threshold_calibrated_summary.json")
    df = pd.read_csv(pred_path)
    raw_labels = ["NORMAL", "RHYTHM_SUSPECT"]
    raw_matrix = confusion_matrix(df["true_label"], df["pred_label"], labels=raw_labels).tolist()
    raw_metrics = metrics_from_conf(raw_labels, raw_matrix)
    normal_t = float(summary.get("BEST_NORMAL_THRESHOLD", 0.42))
    suspect_t = float(summary.get("BEST_SUSPECT_THRESHOLD", 0.60))
    final_pred = []
    for p in df["p_rhythm_suspect"]:
        if p <= normal_t:
            final_pred.append("NORMAL")
        elif p >= suspect_t:
            final_pred.append("RHYTHM_SUSPECT")
        else:
            final_pred.append("OTHER_OR_UNCERTAIN")
    final_labels = ["NORMAL", "RHYTHM_SUSPECT", "OTHER_OR_UNCERTAIN"]
    final_matrix = confusion_matrix(df["true_label"], final_pred, labels=final_labels).tolist()
    final_metrics = metrics_from_conf(final_labels, final_matrix)
    final_metrics["RHYTHM_SUSPECT_RECALL"] = summary.get("RHYTHM_SUSPECT_RECALL")
    final_metrics["SUSPECT_AS_NORMAL_RATE"] = summary.get("SUSPECT_AS_NORMAL_RATE")
    final_metrics["NORMAL_AS_SUSPECT_RATE"] = summary.get("NORMAL_AS_SUSPECT_RATE")
    final_metrics["OTHER_OR_UNCERTAIN_RATE"] = summary.get("OTHER_OR_UNCERTAIN_RATE")
    final_metrics["CROSS_SUBJECT_READY"] = False
    final_metrics["EXPORT_TFLITE_READY"] = False
    final_metrics["GD32_AI_READY"] = False
    write_metrics_outputs("model_c_binary", raw_metrics)
    write_metrics_outputs("model_c_threshold_calibrated", final_metrics)
    combined = {"available": True, "binary": raw_metrics, "threshold_calibrated": final_metrics, "threshold_summary": summary}
    write_json(PACK / "json" / "model_c_metrics.json", combined)
    draw_confusion_matrix(PACK / "figures" / "model_c_binary_confusion_matrix.png", raw_labels, raw_matrix, "Model C Binary Confusion Matrix")
    draw_confusion_matrix(PACK / "figures" / "model_c_three_state_confusion_matrix.png", final_labels, final_matrix, "Model C Three-State Confusion Matrix")
    draw_threshold_tradeoff(PACK / "figures" / "model_c_threshold_tradeoff.png", summary)
    draw_bar_chart(PACK / "figures" / "model_c_label_distribution.png", Counter(final_pred), "Model C Output Distribution")
    df_with_final = df.copy()
    df_with_final["final_pred"] = final_pred
    err = df_with_final[df_with_final["true_label"] != df_with_final["final_pred"]]
    err.to_csv(PACK / "errors" / "model_c_error_cases.csv", index=False, encoding="utf-8")
    write_text(PACK / "reports" / "model_c_metrics_report.md", model_c_report(combined))
    return combined


def model_d() -> Dict[str, Any]:
    raw = read_json(D_DIR / "reports" / "model_d_v0_2_metrics.json")
    best = raw.get("best_model", "LogisticRegression")
    test = raw.get("candidates", {}).get(best, {}).get("test", {})
    metrics = {
        "available": bool(test),
        "best_model": best,
        "accuracy": test.get("accuracy"),
        "macro_precision": sum(v["precision"] for v in test.get("per_class", {}).values()) / max(len(test.get("per_class", {})), 1),
        "macro_recall": sum(v["recall"] for v in test.get("per_class", {}).values()) / max(len(test.get("per_class", {})), 1),
        "macro_f1": test.get("macro_f1"),
        "per_class": test.get("per_class", {}),
        "confusion_matrix": test.get("confusion_matrix", []),
        "confusion_matrix_labels": test.get("confusion_matrix_labels", []),
        "high_risk_error_count": test.get("high_risk_error_count"),
        "rhythm_suspect_retest_recall": test.get("rhythm_suspect_retest_recall"),
        "contact_bad_retest_recall": test.get("contact_bad_retest_recall"),
        "label_distribution": raw.get("split_label_distribution", {}).get("test", {}),
    }
    write_metrics_outputs("model_d", metrics)
    write_json(PACK / "json" / "model_d_metrics.json", metrics)
    labels = metrics["confusion_matrix_labels"]
    matrix = metrics["confusion_matrix"]
    draw_confusion_matrix(PACK / "figures" / "model_d_confusion_matrix.png", labels, matrix, "Model D v0.2 Confusion Matrix")
    draw_bar_chart(PACK / "figures" / "model_d_label_distribution.png", metrics["label_distribution"], "Model D Test Label Distribution")
    aggregate_error_cases(PACK / "errors" / "model_d_error_cases.csv", labels, matrix)
    write_text(PACK / "reports" / "model_d_metrics_report.md", report_for_model("Model D v0.2", "Fusion decision", metrics, "PC fusion candidate with rule/proxy label risk; not firmware-ready."))
    return metrics


def report_for_model(name: str, task: str, metrics: Dict[str, Any], note: str) -> str:
    if not metrics.get("available"):
        return f"# {name} Metrics Report\n\n- Task: {task}\n- Status: METRICS_NOT_AVAILABLE\n- Reason: {metrics.get('reason', 'missing usable source')}\n"
    lines = [
        f"# {name} Metrics Report",
        "",
        f"- Task: {task}",
    ]
    if metrics.get("source_type"):
        lines.append(f"- Source type: {metrics.get('source_type')}")
    if metrics.get("source"):
        lines.append(f"- Source: {metrics.get('source')}")
    if metrics.get("source_report"):
        lines.append(f"- Source report: {metrics.get('source_report')}")
    if metrics.get("source_metrics_json"):
        lines.append(f"- Source metrics JSON: {metrics.get('source_metrics_json')}")
    lines.extend([
        f"- Accuracy: {metrics.get('accuracy'):.4f}",
        f"- Macro precision: {metrics.get('macro_precision'):.4f}",
        f"- Macro recall: {metrics.get('macro_recall'):.4f}",
        f"- Macro F1: {metrics.get('macro_f1'):.4f}",
        f"- Note: {note}",
        "",
        "## Per-Class Metrics",
        "",
    ])
    for label, row in metrics.get("per_class", {}).items():
        lines.append(f"- {label}: precision={row['precision']:.4f}, recall={row['recall']:.4f}, f1={row['f1']:.4f}, support={row['support']}")
    return "\n".join(lines) + "\n"


def model_c_report(metrics: Dict[str, Any]) -> str:
    final = metrics["threshold_calibrated"]
    return f"""# Model C Metrics Report

- Task: ECG rhythm-risk hint candidate.
- Safety boundary: risk hint only; no clinical conclusion.
- CROSS_SUBJECT_READY=False
- EXPORT_TFLITE_READY=False
- GD32_AI_READY=False

## Threshold-Calibrated Metrics

- Accuracy: {final.get('accuracy'):.4f}
- Macro F1: {final.get('macro_f1'):.4f}
- RHYTHM_SUSPECT_RECALL: {final.get('RHYTHM_SUSPECT_RECALL')}
- SUSPECT_AS_NORMAL_RATE: {final.get('SUSPECT_AS_NORMAL_RATE')}
- NORMAL_AS_SUSPECT_RATE: {final.get('NORMAL_AS_SUSPECT_RATE')}
- OTHER_OR_UNCERTAIN_RATE: {final.get('OTHER_OR_UNCERTAIN_RATE')}
"""


def source_inventory() -> None:
    configs = {
        "Model A": A_DIR,
        "Model B": B_DIR,
        "Model C": C_DIR,
        "Model D": D_DIR,
        "ABCD v0.2": ROOT,
    }
    rows = []
    for name, path in configs.items():
        files = list(path.rglob("*"))
        reports = [p for p in files if p.suffix.lower() in {".md", ".json"} and "report" in p.name.lower() or p.name.lower().endswith("metrics.json")]
        pred = [p for p in files if "prediction" in p.name.lower() and p.suffix.lower() == ".csv"]
        test = [p for p in files if p.name.lower() == "test.csv"]
        label_map = [p for p in files if "label" in p.name.lower() and p.suffix.lower() == ".json"]
        rows.append(
            {
                "model_name": name,
                "found_reports": len(reports),
                "found_prediction_table": bool(pred),
                "found_test_split": bool(test),
                "found_label_map": bool(label_map),
                "can_recompute_metrics": name in {"Model C"} or bool((path / "reports").exists()),
                "missing_items": "prediction-level table for some models" if name in {"Model A", "Model B", "Model D"} else "",
            }
        )
    write_csv(PACK / "tables" / "available_metrics_source_inventory.csv", rows, list(rows[0].keys()))
    lines = ["# Available Metrics Source Inventory", "", "| Model | Reports | Prediction Table | Test Split | Label Map | Can Recompute | Missing Items |", "| --- | ---: | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['model_name']} | {r['found_reports']} | {r['found_prediction_table']} | {r['found_test_split']} | {r['found_label_map']} | {r['can_recompute_metrics']} | {r['missing_items']} |")
    write_text(PACK / "reports" / "available_metrics_source_inventory.md", "\n".join(lines) + "\n")


def abcd_smoke() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = ROOT / "outputs" / "abcd_v0_2_smoke_test_outputs_fixed.csv"
    rows = pd.read_csv(path).to_dict("records")
    expected = {
        "all_normal": "MEASURE_OK",
        "ppg_bad_ecg_good": "FINAL_UNCERTAIN",
        "ecg_bad_ppg_good": "RETEST_CONTACT",
        "rhythm_suspect": "RHYTHM_RISK_RETEST",
        "motion_degraded": "KEEP_STILL",
        "contact_bad": "RETEST_CONTACT",
        "recovery_wait": "RETEST_AFTER_RECOVERY",
        "sensor_conflict": "SENSOR_CONFLICT_RETEST",
        "measure_failed": "MEASURE_FAILED",
        "all_uncertain": "FINAL_UNCERTAIN",
    }
    out = []
    for row in rows:
        case = row["window_id"]
        exp = expected.get(case, "")
        actual = row["final_action"]
        out.append({"case_name": case, "expected_final_action": exp, "actual_final_action": actual, "pass_fail": "PASS" if exp == actual else "FAIL", "final_reason": row.get("final_reason", "")})
    write_csv(PACK / "tables" / "abcd_v0_2_smoke_case_results.csv", out, list(out[0].keys()))
    draw_pass_fail(PACK / "figures" / "abcd_v0_2_smoke_case_pass_fail.png", out)
    pass_rate = sum(1 for r in out if r["pass_fail"] == "PASS") / max(len(out), 1)
    metrics = {"available": True, "pass_rate": pass_rate, "passed": pass_rate == 1.0, "case_count": len(out)}
    write_text(PACK / "reports" / "abcd_v0_2_smoke_metrics_report.md", f"# ABCD v0.2 Smoke Metrics Report\n\n- Case count: {len(out)}\n- Pass rate: {pass_rate:.3f}\n- Source: `{path}`\n")
    return out, metrics


def summary_table(a: Dict[str, Any], b: Dict[str, Any], c: Dict[str, Any], d: Dict[str, Any], smoke: Dict[str, Any]) -> List[Dict[str, Any]]:
    c_final = c["threshold_calibrated"]
    a_bad_recall = a.get("per_class", {}).get("BAD", a.get("per_class", {}).get("PPG_BAD", {})).get("recall")
    rows = [
        {"model": "Model A", "task": "PPG quality gate", "labels": "GOOD/BAD/UNCERTAIN", "accuracy": a.get("accuracy"), "macro_f1": a.get("macro_f1"), "key_recall": a_bad_recall, "key_error_rate": "bad_as_good=0 in source", "status": "candidate", "deploy_ready": "PC reference", "main_limitation": "small validation set"},
        {"model": "Model B", "task": "ECG quality gate", "labels": "GOOD/BAD/UNCERTAIN", "accuracy": b.get("accuracy"), "macro_f1": b.get("macro_f1"), "key_recall": b.get("per_class", {}).get("ECG_BAD", {}).get("recall"), "key_error_rate": "bad_as_good=0 in source", "status": "candidate", "deploy_ready": "PC reference", "main_limitation": "selected RF is PC-only"},
        {"model": "Model C", "task": "ECG rhythm risk hint", "labels": "NORMAL/RHYTHM_SUSPECT/OTHER", "accuracy": c_final.get("accuracy"), "macro_f1": c_final.get("macro_f1"), "key_recall": c_final.get("RHYTHM_SUSPECT_RECALL"), "key_error_rate": c_final.get("SUSPECT_AS_NORMAL_RATE"), "status": "smoke usable, not deployable", "deploy_ready": "False", "main_limitation": "cross-subject not ready"},
        {"model": "Model D", "task": "Fusion decision", "labels": "10 fusion classes", "accuracy": d.get("accuracy"), "macro_f1": d.get("macro_f1"), "key_recall": d.get("rhythm_suspect_retest_recall"), "key_error_rate": f"high-risk errors={d.get('high_risk_error_count')}", "status": "PC candidate", "deploy_ready": "False", "main_limitation": "rule/proxy label risk"},
        {"model": "ABCD", "task": "PC integration", "labels": "final_action", "accuracy": smoke.get("pass_rate"), "macro_f1": "", "key_recall": "", "key_error_rate": "", "status": "PC pipeline ready", "deploy_ready": "False for firmware", "main_limitation": "not firmware-ready"},
    ]
    write_csv(PACK / "tables" / "abcd_model_metrics_summary.csv", rows, list(rows[0].keys()))
    write_json(PACK / "json" / "abcd_model_metrics_summary.json", rows)
    return rows


def dashboard(path: Path, summary_rows: List[Dict[str, Any]], c: Dict[str, Any], d: Dict[str, Any], smoke: Dict[str, Any]) -> None:
    img = Image.new("RGB", (1800, 1050), "#f8fafc")
    dr = ImageDraw.Draw(img)
    dr.text((60, 40), "ABCD v0.2 Metrics Dashboard", fill="#111827", font=font(44, True))
    blocks = [
        ("Model A", "PPG quality", f"Acc {float(summary_rows[0]['accuracy']):.3f}\nMacro F1 {float(summary_rows[0]['macro_f1']):.3f}", "#d8f3dc"),
        ("Model B", "ECG quality", f"Acc {float(summary_rows[1]['accuracy']):.3f}\nMacro F1 {float(summary_rows[1]['macro_f1']):.3f}", "#dbeafe"),
        ("Model C", "Risk hint only", f"Recall {c['threshold_calibrated']['RHYTHM_SUSPECT_RECALL']:.3f}\nSuspect->Normal {c['threshold_calibrated']['SUSPECT_AS_NORMAL_RATE']:.3f}", "#fef3c7"),
        ("Model D", "Fusion decision", f"Macro F1 {d['macro_f1']:.3f}\nHigh-risk errors {d['high_risk_error_count']}", "#fee2e2"),
    ]
    for i, (name, task, text, fill) in enumerate(blocks):
        x = 70 + (i % 2) * 850
        y = 140 + (i // 2) * 280
        dr.rectangle([x, y, x + 760, y + 220], fill=fill, outline="#334155", width=3)
        dr.text((x + 30, y + 25), name, fill="#111827", font=font(34, True))
        dr.text((x + 30, y + 72), task, fill="#475569", font=font(24))
        for k, line in enumerate(text.split("\n")):
            dr.text((x + 30, y + 120 + k * 36), line, fill="#111827", font=font(28, True))
    dr.rectangle([70, 720, 1640, 920], fill="#ffffff", outline="#334155", width=3)
    dr.text((100, 750), f"ABCD smoke pass rate: {smoke['pass_rate']:.3f}", fill="#111827", font=font(34, True))
    dr.text((100, 805), "Deployment gate: PC_READY=True    Firmware=False    TFLite=False    GD32=False", fill="#334155", font=font(26))
    dr.text((100, 855), "Safety note: no clinical conclusion output; Model C is risk hint only.", fill="#991b1b", font=font(26, True))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def ppt_assets() -> None:
    required = [
        "model_a_confusion_matrix.png",
        "model_b_confusion_matrix.png",
        "model_c_three_state_confusion_matrix.png",
        "model_d_confusion_matrix.png",
        "abcd_v0_2_smoke_case_pass_fail.png",
        "abcd_v0_2_metrics_dashboard.png",
        "model_a_label_distribution.png",
        "model_b_label_distribution.png",
        "model_c_label_distribution.png",
        "model_d_label_distribution.png",
    ]
    for name in required:
        src = PACK / "figures" / name
        if src.exists():
            shutil.copy2(src, PACK / "ppt_assets" / name)
    lines = ["# PPT Figure Caption Suggestions", ""]
    for name in required:
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Chinese caption: `{name}` 对应的 ABCD v0.2 评估可视化。",
                f"- English caption: ABCD v0.2 evaluation visualization for `{name}`.",
                "- Safe wording note: Describe as signal quality, rhythm-risk hint, or fusion decision; avoid clinical conclusion wording.",
                "",
            ]
        )
    write_text(PACK / "reports" / "ppt_figure_caption_suggestions.md", "\n".join(lines))


def final_report(a: Dict[str, Any], b: Dict[str, Any], c: Dict[str, Any], d: Dict[str, Any], smoke: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "model_a_metrics_available": bool(a.get("available")),
        "model_b_metrics_available": bool(b.get("available")),
        "model_c_metrics_available": bool(c.get("available")),
        "model_d_metrics_available": bool(d.get("available")),
        "abcd_smoke_metrics_available": bool(smoke.get("available")),
        "model_c_rhythm_suspect_recall": c["threshold_calibrated"].get("RHYTHM_SUSPECT_RECALL"),
        "model_c_suspect_as_normal_rate": c["threshold_calibrated"].get("SUSPECT_AS_NORMAL_RATE"),
        "model_c_normal_as_suspect_rate": c["threshold_calibrated"].get("NORMAL_AS_SUSPECT_RATE"),
        "abcd_pc_pipeline_ready": smoke.get("passed"),
        "diagnosis_output_absent": True,
        "firmware_ready": False,
        "tflite_all_ready": False,
        "gd32_ready": False,
    }
    write_json(PACK / "json" / "abcd_v0_2_final_metrics_summary.json", summary)
    report = f"""# ABCD v0.2 Final Metrics Report

## 1. Overview

ABCD v0.2 is a PC-side integrated candidate only. No retraining, export, or firmware changes were performed.

## 2. Model A Metrics

- Source type: {a.get('source_type')}
- Source: {a.get('source')}
- Accuracy: {a.get('accuracy'):.4f}
- Macro F1: {a.get('macro_f1'):.4f}
- Confusion matrix: {a.get('confusion_matrix')}

## 3. Model B Metrics

- Accuracy: {b.get('accuracy'):.4f}
- Macro F1: {b.get('macro_f1'):.4f}

## 4. Model C Metrics

- Rhythm suspect recall: {summary['model_c_rhythm_suspect_recall']}
- Suspect-as-normal rate: {summary['model_c_suspect_as_normal_rate']}
- Normal-as-suspect rate: {summary['model_c_normal_as_suspect_rate']}
- Safety boundary: risk hint candidate only; CROSS_SUBJECT_READY=False.

## 5. Model D Metrics

- Accuracy: {d.get('accuracy'):.4f}
- Macro F1: {d.get('macro_f1'):.4f}
- High-risk error count: {d.get('high_risk_error_count')}

## 6. ABCD Smoke Test Metrics

- Smoke pass rate: {smoke.get('pass_rate'):.4f}

## 7. Final Dashboard

Dashboard figure: `metrics_pack/figures/abcd_v0_2_metrics_dashboard.png`.

## 8. Deployment Gate

PC_ABCD_PIPELINE_READY = {summary['abcd_pc_pipeline_ready']}
FIRMWARE_READY = False
TFLITE_ALL_READY = False
GD32_READY = False

## 9. Source Audit

Model A now uses the standalone v0.4.1 validation prediction table, so its metrics are recomputed from predictions rather than reconstructed from the older v0.4 summary JSON. Model B and Model D remain reconstructed from stored metrics JSON/confusion matrices. Model C prediction-level table was already available.

## 10. Safe Conclusion

ABCD v0.2 is a PC-side integrated candidate and is not firmware-ready. The Model A source selection changes the metrics values and visuals, but it does not change the deployment conclusion.
"""
    write_text(PACK / "reports" / "abcd_v0_2_final_metrics_report.md", report)
    return summary


def integrity(summary: Dict[str, Any]) -> Dict[str, bool]:
    checks = {
        "metrics_csv_files_exist": all((PACK / "tables" / f).exists() for f in ["model_a_metrics.csv", "model_b_metrics.csv", "model_c_threshold_calibrated_metrics.csv", "model_d_metrics.csv", "abcd_model_metrics_summary.csv"]),
        "metrics_json_files_exist": all((PACK / "json" / f).exists() for f in ["model_a_metrics.json", "model_b_metrics.json", "model_c_metrics.json", "model_d_metrics.json", "abcd_v0_2_final_metrics_summary.json"]),
        "confusion_matrix_figures_exist": all((PACK / "figures" / f).exists() for f in ["model_a_confusion_matrix.png", "model_b_confusion_matrix.png", "model_c_three_state_confusion_matrix.png", "model_d_confusion_matrix.png"]),
        "label_distribution_figures_exist": all((PACK / "figures" / f).exists() for f in ["model_a_label_distribution.png", "model_b_label_distribution.png", "model_c_label_distribution.png", "model_d_label_distribution.png"]),
        "dashboard_figure_exists": (PACK / "figures" / "abcd_v0_2_metrics_dashboard.png").exists(),
        "final_metrics_report_exists": (PACK / "reports" / "abcd_v0_2_final_metrics_report.md").exists(),
        "no_forbidden_output_tokens_in_smoke": not any(t in (ROOT / "outputs" / "abcd_v0_2_smoke_test_outputs_fixed.csv").read_text(encoding="utf-8", errors="ignore").upper() for t in FORBIDDEN_OUTPUT_TOKENS),
        "firmware_ready_false": summary["firmware_ready"] is False,
        "tflite_all_ready_false": summary["tflite_all_ready"] is False,
        "gd32_ready_false": summary["gd32_ready"] is False,
    }
    lines = ["# Metrics Pack Integrity Check", ""]
    for key, value in checks.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append(f"Overall pass: {all(checks.values())}")
    write_text(PACK / "reports" / "metrics_pack_integrity_check.md", "\n".join(lines) + "\n")
    return checks


def main() -> int:
    ensure_dirs()
    source_inventory()
    a = model_a()
    b = model_b()
    c = model_c()
    d = model_d()
    smoke_rows, smoke = abcd_smoke()
    rows = summary_table(a, b, c, d, smoke)
    dashboard(PACK / "figures" / "abcd_v0_2_metrics_dashboard.png", rows, c, d, smoke)
    ppt_assets()
    final_summary = final_report(a, b, c, d, smoke)
    checks = integrity(final_summary)
    print(f"MODEL_A_METRICS_READY = {a.get('available')}")
    print(f"MODEL_B_METRICS_READY = {b.get('available')}")
    print(f"MODEL_C_METRICS_READY = {c.get('available')}")
    print(f"MODEL_D_METRICS_READY = {d.get('available')}")
    print(f"ABCD_SMOKE_METRICS_READY = {smoke.get('available')}")
    print(f"CONFUSION_MATRICES_READY = {checks['confusion_matrix_figures_exist']}")
    print(f"DASHBOARD_READY = {checks['dashboard_figure_exists']}")
    print(f"FINAL_METRICS_REPORT = {PACK / 'reports' / 'abcd_v0_2_final_metrics_report.md'}")
    print(f"PPT_ASSETS_DIR = {PACK / 'ppt_assets'}")
    print(f"PC_ABCD_PIPELINE_READY = {final_summary['abcd_pc_pipeline_ready']}")
    print("FIRMWARE_READY = False")
    print("TFLITE_ALL_READY = False")
    print("GD32_READY = False")
    print("NEXT_ACTION = Use metrics_pack figures/tables for paper, PPT, and demo; collect richer prediction-level test tables before making broader claims.")
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
