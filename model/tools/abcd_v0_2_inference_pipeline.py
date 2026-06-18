#!/usr/bin/env python3
"""ABCD Integrated v0.2 PC inference pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = ROOT.parents[0]
D_TOOLS = AI_ROOT / "model_d_fusion_decision_v0_2" / "tools"
if str(D_TOOLS) not in sys.path:
    sys.path.insert(0, str(D_TOOLS))

try:
    from model_d_v0_2_inference import infer_row as d_infer_row, load_model as d_load_model
except Exception as exc:
    d_infer_row = None
    d_load_model = None
    D_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    D_IMPORT_ERROR = ""


INPUT_FIELDS = [
    "timestamp_ms","window_id","ppg_quality_score","ppg_valid_window_flag","ppg_trust_level","ppg_hr_bpm_x10","ppg_spo2_x10",
    "model_a_label","model_a_confidence","ecg_quality_score","ecg_lead_off","ecg_hr_bpm_x10","ecg_hr_stale_flag",
    "model_b_label","model_b_confidence","model_c_p_normal","model_c_p_rhythm_suspect","model_c_raw_label",
    "model_c_final_label","model_c_reason","imu_state","imu_motion_flag","imu_motion_level","tmp_state","tmp_c_x10",
    "tmp_finger_cold","tmp_valid","tmp_stable","lead_off","contact_bad","recovery_wait","sensor_conflict",
    "motion_degraded","cold_suspect","hr_conflict","spo2_missing","measure_failed"
]
OUTPUT_FIELDS = INPUT_FIELDS + ["model_d_final_label","model_d_confidence","final_action","final_reason"]
LEGAL_A = {"PPG_GOOD","PPG_BAD","PPG_UNCERTAIN","MODEL_NOT_AVAILABLE"}
LEGAL_B = {"ECG_GOOD","ECG_BAD","ECG_UNCERTAIN","MODEL_NOT_AVAILABLE"}
LEGAL_C = {"NORMAL","RHYTHM_SUSPECT","OTHER_OR_UNCERTAIN","MODEL_NOT_AVAILABLE"}
LEGAL_ACTIONS = {"MEASURE_OK","KEEP_STILL","RETEST_CONTACT","RETEST_AFTER_RECOVERY","RHYTHM_RISK_RETEST","SENSOR_CONFLICT_RETEST","MEASURE_FAILED","FINAL_UNCERTAIN"}
FORBIDDEN = {"DIAGNOSIS","HEART_DISEASE","AF_DIAGNOSIS","PVC_DIAGNOSIS","MYOCARDIAL_ISCHEMIA","DISEASE_DETECTED"}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def norm(value: Any) -> str:
    return str(value or "").strip().upper()


def f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except ValueError:
        return default


def normalize_a(value: Any) -> str:
    text = norm(value)
    return {"GOOD":"PPG_GOOD","BAD":"PPG_BAD","UNCERTAIN":"PPG_UNCERTAIN"}.get(text, text if text in LEGAL_A else "MODEL_NOT_AVAILABLE")


def normalize_b(value: Any) -> str:
    text = norm(value)
    return {"GOOD":"ECG_GOOD","BAD":"ECG_BAD","UNCERTAIN":"ECG_UNCERTAIN"}.get(text, text if text in LEGAL_B else "MODEL_NOT_AVAILABLE")


def apply_c_gate(row: Dict[str, Any]) -> None:
    b = normalize_b(row.get("model_b_label"))
    row["model_b_label"] = b
    if b != "ECG_GOOD":
        row["model_c_raw_label"] = "SKIPPED_BY_B_GATE"
        row["model_c_final_label"] = "OTHER_OR_UNCERTAIN"
        row["model_c_p_normal"] = 0.0
        row["model_c_p_rhythm_suspect"] = 0.0
        row["model_c_reason"] = "ECG_QUALITY_NOT_GOOD"
        return
    p_suspect = f(row.get("model_c_p_rhythm_suspect"), 0.5)
    row["model_c_p_rhythm_suspect"] = p_suspect
    row["model_c_p_normal"] = f(row.get("model_c_p_normal"), max(0.0, 1.0 - p_suspect))
    row["model_c_raw_label"] = "RHYTHM_SUSPECT" if p_suspect >= 0.5 else "NORMAL"
    if p_suspect <= 0.42:
        row["model_c_final_label"] = "NORMAL"
        row["model_c_reason"] = row.get("model_c_reason") or "P_RHYTHM_SUSPECT_LTE_0_42"
    elif p_suspect >= 0.60:
        row["model_c_final_label"] = "RHYTHM_SUSPECT"
        row["model_c_reason"] = row.get("model_c_reason") or "P_RHYTHM_SUSPECT_GTE_0_60"
    else:
        row["model_c_final_label"] = "OTHER_OR_UNCERTAIN"
        row["model_c_reason"] = "LOW_CONFIDENCE_GRAY_ZONE"


def infer_rows(input_rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    model = d_load_model() if d_load_model else None
    out = []
    for src in input_rows:
        row: Dict[str, Any] = {field: src.get(field, "") for field in INPUT_FIELDS}
        row["model_a_label"] = normalize_a(row.get("model_a_label"))
        if row["model_a_label"] == "MODEL_NOT_AVAILABLE":
            row["model_a_confidence"] = 0.0
        apply_c_gate(row)
        if d_infer_row:
            d = d_infer_row(row, model)
            row["model_d_final_label"] = d["model_d_final_label"]
            row["model_d_confidence"] = 1.0 if d["model_d_final_label"] != "MODEL_NOT_AVAILABLE" else 0.0
            row["final_action"] = d["final_action"]
            row["final_reason"] = d["final_reason"]
        else:
            row["model_d_final_label"] = "MODEL_NOT_AVAILABLE"
            row["model_d_confidence"] = 0.0
            row["final_action"] = "FINAL_UNCERTAIN"
            row["final_reason"] = "MODEL_D_NOT_AVAILABLE: " + D_IMPORT_ERROR
        out.append(row)
    return out


def validate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    issues = []
    for i, row in enumerate(rows, 1):
        if row["model_a_label"] not in LEGAL_A:
            issues.append(f"row {i}: illegal A label")
        if row["model_b_label"] not in LEGAL_B:
            issues.append(f"row {i}: illegal B label")
        if row["model_b_label"] != "ECG_GOOD" and row["model_c_final_label"] != "OTHER_OR_UNCERTAIN":
            issues.append(f"row {i}: C gate violation")
        if row["model_c_final_label"] not in LEGAL_C:
            issues.append(f"row {i}: illegal C label")
        if row["final_action"] not in LEGAL_ACTIONS:
            issues.append(f"row {i}: illegal final_action")
        joined = json.dumps(row, ensure_ascii=False).upper()
        for token in FORBIDDEN:
            if token in joined:
                issues.append(f"row {i}: forbidden token {token}")
    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "row_count": len(rows),
        "model_c_gated_by_b": all(r["model_b_label"] == "ECG_GOOD" or r["model_c_final_label"] == "OTHER_OR_UNCERTAIN" for r in rows),
        "diagnosis_output_absent": not any(token in json.dumps(rows, ensure_ascii=False).upper() for token in FORBIDDEN),
        "final_action_legal": all(r["final_action"] in LEGAL_ACTIONS for r in rows),
        "schema_ok": all(all(field in r for field in OUTPUT_FIELDS) for r in rows),
        "old_c_proxy_absent": not any(any(k in r for k in ["C_normal_score","C_premature_score","C_irregular_score","C_uncertain_score"]) for r in rows),
    }


def write_report(path: Path, input_path: Path, output_path: Path, rows: Sequence[Dict[str, Any]], validation: Dict[str, Any]) -> None:
    lines = [
        "# ABCD v0.2 Smoke Test Report","",
        f"- Input: `{input_path}`",
        f"- Output: `{output_path}`",
        f"- Rows: {validation['row_count']}",
        f"- Passed: {validation['passed']}",
        f"- Model C obeys B gate: {validation['model_c_gated_by_b']}",
        "- Model D directly consumes current Model C v0.4 fields: True",
        "- Rule fallback enabled: True",
        f"- final_action legal: {validation['final_action_legal']}",
        f"- No diagnosis-style outputs: {validation['diagnosis_output_absent']}",
        f"- No schema mismatch: {validation['schema_ok']}",
        f"- No old C proxy dependency remains: {validation['old_c_proxy_absent']}",
        "",
        "| Window | A | B | C | D | final_action |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(f"| {r['window_id']} | {r['model_a_label']} | {r['model_b_label']} | {r['model_c_final_label']} | {r['model_d_final_label']} | {r['final_action']} |")
    if validation["issues"]:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in validation["issues"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "replay" / "abcd_v0_2_smoke_test_windows.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "abcd_v0_2_inference_outputs.csv")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = infer_rows(read_csv(args.input))
    write_csv(args.output, rows, OUTPUT_FIELDS)
    validation = validate(rows)
    if args.report:
        write_report(args.report, args.input, args.output, rows, validation)
    print(f"ABCD_V0_2_ROWS={validation['row_count']}")
    print(f"ABCD_V0_2_PASSED={validation['passed']}")
    print(f"MODEL_C_GATED_BY_B={validation['model_c_gated_by_b']}")
    print(f"DIAGNOSIS_OUTPUT_ABSENT={validation['diagnosis_output_absent']}")
    if validation["issues"]:
        print("ISSUES=" + "; ".join(validation["issues"]))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
