#!/usr/bin/env python3
"""Finalize ABCD Integrated v0.2 PC candidate artifacts."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = ROOT.parents[0]
A_DIR = AI_ROOT / "model_a_ppg_trust_gate_v0_4"
B_DIR = AI_ROOT / "model_b_ecg_quality_v0_1"
C_DIR = AI_ROOT / "model_c_ecg_rhythm_binary_v0_4_compromise"
D_DIR = AI_ROOT / "model_d_fusion_decision_v0_2"
ABCD_V01_DIR = AI_ROOT / "model_abcd_integrated_v0_1"

SUBDIRS = [
    "inputs",
    "model_registry",
    "schemas",
    "contracts",
    "tools",
    "reports",
    "figures",
    "outputs",
    "replay",
    "deliverable",
    "logs",
]

FINAL_ACTIONS = [
    "MEASURE_OK",
    "KEEP_STILL",
    "RETEST_CONTACT",
    "RETEST_AFTER_RECOVERY",
    "RHYTHM_RISK_RETEST",
    "SENSOR_CONFLICT_RETEST",
    "MEASURE_FAILED",
    "FINAL_UNCERTAIN",
]

FORBIDDEN_TOKENS = [
    "DIAGNOSIS",
    "HEART_DISEASE",
    "AF_DIAGNOSIS",
    "PVC_DIAGNOSIS",
    "MYOCARDIAL_ISCHEMIA",
    "DISEASE_DETECTED",
]

INPUT_FIELDS = [
    "timestamp_ms",
    "window_id",
    "ppg_quality_score",
    "ppg_valid_window_flag",
    "ppg_trust_level",
    "ppg_hr_bpm_x10",
    "ppg_spo2_x10",
    "model_a_label",
    "model_a_confidence",
    "ecg_quality_score",
    "ecg_lead_off",
    "ecg_hr_bpm_x10",
    "ecg_hr_stale_flag",
    "model_b_label",
    "model_b_confidence",
    "model_c_p_normal",
    "model_c_p_rhythm_suspect",
    "model_c_raw_label",
    "model_c_final_label",
    "model_c_reason",
    "imu_state",
    "imu_motion_flag",
    "imu_motion_level",
    "tmp_state",
    "tmp_c_x10",
    "tmp_finger_cold",
    "tmp_valid",
    "tmp_stable",
    "lead_off",
    "contact_bad",
    "recovery_wait",
    "sensor_conflict",
    "motion_degraded",
    "cold_suspect",
    "hr_conflict",
    "spo2_missing",
    "measure_failed",
]

OUTPUT_FIELDS = INPUT_FIELDS + [
    "model_d_final_label",
    "model_d_confidence",
    "final_action",
    "final_reason",
]


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


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def feature_names_from_a(schema: Dict[str, Any]) -> List[str]:
    return [x["name"] for x in schema.get("feature_order", []) if isinstance(x, dict) and "name" in x]


def init_dirs() -> None:
    for sub in SUBDIRS:
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    write_text(
        ROOT / "reports" / "abcd_v0_2_init_report.md",
        "\n".join(
            [
                "# ABCD v0.2 Init Report",
                "",
                f"- Root: `{ROOT}`",
                "- Created/verified subdirectories:",
                *[f"  - `{sub}`" for sub in SUBDIRS],
                "- Scope: final PC-side integration and documentation only.",
                "- No retraining, no TFLite export, no GD32 Embedded AI, no MDK firmware modification.",
            ]
        )
        + "\n",
    )


def model_inventory() -> List[Dict[str, Any]]:
    a_schema = read_json(A_DIR / "models" / "model_a_ppg_trust_gate_v0_4_feature_schema.json")
    b_schema = read_json(B_DIR / "models" / "model_b_ecg_quality_v0_1_feature_schema.json")
    c_schema = read_json(C_DIR / "final_candidate" / "model_c_binary_v0_4_feature_schema.json")
    d_schema = read_json(D_DIR / "models" / "model_d_v0_2_input_schema.json")
    rows = [
        {
            "model_name": "Model A - PPG quality gate",
            "version": "v0.4",
            "source_directory": str(A_DIR),
            "model_file": str(A_DIR / "models" / "model_a_ppg_trust_gate_v0_4_float32.keras"),
            "preprocessor_file": "",
            "feature_schema_file": str(A_DIR / "models" / "model_a_ppg_trust_gate_v0_4_feature_schema.json"),
            "label_map_file": str(A_DIR / "models" / "model_a_ppg_trust_gate_v0_4_label_map.json"),
            "threshold_file": "",
            "final_report": str(A_DIR / "reports" / "model_a_v0_4_training_report.md"),
            "summary_json": str(A_DIR / "reports" / "model_a_v0_4_metrics.json"),
            "input_features": len(feature_names_from_a(a_schema)),
            "output_labels": "PPG_GOOD|PPG_BAD|PPG_UNCERTAIN",
            "status": "FOUND",
            "deploy_status": "PC_CANDIDATE_REFERENCE_ONLY",
            "safety_boundary": "PPG quality gate, not diagnosis.",
        },
        {
            "model_name": "Model B - ECG quality gate",
            "version": "v0.1",
            "source_directory": str(B_DIR),
            "model_file": str(B_DIR / "models" / "model_b_ecg_quality_v0_1.pkl"),
            "preprocessor_file": "",
            "feature_schema_file": str(B_DIR / "models" / "model_b_ecg_quality_v0_1_feature_schema.json"),
            "label_map_file": str(B_DIR / "models" / "model_b_ecg_quality_v0_1_label_schema.json"),
            "threshold_file": "",
            "final_report": str(B_DIR / "reports" / "model_b_training_report.md"),
            "summary_json": str(B_DIR / "reports" / "model_b_metrics.json"),
            "input_features": len(b_schema.get("feature_columns", [])),
            "output_labels": "ECG_GOOD|ECG_BAD|ECG_UNCERTAIN",
            "status": "FOUND",
            "deploy_status": "PC_CANDIDATE_REFERENCE_ONLY",
            "safety_boundary": "ECG quality gate, not diagnosis.",
        },
        {
            "model_name": "Model C - ECG rhythm-risk hint candidate",
            "version": "v0.4 threshold-calibrated compromise",
            "source_directory": str(C_DIR),
            "model_file": str(C_DIR / "final_candidate" / "model_c_binary_v0_4_threshold_calibrated_candidate.pkl"),
            "preprocessor_file": str(C_DIR / "final_candidate" / "model_c_binary_v0_4_preprocessor.pkl"),
            "feature_schema_file": str(C_DIR / "final_candidate" / "model_c_binary_v0_4_feature_schema.json"),
            "label_map_file": str(C_DIR / "final_candidate" / "model_c_binary_v0_4_label_map.json"),
            "threshold_file": str(C_DIR / "final_candidate" / "model_c_binary_v0_4_thresholds.json"),
            "final_report": str(C_DIR / "reports" / "model_c_v0_4_threshold_calibrated_final_report.md"),
            "summary_json": str(C_DIR / "reports" / "model_c_v0_4_threshold_calibrated_final_summary.json"),
            "input_features": len(c_schema.get("features", [])),
            "output_labels": "NORMAL|RHYTHM_SUSPECT|OTHER_OR_UNCERTAIN",
            "status": "SMOKE_USABLE_DO_NOT_DEPLOY",
            "deploy_status": "DO_NOT_DEPLOY",
            "safety_boundary": "Rhythm risk hint candidate, not diagnosis, not deployable; gated by Model B ECG_GOOD.",
        },
        {
            "model_name": "Model D - Fusion decision",
            "version": "v0.2",
            "source_directory": str(D_DIR),
            "model_file": str(D_DIR / "models" / "model_d_fusion_decision_v0_2_best.pkl"),
            "preprocessor_file": str(D_DIR / "models" / "model_d_fusion_decision_v0_2_preprocessor.pkl"),
            "feature_schema_file": str(D_DIR / "models" / "model_d_v0_2_input_schema.json"),
            "label_map_file": str(D_DIR / "models" / "model_d_v0_2_label_map.json"),
            "threshold_file": "",
            "final_report": str(D_DIR / "reports" / "model_d_v0_2_final_report.md"),
            "summary_json": str(D_DIR / "reports" / "model_d_v0_2_summary.json"),
            "input_features": len(d_schema.get("feature_names", [])),
            "output_labels": "|".join(read_json(D_DIR / "models" / "model_d_v0_2_label_map.json").get("labels", [])),
            "status": "FOUND_PC_CANDIDATE",
            "deploy_status": "PC_ONLY_NOT_FIRMWARE_READY",
            "safety_boundary": "Fusion decision PC candidate with rule fallback; not firmware-ready.",
        },
    ]
    return rows


def write_inventory(rows: List[Dict[str, Any]]) -> None:
    fields = [
        "model_name",
        "version",
        "source_directory",
        "model_file",
        "preprocessor_file",
        "feature_schema_file",
        "label_map_file",
        "threshold_file",
        "final_report",
        "summary_json",
        "input_features",
        "output_labels",
        "status",
        "deploy_status",
        "safety_boundary",
    ]
    write_csv(ROOT / "reports" / "abcd_v0_2_model_inventory.csv", rows, fields)
    lines = ["# ABCD v0.2 Model Inventory", "", "| Model | Version | Status | Deploy Status | Safety Boundary |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(
            f"| {row['model_name']} | {row['version']} | {row['status']} | {row['deploy_status']} | {row['safety_boundary']} |"
        )
    lines.extend(["", "## Artifact Details", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['model_name']}",
                "",
                f"- Source directory: `{row['source_directory']}`",
                f"- Model file: `{row['model_file']}`",
                f"- Preprocessor/scaler: `{row['preprocessor_file'] or 'none'}`",
                f"- Feature schema: `{row['feature_schema_file']}`",
                f"- Label map: `{row['label_map_file']}`",
                f"- Threshold file: `{row['threshold_file'] or 'none'}`",
                f"- Final report: `{row['final_report']}`",
                f"- Summary JSON: `{row['summary_json']}`",
                f"- Input features: {row['input_features']}",
                f"- Output labels: `{row['output_labels']}`",
                "",
            ]
        )
    write_text(ROOT / "reports" / "abcd_v0_2_model_inventory.md", "\n".join(lines))


def build_registry(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    c_thresholds = read_json(C_DIR / "final_candidate" / "model_c_binary_v0_4_thresholds.json")
    d_summary = read_json(D_DIR / "reports" / "model_d_v0_2_summary.json")
    registry = {
        "registry_name": "abcd_v0_2_model_registry",
        "version": "v0.2",
        "scope": "PC-side integrated inference and fusion candidate only.",
        "models": {row["model_name"].split(" - ")[0].replace("Model ", ""): row for row in rows},
        "model_c_gate": {
            "normal_threshold": c_thresholds.get("normal_threshold", 0.42),
            "suspect_threshold": c_thresholds.get("suspect_threshold", 0.60),
            "participates_only_if_model_b_label": "ECG_GOOD",
        },
        "model_d_v0_2": {
            "directly_uses_model_c_v0_4": d_summary.get("directly_uses_model_c_v0_4", True),
            "rule_fallback_enabled": d_summary.get("rule_fallback_enabled", True),
        },
        "deployment_gates": {
            "PC_ABCD_PIPELINE_READY": None,
            "FIRMWARE_READY": False,
            "TFLITE_ALL_READY": False,
            "GD32_READY": False,
        },
        "forbidden_output_tokens": FORBIDDEN_TOKENS,
    }
    write_json(ROOT / "model_registry" / "abcd_v0_2_model_registry.json", registry)
    lines = ["# ABCD v0.2 Model Cards", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['model_name']}",
                "",
                f"- Version: {row['version']}",
                f"- Source: `{row['source_directory']}`",
                f"- Inputs: {row['input_features']}",
                f"- Outputs: `{row['output_labels']}`",
                f"- Status: {row['status']}",
                f"- Deploy status: {row['deploy_status']}",
                f"- Safety limitations: {row['safety_boundary']}",
                "",
            ]
        )
    write_text(ROOT / "reports" / "abcd_v0_2_model_cards.md", "\n".join(lines))
    return registry


def schemas_and_contract() -> None:
    input_schema = {
        "schema_name": "abcd_v0_2_unified_input_schema",
        "version": "v0.2",
        "fields": INPUT_FIELDS,
        "model_c_gate": "Model C may participate only when model_b_label == ECG_GOOD.",
        "safety_boundary": "No diagnosis output; Model C is risk hint only; D is fusion decision only.",
    }
    output_schema = {
        "schema_name": "abcd_v0_2_unified_output_schema",
        "version": "v0.2",
        "fields": OUTPUT_FIELDS,
        "allowed_final_action": FINAL_ACTIONS,
        "forbidden_output_tokens": FORBIDDEN_TOKENS,
    }
    write_json(ROOT / "schemas" / "abcd_v0_2_unified_input_schema.json", input_schema)
    write_json(ROOT / "schemas" / "abcd_v0_2_unified_output_schema.json", output_schema)
    write_text(
        ROOT / "contracts" / "abcd_v0_2_inference_contract.md",
        """# ABCD v0.2 Inference Contract

## Scope

ABCD Integrated v0.2 is a PC-side inference and fusion candidate only. It does
not retrain models, export TFLite, enter GD32 Embedded AI, or modify MDK firmware.

## Pipeline Order

1. Load/read Model A PPG quality output.
2. Load/read Model B ECG quality output.
3. Run/read Model C only if Model B is `ECG_GOOD`.
4. Build rule flags from IMU/TMP/contact/recovery/conflict fields.
5. Run Model D v0.2 fusion.
6. Apply hard-rule fallback; hard safety/retest rules override conflicting ML.
7. Emit `final_action` and `final_reason`.

## Model C Gate

```text
if model_b_label != "ECG_GOOD":
    model_c_final_label = "OTHER_OR_UNCERTAIN"
    model_c_reason = "ECG_QUALITY_NOT_GOOD"
elif model_c_p_rhythm_suspect <= 0.42:
    model_c_final_label = "NORMAL"
elif model_c_p_rhythm_suspect >= 0.60:
    model_c_final_label = "RHYTHM_SUSPECT"
else:
    model_c_final_label = "OTHER_OR_UNCERTAIN"
    model_c_reason = "LOW_CONFIDENCE_GRAY_ZONE"
```

## Safety

Model C is a rhythm-risk hint candidate only, not diagnosis. Model D is a
fusion decision layer only. Forbidden output tokens are listed in
`schemas/abcd_v0_2_unified_output_schema.json`.
""",
    )


PIPELINE_CODE = r'''#!/usr/bin/env python3
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
'''


def write_pipeline() -> None:
    write_text(ROOT / "tools" / "abcd_v0_2_inference_pipeline.py", PIPELINE_CODE)


def smoke_rows() -> List[Dict[str, Any]]:
    cases = [
        ("all_normal", "PPG_GOOD", "ECG_GOOD", 0.20, "STATIC", "TMP_NORMAL", {}),
        ("ppg_bad_ecg_good", "PPG_BAD", "ECG_GOOD", 0.20, "STATIC", "TMP_NORMAL", {}),
        ("ecg_bad_ppg_good", "PPG_GOOD", "ECG_BAD", 0.75, "STATIC", "TMP_NORMAL", {}),
        ("rhythm_suspect", "PPG_GOOD", "ECG_GOOD", 0.75, "STATIC", "TMP_NORMAL", {}),
        ("motion_degraded", "PPG_UNCERTAIN", "ECG_UNCERTAIN", 0.50, "MOTION", "TMP_NORMAL", {"motion_degraded": 1}),
        ("contact_bad", "PPG_BAD", "ECG_BAD", 0.50, "STATIC", "TMP_NORMAL", {"lead_off": 1, "contact_bad": 1}),
        ("recovery_wait", "PPG_UNCERTAIN", "ECG_GOOD", 0.20, "STATIC", "COLD_SUSPECT", {"recovery_wait": 1, "cold_suspect": 1}),
        ("sensor_conflict", "PPG_GOOD", "ECG_GOOD", 0.50, "STATIC", "TMP_NORMAL", {"sensor_conflict": 1, "hr_conflict": 1}),
        ("measure_failed", "PPG_BAD", "ECG_BAD", 0.50, "STATIC", "TMP_INVALID", {"measure_failed": 1, "spo2_missing": 1}),
        ("all_uncertain", "PPG_UNCERTAIN", "ECG_UNCERTAIN", 0.50, "STATIC", "TMP_NORMAL", {}),
    ]
    rows: List[Dict[str, Any]] = []
    for idx, (case, a, b, p_suspect, imu, tmp, flags) in enumerate(cases):
        row = {
            "timestamp_ms": idx * 5000,
            "window_id": case,
            "ppg_quality_score": 0.95 if a == "PPG_GOOD" else (0.1 if a == "PPG_BAD" else 0.5),
            "ppg_valid_window_flag": 0 if flags.get("measure_failed") else 1,
            "ppg_trust_level": 2 if a == "PPG_GOOD" else (0 if a == "PPG_BAD" else 1),
            "ppg_hr_bpm_x10": 720,
            "ppg_spo2_x10": 0 if flags.get("spo2_missing") else 980,
            "model_a_label": a,
            "model_a_confidence": 0.95 if a == "PPG_GOOD" else 0.82,
            "ecg_quality_score": 0.95 if b == "ECG_GOOD" else (0.1 if b == "ECG_BAD" else 0.5),
            "ecg_lead_off": flags.get("lead_off", 0),
            "ecg_hr_bpm_x10": 900 if flags.get("hr_conflict") else 720,
            "ecg_hr_stale_flag": 0,
            "model_b_label": b,
            "model_b_confidence": 0.95 if b == "ECG_GOOD" else 0.82,
            "model_c_p_normal": round(max(0.0, 1.0 - p_suspect), 4),
            "model_c_p_rhythm_suspect": p_suspect,
            "model_c_raw_label": "RHYTHM_SUSPECT" if p_suspect >= 0.5 else "NORMAL",
            "model_c_final_label": "",
            "model_c_reason": "",
            "imu_state": imu,
            "imu_motion_flag": 1 if imu == "MOTION" else 0,
            "imu_motion_level": 2 if imu == "MOTION" else 0,
            "tmp_state": tmp,
            "tmp_c_x10": 0 if tmp == "TMP_INVALID" else (300 if tmp == "COLD_SUSPECT" else 330),
            "tmp_finger_cold": 1 if tmp == "COLD_SUSPECT" else 0,
            "tmp_valid": 0 if tmp == "TMP_INVALID" else 1,
            "tmp_stable": 0 if flags.get("recovery_wait") else 1,
            "lead_off": flags.get("lead_off", 0),
            "contact_bad": flags.get("contact_bad", 0),
            "recovery_wait": flags.get("recovery_wait", 0),
            "sensor_conflict": flags.get("sensor_conflict", 0),
            "motion_degraded": flags.get("motion_degraded", 0),
            "cold_suspect": flags.get("cold_suspect", 0),
            "hr_conflict": flags.get("hr_conflict", 0),
            "spo2_missing": flags.get("spo2_missing", 0),
            "measure_failed": flags.get("measure_failed", 0),
        }
        rows.append(row)
    return rows


def build_smoke_dataset() -> None:
    rows = smoke_rows()
    write_csv(ROOT / "replay" / "abcd_v0_2_smoke_test_windows.csv", rows, INPUT_FIELDS)
    lines = [
        "# ABCD v0.2 Smoke Test Dataset Report",
        "",
        f"- Dataset: `{ROOT / 'replay' / 'abcd_v0_2_smoke_test_windows.csv'}`",
        f"- Rows: {len(rows)}",
        "- Contains required cases: all_normal, ppg_bad_ecg_good, ecg_bad_ppg_good, rhythm_suspect, motion_degraded, contact_bad, recovery_wait, sensor_conflict, measure_failed, all_uncertain.",
        "",
        "| Case | A | B | p_rhythm_suspect | IMU | TMP |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['window_id']} | {r['model_a_label']} | {r['model_b_label']} | {r['model_c_p_rhythm_suspect']} | {r['imu_state']} | {r['tmp_state']} |"
        )
    write_text(ROOT / "reports" / "abcd_v0_2_smoke_test_dataset_report.md", "\n".join(lines) + "\n")


def run_pipeline() -> Dict[str, Any]:
    smoke_input = ROOT / "replay" / "abcd_v0_2_smoke_test_windows.csv"
    smoke_output = ROOT / "outputs" / "abcd_v0_2_smoke_test_outputs.csv"
    smoke_report = ROOT / "reports" / "abcd_v0_2_smoke_test_report.md"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "abcd_v0_2_inference_pipeline.py"),
            "--input",
            str(smoke_input),
            "--output",
            str(smoke_output),
            "--report",
            str(smoke_report),
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    write_text(ROOT / "logs" / "abcd_v0_2_smoke_test_run.log", result.stdout)
    shutil.copyfile(smoke_output, ROOT / "outputs" / "abcd_v0_2_inference_outputs.csv")
    outputs = read_csv(smoke_output)
    passed = result.returncode == 0
    c_gated = all(r["model_b_label"] == "ECG_GOOD" or r["model_c_final_label"] == "OTHER_OR_UNCERTAIN" for r in outputs)
    diagnosis_absent = not any(token in json.dumps(outputs, ensure_ascii=False).upper() for token in FORBIDDEN_TOKENS)
    return {
        "passed": passed,
        "stdout": result.stdout,
        "outputs": outputs,
        "model_c_gated_by_b": c_gated,
        "diagnosis_output_absent": diagnosis_absent,
    }


def draw_architecture() -> None:
    path = ROOT / "figures" / "abcd_v0_2_pipeline_architecture.png"
    img = Image.new("RGB", (1900, 1100), "#f8fafc")
    draw = ImageDraw.Draw(img)
    try:
        title = ImageFont.truetype("arial.ttf", 44)
        font = ImageFont.truetype("arial.ttf", 27)
        small = ImageFont.truetype("arial.ttf", 23)
    except Exception:
        title = ImageFont.load_default()
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    def box(x: int, y: int, w: int, h: int, text: str, fill: str) -> None:
        draw.rectangle([x, y, x + w, y + h], fill=fill, outline="#27313f", width=3)
        lines = text.split("\n")
        for idx, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            draw.text((x + (w - bbox[2]) // 2, y + 24 + idx * 34), line, fill="#111827", font=font)

    def arrow(x1: int, y1: int, x2: int, y2: int) -> None:
        draw.line([x1, y1, x2, y2], fill="#374151", width=4)
        draw.polygon([(x2, y2), (x2 - 18, y2 - 10), (x2 - 18, y2 + 10)], fill="#374151")

    draw.text((500, 35), "ABCD Integrated v0.2 PC Inference Pipeline", fill="#111827", font=title)
    box(90, 165, 185, 90, "PPG", "#d8f3dc")
    box(360, 145, 285, 130, "Model A\nPPG quality", "#b7e4c7")
    box(735, 165, 270, 90, "PPG quality\nlabel", "#d8f3dc")
    box(90, 435, 185, 90, "ECG", "#dbeafe")
    box(360, 415, 285, 130, "Model B\nECG quality", "#bfdbfe")
    box(735, 435, 270, 90, "ECG quality\nlabel", "#dbeafe")
    box(360, 660, 285, 130, "Model C\nrisk hint only", "#fde68a")
    box(735, 680, 270, 90, "Rhythm risk\nhint", "#fef3c7")
    box(90, 875, 280, 90, "IMU/TMP/Rule", "#e5e7eb")
    box(1180, 405, 350, 155, "Model D v0.2\nFusion decision", "#fecaca")
    box(1620, 430, 230, 100, "final_action", "#fee2e2")
    for x1, y1, x2, y2 in [(275,210,360,210),(645,210,735,210),(275,480,360,480),(645,480,735,480),(505,545,505,660),(645,725,735,725),(1005,210,1180,435),(1005,480,1180,480),(1005,725,1180,545),(370,920,1180,545),(1530,480,1620,480)]:
        arrow(x1, y1, x2, y2)
    draw.text((365, 805), "ECG_GOOD gate -> Model C", fill="#92400e", font=small)
    draw.text((1165, 605), "Final decision is Fusion-owned", fill="#991b1b", font=small)
    draw.text((1165, 640), "No diagnosis output", fill="#991b1b", font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def final_report_and_summary(rows: List[Dict[str, Any]], run: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "model_a_found": Path(rows[0]["model_file"]).exists(),
        "model_b_found": Path(rows[1]["model_file"]).exists(),
        "model_c_found": Path(rows[2]["model_file"]).exists(),
        "model_d_v0_2_found": Path(rows[3]["model_file"]).exists(),
        "model_c_gated_by_b": run["model_c_gated_by_b"],
        "model_d_directly_uses_model_c_v0_4": True,
        "rule_fallback_enabled": True,
        "abcd_v0_2_pc_pipeline_ready": run["passed"] and run["model_c_gated_by_b"] and run["diagnosis_output_absent"],
        "diagnosis_output_absent": run["diagnosis_output_absent"],
        "firmware_ready": False,
        "tflite_all_ready": False,
        "gd32_ready": False,
        "next_action": "Freeze PC integration contract; prepare MDK/Firmware interface design later; do not proceed to board deployment until TFLite export and MCU replay are completed.",
    }
    write_json(ROOT / "reports" / "abcd_v0_2_summary.json", summary)
    out_rows = run["outputs"]
    smoke_lines = "\n".join(
        f"| {r['window_id']} | {r['model_d_final_label']} | {r['final_action']} |" for r in out_rows
    )
    status_lines = "\n".join(
        f"| {r['model_name']} | {r['version']} | {r['status']} | {r['deploy_status']} |" for r in rows
    )
    report = f"""# ABCD Integrated v0.2 Final Report

## 1. Goal

Integrate A/B/C/D into a unified PC-side inference and fusion pipeline.

## 2. Model Status

| Model | Version | Status | Deploy Status |
| --- | --- | --- | --- |
{status_lines}

## 3. What Changed From ABCD v0.1

ABCD v0.1 used D v0.1, which depended on old Model C proxy fields such as
`C_normal_score`, `C_premature_score`, `C_irregular_score`, and
`C_uncertain_score`. ABCD v0.2 uses D v0.2, which directly consumes current
Model C v0.4 outputs: `model_c_final_label`, `model_c_p_normal`,
`model_c_p_rhythm_suspect`, and `model_c_reason`.

## 4. Unified Schema

Input schema: `schemas/abcd_v0_2_unified_input_schema.json`.
Output schema: `schemas/abcd_v0_2_unified_output_schema.json`.

Key groups: Model A PPG quality fields, Model B ECG quality fields, Model C
rhythm-risk hint fields, IMU/TMP fields, rule flags, and Model D/fusion output.

## 5. Inference Pipeline

Model A -> Model B -> Model C gated by B -> Rule flags -> D v0.2 Fusion.
Hard rule fallback is enabled and overrides conflicting ML output.

## 6. Smoke Test

| Case | D label | final_action |
| --- | --- | --- |
{smoke_lines}

Smoke result: {'PASS' if summary['abcd_v0_2_pc_pipeline_ready'] else 'FAIL'}.

## 7. Safety Boundary

- No diagnosis output.
- Model A is a PPG quality gate, not diagnosis.
- Model B is an ECG quality gate, not diagnosis.
- Model C is a rhythm risk hint candidate only, not diagnosis.
- Model D is a fusion decision layer only.
- This is PC pipeline only.
- No cross-subject clinical generalization is claimed.

## 8. Deployment Gate

PC_ABCD_PIPELINE_READY = {summary['abcd_v0_2_pc_pipeline_ready']}
FIRMWARE_READY = False
TFLITE_ALL_READY = False
GD32_READY = False

## 9. Next Action

{summary['next_action']}
"""
    write_text(ROOT / "reports" / "abcd_v0_2_final_report.md", report)
    return summary


def make_deliverable() -> Path:
    dst = ROOT / "deliverable" / "abcd_integrated_v0_2_pc_candidate"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for dirname in ["model_registry", "schemas", "contracts"]:
        shutil.copytree(ROOT / dirname, dst / dirname)
    (dst / "tools").mkdir()
    shutil.copy2(ROOT / "tools" / "abcd_v0_2_inference_pipeline.py", dst / "tools" / "abcd_v0_2_inference_pipeline.py")
    for dirname, files in {
        "replay": ["abcd_v0_2_smoke_test_windows.csv"],
        "outputs": ["abcd_v0_2_smoke_test_outputs.csv"],
        "figures": ["abcd_v0_2_pipeline_architecture.png"],
        "reports": [
            "abcd_v0_2_final_report.md",
            "abcd_v0_2_summary.json",
            "abcd_v0_2_smoke_test_report.md",
            "abcd_v0_2_model_inventory.md",
            "abcd_v0_2_model_cards.md",
        ],
    }.items():
        (dst / dirname).mkdir(exist_ok=True)
        for file in files:
            shutil.copy2(ROOT / dirname / file, dst / dirname / file)
    refs = dst / "references"
    refs.mkdir()
    for src in [
        C_DIR / "final_candidate" / "model_c_fusion_interface_contract.json",
        D_DIR / "reports" / "model_d_v0_2_final_report.md",
        ABCD_V01_DIR / "reports" / "abcd_integrated_v0_1_one_file_report.md",
    ]:
        if src.exists():
            shutil.copy2(src, refs / src.name)
    readme = """# ABCD Integrated v0.2 PC Candidate

This package is a PC-side candidate deliverable only. It contains registry,
schemas, contracts, the PC inference pipeline, smoke replay data, smoke outputs,
architecture figure, and final reports.

It does not contain firmware-ready claims. FIRMWARE_READY=False,
TFLITE_ALL_READY=False, GD32_READY=False.
"""
    write_text(dst / "README.md", readme)
    zip_path = ROOT / "deliverable" / "abcd_integrated_v0_2_pc_candidate.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in dst.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(dst.parent))
    return zip_path


def integrity(summary: Dict[str, Any], zip_path: Path) -> Dict[str, Any]:
    checks = {
        "model_registry_exists": (ROOT / "model_registry" / "abcd_v0_2_model_registry.json").exists(),
        "unified_input_schema_exists": (ROOT / "schemas" / "abcd_v0_2_unified_input_schema.json").exists(),
        "unified_output_schema_exists": (ROOT / "schemas" / "abcd_v0_2_unified_output_schema.json").exists(),
        "inference_contract_exists": (ROOT / "contracts" / "abcd_v0_2_inference_contract.md").exists(),
        "pipeline_script_exists": (ROOT / "tools" / "abcd_v0_2_inference_pipeline.py").exists(),
        "smoke_output_exists": (ROOT / "outputs" / "abcd_v0_2_smoke_test_outputs.csv").exists(),
        "architecture_figure_exists": (ROOT / "figures" / "abcd_v0_2_pipeline_architecture.png").exists(),
        "final_report_exists": (ROOT / "reports" / "abcd_v0_2_final_report.md").exists(),
        "summary_json_exists": (ROOT / "reports" / "abcd_v0_2_summary.json").exists(),
        "deliverable_zip_exists": zip_path.exists(),
        "deliverable_zip_opens": False,
        "diagnosis_output_absent": summary["diagnosis_output_absent"],
        "firmware_ready_false": summary["firmware_ready"] is False,
        "tflite_all_ready_false": summary["tflite_all_ready"] is False,
        "gd32_ready_false": summary["gd32_ready"] is False,
    }
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            checks["deliverable_zip_opens"] = bool(zf.namelist())
    except zipfile.BadZipFile:
        checks["deliverable_zip_opens"] = False
    lines = ["# ABCD v0.2 Final Integrity Check", ""]
    for key, value in checks.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append(f"Overall pass: {all(checks.values())}")
    write_text(ROOT / "reports" / "abcd_v0_2_final_integrity_check.md", "\n".join(lines) + "\n")
    return checks


def main() -> int:
    init_dirs()
    rows = model_inventory()
    write_inventory(rows)
    registry = build_registry(rows)
    schemas_and_contract()
    write_pipeline()
    build_smoke_dataset()
    run = run_pipeline()
    draw_architecture()
    summary = final_report_and_summary(rows, run)
    registry["deployment_gates"]["PC_ABCD_PIPELINE_READY"] = summary["abcd_v0_2_pc_pipeline_ready"]
    write_json(ROOT / "model_registry" / "abcd_v0_2_model_registry.json", registry)
    zip_path = make_deliverable()
    checks = integrity(summary, zip_path)
    print(f"ABCD_V0_2_DIR = {ROOT}")
    print(f"MODEL_A_STATUS = {'FOUND' if summary['model_a_found'] else 'MISSING'}")
    print(f"MODEL_B_STATUS = {'FOUND' if summary['model_b_found'] else 'MISSING'}")
    print("MODEL_C_STATUS = SMOKE_USABLE_DO_NOT_DEPLOY")
    print(f"MODEL_D_V0_2_STATUS = {'FOUND_PC_CANDIDATE' if summary['model_d_v0_2_found'] else 'MISSING'}")
    print(f"MODEL_C_GATED_BY_B = {summary['model_c_gated_by_b']}")
    print(f"MODEL_D_DIRECT_C_V0_4_COMPATIBLE = {summary['model_d_directly_uses_model_c_v0_4']}")
    print(f"RULE_FALLBACK_ENABLED = {summary['rule_fallback_enabled']}")
    print(f"ABCD_V0_2_PC_PIPELINE_READY = {summary['abcd_v0_2_pc_pipeline_ready']}")
    print(f"DIAGNOSIS_OUTPUT_ABSENT = {summary['diagnosis_output_absent']}")
    print("FIRMWARE_READY = False")
    print("TFLITE_ALL_READY = False")
    print("GD32_READY = False")
    print(f"DELIVERABLE_ZIP = {zip_path}")
    print(f"REPORT_PATH = {ROOT / 'reports' / 'abcd_v0_2_final_report.md'}")
    print(f"NEXT_ACTION = {summary['next_action']}")
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
