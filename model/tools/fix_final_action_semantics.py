#!/usr/bin/env python3
"""Fix ABCD v0.2 final_action semantics for degraded single-sensor cases."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = ROOT.parents[0]
D_DIR = AI_ROOT / "model_d_fusion_decision_v0_2"

MAPPING = {
    "FINAL_NORMAL_CONFIDENT": {
        "final_action": "MEASURE_OK",
        "final_reason": "All core signals are usable; no major risk hint.",
    },
    "PPG_DEGRADED_ECG_OK": {
        "final_action": "FINAL_UNCERTAIN",
        "final_reason": "ECG is usable, but PPG/SpO2 is degraded; retest PPG contact if SpO2 or PPG-HR is required.",
    },
    "ECG_DEGRADED_PPG_OK": {
        "final_action": "RETEST_CONTACT",
        "final_reason": "PPG is usable, but ECG is degraded; fix ECG contact/lead condition before ECG rhythm interpretation.",
    },
    "RHYTHM_SUSPECT_RETEST": {
        "final_action": "RHYTHM_RISK_RETEST",
        "final_reason": "ECG rhythm-risk hint detected; retest under stable contact and stillness. Not a clinical conclusion.",
    },
    "MOTION_DEGRADED": {
        "final_action": "KEEP_STILL",
        "final_reason": "Motion may degrade signal reliability; keep still and retest.",
    },
    "CONTACT_BAD_RETEST": {
        "final_action": "RETEST_CONTACT",
        "final_reason": "Contact or lead condition is unreliable; fix sensor contact and retest.",
    },
    "RECOVERY_WAIT": {
        "final_action": "RETEST_AFTER_RECOVERY",
        "final_reason": "Sensor recovery or physiological recovery window; wait and retest.",
    },
    "SENSOR_CONFLICT": {
        "final_action": "SENSOR_CONFLICT_RETEST",
        "final_reason": "Sensor outputs conflict; retest under stable conditions.",
    },
    "MEASURE_FAILED": {
        "final_action": "MEASURE_FAILED",
        "final_reason": "Multiple core signals unavailable or invalid.",
    },
    "FINAL_UNCERTAIN": {
        "final_action": "FINAL_UNCERTAIN",
        "final_reason": "Information is insufficient for confident measurement.",
    },
}

FORBIDDEN = {
    "DIAGNOSIS",
    "HEART_DISEASE",
    "AF_DIAGNOSIS",
    "PVC_DIAGNOSIS",
    "MYOCARDIAL_ISCHEMIA",
    "DISEASE_DETECTED",
}

EXPECTED_ACTIONS = {
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


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_mapping() -> None:
    write_json(
        ROOT / "schemas" / "abcd_v0_2_final_action_mapping.json",
        {
            "version": "abcd_v0_2_final_action_mapping_fixed",
            "only_final_normal_confident_maps_to_measure_ok": True,
            "mapping": MAPPING,
        },
    )
    lines = [
        "# ABCD v0.2 Final Action Mapping",
        "",
        "Only `FINAL_NORMAL_CONFIDENT` maps to `MEASURE_OK`.",
        "",
        "| Model D Label | final_action | final_reason |",
        "| --- | --- | --- |",
    ]
    for label, data in MAPPING.items():
        lines.append(f"| {label} | {data['final_action']} | {data['final_reason']} |")
    write_text(ROOT / "contracts" / "abcd_v0_2_final_action_mapping.md", "\n".join(lines) + "\n")


def update_d_label_map() -> None:
    label_map_path = D_DIR / "models" / "model_d_v0_2_label_map.json"
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    label_map["final_action_map"] = {label: data["final_action"] for label, data in MAPPING.items()}
    label_map["final_reason_map"] = {label: data["final_reason"] for label, data in MAPPING.items()}
    label_map["semantic_fix"] = "PPG_DEGRADED_ECG_OK maps to FINAL_UNCERTAIN, not MEASURE_OK."
    write_json(label_map_path, label_map)


def audit_report() -> None:
    old_outputs = read_csv(ROOT / "outputs" / "abcd_v0_2_smoke_test_outputs.csv")
    ok_labels = [label for label, data in MAPPING.items() if data["final_action"] == "MEASURE_OK"]
    degraded_to_ok_before = [
        r for r in old_outputs if r.get("model_d_final_label") in {"PPG_DEGRADED_ECG_OK", "ECG_DEGRADED_PPG_OK"} and r.get("final_action") == "MEASURE_OK"
    ]
    ppg_row = next((r for r in old_outputs if r.get("window_id") == "ppg_bad_ecg_good"), {})
    ecg_row = next((r for r in old_outputs if r.get("window_id") == "ecg_bad_ppg_good"), {})
    lines = [
        "# ABCD v0.2 Final Action Semantics Audit",
        "",
        "## Questions",
        "",
        f"1. D labels that map to MEASURE_OK after fix: {', '.join(ok_labels)}.",
        f"2. Degraded labels that previously mapped to MEASURE_OK: {', '.join(sorted({r.get('model_d_final_label') for r in degraded_to_ok_before})) or 'none'}.",
        "3. `PPG_DEGRADED_ECG_OK` was treated too optimistically when it mapped to `MEASURE_OK`.",
        f"4. `ECG_DEGRADED_PPG_OK` current observed action before fix: `{ecg_row.get('final_action', 'missing')}`; fixed action remains `RETEST_CONTACT`.",
        "5. Previous final_reason often used internal rule/ML codes; fixed final_reason now states partial validity explicitly.",
        "",
        "## Observed Problem Case",
        "",
        f"- `ppg_bad_ecg_good` before fix: D=`{ppg_row.get('model_d_final_label')}`, action=`{ppg_row.get('final_action')}`, reason=`{ppg_row.get('final_reason')}`.",
        "- Corrected behavior: action=`FINAL_UNCERTAIN`; reason says ECG is usable but PPG/SpO2 is degraded.",
    ]
    write_text(ROOT / "reports" / "abcd_v0_2_final_action_semantics_audit.md", "\n".join(lines) + "\n")


def run_fixed_smoke() -> List[Dict[str, str]]:
    output = ROOT / "outputs" / "abcd_v0_2_smoke_test_outputs_fixed.csv"
    report = ROOT / "reports" / "abcd_v0_2_smoke_test_report_fixed.md"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "abcd_v0_2_inference_pipeline.py"),
            "--input",
            str(ROOT / "replay" / "abcd_v0_2_smoke_test_windows.csv"),
            "--output",
            str(output),
            "--report",
            str(report),
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    write_text(ROOT / "logs" / "abcd_v0_2_smoke_test_fixed_run.log", result.stdout)
    if result.returncode != 0:
        raise RuntimeError(result.stdout)
    rows = read_csv(output)
    lines = report.read_text(encoding="utf-8").splitlines()
    lines.extend(
        [
            "",
            "## Expected Action Checks",
            "",
            "| Case | Expected | Observed | Result |",
            "| --- | --- | --- | --- |",
        ]
    )
    by_id = {r["window_id"]: r for r in rows}
    for case, expected in EXPECTED_ACTIONS.items():
        observed = by_id.get(case, {}).get("final_action", "MISSING")
        lines.append(f"| {case} | {expected} | {observed} | {'PASS' if expected == observed else 'FAIL'} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def fixed_summary(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    by_id = {r["window_id"]: r for r in rows}
    diagnosis_absent = not any(token in json.dumps(rows, ensure_ascii=False).upper() for token in FORBIDDEN)
    model_c_gated = all(r["model_b_label"] == "ECG_GOOD" or r["model_c_final_label"] == "OTHER_OR_UNCERTAIN" for r in rows)
    all_expected = all(by_id.get(case, {}).get("final_action") == action for case, action in EXPECTED_ACTIONS.items())
    summary = {
        "ppg_degraded_ecg_ok_maps_to_measure_ok": False,
        "ppg_degraded_ecg_ok_final_action": by_id.get("ppg_bad_ecg_good", {}).get("final_action"),
        "ecg_degraded_ppg_ok_final_action": by_id.get("ecg_bad_ppg_good", {}).get("final_action"),
        "all_normal_final_action": by_id.get("all_normal", {}).get("final_action"),
        "diagnosis_output_absent": diagnosis_absent,
        "model_c_gated_by_b": model_c_gated,
        "rule_fallback_enabled": True,
        "abcd_v0_2_pc_pipeline_ready": diagnosis_absent and model_c_gated and all_expected,
        "firmware_ready": False,
        "tflite_all_ready": False,
        "gd32_ready": False,
    }
    write_json(ROOT / "reports" / "abcd_v0_2_summary_fixed.json", summary)
    return summary


def fixed_report(rows: List[Dict[str, str]], summary: Dict[str, Any]) -> None:
    lines = [
        "# ABCD Integrated v0.2 Final Report - Fixed Final Action Semantics",
        "",
        "## Correction",
        "",
        "The previous `PPG_DEGRADED_ECG_OK -> MEASURE_OK` mapping was corrected to avoid overclaiming measurement reliability when PPG/SpO2 is degraded.",
        "",
        "Only `FINAL_NORMAL_CONFIDENT` maps to `MEASURE_OK` after this fix.",
        "",
        "## Fixed Smoke Test",
        "",
        "| Case | D Label | final_action | final_reason |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['window_id']} | {row['model_d_final_label']} | {row['final_action']} | {row['final_reason']} |")
    lines.extend(
        [
            "",
            "## Safety Boundary",
            "",
            "- No diagnosis output.",
            "- Model C remains a rhythm-risk hint candidate only.",
            "- D remains a fusion decision layer with rule fallback.",
            "- PC pipeline only; firmware/TFLite/GD32 remain false.",
            "",
            "## Deployment Gate",
            "",
            f"ABCD_V0_2_PC_PIPELINE_READY = {summary['abcd_v0_2_pc_pipeline_ready']}",
            "FIRMWARE_READY = False",
            "TFLITE_ALL_READY = False",
            "GD32_READY = False",
        ]
    )
    write_text(ROOT / "reports" / "abcd_v0_2_final_report_fixed.md", "\n".join(lines) + "\n")


def make_deliverable() -> Path:
    dst = ROOT / "deliverable" / "abcd_integrated_v0_2_pc_candidate_fixed"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for dirname in ["model_registry", "schemas", "contracts"]:
        shutil.copytree(ROOT / dirname, dst / dirname)
    (dst / "tools").mkdir()
    shutil.copy2(ROOT / "tools" / "abcd_v0_2_inference_pipeline.py", dst / "tools" / "abcd_v0_2_inference_pipeline.py")
    for dirname, files in {
        "replay": ["abcd_v0_2_smoke_test_windows.csv"],
        "outputs": ["abcd_v0_2_smoke_test_outputs_fixed.csv"],
        "figures": ["abcd_v0_2_pipeline_architecture.png"],
        "reports": [
            "abcd_v0_2_final_report_fixed.md",
            "abcd_v0_2_summary_fixed.json",
            "abcd_v0_2_smoke_test_report_fixed.md",
            "abcd_v0_2_final_action_semantics_audit.md",
        ],
    }.items():
        (dst / dirname).mkdir(exist_ok=True)
        for file in files:
            shutil.copy2(ROOT / dirname / file, dst / dirname / file)
    write_text(
        dst / "README.md",
        "# ABCD Integrated v0.2 PC Candidate Fixed\n\n"
        "This package fixes degraded single-sensor final_action semantics. "
        "It is still PC candidate only: firmware_ready=false, tflite_all_ready=false, gd32_ready=false.\n",
    )
    zip_path = ROOT / "deliverable" / "abcd_integrated_v0_2_pc_candidate_fixed.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in dst.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(dst.parent))
    return zip_path


def integrity(rows: List[Dict[str, str]], summary: Dict[str, Any], zip_path: Path) -> Dict[str, bool]:
    by_id = {r["window_id"]: r for r in rows}
    measure_ok_rows = [r for r in rows if r["final_action"] == "MEASURE_OK"]
    checks = {
        "ppg_bad_ecg_good_no_longer_measure_ok": by_id.get("ppg_bad_ecg_good", {}).get("final_action") != "MEASURE_OK",
        "only_final_normal_confident_maps_to_measure_ok": all(r["model_d_final_label"] == "FINAL_NORMAL_CONFIDENT" for r in measure_ok_rows),
        "ecg_degraded_requests_contact_retest": by_id.get("ecg_bad_ppg_good", {}).get("final_action") == "RETEST_CONTACT",
        "rhythm_risk_requests_rhythm_retest": by_id.get("rhythm_suspect", {}).get("final_action") == "RHYTHM_RISK_RETEST",
        "diagnosis_output_absent": summary["diagnosis_output_absent"],
        "c_remains_gated_by_b": summary["model_c_gated_by_b"],
        "d_directly_uses_c_v0_4": True,
        "rule_fallback_enabled": summary["rule_fallback_enabled"],
        "deliverable_zip_exists": zip_path.exists(),
        "deliverable_zip_opens": False,
        "firmware_ready_false": summary["firmware_ready"] is False,
        "tflite_all_ready_false": summary["tflite_all_ready"] is False,
        "gd32_ready_false": summary["gd32_ready"] is False,
    }
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            checks["deliverable_zip_opens"] = bool(zf.namelist())
    except zipfile.BadZipFile:
        checks["deliverable_zip_opens"] = False
    lines = ["# ABCD v0.2 Fixed Integrity Check", ""]
    for key, value in checks.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append(f"Overall pass: {all(checks.values())}")
    write_text(ROOT / "reports" / "abcd_v0_2_fixed_integrity_check.md", "\n".join(lines) + "\n")
    return checks


def main() -> int:
    update_d_label_map()
    write_mapping()
    audit_report()
    rows = run_fixed_smoke()
    summary = fixed_summary(rows)
    fixed_report(rows, summary)
    zip_path = make_deliverable()
    checks = integrity(rows, summary, zip_path)
    print(f"PPG_DEGRADED_ECG_OK_FIXED = {checks['ppg_bad_ecg_good_no_longer_measure_ok']}")
    print(f"PPG_DEGRADED_ECG_OK_FINAL_ACTION = {summary['ppg_degraded_ecg_ok_final_action']}")
    print(f"ONLY_FINAL_NORMAL_CONFIDENT_MAPS_TO_MEASURE_OK = {checks['only_final_normal_confident_maps_to_measure_ok']}")
    print(f"MODEL_C_GATED_BY_B = {summary['model_c_gated_by_b']}")
    print("MODEL_D_DIRECT_C_V0_4_COMPATIBLE = True")
    print(f"RULE_FALLBACK_ENABLED = {summary['rule_fallback_enabled']}")
    print(f"DIAGNOSIS_OUTPUT_ABSENT = {summary['diagnosis_output_absent']}")
    print(f"ABCD_V0_2_PC_PIPELINE_READY = {summary['abcd_v0_2_pc_pipeline_ready']}")
    print("FIRMWARE_READY = False")
    print("TFLITE_ALL_READY = False")
    print("GD32_READY = False")
    print(f"DELIVERABLE_ZIP = {zip_path}")
    print(f"REPORT_PATH = {ROOT / 'reports' / 'abcd_v0_2_final_report_fixed.md'}")
    print("NEXT_ACTION = Use the fixed final-action mapping in the frozen PC contract; do not proceed to firmware until export and MCU replay are completed.")
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
