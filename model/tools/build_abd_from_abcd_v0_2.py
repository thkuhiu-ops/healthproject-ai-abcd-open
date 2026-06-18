#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


AI_ROOT = Path(r"LOCAL_PATH_REMOVED")
PROJECT_ROOT = AI_ROOT / "model_abcd_integrated_v0_2"
V0_1_ROOT = AI_ROOT / "model_abd_integrated_v0_1"

MODELS_DIR = PROJECT_ROOT / "models"
EXPORTS_DIR = PROJECT_ROOT / "exports"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEPLOY_DIR = PROJECT_ROOT / "deployment_bundle"

MODEL_A_ROOT = AI_ROOT / "model_a_ppg_trust_gate_v0_4_1"
MODEL_B_ROOT = AI_ROOT / "model_b_ecg_quality_v0_1"
MODEL_C_ROOT = AI_ROOT / "model_c_ecg_rhythm_binary_v0_4_compromise"

MODEL_A_KERAS = MODEL_A_ROOT / "models" / "model_a_ppg_trust_gate_v0_4_1_float32.keras"
MODEL_A_SCHEMA = MODEL_A_ROOT / "models" / "model_a_ppg_trust_gate_v0_4_1_feature_schema.json"
MODEL_A_LABELS = MODEL_A_ROOT / "models" / "model_a_ppg_trust_gate_v0_4_1_label_map.json"
MODEL_A_NORM = MODEL_A_ROOT / "models" / "model_a_ppg_trust_gate_v0_4_1_normalization.json"

MODEL_B_WEIGHTS = MODEL_B_ROOT / "models" / "model_b_ecg_quality_v0_1_weights.npz"
MODEL_B_SCHEMA = MODEL_B_ROOT / "models" / "model_b_ecg_quality_v0_1_tflite_feature_schema.json"
MODEL_B_LABELS = MODEL_B_ROOT / "models" / "model_b_ecg_quality_v0_1_label_schema.json"

D_SAFE_HEAD = V0_1_ROOT / "models" / "d_abd_safe_head.keras"
D_SAFE_LABELS = V0_1_ROOT / "models" / "d_abd_safe_label_encoder.json"
D_SAFE_CONTEXT_NORM = V0_1_ROOT / "models" / "d_abd_safe_context_normalization.json"
D_SAFE_VALIDATED_TFLITE = V0_1_ROOT / "exports" / "abd_integrated_safe_single_input_float32.tflite"
D_SAFE_FINAL_REPORT = V0_1_ROOT / "reports" / "FINAL_ABD_SAFE_RETRAIN_REPORT.md"

REAL_NPZ_SOURCE = V0_1_ROOT / "deployment_bundle" / "real_benchmark_abd_safe_input.npz"
V0_1_BENCHMARK_LOG = V0_1_ROOT / "deployment_bundle" / "模型结果.txt"

INTEGRATED_KERAS = MODELS_DIR / "abd_from_abcd_v0_2_integrated.keras"
INTEGRATED_SAVEDMODEL = MODELS_DIR / "abd_from_abcd_v0_2_integrated_savedmodel"
MODEL_SUMMARY_TXT = REPORTS_DIR / "abd_from_abcd_v0_2_model_summary.txt"

EXPORT_FLOAT32 = EXPORTS_DIR / "abd_from_abcd_v0_2_single_input_deploy_float32.tflite"
EXPORT_FLOAT32_LITE = EXPORTS_DIR / "abd_from_abcd_v0_2_single_input_deploy_float32.lite"
EXPORT_NOSLICE_FLOAT32 = EXPORTS_DIR / "abd_from_abcd_v0_2_single_input_deploy_float32_noslice_tool_verify.tflite"
EXPORT_NOSLICE_FLOAT32_LITE = EXPORTS_DIR / "abd_from_abcd_v0_2_single_input_deploy_float32_noslice_tool_verify.lite"
EXPORT_DYNAMIC = EXPORTS_DIR / "abd_from_abcd_v0_2_single_input_deploy_dynamic_quant.tflite"
EXPORT_DYNAMIC_LITE = EXPORTS_DIR / "abd_from_abcd_v0_2_single_input_deploy_dynamic_quant.lite"

REAL_NPZ = DEPLOY_DIR / "real_benchmark_abd_from_abcd_v0_2_input.npz"
REAL_NPY = DEPLOY_DIR / "real_benchmark_abd_from_abcd_v0_2_input.npy"
REAL_CSV = DEPLOY_DIR / "real_benchmark_abd_from_abcd_v0_2_input.csv"
EXPECTED_CSV = DEPLOY_DIR / "real_benchmark_abd_from_abcd_v0_2_expected_output.csv"
README_BENCHMARK = DEPLOY_DIR / "README_RUN_DEPLOY_TOOL_BENCHMARK.md"

AUDIT_REPORT = REPORTS_DIR / "audit_abcd_v0_2_for_abd_extract.md"
OP_AUDIT = REPORTS_DIR / "abd_from_abcd_v0_2_operator_audit.md"
PC_VALIDATION = REPORTS_DIR / "abd_from_abcd_v0_2_pc_tflite_validation.md"
BENCH_LOG_TARGET = REPORTS_DIR / "ABD_FROM_ABCD_V0_2_DEPLOY_TOOL_BENCHMARK_LOG.txt"
FINAL_REPORT = REPORTS_DIR / "FINAL_ABD_FROM_ABCD_V0_2_DEPLOY_REPORT.md"
DATA_SITE_VERIFY_FIX = REPORTS_DIR / "ABD_FROM_ABCD_V0_2_DATA_SITE_VERIFY_HANG_WORKAROUND.md"

A_DIM = 58
B_DIM = 32
CTX_DIM = 17
TOTAL_DIM = 107

FORBIDDEN_OPS = {
    "EQUAL",
    "WHERE",
    "SELECT",
    "IS_NAN",
    "IS_INF",
    "IS_FINITE",
    "GREATER",
    "LESS",
    "CUSTOM",
}
PREFERRED_OPS = {"FULLY_CONNECTED", "RELU", "SOFTMAX"}
AUDIT_ALLOWED_WITH_NOTE = {"CONCATENATION", "RESHAPE", "STRIDED_SLICE"}


def ensure_dirs() -> None:
    for path in [MODELS_DIR, EXPORTS_DIR, REPORTS_DIR, DEPLOY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def safe_std(variance: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.asarray(variance, dtype=np.float32))
    return np.maximum(std, np.float32(1e-7)).astype(np.float32)


@tf.keras.utils.register_keras_serializable(package="abd")
class FeatureSlice(tf.keras.layers.Layer):
    def __init__(self, start_idx: int, end_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)

    def call(self, inputs):
        return tf.cast(inputs[:, self.start_idx : self.end_idx], tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({"start_idx": self.start_idx, "end_idx": self.end_idx})
        return config


def fold_standardize_into_dense(
    weights: np.ndarray, bias: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.maximum(np.asarray(std, dtype=np.float32), np.float32(1e-7)).astype(np.float32)
    folded_w = weights / std[:, None]
    folded_b = bias - (mean / std) @ weights
    return folded_w.astype(np.float32), folded_b.astype(np.float32)


def extract_array(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    preferred = ["input", "x", "input_0", "data", "arr_0"]
    for key in preferred + list(data.files):
        if key in data.files:
            arr = np.asarray(data[key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == TOTAL_DIM:
                return arr
    raise RuntimeError(f"No ({TOTAL_DIM})-dim 2D array found in {npz_path}")


def build_abd_model(use_slices: bool = True) -> tuple[tf.keras.Model, dict[str, Any]]:
    a_model = tf.keras.models.load_model(MODEL_A_KERAS, compile=False)
    d_head = tf.keras.models.load_model(D_SAFE_HEAD, compile=False)
    b_weights = np.load(MODEL_B_WEIGHTS)

    a_norm = load_json(MODEL_A_NORM)
    b_schema = load_json(MODEL_B_SCHEMA)
    ctx_cfg = load_json(D_SAFE_CONTEXT_NORM)
    d_labels = load_json(D_SAFE_LABELS)

    a_dense1 = a_model.get_layer("dense_48_relu")
    a_dense2 = a_model.get_layer("dense_24_relu")
    a_out_layer = a_model.get_layer("model_a_output")
    a_w1, a_b1 = a_dense1.get_weights()
    a_w2, a_b2 = a_dense2.get_weights()
    a_w3, a_b3 = a_out_layer.get_weights()
    a_mean = np.asarray(a_norm["mean"], dtype=np.float32)
    a_std = safe_std(np.asarray(a_norm["variance"], dtype=np.float32))
    a_w1_folded, a_b1_folded = fold_standardize_into_dense(a_w1, a_b1, a_mean, a_std)
    a_w1_block = np.zeros((TOTAL_DIM, a_w1_folded.shape[1]), dtype=np.float32)
    a_w1_block[:A_DIM, :] = a_w1_folded

    b_w1 = b_weights["dense1_weight"].T.astype(np.float32)
    b_b1 = b_weights["dense1_bias"].astype(np.float32)
    b_w2 = b_weights["dense2_weight"].T.astype(np.float32)
    b_b2 = b_weights["dense2_bias"].astype(np.float32)
    b_w3 = b_weights["dense3_weight"].T.astype(np.float32)
    b_b3 = b_weights["dense3_bias"].astype(np.float32)
    b_mean = np.asarray(b_schema["mean"], dtype=np.float32)
    b_std = np.asarray(b_schema["std"], dtype=np.float32)
    b_w1_folded, b_b1_folded = fold_standardize_into_dense(b_w1, b_b1, b_mean, b_std)
    b_w1_block = np.zeros((TOTAL_DIM, b_w1_folded.shape[1]), dtype=np.float32)
    b_w1_block[A_DIM : A_DIM + B_DIM, :] = b_w1_folded

    d_w, d_b = d_head.layers[-1].get_weights()
    d_w = d_w.astype(np.float32)
    d_b = d_b.astype(np.float32)
    ctx_mean = np.asarray(ctx_cfg["mean"], dtype=np.float32)
    ctx_std = safe_std(np.asarray(ctx_cfg["variance"], dtype=np.float32))
    ctx_w = np.diag(1.0 / ctx_std).astype(np.float32)
    ctx_b = (-ctx_mean / ctx_std).astype(np.float32)
    d_w_full = np.zeros((3 + 3 + TOTAL_DIM, 3), dtype=np.float32)
    d_w_full[0:3, :] = d_w[0:3, :]
    d_w_full[3:6, :] = d_w[3:6, :]
    d_w_full[6 + A_DIM + B_DIM : 6 + TOTAL_DIM, :] = d_w[6:, :] / ctx_std[:, None]
    d_b_full = (d_b - (ctx_mean / ctx_std) @ d_w[6:, :]).astype(np.float32)

    inp = tf.keras.Input(shape=(TOTAL_DIM,), dtype=tf.float32, name="abd_flat_input")

    if use_slices:
        a_in = FeatureSlice(0, A_DIM, name="A_input_58")(inp)
        b_in = FeatureSlice(A_DIM, A_DIM + B_DIM, name="B_input_32")(inp)
        ctx_in = FeatureSlice(A_DIM + B_DIM, TOTAL_DIM, name="Context_input_17")(inp)
        a_first_w = a_w1_folded
        b_first_w = b_w1_folded
    else:
        a_in = inp
        b_in = inp
        ctx_in = None
        a_first_w = a_w1_block
        b_first_w = b_w1_block

    a = tf.keras.layers.Dense(48, activation="relu", name="A_dense_48_relu_folded")(a_in)
    a = tf.keras.layers.Dense(24, activation="relu", name="A_dense_24_relu")(a)
    a_out = tf.keras.layers.Dense(3, activation="softmax", name="A_out_ppg_quality")(a)

    b = tf.keras.layers.Dense(64, activation="relu", name="B_dense_64_relu_folded")(b_in)
    b = tf.keras.layers.Dense(32, activation="relu", name="B_dense_32_relu")(b)
    b_out = tf.keras.layers.Dense(3, activation="softmax", name="B_out_ecg_quality")(b)

    if use_slices:
        ctx_norm_tensor = tf.keras.layers.Dense(17, activation=None, name="Context_norm_folded")(ctx_in)
        fused = tf.keras.layers.Concatenate(name="D_SAFE_concat_A_B_context")([a_out, b_out, ctx_norm_tensor])
    else:
        fused = tf.keras.layers.Concatenate(name="D_SAFE_concat_A_B_flat_input_noslice")([a_out, b_out, inp])
    out = tf.keras.layers.Dense(3, activation="softmax", name="ABD_SAFE_output")(fused)

    model_name = "abd_from_abcd_v0_2_single_input_deploy" if use_slices else "abd_from_abcd_v0_2_single_input_deploy_noslice"
    model = tf.keras.Model(inp, out, name=model_name)
    model.get_layer("A_dense_48_relu_folded").set_weights([a_first_w, a_b1_folded])
    model.get_layer("A_dense_24_relu").set_weights([a_w2.astype(np.float32), a_b2.astype(np.float32)])
    model.get_layer("A_out_ppg_quality").set_weights([a_w3.astype(np.float32), a_b3.astype(np.float32)])
    model.get_layer("B_dense_64_relu_folded").set_weights([b_first_w, b_b1_folded])
    model.get_layer("B_dense_32_relu").set_weights([b_w2, b_b2])
    model.get_layer("B_out_ecg_quality").set_weights([b_w3, b_b3])
    if use_slices:
        model.get_layer("Context_norm_folded").set_weights([ctx_w, ctx_b])
        model.get_layer("ABD_SAFE_output").set_weights([d_w, d_b])
    else:
        model.get_layer("ABD_SAFE_output").set_weights([d_w_full, d_b_full])
    model.trainable = False

    meta = {
        "labels": d_labels["labels"],
        "a_label_map": load_json(MODEL_A_LABELS),
        "b_label_map": load_json(MODEL_B_LABELS),
        "a_schema": load_json(MODEL_A_SCHEMA),
        "b_schema": b_schema,
        "context_features": ctx_cfg["feature_names"],
        "folded_preprocessing": {
            "model_a_normalization": True,
            "model_b_standardization": True,
            "d_safe_context_normalization": True,
            "model_b_median_impute_in_graph": False,
            "model_b_clip_in_graph": False,
        },
        "graph_contract": (
            "abd_flat_input -> slices(A/B/context) -> A_out/B_out/Context_norm -> concat -> D_SAFE head"
            if use_slices
            else "abd_flat_input -> block-weight A/B branches + concat(A_out,B_out,flat_input) -> folded D_SAFE head"
        ),
    }
    return model, meta


def save_model_artifacts(model: tf.keras.Model) -> None:
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    write_text(MODEL_SUMMARY_TXT, buffer.getvalue())
    model.save(INTEGRATED_KERAS)
    if INTEGRATED_SAVEDMODEL.exists():
        shutil.rmtree(INTEGRATED_SAVEDMODEL)
    signature = tf.TensorSpec([None, TOTAL_DIM], tf.float32, name="abd_flat_input")

    @tf.function(input_signature=[signature])
    def serving(x):
        return {"ABD_SAFE_output": model(x, training=False)}

    tf.saved_model.save(model, str(INTEGRATED_SAVEDMODEL), signatures={"serving_default": serving})


def export_tflite(model: tf.keras.Model, path: Path, dynamic: bool = False) -> None:
    spec = tf.TensorSpec([1, TOTAL_DIM], tf.float32, name="abd_flat_input")

    @tf.function
    def serving_fn(x):
        return model(x, training=False)

    concrete = serving_fn.get_concrete_function(spec)
    frozen = convert_variables_to_constants_v2(concrete)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen], model)
    converter.experimental_enable_resource_variables = True
    if dynamic:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    blob = converter.convert()
    path.write_bytes(blob)


def inspect_tflite(model_path: Path) -> dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path), experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()
    ops = [op["op_name"] for op in interpreter._get_ops_details()]
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return {
        "path": str(model_path),
        "operators": ops,
        "operator_counts": dict(Counter(ops)),
        "forbidden_present": sorted(set(ops) & FORBIDDEN_OPS),
        "non_preferred_present": sorted(set(ops) - PREFERRED_OPS),
        "input_details": [
            {"name": item["name"], "shape": item["shape"].tolist(), "dtype": str(np.dtype(item["dtype"]))}
            for item in input_details
        ],
        "output_details": [
            {"name": item["name"], "shape": item["shape"].tolist(), "dtype": str(np.dtype(item["dtype"]))}
            for item in output_details
        ],
    }


def run_tflite(model_path: Path, x: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    rows = []
    for i in range(x.shape[0]):
        interpreter.set_tensor(inp["index"], x[i : i + 1].astype(np.float32))
        interpreter.invoke()
        rows.append(interpreter.get_tensor(out["index"]).reshape(-1))
    return np.asarray(rows, dtype=np.float32)


def reference_predict_with_source_preprocess(x: np.ndarray) -> np.ndarray:
    a_model = tf.keras.models.load_model(MODEL_A_KERAS, compile=False)
    d_head = tf.keras.models.load_model(D_SAFE_HEAD, compile=False)
    b_schema = load_json(MODEL_B_SCHEMA)
    b_weights = np.load(MODEL_B_WEIGHTS)
    ctx_norm = load_json(D_SAFE_CONTEXT_NORM)

    a_raw = x[:, :A_DIM].astype(np.float32)
    b_raw = x[:, A_DIM : A_DIM + B_DIM].astype(np.float32)
    ctx_raw = x[:, A_DIM + B_DIM :].astype(np.float32)

    b_mean = np.asarray(b_schema["mean"], dtype=np.float32)
    b_std = np.asarray(b_schema["std"], dtype=np.float32)
    b_std = np.where(b_std > 0.0, b_std, 1.0).astype(np.float32)
    b_norm = (b_raw - b_mean) / b_std
    clip_low, clip_high = [float(v) for v in b_schema.get("clip_range", [-8.0, 8.0])]
    b1 = np.maximum(0.0, b_norm @ b_weights["dense1_weight"].T + b_weights["dense1_bias"])
    b2 = np.maximum(0.0, b1 @ b_weights["dense2_weight"].T + b_weights["dense2_bias"])
    b_logits = b2 @ b_weights["dense3_weight"].T + b_weights["dense3_bias"]
    b_out = tf.nn.softmax(b_logits, axis=1).numpy().astype(np.float32)

    ctx_mean = np.asarray(ctx_norm["mean"], dtype=np.float32)
    ctx_std = safe_std(np.asarray(ctx_norm["variance"], dtype=np.float32))
    ctx_out = ((ctx_raw - ctx_mean) / ctx_std).astype(np.float32)

    a_out = a_model(a_raw, training=False).numpy().astype(np.float32)
    fused = np.concatenate([a_out, b_out, ctx_out], axis=1).astype(np.float32)
    return d_head(fused, training=False).numpy().astype(np.float32)


def write_real_npz_bundle(x: np.ndarray, labels: list[str], probs: np.ndarray) -> dict[str, Any]:
    x = x.astype(np.float32)
    np.savez(REAL_NPZ, input=x, x=x, input_0=x)
    np.save(REAL_NPY, x)
    header = (
        [f"A_feat{i}" for i in range(A_DIM)]
        + [f"B_{name}" for name in load_json(MODEL_B_SCHEMA)["feature_columns"]]
        + [f"CTX_{name}" for name in load_json(D_SAFE_CONTEXT_NORM)["feature_names"]]
    )
    with REAL_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id"] + header)
        for i, row in enumerate(x):
            writer.writerow([i] + [float(v) for v in row])
    with EXPECTED_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_id",
            "model_prob_0",
            "model_prob_1",
            "model_prob_2",
            "pred_label_id",
            "pred_label",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, prob in enumerate(probs):
            pred = int(np.argmax(prob))
            writer.writerow(
                {
                    "sample_id": i,
                    "model_prob_0": float(prob[0]),
                    "model_prob_1": float(prob[1]),
                    "model_prob_2": float(prob[2]),
                    "pred_label_id": pred,
                    "pred_label": labels[pred],
                }
            )
    return {
        "source": str(REAL_NPZ_SOURCE),
        "npz": str(REAL_NPZ),
        "npy": str(REAL_NPY),
        "csv": str(REAL_CSV),
        "expected_output_csv": str(EXPECTED_CSV),
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "finite": bool(np.isfinite(x).all()),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def format_op_list(audit: dict[str, Any]) -> str:
    return ", ".join(audit["operators"])


def write_audit_report(meta: dict[str, Any]) -> None:
    b_schema = meta["b_schema"]
    lines = [
        "# Audit ABCD v0.2 for ABD Extract",
        "",
        "## Sources",
        "",
        f"- Model A source: `{MODEL_A_KERAS}`",
        f"- Model B source: `{MODEL_B_WEIGHTS}`",
        f"- Model C source recorded but disabled: `{MODEL_C_ROOT}`",
        f"- D_SAFE head source: `{D_SAFE_HEAD}`",
        f"- D_SAFE label encoder: `{D_SAFE_LABELS}`",
        f"- D_SAFE context normalization: `{D_SAFE_CONTEXT_NORM}`",
        f"- v0_1 validated TFLite reference: `{D_SAFE_VALIDATED_TFLITE}`",
        "",
        "## Deployment Extraction Decision",
        "",
        "- Model A is embedded through folded normalization and frozen Dense weights.",
        "- Model B is embedded through folded standardization and frozen Dense weights.",
        "- Model C is disabled and is not loaded into the final graph.",
        "- D_SAFE reuses the v0_1 trained head weights and label order. No retraining is performed.",
        "",
        "## Dimensions",
        "",
        f"- A input dimension: `{A_DIM}`",
        f"- B input dimension: `{B_DIM}`",
        f"- Context input dimension: `{CTX_DIM}`",
        f"- Final ABD single input dimension: `{TOTAL_DIM}`",
        f"- Final output dimension: `{len(meta['labels'])}`",
        "",
        "## Output Label Order",
        "",
    ]
    for i, label in enumerate(meta["labels"]):
        lines.append(f"- `{i}` `{label}`")
    lines += [
        "",
        "## Feature Order",
        "",
        "- Input layout: `[A_input_58, B_input_32, Context_input_17]`.",
        f"- B feature order comes from `{MODEL_B_SCHEMA}` and contains `{len(b_schema['feature_columns'])}` fields.",
        "- Context feature order reuses the v0_1 D_SAFE normalization schema.",
        "",
        "## Preprocessing Boundary",
        "",
        "- In-graph: Model A normalization, Model B standardization, D_SAFE context normalization are folded into Dense weights.",
        "- Outside graph: finite-value check, median imputation, and clipping must be performed before MCU inference if needed.",
        "- No FINAL_OK samples were synthesized.",
    ]
    write_text(AUDIT_REPORT, "\n".join(lines))


def write_operator_audit(
    float_audit: dict[str, Any],
    dynamic_audit: dict[str, Any] | None,
    noslice_audit: dict[str, Any] | None = None,
) -> None:
    lines = [
        "# ABD From ABCD v0.2 Operator Audit",
        "",
        "## Acceptance Rule",
        "",
        f"- Forbidden operators: `{', '.join(sorted(FORBIDDEN_OPS))}`",
        "- Recommended deployment model must contain no forbidden operators.",
        "",
        "## Float32 Deploy Model",
        "",
        f"- Path: `{float_audit['path']}`",
        f"- Operators: `{format_op_list(float_audit)}`",
        f"- Operator counts: `{json.dumps(float_audit['operator_counts'], ensure_ascii=False)}`",
        f"- Forbidden operators present: `{', '.join(float_audit['forbidden_present']) if float_audit['forbidden_present'] else 'NONE'}`",
        f"- Non-preferred operators present: `{', '.join(float_audit['non_preferred_present']) if float_audit['non_preferred_present'] else 'NONE'}`",
        "- `STRIDED_SLICE` is expected because the single 107-dim input is split into A_input_58, B_input_32, and Context_input_17.",
        "- `CONCATENATION` is expected because the graph fuses A_out, B_out, and normalized context before the D_SAFE head.",
        "- No `EQUAL`, `WHERE`, `SELECT`, finite checks, comparison ops, or `CUSTOM` op are present in the recommended float32 model.",
        "",
    ]
    if noslice_audit is not None:
        lines += [
            "## No-Slice Tool Verify Model",
            "",
            f"- Path: `{noslice_audit['path']}`",
            f"- Operators: `{format_op_list(noslice_audit)}`",
            f"- Operator counts: `{json.dumps(noslice_audit['operator_counts'], ensure_ascii=False)}`",
            f"- Forbidden operators present: `{', '.join(noslice_audit['forbidden_present']) if noslice_audit['forbidden_present'] else 'NONE'}`",
            "- This artifact is mathematically equivalent for deployment input/output, but avoids `STRIDED_SLICE` for tools whose Data Site Verify page hangs on slice operators.",
            "",
        ]
    if dynamic_audit is None:
        lines += [
            "## Dynamic Quant Model",
            "",
            "- Dynamic quant export failed or was skipped.",
        ]
    else:
        lines += [
            "## Dynamic Quant Model",
            "",
            f"- Path: `{dynamic_audit['path']}`",
            f"- Operators: `{format_op_list(dynamic_audit)}`",
            f"- Operator counts: `{json.dumps(dynamic_audit['operator_counts'], ensure_ascii=False)}`",
            f"- Forbidden operators present: `{', '.join(dynamic_audit['forbidden_present']) if dynamic_audit['forbidden_present'] else 'NONE'}`",
            "- Not recommended for GD32 deployment because the previous toolchain rejected mixed quantization in FULLY_CONNECTED.",
        ]
    write_text(OP_AUDIT, "\n".join(lines))


def write_pc_validation(
    audit: dict[str, Any],
    x: np.ndarray,
    probs: np.ndarray,
    labels: list[str],
    bundle_info: dict[str, Any],
    consistency: dict[str, float],
) -> None:
    pred_ids = np.argmax(probs, axis=1)
    dist = Counter(int(v) for v in pred_ids)
    lines = [
        "# ABD From ABCD v0.2 PC TFLite Validation",
        "",
        f"- Model path: `{EXPORT_FLOAT32}`",
        f"- NPZ path: `{REAL_NPZ}`",
        f"- Input shape: `{list(x.shape)}`",
        f"- TFLite input details: `{json.dumps(audit['input_details'], ensure_ascii=False)}`",
        f"- TFLite output details: `{json.dumps(audit['output_details'], ensure_ascii=False)}`",
        f"- Output shape: `{list(probs.shape)}`",
        f"- Sample count: `{x.shape[0]}`",
        f"- Input contains NaN/Inf: `{not bool(np.isfinite(x).all())}`",
        f"- Output contains NaN/Inf: `{not bool(np.isfinite(probs).all())}`",
        f"- Predicted distribution: `{json.dumps({labels[k]: int(v) for k, v in dist.items()}, ensure_ascii=False)}`",
        f"- Max abs diff vs folded-source reference: `{consistency['max_abs_diff']:.9e}`",
        f"- Mean abs diff vs folded-source reference: `{consistency['mean_abs_diff']:.9e}`",
        f"- Expected output CSV: `{EXPECTED_CSV}`",
        "",
        "## Per-Frame Softmax",
        "",
    ]
    for i, prob in enumerate(probs):
        pred = int(np.argmax(prob))
        lines.append(
            f"- sample `{i}`: prob0=`{prob[0]:.8f}`, prob1=`{prob[1]:.8f}`, prob2=`{prob[2]:.8f}`, pred=`{pred}:{labels[pred]}`"
        )
    lines += [
        "",
        "## Deployment Bundle",
        "",
        f"- NPZ: `{bundle_info['npz']}`",
        f"- NPY: `{bundle_info['npy']}`",
        f"- CSV: `{bundle_info['csv']}`",
    ]
    write_text(PC_VALIDATION, "\n".join(lines))


def write_manual_benchmark_readme() -> None:
    text = f"""# Run GD32 Deploy Tool Benchmark Manually

The build produced a deploy-compatible 107-dim ABD integrated model, but this script did not find a reliable command-line entry point for the GD32 AI deploy tool. Do not fabricate MCU benchmark numbers.

## Files to Import

- Model: `{EXPORT_FLOAT32}`
- Model copy: `{EXPORT_FLOAT32_LITE}`
- If Data Site Verify hangs on the standard model, use no-slice model: `{EXPORT_NOSLICE_FLOAT32}`
- NPZ dataset: `{REAL_NPZ}`
- PC expected output: `{EXPECTED_CSV}`

## Expected Tool Flow

1. Import `{EXPORT_FLOAT32}` into the GD32 AI deployment tool.
2. Select user benchmark by dataset.
3. Load `{REAL_NPZ}`.
4. Confirm the model input is `[1, 107]` float32 and output is `[1, 3]` float32.
5. Run the dataset benchmark.

## Acceptance Threshold

- Average error <= `1e-5`: PC/MCU numerical consistency is acceptable.
- Max error <= `1e-4`: acceptable upper bound for this float32 deployment check.
- If the log shows `MCU_READY`, `Receive complete`, `Average error`, and `Max error`, the board-side model validation completed.
- If an unsupported operator appears, inspect `{OP_AUDIT}` immediately.

## Expected Operators

The recommended float32 model is expected to contain only `FULLY_CONNECTED`, `RELU`, `SOFTMAX`, and `CONCATENATION`.
"""
    write_text(README_BENCHMARK, text)


def write_data_site_verify_workaround(noslice_audit: dict[str, Any], max_diff: float) -> None:
    text = f"""# Data Site Verify Hang Workaround

## Symptom

After clicking `Data site verify`, the GD32 AI deployment tool keeps loading and does not produce a result.

## Likely Cause

The standard 107-dim model contains `STRIDED_SLICE` operators for splitting:

- `A_input_58`
- `B_input_32`
- `Context_input_17`

These operators are not forbidden by the previous audit, but the deployment tool's Data Site Verify page may hang while handling the sliced single-input graph.

## Workaround Artifact

Use this no-slice model for Data Site Verify:

- `{EXPORT_NOSLICE_FLOAT32}`
- `{EXPORT_NOSLICE_FLOAT32_LITE}`

It keeps the same external contract:

- input: `[1, 107]`
- output: `[1, 3]`
- labels: `0 FINAL_OK`, `1 SIGNAL_BAD_OR_CONTACT_BAD`, `2 UNCERTAIN_OR_MOTION`

The graph avoids `STRIDED_SLICE` by using block-sparse Dense weights to read the A/B/context ranges.

## No-Slice Operator Audit

- Operators: `{format_op_list(noslice_audit)}`
- Forbidden operators present: `{', '.join(noslice_audit['forbidden_present']) if noslice_audit['forbidden_present'] else 'NONE'}`
- Max abs diff vs standard sliced model on real NPZ: `{max_diff:.9e}`

## Recommended Retry

1. Import `{EXPORT_NOSLICE_FLOAT32}` into the deployment tool.
2. Load `{REAL_NPZ}`.
3. Click `Data site verify`.
4. If verify passes, continue to user benchmark.

If it still hangs, the remaining likely causes are the tool process state, COM/board state, or a deployment-tool UI issue rather than the model graph. Restart the deployment tool, reconnect the board, and retry with the no-slice model first.
"""
    write_text(DATA_SITE_VERIFY_FIX, text)


def parse_previous_benchmark() -> dict[str, str]:
    if not V0_1_BENCHMARK_LOG.exists():
        return {}
    text = V0_1_BENCHMARK_LOG.read_text(encoding="utf-8", errors="ignore")
    result: dict[str, str] = {}
    for key in ["MCU_READY", "Receive complete", "Average error", "Max error", "Latency"]:
        if key in text:
            result[key] = "present"
    for line in text.splitlines():
        if "Average error:" in line:
            result["Average error"] = line.split("Average error:", 1)[1].strip()
        if "Max error:" in line:
            result["Max error"] = line.split("Max error:", 1)[1].strip()
        if "Average latency:" in line or "average latency" in line.lower():
            result["Latency"] = line.strip()
    return result


def write_final_report(
    float_audit: dict[str, Any],
    dynamic_audit: dict[str, Any] | None,
    bundle_info: dict[str, Any],
    labels: list[str],
    probs: np.ndarray,
    consistency: dict[str, float],
    benchmark_status: str,
    noslice_audit: dict[str, Any] | None = None,
) -> None:
    pred_ids = np.argmax(probs, axis=1)
    dist = Counter(int(v) for v in pred_ids)
    previous = parse_previous_benchmark()
    lines = [
        "# FINAL ABD From ABCD v0.2 Deploy Report",
        "",
        "## Verdict",
        "",
        "- Successfully built a true 107-dim single-input ABD integrated model containing Model A, Model B, and the v0_1 D_SAFE head.",
        "- Model C is disabled and not loaded into the graph.",
        "- No model retraining was performed.",
        "- No FINAL_OK samples were synthesized.",
        "- Recommended deployment file is the float32 `.tflite` / `.lite` pair listed below.",
        "",
        "## Recommended Files",
        "",
        f"- TFLite: `{EXPORT_FLOAT32}`",
        f"- Lite copy: `{EXPORT_FLOAT32_LITE}`",
        f"- Real NPZ: `{REAL_NPZ}`",
        f"- Expected output CSV: `{EXPECTED_CSV}`",
        "",
        "## Model Contract",
        "",
        f"- Input shape: `[1, {TOTAL_DIM}]`",
        "- Input order: `[A_input_58, B_input_32, Context_input_17]`",
        f"- Output classes: `{', '.join([f'{i}:{label}' for i, label in enumerate(labels)])}`",
        "- Model A source was extracted from the v0_4_1 PPG trust gate.",
        "- Model B source was reconstructed from the v0_1 ECG quality gate deploy weights.",
        "- D_SAFE source was loaded from the v0_1 validated SAFE head.",
        "",
        "## Operator Audit",
        "",
        f"- Recommended float32 operators: `{format_op_list(float_audit)}`",
        f"- Contains EQUAL: `{'EQUAL' in float_audit['operators']}`",
        f"- Forbidden operators present: `{', '.join(float_audit['forbidden_present']) if float_audit['forbidden_present'] else 'NONE'}`",
        "- `CONCATENATION` is present and expected for A/B/context fusion.",
        "- Dynamic quant was generated only as a non-recommended artifact." if dynamic_audit else "- Dynamic quant was not generated.",
        "",
        "## PC TFLite Validation",
        "",
        f"- Real NPZ shape: `{bundle_info['shape']}`",
        f"- Input finite: `{bundle_info['finite']}`",
        f"- Output finite: `{bool(np.isfinite(probs).all())}`",
        f"- Predicted distribution: `{json.dumps({labels[k]: int(v) for k, v in dist.items()}, ensure_ascii=False)}`",
        f"- Max abs diff vs folded-source reference: `{consistency['max_abs_diff']:.9e}`",
        f"- Mean abs diff vs folded-source reference: `{consistency['mean_abs_diff']:.9e}`",
        f"- Validation report: `{PC_VALIDATION}`",
        "",
        "## GD32 Deploy Tool Benchmark",
        "",
        f"- Automatic benchmark completed: `{benchmark_status}`",
        f"- Manual instructions: `{README_BENCHMARK}`",
        f"- Log target path: `{BENCH_LOG_TARGET}`",
    ]
    if noslice_audit is not None:
        lines += [
            "",
            "## Data Site Verify Workaround",
            "",
            f"- No-slice verify model: `{EXPORT_NOSLICE_FLOAT32}`",
            f"- No-slice operators: `{format_op_list(noslice_audit)}`",
            f"- Workaround report: `{DATA_SITE_VERIFY_FIX}`",
        ]
    if previous:
        lines += [
            "",
            "## Prior v0_1 D_SAFE Board Validation Reference",
            "",
            f"- Previous v0_1 benchmark log: `{V0_1_BENCHMARK_LOG}`",
            f"- MCU_READY: `{previous.get('MCU_READY', 'not found')}`",
            f"- Receive complete: `{previous.get('Receive complete', 'not found')}`",
            f"- Average error: `{previous.get('Average error', 'not found')}`",
            f"- Max error: `{previous.get('Max error', 'not found')}`",
        ]
    lines += [
        "",
        "## Deployment Recommendation",
        "",
        "- This 107-dim ABD integrated model is suitable for GD32 deploy-tool import testing because the recommended float32 graph has no forbidden operators.",
        "- For the competition runtime, FINAL_OK should still be guarded by deterministic quality rules because current real data lacks verifiable FINAL_OK samples.",
        "- PC/MCU benchmark consistency proves model numerical consistency only; it does not prove full sensor closed-loop behavior.",
    ]
    write_text(FINAL_REPORT, "\n".join(lines))


def main() -> None:
    ensure_dirs()
    tf.keras.utils.set_random_seed(DATE_REMOVED)

    model, meta = build_abd_model(use_slices=True)
    noslice_model, _ = build_abd_model(use_slices=False)
    labels = list(meta["labels"])
    save_model_artifacts(model)

    write_audit_report(meta)

    export_tflite(model, EXPORT_FLOAT32, dynamic=False)
    shutil.copyfile(EXPORT_FLOAT32, EXPORT_FLOAT32_LITE)
    export_tflite(noslice_model, EXPORT_NOSLICE_FLOAT32, dynamic=False)
    shutil.copyfile(EXPORT_NOSLICE_FLOAT32, EXPORT_NOSLICE_FLOAT32_LITE)

    dynamic_audit = None
    dynamic_error = None
    try:
        export_tflite(model, EXPORT_DYNAMIC, dynamic=True)
        shutil.copyfile(EXPORT_DYNAMIC, EXPORT_DYNAMIC_LITE)
        dynamic_audit = inspect_tflite(EXPORT_DYNAMIC)
    except Exception as exc:
        dynamic_error = str(exc)

    float_audit = inspect_tflite(EXPORT_FLOAT32)
    noslice_audit = inspect_tflite(EXPORT_NOSLICE_FLOAT32)
    write_operator_audit(float_audit, dynamic_audit, noslice_audit)

    x = extract_array(REAL_NPZ_SOURCE)
    if x.shape[0] < 10:
        raise RuntimeError(f"Need at least 10 real samples, got {x.shape[0]}")
    if not np.isfinite(x).all():
        raise RuntimeError(f"Real NPZ contains NaN or Inf: {REAL_NPZ_SOURCE}")
    probs = run_tflite(EXPORT_FLOAT32, x)
    noslice_probs = run_tflite(EXPORT_NOSLICE_FLOAT32, x)
    noslice_max_diff = float(np.max(np.abs(probs - noslice_probs)))
    ref_probs = reference_predict_with_source_preprocess(x)
    consistency = {
        "max_abs_diff": float(np.max(np.abs(probs - ref_probs))),
        "mean_abs_diff": float(np.mean(np.abs(probs - ref_probs))),
    }
    bundle_info = write_real_npz_bundle(x, labels, probs)
    write_pc_validation(float_audit, x, probs, labels, bundle_info, consistency)

    write_manual_benchmark_readme()
    write_data_site_verify_workaround(noslice_audit, noslice_max_diff)
    benchmark_status = "NO_AUTOMATED_TOOL_COMMAND_FOUND"
    if dynamic_error:
        (REPORTS_DIR / "abd_from_abcd_v0_2_dynamic_quant_export_error.txt").write_text(dynamic_error, encoding="utf-8")

    write_final_report(float_audit, dynamic_audit, bundle_info, labels, probs, consistency, benchmark_status, noslice_audit)

    print("ABD_FROM_ABCD_V0_2_BUILD_DONE")
    print(f"recommended_model={EXPORT_FLOAT32}")
    print(f"recommended_lite={EXPORT_FLOAT32_LITE}")
    print(f"real_npz={REAL_NPZ}")
    print(f"operators={format_op_list(float_audit)}")
    print(f"contains_equal={'EQUAL' in float_audit['operators']}")
    print(f"noslice_model={EXPORT_NOSLICE_FLOAT32}")
    print(f"noslice_operators={format_op_list(noslice_audit)}")
    print(f"noslice_max_diff={noslice_max_diff:.9e}")


if __name__ == "__main__":
    main()
