from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL = Path(os.environ.get("HP_ABCD_MODEL_DIR", ROOT.parent / "model_abcd_integrated_v0_2"))
SOURCE_TRAIN = Path(os.environ.get("HP_ABCD_TRAIN_DIR", ROOT.parent / "source_train"))
SOURCE_TEST = Path(os.environ.get("HP_ABCD_TEST_DIR", ROOT.parent / "source_test"))

TEXT_SUFFIXES = {
    ".c",
    ".csv",
    ".h",
    ".json",
    ".ld",
    ".md",
    ".py",
    ".s",
    ".S",
    ".txt",
    ".xml",
}

MODEL_COPY_ROOTS = [
    ("contracts", "model/contracts"),
    ("figures", "model/figures"),
    ("json", "model/json"),
    ("metrics_pack_verified", "model/metrics_pack_verified"),
    ("model_registry", "model/model_registry"),
    ("models", "model/models"),
    ("reports", "model/reports"),
    ("schemas", "model/schemas"),
    ("tables", "model/tables"),
    ("tools", "model/tools"),
    ("GD_Embedded_AI/Core/Include", "embedded/GD_Embedded_AI/Core/Include"),
    ("GD_Embedded_AI/Core/Source", "embedded/GD_Embedded_AI/Core/Source"),
    ("GD_Embedded_AI/User_model/cur_tflite", "embedded/GD_Embedded_AI/User_model/cur_tflite"),
    ("GD_Embedded_AI/Utilities", "embedded/GD_Embedded_AI/Utilities"),
]

SKIP_DIR_NAMES = {
    "__pycache__",
    ".git",
    "output",
    "list",
    "Objects",
    "Listings",
}

SKIP_SUFFIXES = {
    ".Administrator",
    ".axf",
    ".bak",
    ".dep",
    ".d",
    ".exe",
    ".htm",
    ".lnp",
    ".log",
    ".map",
    ".o",
    ".pack",
    ".pdf",
    ".pyc",
    ".rar",
    ".uvguix",
    ".zip",
}


def sanitize_text(text: str) -> str:
    text = re.sub(r"(?<![A-Za-z0-9])S(\d{2})(?![A-Za-z0-9])", lambda m: f"P{int(m.group(1)):03d}", text)
    text = re.sub(r"20\d{6}", "DATE_REMOVED", text)
    text = re.sub(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\s+20\d{2}\b",
        "BUILD_DATE_REMOVED",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b\d{1,2}:\d{2}:\d{2}\b", "TIME_REMOVED", text)
    text = re.sub(r"[A-Za-z]:\\[^\s,;\"']+", "LOCAL_PATH_REMOVED", text)
    return text


def sanitize_path_part(part: str) -> str:
    replacements = {
        "ECG训练集": "ecg_dataset",
        "PPG训练集": "ppg_dataset",
    }
    part = replacements.get(part, part)
    part = re.sub(r"(?<![A-Za-z0-9])S(\d{2})(?![A-Za-z0-9])", lambda m: f"P{int(m.group(1)):03d}", part)
    part = re.sub(r"20\d{6}", "DATE_REMOVED", part)
    part = part.replace("YYYYMMDD", "DATE_REMOVED")
    part = part.replace(".csv.csv", ".csv")
    return part


def clean_generated_content() -> None:
    for name in ("model", "embedded", "data"):
        target = ROOT / name
        if target.exists():
            shutil.rmtree(target)
    for manifest in (ROOT / "docs" / "dataset_manifest.csv", ROOT / "docs" / "release_manifest.csv"):
        if manifest.exists():
            manifest.unlink()


def should_skip(path: Path) -> bool:
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return True
    if path.suffix in SKIP_SUFFIXES:
        return True
    return False


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix in TEXT_SUFFIXES:
        try:
            text = src.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = src.read_text(encoding="gb18030", errors="replace")
        dst.write_text(sanitize_text(text), encoding="utf-8", newline="")
    else:
        shutil.copy2(src, dst)


def copy_model_release() -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    for source_rel, dest_rel in MODEL_COPY_ROOTS:
        src_root = SOURCE_MODEL / source_rel
        if not src_root.exists():
            continue
        for src in src_root.rglob("*"):
            if not src.is_file() or should_skip(src):
                continue
            rel = src.relative_to(src_root)
            dest = ROOT / dest_rel / Path(*[sanitize_path_part(p) for p in rel.parts])
            copy_file(src, dest)
            manifest.append(
                {
                    "area": "model",
                    "path": dest.relative_to(ROOT).as_posix(),
                    "bytes": str(dest.stat().st_size),
                    "sha256": sha256_file(dest),
                }
            )
    return manifest


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 2
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def sanitize_dataset(source_root: Path, split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for src in sorted(source_root.rglob("*.csv")):
        rel = src.relative_to(source_root)
        sanitized_parts = [sanitize_path_part(part) for part in rel.parts]
        dst = unique_path(ROOT / "data" / split / Path(*sanitized_parts))
        copy_file(src, dst)
        line_count = count_lines(dst)
        rows.append(
            {
                "split": split,
                "path": dst.relative_to(ROOT).as_posix(),
                "scenario": scenario_from_name(dst.name),
                "lines": str(line_count),
                "bytes": str(dst.stat().st_size),
                "sha256": sha256_file(dst),
            }
        )
    return rows


def scenario_from_name(name: str) -> str:
    base = Path(name).stem
    base = re.sub(r"^P\d{3}_R\d{2}_", "", base)
    base = re.sub(r"^P\d{3}_", "", base)
    base = base.replace("_DATE_REMOVED", "")
    return base


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for label, path in {
        "HP_ABCD_MODEL_DIR": SOURCE_MODEL,
        "HP_ABCD_TRAIN_DIR": SOURCE_TRAIN,
        "HP_ABCD_TEST_DIR": SOURCE_TEST,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} source directory not found: {path}")
    clean_generated_content()
    release_manifest = copy_model_release()
    dataset_manifest = []
    dataset_manifest.extend(sanitize_dataset(SOURCE_TRAIN, "train"))
    dataset_manifest.extend(sanitize_dataset(SOURCE_TEST, "test"))
    release_manifest.extend(
        {
            "area": "data",
            "path": row["path"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in dataset_manifest
    )
    write_csv(
        ROOT / "docs" / "dataset_manifest.csv",
        dataset_manifest,
        ["split", "path", "scenario", "lines", "bytes", "sha256"],
    )
    write_csv(
        ROOT / "docs" / "release_manifest.csv",
        release_manifest,
        ["area", "path", "bytes", "sha256"],
    )
    print(f"model/data release built at {ROOT}")
    print(f"dataset csv files: {len(dataset_manifest)}")
    print(f"release files: {len(release_manifest)}")


if __name__ == "__main__":
    main()
