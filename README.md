# HealthProject AI ABCD Integrated Model v0.2

Open release package for the HealthProject ABCD integrated physiological signal model.

This repository contains:

- Inference and audit tooling for the ABCD integrated model.
- Model artifacts for the ABD-from-ABCD deployment candidate.
- Public schemas, contracts, model registry, and verified metrics.
- De-identified CSV training and test data generated from local acquisition runs.
- Embedded deployment excerpts limited to project-specific source/model files.

## Repository Layout

```text
model/
  tools/                  Python inference, analysis, and audit scripts
  models/                 Keras/SavedModel artifacts
  schemas/                Public input/output schemas
  contracts/              Inference and action mapping contracts
  metrics_pack_verified/  Verified metrics, figures, and provenance reports
embedded/
  GD_Embedded_AI/         Project-specific embedded source excerpts
data/
  train/                  De-identified training CSV files
  test/                   De-identified test CSV files
docs/
  dataset_manifest.csv    Public data file manifest with hashes
  release_manifest.csv    Public release file manifest with hashes
```

## De-identification

The public CSV data was generated from the original local collection folders with `scripts/build_open_release.py`.
The release process:

- replaces private subject/run directory identifiers with stable public identifiers such as `P001`;
- removes date tokens from file names and file contents;
- removes firmware build dates and clock times from CSV metadata lines;
- removes local Windows paths when they appear in text artifacts;
- excludes Word experiment-flow documents from the public release.

The de-identification process does not publish the private subject mapping table.

## Rebuild The Release

Run from the repository root on the original workstation. Set the source directory environment variables before rebuilding:

```powershell
$env:HP_ABCD_MODEL_DIR="path\to\model_abcd_integrated_v0_2"
$env:HP_ABCD_TRAIN_DIR="path\to\training_csv_folder"
$env:HP_ABCD_TEST_DIR="path\to\test_csv_folder"
python scripts/build_open_release.py
```

The script regenerates `model/`, `embedded/`, `data/`, and the public manifests from the local source directories.
If the variables are omitted, the script looks for sibling folders named `model_abcd_integrated_v0_2`, `source_train`, and `source_test`.

## Notes

The `embedded/` folder intentionally excludes vendor SDK packages, IDE build outputs, and proprietary prebuilt libraries.
To rebuild firmware, install the required GigaDevice GD32H7xx SDK and AI middleware separately, then merge these project-specific files into the vendor project structure.

## License

Code and documentation are released under the Apache License 2.0. De-identified data and model artifacts are released for research use; review `DATA_LICENSE.md` before public publication.
