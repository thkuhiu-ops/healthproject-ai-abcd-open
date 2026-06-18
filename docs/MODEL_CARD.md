# Model Card

## Model

HealthProject ABCD integrated model v0.2, including an ABD-from-ABCD deployment candidate.

## Inputs And Outputs

Public schemas and contracts are available in:

- `model/schemas/`
- `model/contracts/`

## Artifacts

Model artifacts are available in `model/models/`.
Project-specific embedded deployment excerpts are available in `embedded/`.

## Evaluation

Verified metrics, figures, and provenance reports are available in `model/metrics_pack_verified/`.
The release manifest in `docs/release_manifest.csv` records file hashes for reproducibility.

## Intended Use

The model is intended for research and engineering evaluation of physiological signal quality, rhythm screening, and fusion-decision behavior.
It is not cleared for clinical diagnosis or medical-device deployment.

## Limitations

The model should be validated on independent data before any safety-critical or clinical workflow.
Embedded rebuilds require separately licensed vendor SDK and middleware dependencies that are not redistributed in this repository.
