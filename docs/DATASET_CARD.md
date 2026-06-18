# Dataset Card

## Summary

This dataset contains de-identified CSV recordings collected for HealthProject AI model development and testing.
The files include ECG, PPG, temperature, IMU, and derived firmware feature streams depending on the acquisition scenario.

## Splits

- `data/train`: de-identified training recordings from the new training set.
- `data/test`: de-identified test recordings from the test-data folder.

See `docs/dataset_manifest.csv` for file names, scenarios, line counts, sizes, and SHA-256 checksums.

## De-identification

Subject-style identifiers were replaced with stable public IDs.
Collection dates, firmware build dates, wall-clock times, and local filesystem paths were removed from file names and text contents.
Original Word experiment documents are not included.

## Intended Use

The dataset is intended for model reproducibility, signal-quality experiments, and non-clinical algorithm evaluation.
It is not intended for clinical diagnosis, treatment decisions, or medical-device validation without additional controlled verification.

## Known Limitations

The public IDs preserve within-release grouping for reproducibility, so repeated recordings from the same public participant code can still be grouped.
The data is collected from a limited local protocol and may not represent broader populations, devices, or acquisition environments.
