# ABCD v0.2 Polished Figure Notes

The earlier generated figures were engineering diagnostics and were visually weaker than the original single-purpose confusion matrices. The updated figures separate the reporting purposes clearly:

- Source model validation/test confusion matrices: `LOCAL_PATH_REMOVED
- Deployment consistency matrix: `LOCAL_PATH_REMOVED
- Internal A/B/D branch distribution: `LOCAL_PATH_REMOVED

Important distinction:

- A/B/D source confusion matrices use ground-truth labels.
- The ABCD deployment matrix uses PC output vs MCU output and has no ground-truth labels.
- The deployed TFLite exposes only final D output, while A and B remain internal branches.
