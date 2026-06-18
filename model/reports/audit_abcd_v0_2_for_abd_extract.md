# Audit ABCD v0.2 for ABD Extract

## Sources

- Model A source: `LOCAL_PATH_REMOVED
- Model B source: `LOCAL_PATH_REMOVED
- Model C source recorded but disabled: `LOCAL_PATH_REMOVED
- D_SAFE head source: `LOCAL_PATH_REMOVED
- D_SAFE label encoder: `LOCAL_PATH_REMOVED
- D_SAFE context normalization: `LOCAL_PATH_REMOVED
- v0_1 validated TFLite reference: `LOCAL_PATH_REMOVED

## Deployment Extraction Decision

- Model A is embedded through folded normalization and frozen Dense weights.
- Model B is embedded through folded standardization and frozen Dense weights.
- Model C is disabled and is not loaded into the final graph.
- D_SAFE reuses the v0_1 trained head weights and label order. No retraining is performed.

## Dimensions

- A input dimension: `58`
- B input dimension: `32`
- Context input dimension: `17`
- Final ABD single input dimension: `107`
- Final output dimension: `3`

## Output Label Order

- `0` `FINAL_OK`
- `1` `SIGNAL_BAD_OR_CONTACT_BAD`
- `2` `UNCERTAIN_OR_MOTION`

## Feature Order

- Input layout: `[A_input_58, B_input_32, Context_input_17]`.
- B feature order comes from `LOCAL_PATH_REMOVED and contains `32` fields.
- Context feature order reuses the v0_1 D_SAFE normalization schema.

## Preprocessing Boundary

- In-graph: Model A normalization, Model B standardization, D_SAFE context normalization are folded into Dense weights.
- Outside graph: finite-value check, median imputation, and clipping must be performed before MCU inference if needed.
- No FINAL_OK samples were synthesized.
