# Data Site Verify Hang Workaround

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

- `LOCAL_PATH_REMOVED
- `LOCAL_PATH_REMOVED

It keeps the same external contract:

- input: `[1, 107]`
- output: `[1, 3]`
- labels: `0 FINAL_OK`, `1 SIGNAL_BAD_OR_CONTACT_BAD`, `2 UNCERTAIN_OR_MOTION`

The graph avoids `STRIDED_SLICE` by using block-sparse Dense weights to read the A/B/context ranges.

## No-Slice Operator Audit

- Operators: `FULLY_CONNECTED, FULLY_CONNECTED, FULLY_CONNECTED, SOFTMAX, FULLY_CONNECTED, FULLY_CONNECTED, FULLY_CONNECTED, SOFTMAX, CONCATENATION, FULLY_CONNECTED, SOFTMAX`
- Forbidden operators present: `NONE`
- Max abs diff vs standard sliced model on real NPZ: `1.192092896e-07`

## Recommended Retry

1. Import `LOCAL_PATH_REMOVED into the deployment tool.
2. Load `LOCAL_PATH_REMOVED
3. Click `Data site verify`.
4. If verify passes, continue to user benchmark.

If it still hangs, the remaining likely causes are the tool process state, COM/board state, or a deployment-tool UI issue rather than the model graph. Restart the deployment tool, reconnect the board, and retry with the no-slice model first.
