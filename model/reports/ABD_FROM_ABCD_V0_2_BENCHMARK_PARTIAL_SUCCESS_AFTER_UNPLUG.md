# ABD From ABCD v0.2 Benchmark Partial Success After Serial Unplug

## Observed Log

The benchmark log reached these success markers:

- `NPZ file Number: 1 sent success`
- `MCU Response: MCU_READY`
- `Receiving No.20 mcu invoke result, total:20`
- `Receive complete`

Then the serial cable was unplugged, and the PC tool printed:

```text
frame idx = 21 Receive failed!
ERROR: Benchmark, reason maybe serial com be occupied / not open / MCU not reset/PC tensorflow invoke error/npz error
```

## Interpretation

This is not a model import failure, NPZ failure, or unsupported-operator failure.

The tool had already:

1. Sent the NPZ count to MCU.
2. Started the real NPZ benchmark.
3. Received `MCU_READY`.
4. Sent all 20 frames.
5. Received all 20 MCU frame outputs.

The failure happened after `Receive complete`, while the PC tool was still waiting for the remaining result/summary bytes and final evaluation stage. Unplugging the serial cable at that moment prevents the PC tool from printing:

- MCU invoke result
- Step3 evaluation
- Inference average latency
- Average error
- Max error

## PC Model Confirmation

The PC outputs in the pasted log match the local no-slice TFLite model:

- Model: `LOCAL_PATH_REMOVED
- NPZ: `LOCAL_PATH_REMOVED

So the correct model/NPZ pair was used.

## Required Retry

Run the benchmark again and do not unplug or reset after `Receive complete`.

Wait until the tool prints:

- `Step3. Evaluate results.`
- `Inference average latency`
- `Average error`
- `Max error`

Only after those lines appear should the board or serial cable be disconnected.

## Acceptance Criteria

The benchmark is complete only if the log contains:

- `Receive complete`
- all 20 MCU inference result lines
- `Inference times: 20`
- `Inference average latency`
- `Average error`
- `Max error`

If the tool remains stuck for more than 2 minutes after `Receive complete` without unplugging, preserve the serial connection and capture the log exactly as-is; that would indicate a final-result packet handling issue rather than the previous handshake or data-site-verify issue.
