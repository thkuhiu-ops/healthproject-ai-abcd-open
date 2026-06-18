# ABD From ABCD v0.2 Benchmark Handshake Fix

## Observed Failure

The GD32 AI deployment tool stopped at the first benchmark handshake:

```text
Model user benchmarking by dataset.....
Open serial and PC sent NPZ file number to mcu...
ERROR: PC can not read bytes from MCU, please reset MCU first
ERROR: MCU GET NPZ NUMBER
```

This failure occurs before the dataset frames are transmitted. It is not evidence of a TFLite operator failure, NPZ shape failure, or model inference mismatch.

## Protocol Point

The PC first sends an 8-byte header:

- bytes 0..3: `NPZN`
- bytes 4..7: NPZ file count

The MCU must respond with:

- `GET_NPZN`

The failure means the PC did not read this response.

## Local Audit Result

- v0.2 generated firmware exists and builds successfully.
- `INPUT_SIZE` is `107`.
- `OUTPUT_SIZE` is `3`.
- `BENCHMARK` is enabled.
- The benchmark code is present in `GD_Embedded_AI\Core\Source\nn_model_benchmark.c`.

## Patch Applied

Two robustness changes were applied to the v0.2 generated firmware source:

1. `nn_benchmark_init()` now initializes `benchmark->npz_number = 0` and sends `GET_NPZE` if the first header is not `NPZN`.
2. `main.c` benchmark mode now loops back to wait for the next PC benchmark handshake instead of exiting after the first failed attempt.

Patched files:

- `LOCAL_PATH_REMOVED
- `LOCAL_PATH_REMOVED

## Required Next Step

Rebuild and download the patched firmware:

- Project: `LOCAL_PATH_REMOVED
- Existing last build log shows `0 Error(s), 0 Warning(s)`.
- After flashing, press reset once before starting the PC benchmark.

Then rerun benchmark with:

- Model: `LOCAL_PATH_REMOVED
- NPZ: `LOCAL_PATH_REMOVED

## If It Still Fails

Check the following in order:

1. Close any serial terminal or IDE serial monitor occupying the COM port.
2. Confirm the tool is using the same COM port as USART0.
3. Press board reset after the deployment tool opens the benchmark flow, then start the dataset benchmark.
4. Confirm the PC tool expects baud rate `1152000`.
5. If the tool receives `GET_NPZE`, the first 8 bytes are not `NPZN + count`, so the benchmark PC-side protocol is misaligned.

## Expected Success Markers

The successful log should contain:

- `NPZ file Number: 1 sent success`
- `MCU Response: MCU_READY`
- `Receive complete`
- `Inference average latency`
- `Average error`
- `Max error`
