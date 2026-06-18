# ABD From ABCD v0.2 Benchmark Stall After Receive Complete Fix

## Symptom

After `Receive complete`, the GD32 AI deployment tool keeps printing progress markers, roughly one per second, and does not print:

- `Step3. Evaluate results.`
- `Inference average latency`
- `Average error`
- `Max error`

This is not normal if it lasts longer than a few seconds.

## Likely Cause

The previous robustness patch made benchmark mode loop forever:

1. wait for PC handshake,
2. run benchmark,
3. if benchmark succeeds, immediately wait for another PC handshake.

That is useful after a failed attempt, but too aggressive after a successful benchmark. The MCU may reinitialize USART/DMA immediately after sending the final 8-byte result packet, while the PC tool is still reading and evaluating the final packet.

## Fix Applied

`main.c` was updated so that:

- failed `nn_benchmark_init()` or `nn_benchmark_start()` still retries;
- successful benchmark breaks out of the retry loop;
- the MCU then falls through to the final idle `while(1)` and stays silent.

Patched file:

- `LOCAL_PATH_REMOVED

## Required Next Step

Rebuild and flash:

- `LOCAL_PATH_REMOVED

Then rerun with:

- Model: `LOCAL_PATH_REMOVED
- NPZ: `LOCAL_PATH_REMOVED

## Expected Result

After `Receive complete`, the PC tool should proceed to:

- print MCU invoke result frames;
- print `Step3. Evaluate results.`;
- print latency and error statistics.

If it still stalls, the next suspect is final 8-byte result packet timing/format, not the model graph or NPZ.
