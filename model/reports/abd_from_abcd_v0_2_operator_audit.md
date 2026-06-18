# ABD From ABCD v0.2 Operator Audit

## Acceptance Rule

- Forbidden operators: `CUSTOM, EQUAL, GREATER, IS_FINITE, IS_INF, IS_NAN, LESS, SELECT, WHERE`
- Recommended deployment model must contain no forbidden operators.

## Float32 Deploy Model

- Path: `LOCAL_PATH_REMOVED
- Operators: `STRIDED_SLICE, FULLY_CONNECTED, FULLY_CONNECTED, FULLY_CONNECTED, SOFTMAX, STRIDED_SLICE, FULLY_CONNECTED, FULLY_CONNECTED, FULLY_CONNECTED, SOFTMAX, STRIDED_SLICE, FULLY_CONNECTED, CONCATENATION, FULLY_CONNECTED, SOFTMAX`
- Operator counts: `{"STRIDED_SLICE": 3, "FULLY_CONNECTED": 8, "SOFTMAX": 3, "CONCATENATION": 1}`
- Forbidden operators present: `NONE`
- Non-preferred operators present: `CONCATENATION, STRIDED_SLICE`
- `STRIDED_SLICE` is expected because the single 107-dim input is split into A_input_58, B_input_32, and Context_input_17.
- `CONCATENATION` is expected because the graph fuses A_out, B_out, and normalized context before the D_SAFE head.
- No `EQUAL`, `WHERE`, `SELECT`, finite checks, comparison ops, or `CUSTOM` op are present in the recommended float32 model.

## No-Slice Tool Verify Model

- Path: `LOCAL_PATH_REMOVED
- Operators: `FULLY_CONNECTED, FULLY_CONNECTED, FULLY_CONNECTED, SOFTMAX, FULLY_CONNECTED, FULLY_CONNECTED, FULLY_CONNECTED, SOFTMAX, CONCATENATION, FULLY_CONNECTED, SOFTMAX`
- Operator counts: `{"FULLY_CONNECTED": 7, "SOFTMAX": 3, "CONCATENATION": 1}`
- Forbidden operators present: `NONE`
- This artifact is mathematically equivalent for deployment input/output, but avoids `STRIDED_SLICE` for tools whose Data Site Verify page hangs on slice operators.

## Dynamic Quant Model

- Path: `LOCAL_PATH_REMOVED
- Operators: `STRIDED_SLICE, FULLY_CONNECTED, FULLY_CONNECTED, FULLY_CONNECTED, SOFTMAX, STRIDED_SLICE, FULLY_CONNECTED, FULLY_CONNECTED, FULLY_CONNECTED, SOFTMAX, STRIDED_SLICE, FULLY_CONNECTED, CONCATENATION, FULLY_CONNECTED, SOFTMAX`
- Operator counts: `{"STRIDED_SLICE": 3, "FULLY_CONNECTED": 8, "SOFTMAX": 3, "CONCATENATION": 1}`
- Forbidden operators present: `NONE`
- Not recommended for GD32 deployment because the previous toolchain rejected mixed quantization in FULLY_CONNECTED.
