# D-01 — Lower `nnsimple.sub` to `linalg.sub`

**Concepts**: `OpConversionPattern`, dialect conversion, `TypeConverter`.

**Time**: ~2h.

## Setup

F-01's `nnsimple.sub` op is already pre-filled in this starter (so you don't have to re-do F-01 here). Your job is to teach `-nnsimple-lower-to-linalg` how to lower it.

## Task

When `-nnsimple-lower-to-linalg` runs, a `sub` on the dialect's tensor type should become a `linalg.sub` on standard ranked tensors (with the layout stripped by the existing type converter):

```mlir
// Before:
func.func @f(%a: !nnsimple.tensor<f32, [4, 4], NCHW>, %b: ...) -> ... {
    %0 = nnsimple.sub %a, %b : ... -> ...
    return %0 : ...
}
// After:
func.func @f(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %init = tensor.empty() : tensor<4x4xf32>
    %0 = linalg.sub ins(%a, %b : ...) outs(%init : ...) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
}
```

## Files to edit

| File | What |
|---|---|
| `lib/NNSimple/NNSimpleLinalgLowering.cpp` | Define `SubOpLowering` (mirror `AddOpLowering`), register it in the `patterns.add<...>` call near the bottom. Stub shown above `MulOpLowering`. |

## Hints

- `AddOpLowering` right above your stub is the exact template. Differences:
  - Replace `AddOp` with `SubOp` and `linalg::AddOp` with `linalg::SubOp`.
  - Everything else (grabbing the result type through the type converter, creating a `tensor.empty`, building the linalg op) stays identical.
- Don't forget to register the new pattern in the call at the bottom:
  ```cpp
  patterns.add<AddOpLowering, SubOpLowering, MulOpLowering, ReluOpLowering,
               ConstOpLowering, FusedAddReluOpLowering>(typeConverter, &getContext());
  ```
- The `TypeConverter` at the top of the file already handles `!nnsimple.tensor<f32, [4, 4], NCHW> → tensor<4x4xf32>`. No type-conversion code to write.

## Done when

```bash
cd build && ninja check-nnsimple
```

passes. `test/NNSimple/sub-lowering.mlir` checks that the lowered IR contains `tensor.empty` + `linalg.sub`.

## Stretch

- Add lowerings for all the other ops you've introduced in this curriculum: `neg` (lower to `linalg.generic` with `arith.negf`), `matmul` (lower to `linalg.matmul` — needs a zero-filled init), `fused_mul_add` (lower to `linalg.generic` with `arith.mulf + arith.addf` in the region).
- End-to-end test: pipe through `-nnsimple-pipeline -convert-linalg-to-loops -convert-scf-to-cf -convert-to-llvm | mlir-runner` and check numeric output (see D-02).
