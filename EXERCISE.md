# A-01 — `nnsimple.matmul` with shape-compatibility verifier

**Concepts**: TableGen op with a non-trivial verifier, inspecting type parameters in C++.

**Time**: ~2-3h.

## Task

Add a matrix-multiply op:

```mlir
// [M, K] * [K, N] -> [M, N]
%c = nnsimple.matmul %a, %b
    : (!nnsimple.tensor<f32, [2, 3], RowMajor>,
       !nnsimple.tensor<f32, [3, 4], RowMajor>)
   -> !nnsimple.tensor<f32, [2, 4], RowMajor>
```

Only 2D (no batch). The verifier must reject:
1. Any operand whose shape has rank ≠ 2 → `operands must be rank-2`
2. `lhs.shape[1] != rhs.shape[0]` → `inner dimensions must match`
3. Result shape ≠ `[lhs.shape[0], rhs.shape[1]]` → `result shape must be [lhs.shape[0], rhs.shape[1]]`

(Element-type checks, layout checks, batch dims — out of scope for this exercise. Keep it tight.)

## Files to edit

| File | What |
|---|---|
| `include/NNSimple/NNSimpleOps.td` | Define `NNSimple_MatMulOp`. Stub comment shows where. |
| `lib/NNSimple/NNSimpleOps.cpp` | Implement `MatMulOp::verify`. Stub comment shows where. |

## Hints

- Mirror `NNSimple_AddOp` (td lines 17-34) for the op skeleton. Two operands, one result, Pure, `hasVerifier = 1`. Not Commutative (matrix mul isn't). No folder, no canonicalizer.
- In the verifier, cast operand types to the dialect type to access shape:
  ```cpp
  auto lhsType = llvm::cast<nnsimple::TensorType>(getLhs().getType());
  llvm::ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ```
- `ArrayRef<int64_t>::size()` tells you rank. `shape[0]` and `shape[1]` are the two dims.
- See `AddOp::verify` in `NNSimpleOps.cpp` — same pattern, simpler check.

## Done when

```bash
cd build && ninja check-nnsimple
```

passes. The test file is `test/NNSimple/matmul.mlir` — 1 positive, 3 negatives (rank, inner dim, result shape).

## Stretch

- Support batched matmul: `[B, M, K] * [B, K, N] -> [B, M, N]` — verifier checks all three dims including broadcast of B=1.
- Add a `TransposeAttr` option: `matmul %a, %b { transpose_lhs = true } : ...`.
