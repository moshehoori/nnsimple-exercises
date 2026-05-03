# F-01 — Add the `nnsimple.sub` op

**Concepts**: TableGen op definition, C++ verifier, op traits.

**Time**: ~1.5h.

## Task

Add a new element-wise subtraction op to the NNSimple dialect:

```mlir
%0 = nnsimple.sub %a, %b : (!nnsimple.tensor<f32, [1, 128], NCHW>,
                            !nnsimple.tensor<f32, [1, 128], NCHW>)
                        -> !nnsimple.tensor<f32, [1, 128], NCHW>
```

It should be **Pure** (no side effects) but **not Commutative** (order matters: `a - b ≠ b - a`). The op must verify that:
1. lhs type == rhs type — error: `operand types don't match`
2. lhs type == result type — error: `operand type doesn't match result type`

## Files to edit

| File | What to do |
|---|---|
| `include/NNSimple/NNSimpleOps.td` | Define `NNSimple_SubOp`. Stub comment shows where. |
| `lib/NNSimple/NNSimpleOps.cpp` | Implement `SubOp::verify()`. Stub comment shows where. |

## Hints

- Look at `NNSimple_AddOp` in `NNSimpleOps.td` (lines ~17-34) — the new op is almost identical but drops the `Commutative` trait and drops `hasFolder`/`hasCanonicalizer`.
- Look at `AddOp::verify` in `NNSimpleOps.cpp` (just below where you'll add `SubOp::verify`) — same idea, verbatim.
- Don't forget `let hasVerifier = 1;` on the TableGen side.

## Done when

```bash
cd build && ninja check-nnsimple
```

passes. The test file is `test/NNSimple/sub-ops.mlir` — it has one positive parse test and two negative tests (mismatched operand shapes, mismatched result shape) that exercise both verifier diagnostics.

To run just this test:
```bash
llvm-lit -v ../test/NNSimple/sub-ops.mlir
```

## Stretch (optional)

- Add a folder `sub(x, x) → 0` using `hasFolder = 1` and `SubOp::fold` (mirror `AddOp::fold`). Return a zero-splat `DenseElementsAttr` when the two operands alias.
- Add a canonicalization `sub(x, 0) → x` (mirror `AddZeroElimination`).
