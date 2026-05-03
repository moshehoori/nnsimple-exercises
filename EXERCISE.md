# B-02 — Canonicalizations `mul(x, 1) → x` and `mul(x, 0) → 0`

**Concepts**: imperative C++ `OpRewritePattern`, replacing ops with other ops, Commutative trait.

**Time**: ~2h.

## Task

Add two canonicalization patterns for `nnsimple.mul`:

```mlir
// Before -canonicalize:
%one = nnsimple.const dense<1.0> : ...
%0   = nnsimple.mul %arg0, %one : ...
// After:
// (no op; uses of %0 are redirected to %arg0)

// Before:
%zero = nnsimple.const dense<0.0> : ...
%0    = nnsimple.mul %arg0, %zero : ...
// After:
%0 = nnsimple.const dense<0.0> : ...
```

Because `MulOp` has the `Commutative` trait, the canonicalizer moves constants to the rhs automatically — you only need to check `op.getRhs()`.

## Files to edit

| File | What |
|---|---|
| `include/NNSimple/NNSimpleOps.td` | Uncomment `let hasCanonicalizer = 1;` inside `NNSimple_MulOp`. |
| `lib/NNSimple/NNSimpleOps.cpp` | Implement `MulOneElimination`, `MulZeroElimination`, and `MulOp::getCanonicalizationPatterns`. Stubs are in the file. |

## Hints

- `AddZeroElimination` just above your stubs is the exact template for `MulOneElimination` (swap `.isZero()` → `.isOne()`, i.e. `floatVal.convertToDouble() == 1.0` or `APFloat::getOne(...).bitwiseIsEqual(floatVal)`).
- For `MulZeroElimination`, you need to produce a *new* constant (the zero), not reuse an operand. Use:
  ```cpp
  auto zeroAttr = DenseElementsAttr::get(denseAttr.getType(),
                                          APFloat::getZero(floatVal.getSemantics()));
  rewriter.replaceOpWithNewOp<ConstOp>(op, op.getResult().getType(), zeroAttr);
  ```
- A simpler trick: when rhs is already a zero-splat constant with the right type, you can just replace the op with the rhs value directly: `rewriter.replaceOp(op, op.getRhs());` — but this only works if the result type matches the rhs type (which for nnsimple.mul it does by construction).

## Done when

```bash
cd build && ninja check-nnsimple
```

passes. `test/NNSimple/mul-canonicalize.mlir` has 3 cases: `x*1`, `1*x` (commutative swap), `x*0`.

## Stretch

- `mul(x, -1) → neg(x)` (depends on F-02's neg op).
- `mul(x, x) → square(x)` if you add a `square` op.
