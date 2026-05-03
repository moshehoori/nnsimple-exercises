# B-01 — Implement `MulOp::fold` for constant × constant

**Concepts**: constant folding with `OpFoldResult`, `DenseElementsAttr`, `APFloat`.

**Time**: ~1-1.5h.

## Task

When both operands of `nnsimple.mul` are dense-constant tensors of the same type, the op should fold to a single `nnsimple.const` holding the element-wise product.

```mlir
// Before -canonicalize:
%a = nnsimple.const dense<2.0> : tensor<2x2xf32> : !nnsimple.tensor<...>
%b = nnsimple.const dense<3.0> : tensor<2x2xf32> : !nnsimple.tensor<...>
%c = nnsimple.mul %a, %b : ...
// After:
%c = nnsimple.const dense<6.0> : tensor<2x2xf32> : !nnsimple.tensor<...>
```

When either operand is non-constant, the fold must NOT fire.

## Files to edit

| File | What |
|---|---|
| `include/NNSimple/NNSimpleOps.td` | Uncomment `let hasFolder = 1;` inside `NNSimple_MulOp`. |
| `lib/NNSimple/NNSimpleOps.cpp` | Implement `MulOp::fold`. Stub shown above `AddOp::fold`. |

## Hints

- `AddOp::fold` a few lines below your stub is the exact template. Only difference: use `APFloat::multiply(..., APFloat::rmNearestTiesToEven)` instead of `.add(...)`.
- FoldAdaptor gives you `adaptor.getLhs()` / `adaptor.getRhs()` as `Attribute` (already resolved through any constant operands). If the operand isn't constant, they'll be null. Use `dyn_cast_or_null<DenseElementsAttr>`.
- Build the result with `DenseElementsAttr::get(lhsType, results)` where `lhsType` is the attribute's type.
- The `nnsimple.const` op has a constant materializer that turns your returned `ElementsAttr` into a fresh `nnsimple.const` — you don't need to create the `const` yourself.

## Done when

```bash
cd build && ninja check-nnsimple
```

passes. `test/NNSimple/mul-fold.mlir` has one fold case and one no-fold case.

## Stretch

- Also fold when one operand is a splat `1.0` (like `AddZeroElimination` but for `mul-by-one`) — but consider: is that a fold or a canonicalization? (Hint: folders return `OpFoldResult`, which can be a Value; canonicalizations use `OpRewritePattern`. The `mul(x, 1) → x` rewrite is more naturally a canonicalization — see B-02.)
