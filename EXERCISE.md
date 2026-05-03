# C-01 — `-nnsimple-fuse-mul-add` pass

**Concepts**: TableGen-declared passes, `OpRewritePattern`, greedy pattern rewrite driver, multi-operand pattern matching.

**Time**: ~3h.

## Task

Write an optimization pass that fuses `add(mul(a, b), c)` into a single `fused_mul_add(a, b, c)`:

```mlir
// Before -nnsimple-fuse-mul-add:
%0 = nnsimple.mul %a, %b : ...
%1 = nnsimple.add %0, %c : ...
// After:
%1 = nnsimple.fused_mul_add %a, %b, %c : ...
```

**Safety**: only fuse when `%0` has exactly one use. Otherwise another user of the mul would lose its result.

## Files to edit

| File | What |
|---|---|
| `include/NNSimple/NNSimpleOps.td` | Uncomment & fill `NNSimple_FusedMulAddOp`. Three operands. |
| `include/NNSimple/NNSimplePasses.td` | Uncomment `NNSimpleFuseMulAdd`. |
| `lib/NNSimple/NNSimpleFuseMulAdd.cpp` | Implement the pattern and the pass body. Starter skeleton provided. |

The pass is auto-registered with `nnsimple-opt` once declared in the `.td` — no changes needed to `nnsimple-opt.cpp`.

## Hints

- **The op**: three-operand equivalent of `NNSimple_FusedAddReluOp` (lines ~83-96 of `NNSimpleOps.td`). Use `(ins NNSimple_TensorType:$a, NNSimple_TensorType:$b, NNSimple_TensorType:$c)`. Assembly format extends naturally:
  ```tablegen
  let assemblyFormat = [{
      $a `,` $b `,` $c attr-dict `:` `(` type($a) `,` type($b) `,` type($c) `)` `->` type($result)
  }];
  ```
- **The pattern**: mirror `NNSimpleFuseAddReluRewriter` in `lib/NNSimple/NNSimplePasses.cpp` lines 24-42. Match `AddOp`, check that either operand comes from a single-use `MulOp`:
  ```cpp
  class FuseMulAddRewriter : public OpRewritePattern<AddOp> {
    LogicalResult matchAndRewrite(AddOp addOp, PatternRewriter &rewriter) const override {
      // Try lhs-is-mul first, then rhs-is-mul.
      for (auto [maybeMul, other] : {std::pair{addOp.getLhs(), addOp.getRhs()},
                                     std::pair{addOp.getRhs(), addOp.getLhs()}}) {
        auto mulOp = maybeMul.getDefiningOp<MulOp>();
        if (!mulOp || !mulOp->hasOneUse()) continue;
        auto fused = FusedMulAddOp::create(rewriter, addOp.getLoc(), addOp.getResult().getType(),
                                            mulOp.getLhs(), mulOp.getRhs(), other);
        rewriter.replaceOp(addOp, fused.getResult());
        rewriter.eraseOp(mulOp);
        return success();
      }
      return failure();
    }
  };
  ```
- **The pass body**: look at `NNSimpleFuseAddRelu::runOnOperation` (same file, lines 50-57). Build a `RewritePatternSet`, add your rewriter, call `applyPatternsGreedily(func, ...)`.

## Done when

```bash
cd build && ninja check-nnsimple
```

passes. `test/NNSimple/fuse-mul-add.mlir` has a positive case (simple `(a*b)+c`) and a negative case (multi-use mul — must NOT fuse).

## Stretch

- Fuse across commuted pattern: `add(c, mul(a, b))` (the `for` loop above already handles this).
- Add a `negate_c` boolean pass option: when set, treat the pass as fusing `add(mul(a,b), neg(c))` into a new `fused_mul_sub` op. Demonstrates pass options (see the `enable-fuse-add-relu` flag in `NNSimplePasses.cpp:61-66` for the pattern).
