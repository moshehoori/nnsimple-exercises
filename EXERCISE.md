# C-01 — `-nnsimple-fuse-mul-add` pass

**Concepts**: TableGen-declared passes, `OpRewritePattern`, greedy pattern rewrite driver, multi-operand pattern matching.

**Time**: ~1.5h.

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

The test for this exercise (`test/NNSimple/fuse-mul-add.mlir`) is already in the repo — don't edit it. Just make it pass.

The pass is auto-registered with `nnsimple-opt` once declared in the `.td` — no changes needed to `nnsimple-opt.cpp`.

## Hints

- **The op**: three-operand equivalent of `NNSimple_FusedAddReluOp` (lines ~83-96 of `NNSimpleOps.td`). The existing op has two operands — yours has three. The assembly format extends the same way.
- **The pattern**: mirror `NNSimpleFuseAddReluRewriter` in `lib/NNSimple/NNSimplePasses.cpp` (lines 24-42). That one matches `ReluOp` and looks for an `AddOp` producing its input. Yours matches `AddOp` and looks for a single-use `MulOp` producing either operand.
  - Use `Value::getDefiningOp<MulOp>()` to check the producing op.
  - Use `op->hasOneUse()` for the safety check.
  - Use `rewriter.replaceOp(addOp, fused.getResult())` + `rewriter.eraseOp(mulOp)` to swap in the new op.
- **Commuted pattern**: remember `add(c, mul(a,b))` should also fuse — try each operand of the add.
- **The pass body**: look at `NNSimpleFuseAddRelu::runOnOperation` (same file, lines 50-57). Build a `RewritePatternSet`, add your rewriter, call `applyPatternsGreedily(func, ...)`.

## Done when

`test/NNSimple/fuse-mul-add.mlir` goes from red to green. It has a positive case (simple `(a*b)+c`) and a negative case (multi-use mul — must NOT fuse).

```bash
cd build && ninja nnsimple-opt
llvm-lit -v ../test/NNSimple/fuse-mul-add.mlir   # should PASS
```

Or manually:
```bash
./bin/nnsimple-opt ../test/NNSimple/fuse-mul-add.mlir -nnsimple-fuse-mul-add \
    | /path/to/llvm-project/build/bin/FileCheck ../test/NNSimple/fuse-mul-add.mlir
```

## Stretch

- Add a `negate_c` boolean pass option: when set, treat the pass as fusing `add(mul(a,b), neg(c))` into a new `fused_mul_sub` op. Demonstrates pass options (see the `enable-fuse-add-relu` flag in `NNSimplePasses.cpp:61-66` for the pattern).
- Also handle `sub(mul(a, b), c) → fused_mul_sub(a, b, c)` once F-01's `sub` op is in place.
