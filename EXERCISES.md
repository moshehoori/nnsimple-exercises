# NNSimple MLIR Exercises — Consolidated Reference

Every learner works through all 10 exercises individually, roughly easiest → hardest. For the 2-day schedule see [CURRICULUM.md](CURRICULUM.md). Per exercise, `git checkout starter/<id>` to see the stubs and failing test; the `EXERCISE.md` on each starter branch is the same content as below.

## Contents

**Foundation**
- [F-01 — Add the `nnsimple.sub` op](#f-01--add-the-nnsimplesub-op)
- [F-02 — Add `nnsimple.neg` with a declarative canonicalization](#f-02--add-nnsimpleneg-with-a-declarative-canonicalization)

**Ops & Type System**
- [A-01 — `nnsimple.matmul` with shape-compatibility verifier](#a-01--nnsimplematmul-with-shape-compatibility-verifier)
- [A-02 — `!nnsimple.quantized` type](#a-02--nnsimplequantized-type)

**Folders & Canonicalization**
- [B-01 — Implement `MulOp::fold` for constant × constant](#b-01--implement-mulopfold-for-constant--constant)
- [B-02 — Canonicalizations `mul(x, 1) → x` and `mul(x, 0) → 0`](#b-02--canonicalizations-mulx-1--x-and-mulx-0--0)

**Transformation Passes**
- [C-01 — `-nnsimple-fuse-mul-add` pass](#c-01---nnsimple-fuse-mul-add-pass)
- [C-02 — `-nnsimple-dce` dead-op elimination pass](#c-02---nnsimple-dce-dead-op-elimination-pass)

**Lowering & E2E**
- [D-01 — Lower `nnsimple.sub` to `linalg.sub`](#d-01--lower-nnsimplesub-to-linalgsub)
- [D-02 — Lower all the way to LLVM](#d-02--lower-all-the-way-to-llvm)

---

## F-01 — Add the `nnsimple.sub` op

**Concepts**: TableGen op definition, C++ verifier, op traits.

**Time**: ~1.5h.

### Task

Add a new element-wise subtraction op to the NNSimple dialect:

```mlir
%0 = nnsimple.sub %a, %b : (!nnsimple.tensor<f32, [1, 128], NCHW>,
                            !nnsimple.tensor<f32, [1, 128], NCHW>)
                        -> !nnsimple.tensor<f32, [1, 128], NCHW>
```

It should be **Pure** (no side effects) but **not Commutative** (order matters: `a - b ≠ b - a`). The op must verify that:
1. lhs type == rhs type — error: `operand types don't match`
2. lhs type == result type — error: `operand type doesn't match result type`

### Files to edit

| File | What to do |
|---|---|
| `include/NNSimple/NNSimpleOps.td` | Define `NNSimple_SubOp`. Stub comment shows where. |
| `lib/NNSimple/NNSimpleOps.cpp` | Implement `SubOp::verify()`. Stub comment shows where. |

### Hints

- Look at `NNSimple_AddOp` in `NNSimpleOps.td` (lines ~17-34) — the new op is almost identical but drops the `Commutative` trait and drops `hasFolder`/`hasCanonicalizer`.
- Look at `AddOp::verify` in `NNSimpleOps.cpp` (just below where you'll add `SubOp::verify`) — same idea, verbatim.
- Don't forget `let hasVerifier = 1;` on the TableGen side.

### Done when

```bash
cd build && ninja check-nnsimple
```

passes. The test file is `test/NNSimple/sub-ops.mlir` — it has one positive parse test and two negative tests (mismatched operand shapes, mismatched result shape) that exercise both verifier diagnostics.

To run just this test:
```bash
llvm-lit -v ../test/NNSimple/sub-ops.mlir
```

### Stretch (optional)

- Add a folder `sub(x, x) → 0` using `hasFolder = 1` and `SubOp::fold` (mirror `AddOp::fold`). Return a zero-splat `DenseElementsAttr` when the two operands alias.
- Add a canonicalization `sub(x, 0) → x` (mirror `AddZeroElimination`).


---

## F-02 — Add `nnsimple.neg` with a declarative canonicalization

**Concepts**: TableGen op definition, declarative rewrite rules (DRR), canonicalization.

**Time**: ~1.5h.

### Task

Add an element-wise negation op, then write a declarative rewrite rule that collapses double-negation: `neg(neg(x)) → x`.

```mlir
// Before -canonicalize:
%0 = nnsimple.neg %x : ...
%1 = nnsimple.neg %0 : ...
// After -canonicalize:
// (nothing — both negs are eliminated; uses of %1 are redirected to %x)
```

### Files to edit

| File | What to do |
|---|---|
| `include/NNSimple/NNSimpleOps.td` | Define `NNSimple_NegOp` and a `Pat<>` named `NegNegElimination`. Stubs show where. |
| `lib/NNSimple/NNSimpleOps.cpp` | Register the generated pattern with `NegOp::getCanonicalizationPatterns`. Stub shows where. |

### Hints

- Look at `NNSimple_ReluOp` in `NNSimpleOps.td` (lines ~50-64) — your op is structurally identical: single operand, single result, Pure, `hasCanonicalizer = 1`.
- Look at `ReluReluElimination` in the same file (the `Pat<>` block near the bottom). Your pattern's RHS uses the DRR builtin `replaceWithValue` to replace the outer op with the inner op's original input:
  ```tablegen
  def NegNegElimination : Pat<
    (NNSimple_NegOp (NNSimple_NegOp $input)),
    (replaceWithValue $input)>;
  ```
  Without `replaceWithValue`, you'd create a new op; this builtin just reuses the SSA value.
- The generated include `NNSimple/NNSimpleCanonicalization.inc` is already pulled into `NNSimpleOps.cpp` — the pattern class name `NegNegElimination` will exist as a C++ symbol once the `.td` pattern is defined.
- Look at `ReluOp::getCanonicalizationPatterns` (near the bottom of `NNSimpleOps.cpp`) — yours is one-liner identical.

### Done when

```bash
cd build && ninja check-nnsimple
```

passes. The test file is `test/NNSimple/neg-canonicalize.mlir` — a positive test (`neg(neg(x))` is gone after `-canonicalize`) and a sanity test (a single `neg` survives).

Single-file:
```bash
llvm-lit -v ../test/NNSimple/neg-canonicalize.mlir
```

### Stretch (optional)

- Add a folder: `neg(const)` folds to a new constant with all elements negated. Mirror `AddOp::fold`.
- Add a DRR for `neg(sub(a, b)) → sub(b, a)` (requires F-01's `sub` op merged in).


---

## A-01 — `nnsimple.matmul` with shape-compatibility verifier

**Concepts**: TableGen op with a non-trivial verifier, inspecting type parameters in C++.

**Time**: ~2-3h.

### Task

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

### Files to edit

| File | What |
|---|---|
| `include/NNSimple/NNSimpleOps.td` | Define `NNSimple_MatMulOp`. Stub comment shows where. |
| `lib/NNSimple/NNSimpleOps.cpp` | Implement `MatMulOp::verify`. Stub comment shows where. |

### Hints

- Mirror `NNSimple_AddOp` (td lines 17-34) for the op skeleton. Two operands, one result, Pure, `hasVerifier = 1`. Not Commutative (matrix mul isn't). No folder, no canonicalizer.
- In the verifier, cast operand types to the dialect type to access shape:
  ```cpp
  auto lhsType = llvm::cast<nnsimple::TensorType>(getLhs().getType());
  llvm::ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ```
- `ArrayRef<int64_t>::size()` tells you rank. `shape[0]` and `shape[1]` are the two dims.
- See `AddOp::verify` in `NNSimpleOps.cpp` — same pattern, simpler check.

### Done when

```bash
cd build && ninja check-nnsimple
```

passes. The test file is `test/NNSimple/matmul.mlir` — 1 positive, 3 negatives (rank, inner dim, result shape).

### Stretch

- Support batched matmul: `[B, M, K] * [B, K, N] -> [B, M, N]` — verifier checks all three dims including broadcast of B=1.
- Add a `TransposeAttr` option: `matmul %a, %b { transpose_lhs = true } : ...`.


---

## A-02 — `!nnsimple.quantized` type

**Concepts**: custom types with parameters (TypeDef), declarative `assemblyFormat`.

**Time**: ~1.5-2h.

### Task

Add a new type representing a quantized tensor. Input syntax (what a user writes):

```mlir
!nnsimple.quantized<f32, [4], 0.1, 128>
!nnsimple.quantized<f32, [2, 3], 0.5, 0>
```

Printed form (after the op goes through `nnsimple-opt`) — floats get their type suffix:

```mlir
!nnsimple.quantized<f32, [4], 1.000000e-01 : f64, 128>
```

Parameters:
- `elementType`: float type (e.g. `f32`)
- `shape`: static dims, `ArrayRef<int64_t>`
- `scale`: `mlir::FloatAttr` (see hints — plain C++ `double` doesn't work)
- `zeroPoint`: C++ `int64_t`

### Files to edit

| File | What |
|---|---|
| `include/NNSimple/NNSimpleTypes.td` | Define `NNSimple_QuantizedType`. Stub comment shows where. |

Nothing else — types auto-register via `GET_TYPEDEF_LIST` in `NNSimpleTypes.cpp`.

### Hints

- Look at `NNSimple_TensorType` right above your stub — it's the closest match. Copy its structure and swap out the parameter list.
- **Why `FloatAttr` and not `double`?** TypeDef auto-generates a hash of all parameters for uniquing. `llvm::hash_combine` doesn't accept plain `double` (it requires integer-hashable types). `FloatAttr` is a hashable MLIR attribute — the clean way to hold a float in a type parameter. Upstream MLIR types that hold floats (e.g. the Complex dialect) use either `FloatAttr` or `APFloatParameter` with custom assembly.
- Parameter list:
  ```tablegen
  let parameters = (ins
      TypeParameter<"::mlir::FloatType", "float element type">:$elementType,
      ArrayRefParameter<"int64_t", "shape">:$shape,
      "::mlir::FloatAttr":$scale,
      "int64_t":$zeroPoint);
  ```
- `assemblyFormat` — the DSL handles `FloatAttr` natively (prints as `0.1 : f64`, parses bare `0.1` too):
  ```tablegen
  let assemblyFormat = "`<` $elementType `,` `[` $shape `]` `,` $scale `,` $zeroPoint `>`";
  ```

### Done when

```bash
cd build && ninja check-nnsimple
```

passes. The test `test/NNSimple/quantized-type.mlir` parses `<f32, [4], 0.1, 128>`, re-prints, and parses again through `nnsimple-opt | nnsimple-opt`. If your printer and parser agree, FileCheck passes.

### Stretch

- Write a hand-written parser/printer for the quantized type in `NNSimpleTypes.cpp` (removing `assemblyFormat` and setting `let hasCustomAssemblyFormat = 1;`) — useful for learning the C++ API when you need logic in parsing (e.g. default values for missing `zeroPoint`).
- Add a `nnsimple.quantize` op that takes an `!nnsimple.tensor<f32, ...>` and produces an `!nnsimple.quantized<...>` (just a type coercion for now — no actual quantization math).


---

## B-01 — Implement `MulOp::fold` for constant × constant

**Concepts**: constant folding with `OpFoldResult`, `DenseElementsAttr`, `APFloat`.

**Time**: ~1-1.5h.

### Task

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

### Files to edit

| File | What |
|---|---|
| `include/NNSimple/NNSimpleOps.td` | Uncomment `let hasFolder = 1;` inside `NNSimple_MulOp`. |
| `lib/NNSimple/NNSimpleOps.cpp` | Implement `MulOp::fold`. Stub shown above `AddOp::fold`. |

### Hints

- `AddOp::fold` a few lines below your stub is the exact template. Only difference: use `APFloat::multiply(..., APFloat::rmNearestTiesToEven)` instead of `.add(...)`.
- FoldAdaptor gives you `adaptor.getLhs()` / `adaptor.getRhs()` as `Attribute` (already resolved through any constant operands). If the operand isn't constant, they'll be null. Use `dyn_cast_or_null<DenseElementsAttr>`.
- Build the result with `DenseElementsAttr::get(lhsType, results)` where `lhsType` is the attribute's type.
- The `nnsimple.const` op has a constant materializer that turns your returned `ElementsAttr` into a fresh `nnsimple.const` — you don't need to create the `const` yourself.

### Done when

```bash
cd build && ninja check-nnsimple
```

passes. `test/NNSimple/mul-fold.mlir` has one fold case and one no-fold case.

### Stretch

- Also fold when one operand is a splat `1.0` (like `AddZeroElimination` but for `mul-by-one`) — but consider: is that a fold or a canonicalization? (Hint: folders return `OpFoldResult`, which can be a Value; canonicalizations use `OpRewritePattern`. The `mul(x, 1) → x` rewrite is more naturally a canonicalization — see B-02.)


---

## B-02 — Canonicalizations `mul(x, 1) → x` and `mul(x, 0) → 0`

**Concepts**: imperative C++ `OpRewritePattern`, replacing ops with other ops, Commutative trait.

**Time**: ~2h.

### Task

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

### Files to edit

| File | What |
|---|---|
| `include/NNSimple/NNSimpleOps.td` | Uncomment `let hasCanonicalizer = 1;` inside `NNSimple_MulOp`. |
| `lib/NNSimple/NNSimpleOps.cpp` | Implement `MulOneElimination`, `MulZeroElimination`, and `MulOp::getCanonicalizationPatterns`. Stubs are in the file. |

### Hints

- `AddZeroElimination` just above your stubs is the exact template for `MulOneElimination` (swap `.isZero()` → `.isOne()`, i.e. `floatVal.convertToDouble() == 1.0` or `APFloat::getOne(...).bitwiseIsEqual(floatVal)`).
- For `MulZeroElimination`, you need to produce a *new* constant (the zero), not reuse an operand. Use:
  ```cpp
  auto zeroAttr = DenseElementsAttr::get(denseAttr.getType(),
                                          APFloat::getZero(floatVal.getSemantics()));
  rewriter.replaceOpWithNewOp<ConstOp>(op, op.getResult().getType(), zeroAttr);
  ```
- A simpler trick: when rhs is already a zero-splat constant with the right type, you can just replace the op with the rhs value directly: `rewriter.replaceOp(op, op.getRhs());` — but this only works if the result type matches the rhs type (which for nnsimple.mul it does by construction).

### Done when

```bash
cd build && ninja check-nnsimple
```

passes. `test/NNSimple/mul-canonicalize.mlir` has 3 cases: `x*1`, `1*x` (commutative swap), `x*0`.

### Stretch

- `mul(x, -1) → neg(x)` (depends on F-02's neg op).
- `mul(x, x) → square(x)` if you add a `square` op.


---

## C-01 — `-nnsimple-fuse-mul-add` pass

**Concepts**: TableGen-declared passes, `OpRewritePattern`, greedy pattern rewrite driver, multi-operand pattern matching.

**Time**: ~3h.

### Task

Write an optimization pass that fuses `add(mul(a, b), c)` into a single `fused_mul_add(a, b, c)`:

```mlir
// Before -nnsimple-fuse-mul-add:
%0 = nnsimple.mul %a, %b : ...
%1 = nnsimple.add %0, %c : ...
// After:
%1 = nnsimple.fused_mul_add %a, %b, %c : ...
```

**Safety**: only fuse when `%0` has exactly one use. Otherwise another user of the mul would lose its result.

### Files to edit

| File | What |
|---|---|
| `include/NNSimple/NNSimpleOps.td` | Uncomment & fill `NNSimple_FusedMulAddOp`. Three operands. |
| `include/NNSimple/NNSimplePasses.td` | Uncomment `NNSimpleFuseMulAdd`. |
| `lib/NNSimple/NNSimpleFuseMulAdd.cpp` | Implement the pattern and the pass body. Starter skeleton provided. |

The pass is auto-registered with `nnsimple-opt` once declared in the `.td` — no changes needed to `nnsimple-opt.cpp`.

### Hints

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

### Done when

```bash
cd build && ninja check-nnsimple
```

passes. `test/NNSimple/fuse-mul-add.mlir` has a positive case (simple `(a*b)+c`) and a negative case (multi-use mul — must NOT fuse).

### Stretch

- Fuse across commuted pattern: `add(c, mul(a, b))` (the `for` loop above already handles this).
- Add a `negate_c` boolean pass option: when set, treat the pass as fusing `add(mul(a,b), neg(c))` into a new `fused_mul_sub` op. Demonstrates pass options (see the `enable-fuse-add-relu` flag in `NNSimplePasses.cpp:61-66` for the pattern).


---

## C-02 — `-nnsimple-dce` dead-op elimination pass

**Concepts**: op traits (`Pure`), `op.use_empty()`, IR walks, safe erasure during traversal.

**Time**: ~2h.

### Task

Write a dead-code-elimination pass that walks a function and erases every op that is **both**:
- `Pure` (no side effects) — side-effecting ops cannot be safely removed even if unused.
- Uses-empty (`op->use_empty()`) — nobody reads its result.

```mlir
// Before -nnsimple-dce:
func.func @f(%a: !nnsimple.tensor<f32, [4], NCHW>, %b: ...) {
    %0 = nnsimple.add %a, %b : ... // dead — %0 unused
    return
}
// After: only `return` remains.
```

Don't touch non-`Pure` ops even if their result is unused (e.g. `func.call` to a function with side effects must not be removed).

### Files to edit

| File | What |
|---|---|
| `lib/NNSimple/NNSimpleDCE.cpp` | Implement `runOnOperation`. Starter skeleton provided. |

The pass is already declared in `NNSimplePasses.td` and wired into CMake — auto-registered with `nnsimple-opt`.

### Hints

- Walking the IR:
  ```cpp
  // #include "mlir/Interfaces/SideEffectInterfaces.h"  — for mlir::isPure
  SmallVector<Operation *> toErase;
  func.walk([&](Operation *op) {
    // mlir::isPure(op) is the free-function way to query the Pure trait.
    // Also guard against ops with no results (like func.return) — they
    // satisfy use_empty() trivially but you don't want to erase terminators.
    if (mlir::isPure(op) && op->use_empty() && op->getNumResults() > 0)
      toErase.push_back(op);
  });
  for (Operation *op : toErase) op->erase();
  ```
- **Collect-then-erase, in a fixed-point loop**: `walk()` misbehaves if you erase during traversal. Collect first, erase after. And because erasing one op can leave its operands dead, you need a fixed-point loop (the test's third case — `drop_dead_chain` — has a `relu` whose operand `add` is dead; one pass erases the `relu`, the second erases the `add`):
  ```cpp
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> toErase;
    func.walk([&](Operation *op) {
      if (mlir::isPure(op) && op->use_empty() && op->getNumResults() > 0)
        toErase.push_back(op);
    });
    for (Operation *op : toErase) { op->erase(); changed = true; }
  }
  ```
- `func::FuncOp` (the pass's target op) comes from `getOperation()`.

### Done when

```bash
cd build && ninja check-nnsimple
```

passes. `test/NNSimple/dce.mlir` has 3 cases: simple dead op, used op preserved, chain of dead ops (requires the loop).

### Stretch

- Add a pass option `--preserve-ops=nnsimple.const` that skips ops of named kinds even when dead.
- Compare your implementation to MLIR's built-in `-canonicalize` (which does DCE as part of its work). What does `-canonicalize` do differently?
- Why didn't we use `OpRewritePattern` for this? (Hint: erasing an op isn't a *rewrite* — it's a deletion. Greedy driver can do both, but a hand-written walk makes the intent clearer for DCE.)


---

## D-01 — Lower `nnsimple.sub` to `linalg.sub`

**Concepts**: `OpConversionPattern`, dialect conversion, `TypeConverter`.

**Time**: ~2h.

### Setup

F-01's `nnsimple.sub` op is already pre-filled in this starter (so Track D members don't have to do F-01 first). Your job is to teach `-nnsimple-lower-to-linalg` how to lower it.

### Task

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

### Files to edit

| File | What |
|---|---|
| `lib/NNSimple/NNSimpleLinalgLowering.cpp` | Define `SubOpLowering` (mirror `AddOpLowering`), register it in the `patterns.add<...>` call near the bottom. Stub shown above `MulOpLowering`. |

### Hints

- `AddOpLowering` right above your stub is the exact template. Differences:
  - Replace `AddOp` with `SubOp` and `linalg::AddOp` with `linalg::SubOp`.
  - Everything else (grabbing the result type through the type converter, creating a `tensor.empty`, building the linalg op) stays identical.
- Don't forget to register the new pattern in the call at the bottom:
  ```cpp
  patterns.add<AddOpLowering, SubOpLowering, MulOpLowering, ReluOpLowering,
               ConstOpLowering, FusedAddReluOpLowering>(typeConverter, &getContext());
  ```
- The `TypeConverter` at the top of the file already handles `!nnsimple.tensor<f32, [4, 4], NCHW> → tensor<4x4xf32>`. No type-conversion code to write.

### Done when

```bash
cd build && ninja check-nnsimple
```

passes. `test/NNSimple/sub-lowering.mlir` checks that the lowered IR contains `tensor.empty` + `linalg.sub`.

### Stretch

- Add lowerings for all the other ops you've introduced in this curriculum: `neg` (lower to `linalg.generic` with `arith.negf`), `matmul` (lower to `linalg.matmul` — needs a zero-filled init), `fused_mul_add` (lower to `linalg.generic` with `arith.mulf + arith.addf` in the region).
- End-to-end test: pipe through `-nnsimple-pipeline -convert-linalg-to-loops -convert-scf-to-cf -convert-to-llvm | mlir-runner` and check numeric output (see D-02).


---

## D-02 — Lower all the way to LLVM

**Concepts**: multi-step lowering pipeline, `mlir-opt` conversion passes, bufferization.

**Time**: ~3h.

### Task

Take an nnsimple-dialect function and lower it all the way to the LLVM dialect (ready for JIT execution). Your job is to compose the right sequence of conversion passes so that after your pipeline, the module contains only LLVM-dialect ops — no `nnsimple`, no `linalg`, no `tensor`.

### Files to edit

| File | What |
|---|---|
| `test/Integration/add-relu-e2e.mlir` | Replace the `// RUN:` line with the correct pipeline. |

Only the RUN line changes — no C++.

### Pipeline

`-nnsimple-pipeline` handles nnsimple → linalg. After that you still have linalg, arith, tensor, and func ops. The chain that takes it to LLVM:

```
nnsimple-opt %s -nnsimple-pipeline \
  | mlir-opt --one-shot-bufferize="bufferize-function-boundaries" \
             --convert-linalg-to-loops \
             --convert-scf-to-cf \
             --finalize-memref-to-llvm \
             --convert-func-to-llvm \
             --convert-arith-to-llvm \
             --convert-cf-to-llvm \
             --reconcile-unrealized-casts \
  | FileCheck %s
```

### Why each pass

| Pass | Purpose |
|---|---|
| `-nnsimple-pipeline` | Your own dialect → linalg + arith + tensor |
| `--one-shot-bufferize="bufferize-function-boundaries"` | tensor → memref (necessary before linalg can lower to loops) |
| `--convert-linalg-to-loops` | linalg.generic and friends → scf.for + memref loads/stores |
| `--convert-scf-to-cf` | scf.for → cf.br/cf.cond_br (LLVM has no structured CF) |
| `--finalize-memref-to-llvm` | memref → LLVM struct with base ptr, offset, strides |
| `--convert-func-to-llvm` | func.func → llvm.func |
| `--convert-arith-to-llvm` | arith.addf, arith.maximumf → llvm.fadd, llvm.intr.maximum |
| `--convert-cf-to-llvm` | cf.br → llvm.br |
| `--reconcile-unrealized-casts` | drops the cross-dialect cast ops left behind by partial conversions |

### Done when

```bash
cd build && ninja check-nnsimple
```

passes. The test file contains CHECK lines looking for `llvm.func @kernel`, `llvm.fadd`, `llvm.intr.maximum`, and CHECK-NOT lines making sure there's no leftover `nnsimple.`, `linalg.`, or `tensor.empty` in the output.

### Stretch — actually JIT-execute the function and print the result

Wire an `@main` that calls `@kernel` with constant inputs and uses `mlir-runner` to JIT + print:

```mlir
func.func @main() {
    // ... construct two constant tensors, call @kernel, cast the result to
    // tensor<*xf32>, call printMemrefF32 ...
}
func.func private @printMemrefF32(tensor<*xf32>)
```

RUN line:
```
nnsimple-opt %s -nnsimple-pipeline | mlir-opt --one-shot-bufferize=... ...
  | mlir-runner -e main -entry-point-result=void --shared-libs=%mlir_runner_libs
  | FileCheck %s
```

Expected output (for `[1, -2, 3] + [2, 2, -1]` then relu):
```
CHECK: [3,   0,   2]
```

**Why is this a stretch?** Crossing the dialect boundary cleanly (`nnsimple.tensor` ↔ `tensor<1x3xf32>`) without leaving an `unrealized_conversion_cast` that the conversion pass can't materialize requires either (a) writing @main entirely in nnsimple ops or (b) adding source/target materializers to the `NNSimpleTypeConverter` in `lib/NNSimple/NNSimpleLinalgLowering.cpp`. Option (b) is the right answer — look up `TypeConverter::addSourceMaterialization` and `addTargetMaterialization`.

### Why we're not JIT-running as the primary task

Short answer: making @main's standard-tensor inputs flow cleanly into @kernel's nnsimple-tensor signature through the partial conversion requires type-converter plumbing that's out of scope for a 2-day exercise. Lowering to LLVM is the bigger-value learning moment anyway — once you have LLVM IR you're one `mlir-runner` invocation away from execution.


---

