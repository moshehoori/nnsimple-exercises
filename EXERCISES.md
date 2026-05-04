# NNSimple MLIR Exercises — Consolidated Reference

Every learner works through all 10 exercises individually, roughly easiest → hardest. For the 2-day schedule see [CURRICULUM.md](CURRICULUM.md). Per exercise, `git checkout starter/<id>` to see the stubs and failing test; the `EXERCISE.md` on each starter branch is the same content as below.

## Contents

**Foundation**
- [F-01 — Add the `nnsimple.sub` op](#f-01--add-the-nnsimplesub-op)
- [F-02 — Add `nnsimple.neg` with a declarative canonicalization](#f-02--add-nnsimpleneg-with-a-declarative-canonicalization)

**Ops & Type System**
- [A-01 — `nnsimple.matmul` with shape-compatibility verifier](#a-01--nnsimplematmul-with-shape-compatibility-verifier)
- [A-02 — Quantization on `!nnsimple.tensor` + `nnsimple.quantize` op](#a-02--quantization-on-nnsimpletensor--nnsimplequantize-op)

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

The test for this exercise (`test/NNSimple/sub-ops.mlir`) is already in the repo — don't edit it. Just make it pass.

### Hints

- Look at `NNSimple_AddOp` in `NNSimpleOps.td` (lines ~17-34) — the new op is almost identical but drops the `Commutative` trait and drops `hasFolder`/`hasCanonicalizer`.
- Look at `AddOp::verify` in `NNSimpleOps.cpp` (just below where you'll add `SubOp::verify`) — same idea, verbatim.
- Don't forget `let hasVerifier = 1;` on the TableGen side.

### Done when

`test/NNSimple/sub-ops.mlir` goes from red to green. It's one positive parse test and two negative tests (mismatched operand shapes, mismatched result shape) that exercise both verifier diagnostics.

```bash
cd build && ninja nnsimple-opt
llvm-lit -v ../test/NNSimple/sub-ops.mlir   # should PASS
```

Or run it manually (useful if `llvm-lit` isn't on your PATH):
```bash
./bin/nnsimple-opt ../test/NNSimple/sub-ops.mlir -split-input-file -verify-diagnostics \
    | /path/to/llvm-project/build/bin/FileCheck ../test/NNSimple/sub-ops.mlir
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

The test for this exercise (`test/NNSimple/neg-canonicalize.mlir`) is already in the repo — don't edit it. Just make it pass.

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

`test/NNSimple/neg-canonicalize.mlir` goes from red to green. It's a positive test (`neg(neg(x))` is gone after `-canonicalize`) and a sanity test (a single `neg` survives).

```bash
cd build && ninja nnsimple-opt
llvm-lit -v ../test/NNSimple/neg-canonicalize.mlir   # should PASS
```

Or manually:
```bash
./bin/nnsimple-opt ../test/NNSimple/neg-canonicalize.mlir -canonicalize \
    | /path/to/llvm-project/build/bin/FileCheck ../test/NNSimple/neg-canonicalize.mlir
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

The test for this exercise (`test/NNSimple/matmul.mlir`) is already in the repo — don't edit it. Just make it pass.

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

`test/NNSimple/matmul.mlir` goes from red to green — 1 positive, 3 negatives (rank, inner dim, result shape).

```bash
cd build && ninja nnsimple-opt
llvm-lit -v ../test/NNSimple/matmul.mlir   # should PASS
```

### Stretch

- Support batched matmul: `[B, M, K] * [B, K, N] -> [B, M, N]` — verifier checks all three dims including broadcast of B=1.
- Add a `TransposeAttr` option: `matmul %a, %b { transpose_lhs = true } : ...`.


---

## A-02 — Quantization on `!nnsimple.tensor` + `nnsimple.quantize` op

**Concepts**: extending an existing custom type with optional parameters, hand-written parse/print, verifier that reads type parameters.

**Time**: ~3h.

### Task

Extend the existing `!nnsimple.tensor` type with **optional** `scale` and `zero_point` fields, so a tensor can be marked as quantized. Then add a `nnsimple.quantize` op that attaches quantization info to an unquantized tensor.

Target syntax (both forms must coexist — the plain form still works for `add`/`mul`/... as before):

```mlir
// Plain (existing):
!nnsimple.tensor<f32, [4, 4], NHWC>

// Quantized (new):
!nnsimple.tensor<f32, [4], NCHW, scale 0.1, zero_point 128>

// The new op:
%q = nnsimple.quantize %x
    : !nnsimple.tensor<f32, [4], NCHW>
   -> !nnsimple.tensor<f32, [4], NCHW, scale 0.1, zero_point 128>
```

### What you'll build (3 parts)

#### Part 1 — Extend `!nnsimple.tensor` with optional params
In `include/NNSimple/NNSimpleTypes.td`, add two `OptionalParameter` entries to the `NNSimple_TensorType` definition:
```tablegen
OptionalParameter<"::mlir::FloatAttr">:$scale,
OptionalParameter<"::mlir::IntegerAttr">:$zeroPoint
```
Optional means existing tensors like `!nnsimple.tensor<f32, [4, 4], NHWC>` keep parsing with `scale`/`zeroPoint` as null.

Also uncomment the `extraClassDeclaration` block that adds a `isQuantized()` helper. You'll use it from the op verifier.

#### Part 2 — Hand-write parse/print
The moment you add optional params, the existing declarative `assemblyFormat` won't handle the optional tail well. Replace it with `let hasCustomAssemblyFormat = 1;` in `.td` and implement `TensorType::parse` + `TensorType::print` in `lib/NNSimple/NNSimpleTypes.cpp`.

Grammar:
```
<elementType , [ shape ] , layout-keyword ( , scale float-attr , zero_point int-attr )? >
```

This is the most educational part of the exercise — you learn the actual C++ API that the `assemblyFormat` DSL compiles down to.

#### Part 3 — Add `nnsimple.quantize` op
In `include/NNSimple/NNSimpleOps.td`, uncomment `NNSimple_QuantizeOp`. It takes one `NNSimple_TensorType` input and produces one `NNSimple_TensorType` output; `Pure`; `hasVerifier = 1`.

Then in `lib/NNSimple/NNSimpleOps.cpp`, implement `QuantizeOp::verify` to enforce:
- `input` and `output` share element type, shape, and layout → `"input and output must share element type, shape, and layout"`
- `input` must NOT be quantized → `"input must not already be quantized"`
- `output` must be quantized → `"output must be quantized (carry scale and zero_point)"`

### Files to edit

| File | What |
|---|---|
| `include/NNSimple/NNSimpleTypes.td` | Add optional params, switch to `hasCustomAssemblyFormat`, uncomment helper |
| `lib/NNSimple/NNSimpleTypes.cpp` | Implement `TensorType::parse` and `TensorType::print` |
| `include/NNSimple/NNSimpleOps.td` | Uncomment `NNSimple_QuantizeOp` |
| `lib/NNSimple/NNSimpleOps.cpp` | Implement `QuantizeOp::verify` |

The test (`test/NNSimple/quantized-type.mlir`) is already in the repo — don't edit it. Just make it pass.

### Hints

**Why optional params instead of a separate `!nnsimple.quantized` type?** Reuses the existing type and all the ops/passes that already accept `NNSimple_TensorType` automatically work for quantized tensors too. A separate type would require duplicating every op's type constraint or writing a type-constraint union.

**Why FloatAttr / IntegerAttr and not plain `double` / `int64_t`?** TypeDef auto-hashes every parameter for uniquing. `llvm::hash_combine` doesn't accept plain `double`; MLIR's attribute types (which are hashable) are the idiomatic workaround.

**Parser tips:**
- `parser.parseLess()`, `parser.parseGreater()` — angle brackets.
- `parser.parseType(someFloatType)` — parses `f32`, `f64`, etc.
- `parser.parseLSquare()`, `parser.parseRSquare()`, `parser.parseCommaSeparatedList(lambda)` — for the shape `[...]`.
- `parser.parseKeyword(&stringRef)` — for `NCHW`, `NHWC`, `scale`, `zero_point`.
- `symbolizeDataLayout(keyword)` — turns a layout keyword back into the `DataLayout` enum (defined by the `I32EnumAttr` in `NNSimpleTypes.td`).
- `parser.parseAttribute(floatAttrOrIntegerAttr)` — parses `0.1` as `FloatAttr`, `128` as `IntegerAttr`.
- `parser.parseOptionalComma()` — the quant tail is optional.

**Printer tips:**
- `printer << "<" << getElementType() << ", [";`
- `llvm::interleaveComma(getShape(), printer);`
- `printer << "], " << stringifyDataLayout(getLayout());`
- `if (isQuantized()) printer << ", scale " << getScale() << ", zero_point " << getZeroPoint();`

**Verifier tips:** `llvm::cast<nnsimple::TensorType>(getInput().getType())` gets you the typed handle; then `.getElementType()`, `.getShape()`, `.getLayout()`, `.isQuantized()`, `.getScale()`, `.getZeroPoint()`.

### Done when

`test/NNSimple/quantized-type.mlir` goes from red to green. It has 6 cases:
1. Plain tensor roundtrip (backward compat)
2. Quantized tensor roundtrip
3. `nnsimple.quantize` positive case
4. Double-quantize rejected
5. Unquantized output rejected
6. Shape mismatch rejected

```bash
cd build && ninja nnsimple-opt
llvm-lit -v ../test/NNSimple/quantized-type.mlir   # should PASS
```

Or manually:
```bash
./bin/nnsimple-opt ../test/NNSimple/quantized-type.mlir -split-input-file -verify-diagnostics \
    | /path/to/llvm-project/build/bin/FileCheck ../test/NNSimple/quantized-type.mlir
```

### Stretch

- Print `scale` and `zero_point` without the `: f64`/`: i64` type suffix for nicer output (you'll need to reach into the `FloatAttr`/`IntegerAttr` APIs: `.getValueAsDouble()`, `.getInt()`).
- Add a `nnsimple.dequantize` op (inverse of `quantize`).
- Propagate quantization through the lowering: make `-nnsimple-lower-to-linalg` carry scale/zero_point as attributes on the resulting linalg op (requires touching the type converter).


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

The test for this exercise (`test/NNSimple/mul-fold.mlir`) is already in the repo — don't edit it. Just make it pass.

### Hints

- `AddOp::fold` a few lines below your stub is the exact template. Only difference: use `APFloat::multiply(..., APFloat::rmNearestTiesToEven)` instead of `.add(...)`.
- FoldAdaptor gives you `adaptor.getLhs()` / `adaptor.getRhs()` as `Attribute` (already resolved through any constant operands). If the operand isn't constant, they'll be null. Use `dyn_cast_or_null<DenseElementsAttr>`.
- Build the result with `DenseElementsAttr::get(lhsType, results)` where `lhsType` is the attribute's type.
- The `nnsimple.const` op has a constant materializer that turns your returned `ElementsAttr` into a fresh `nnsimple.const` — you don't need to create the `const` yourself.

### Done when

`test/NNSimple/mul-fold.mlir` goes from red to green. It has one fold case and one no-fold case.

```bash
cd build && ninja nnsimple-opt
llvm-lit -v ../test/NNSimple/mul-fold.mlir   # should PASS
```

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

The test for this exercise (`test/NNSimple/mul-canonicalize.mlir`) is already in the repo — don't edit it. Just make it pass.

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

`test/NNSimple/mul-canonicalize.mlir` goes from red to green. It has 3 cases: `x*1`, `1*x` (commutative swap), `x*0`.

```bash
cd build && ninja nnsimple-opt
llvm-lit -v ../test/NNSimple/mul-canonicalize.mlir   # should PASS
```

### Stretch

- `mul(x, -1) → neg(x)` (depends on F-02's neg op).
- `mul(x, x) → square(x)` if you add a `square` op.


---

## C-01 — `-nnsimple-fuse-mul-add` pass

**Concepts**: TableGen-declared passes, `OpRewritePattern`, greedy pattern rewrite driver, multi-operand pattern matching.

**Time**: ~1.5h.

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

The test for this exercise (`test/NNSimple/fuse-mul-add.mlir`) is already in the repo — don't edit it. Just make it pass.

The pass is auto-registered with `nnsimple-opt` once declared in the `.td` — no changes needed to `nnsimple-opt.cpp`.

### Hints

- **The op**: three-operand equivalent of `NNSimple_FusedAddReluOp` (lines ~83-96 of `NNSimpleOps.td`). The existing op has two operands — yours has three. The assembly format extends the same way.
- **The pattern**: mirror `NNSimpleFuseAddReluRewriter` in `lib/NNSimple/NNSimplePasses.cpp` (lines 24-42). That one matches `ReluOp` and looks for an `AddOp` producing its input. Yours matches `AddOp` and looks for a single-use `MulOp` producing either operand.
  - Use `Value::getDefiningOp<MulOp>()` to check the producing op.
  - Use `op->hasOneUse()` for the safety check.
  - Use `rewriter.replaceOp(addOp, fused.getResult())` + `rewriter.eraseOp(mulOp)` to swap in the new op.
- **Commuted pattern**: remember `add(c, mul(a,b))` should also fuse — try each operand of the add.
- **The pass body**: look at `NNSimpleFuseAddRelu::runOnOperation` (same file, lines 50-57). Build a `RewritePatternSet`, add your rewriter, call `applyPatternsGreedily(func, ...)`.

### Done when

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

### Stretch

- Add a `negate_c` boolean pass option: when set, treat the pass as fusing `add(mul(a,b), neg(c))` into a new `fused_mul_sub` op. Demonstrates pass options (see the `enable-fuse-add-relu` flag in `NNSimplePasses.cpp:61-66` for the pattern).
- Also handle `sub(mul(a, b), c) → fused_mul_sub(a, b, c)` once F-01's `sub` op is in place.


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

The test for this exercise (`test/NNSimple/dce.mlir`) is already in the repo — don't edit it. Just make it pass.

The pass is already declared in `NNSimplePasses.td` and wired into CMake — auto-registered with `nnsimple-opt`.

### Hints

- The pass's target op comes from `getOperation()` — for this pass that's a `func::FuncOp`.
- Read [Understanding the IR Structure — Operation Walkers](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/#operation-walkers) for how to traverse the IR with `op.walk(...)`.
- To query the `Pure` trait, `#include "mlir/Interfaces/SideEffectInterfaces.h"` and call the free function `mlir::isPure(op)`.
- To check whether an op's result is consumed: `op->use_empty()`.
- Two gotchas to think through yourself:
  - If you call `op->erase()` from inside `walk()`, things break. Why? How would you work around it?
  - `func.return` satisfies `use_empty()` trivially (it has no results). You don't want to erase terminators. What extra check keeps it alive?
- **Fixed-point loop**: the third test case (`drop_dead_chain`) has a `relu` whose operand `add` is also dead. A single DCE pass erases the `relu`, but at that point the `add` still existed when you collected dead ops, so it survives that iteration. Loop until nothing changes.

### Done when

`test/NNSimple/dce.mlir` goes from red to green. It has 3 cases: simple dead op, used op preserved, chain of dead ops (requires the fixed-point loop).

```bash
cd build && ninja nnsimple-opt
llvm-lit -v ../test/NNSimple/dce.mlir   # should PASS
```

Or manually:
```bash
./bin/nnsimple-opt ../test/NNSimple/dce.mlir -nnsimple-dce \
    | /path/to/llvm-project/build/bin/FileCheck ../test/NNSimple/dce.mlir
```

### Stretch

- Add a pass option `--preserve-ops=nnsimple.const` that skips ops of named kinds even when dead.
- Compare your implementation to MLIR's built-in `-canonicalize` (which does DCE as part of its work). What does `-canonicalize` do differently?
- Why didn't we use `OpRewritePattern` for this? (Hint: erasing an op isn't a *rewrite* — it's a deletion. Greedy driver can do both, but a hand-written walk makes the intent clearer for DCE.)


---

## D-01 — Lower `nnsimple.sub` to `linalg.sub`

**Concepts**: `OpConversionPattern`, dialect conversion, `TypeConverter`.

**Time**: ~2h.

> ⚠️ **D-01 and D-02 are the toughest exercises in the set.** If your MLIR background is brand new, don't worry if you don't finish D-02 — the conceptual walk-through is more important than a green test. Please do make it to D-01 at least; the lowering pattern here shows up in every real MLIR dialect.

### Setup

F-01's `nnsimple.sub` op is already pre-filled in this starter (so you don't have to re-do F-01 here). Your job is to teach `-nnsimple-lower-to-linalg` how to lower it.

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

The test for this exercise (`test/NNSimple/sub-lowering.mlir`) is already in the repo — don't edit it. Just make it pass.

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

`test/NNSimple/sub-lowering.mlir` goes from red to green. The CHECK lines verify the lowered IR contains `tensor.empty` + `linalg.sub`.

```bash
cd build && ninja nnsimple-opt
llvm-lit -v ../test/NNSimple/sub-lowering.mlir   # should PASS
```

Or manually:
```bash
./bin/nnsimple-opt ../test/NNSimple/sub-lowering.mlir -nnsimple-lower-to-linalg \
    | /path/to/llvm-project/build/bin/FileCheck ../test/NNSimple/sub-lowering.mlir
```

### Stretch

- Add lowerings for all the other ops you've introduced in this curriculum: `neg` (lower to `linalg.generic` with `arith.negf`), `matmul` (lower to `linalg.matmul` — needs a zero-filled init), `fused_mul_add` (lower to `linalg.generic` with `arith.mulf + arith.addf` in the region).
- End-to-end test: pipe through `-nnsimple-pipeline -convert-linalg-to-loops -convert-scf-to-cf -convert-to-llvm | mlir-runner` and check numeric output (see D-02).


---

## D-02 — Lower all the way to LLVM

**Concepts**: multi-step lowering pipeline, `mlir-opt` conversion passes, bufferization.

**Time**: ~3h.

> ⚠️ **D-02 is the hardest exercise in the set.** It's mostly reading (figuring out which passes are needed, in which order, and why), not writing C++. If you're running short on day 2, it's fine to only read through the pipeline and understand *why* each pass is there — the CHECK test is a bonus goal, not a gate.
>
> **Recommended reading**: [MLIR Bufferization docs](https://mlir.llvm.org/docs/Bufferization/) — explains the tensor→memref conversion that `--one-shot-bufferize` performs, which is the most confusing pass in the pipeline.

### Task

Take an nnsimple-dialect function and lower it all the way to the LLVM dialect (ready for JIT execution). Your job is to compose the right sequence of conversion passes so that after your pipeline, the module contains only LLVM-dialect ops — no `nnsimple`, no `linalg`, no `tensor`.

### Files to edit

| File | What |
|---|---|
| `test/Integration/add-relu-e2e.mlir` | Replace the `// RUN:` line with the correct pipeline. |

Only the RUN line changes — no C++. The `.mlir` body and the CHECK lines at the bottom are already in the repo — don't edit them. Just make the RUN line run the right pipeline.

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

`test/Integration/add-relu-e2e.mlir` goes from red to green. The CHECK lines look for `llvm.func @kernel`, `llvm.fadd`, `llvm.intr.maximum`, with CHECK-NOT lines making sure there's no leftover `nnsimple.`, `linalg.`, or `tensor.empty`.

```bash
cd build && ninja nnsimple-opt
llvm-lit -v ../test/Integration/add-relu-e2e.mlir   # should PASS
```

Or run the pipeline manually to see the intermediate IR:
```bash
./bin/nnsimple-opt ../test/Integration/add-relu-e2e.mlir -nnsimple-pipeline \
    | /path/to/llvm-project/build/bin/mlir-opt --one-shot-bufferize=... ...
```

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

