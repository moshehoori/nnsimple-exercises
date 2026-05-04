# A-02 — Quantization on `!nnsimple.tensor` + `nnsimple.quantize` op

**Concepts**: extending an existing custom type with optional parameters, hand-written parse/print, verifier that reads type parameters.

**Time**: ~3h.

## Task

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

## What you'll build (3 parts)

### Part 1 — Extend `!nnsimple.tensor` with optional params
In `include/NNSimple/NNSimpleTypes.td`, add two `OptionalParameter` entries to the `NNSimple_TensorType` definition:
```tablegen
OptionalParameter<"::mlir::FloatAttr">:$scale,
OptionalParameter<"::mlir::IntegerAttr">:$zeroPoint
```
Optional means existing tensors like `!nnsimple.tensor<f32, [4, 4], NHWC>` keep parsing with `scale`/`zeroPoint` as null.

Also uncomment the `extraClassDeclaration` block that adds a `isQuantized()` helper. You'll use it from the op verifier.

### Part 2 — Hand-write parse/print
The moment you add optional params, the existing declarative `assemblyFormat` won't handle the optional tail well. Replace it with `let hasCustomAssemblyFormat = 1;` in `.td` and implement `TensorType::parse` + `TensorType::print` in `lib/NNSimple/NNSimpleTypes.cpp`.

Grammar:
```
<elementType , [ shape ] , layout-keyword ( , scale float-attr , zero_point int-attr )? >
```

This is the most educational part of the exercise — you learn the actual C++ API that the `assemblyFormat` DSL compiles down to.

### Part 3 — Add `nnsimple.quantize` op
In `include/NNSimple/NNSimpleOps.td`, uncomment `NNSimple_QuantizeOp`. It takes one `NNSimple_TensorType` input and produces one `NNSimple_TensorType` output; `Pure`; `hasVerifier = 1`.

Then in `lib/NNSimple/NNSimpleOps.cpp`, implement `QuantizeOp::verify` to enforce:
- `input` and `output` share element type, shape, and layout → `"input and output must share element type, shape, and layout"`
- `input` must NOT be quantized → `"input must not already be quantized"`
- `output` must be quantized → `"output must be quantized (carry scale and zero_point)"`

## Files to edit

| File | What |
|---|---|
| `include/NNSimple/NNSimpleTypes.td` | Add optional params, switch to `hasCustomAssemblyFormat`, uncomment helper |
| `lib/NNSimple/NNSimpleTypes.cpp` | Implement `TensorType::parse` and `TensorType::print` |
| `include/NNSimple/NNSimpleOps.td` | Uncomment `NNSimple_QuantizeOp` |
| `lib/NNSimple/NNSimpleOps.cpp` | Implement `QuantizeOp::verify` |

The test (`test/NNSimple/quantized-type.mlir`) is already in the repo — don't edit it. Just make it pass.

## Hints

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

## Done when

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

## Stretch

- Print `scale` and `zero_point` without the `: f64`/`: i64` type suffix for nicer output (you'll need to reach into the `FloatAttr`/`IntegerAttr` APIs: `.getValueAsDouble()`, `.getInt()`).
- Add a `nnsimple.dequantize` op (inverse of `quantize`).
- Propagate quantization through the lowering: make `-nnsimple-lower-to-linalg` carry scale/zero_point as attributes on the resulting linalg op (requires touching the type converter).
