# A-02 — `!nnsimple.quantized` type

**Concepts**: custom types with parameters (TypeDef), declarative `assemblyFormat`.

**Time**: ~1.5-2h.

## Task

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

## Files to edit

| File | What |
|---|---|
| `include/NNSimple/NNSimpleTypes.td` | Define `NNSimple_QuantizedType`. Stub comment shows where. |

Nothing else — types auto-register via `GET_TYPEDEF_LIST` in `NNSimpleTypes.cpp`.

## Hints

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

## Done when

```bash
cd build && ninja check-nnsimple
```

passes. The test `test/NNSimple/quantized-type.mlir` parses `<f32, [4], 0.1, 128>`, re-prints, and parses again through `nnsimple-opt | nnsimple-opt`. If your printer and parser agree, FileCheck passes.

## Stretch

- Write a hand-written parser/printer for the quantized type in `NNSimpleTypes.cpp` (removing `assemblyFormat` and setting `let hasCustomAssemblyFormat = 1;`) — useful for learning the C++ API when you need logic in parsing (e.g. default values for missing `zeroPoint`).
- Add a `nnsimple.quantize` op that takes an `!nnsimple.tensor<f32, ...>` and produces an `!nnsimple.quantized<...>` (just a type coercion for now — no actual quantization math).
