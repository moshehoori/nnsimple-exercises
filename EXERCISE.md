# D-02 — Lower all the way to LLVM

**Concepts**: multi-step lowering pipeline, `mlir-opt` conversion passes, bufferization.

**Time**: ~3h.

## Task

Take an nnsimple-dialect function and lower it all the way to the LLVM dialect (ready for JIT execution). Your job is to compose the right sequence of conversion passes so that after your pipeline, the module contains only LLVM-dialect ops — no `nnsimple`, no `linalg`, no `tensor`.

## Files to edit

| File | What |
|---|---|
| `test/Integration/add-relu-e2e.mlir` | Replace the `// RUN:` line with the correct pipeline. |

Only the RUN line changes — no C++.

## Pipeline

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

## Why each pass

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

## Done when

```bash
cd build && ninja check-nnsimple
```

passes. The test file contains CHECK lines looking for `llvm.func @kernel`, `llvm.fadd`, `llvm.intr.maximum`, and CHECK-NOT lines making sure there's no leftover `nnsimple.`, `linalg.`, or `tensor.empty` in the output.

## Stretch — actually JIT-execute the function and print the result

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

## Why we're not JIT-running as the primary task

Short answer: making @main's standard-tensor inputs flow cleanly into @kernel's nnsimple-tensor signature through the partial conversion requires type-converter plumbing that's out of scope for a 2-day exercise. Lowering to LLVM is the bigger-value learning moment anyway — once you have LLVM IR you're one `mlir-runner` invocation away from execution.
