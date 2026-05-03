// EXERCISE D-02: Lower nnsimple all the way to LLVM dialect (+ bonus: JIT run).
//
// Your job: fill in the RUN line below so the full pipeline runs. The CHECK
// lines near the bottom verify that the final IR is entirely in the LLVM
// dialect — no linalg, no tensor, no nnsimple left.
//
// Good starting pipeline:
//   nnsimple-opt %s -nnsimple-pipeline \
//     | mlir-opt --one-shot-bufferize="bufferize-function-boundaries" \
//                --convert-linalg-to-loops --convert-scf-to-cf \
//                --finalize-memref-to-llvm --convert-func-to-llvm \
//                --convert-arith-to-llvm --convert-cf-to-llvm \
//                --reconcile-unrealized-casts \
//     | FileCheck %s
//
// RUN: nnsimple-opt %s | FileCheck %s
// (TODO: replace the RUN line above. Look at the hint block above.)

func.func @kernel(%a: !nnsimple.tensor<f32, [1, 3], NCHW>,
                  %b: !nnsimple.tensor<f32, [1, 3], NCHW>)
    -> !nnsimple.tensor<f32, [1, 3], NCHW> {
    %sum = nnsimple.add %a, %b : (!nnsimple.tensor<f32, [1, 3], NCHW>, !nnsimple.tensor<f32, [1, 3], NCHW>) -> !nnsimple.tensor<f32, [1, 3], NCHW>
    %out = nnsimple.relu %sum : !nnsimple.tensor<f32, [1, 3], NCHW> -> !nnsimple.tensor<f32, [1, 3], NCHW>
    return %out : !nnsimple.tensor<f32, [1, 3], NCHW>
}

// After the full pipeline, @kernel should be an LLVM function with arithmetic
// and control flow in the LLVM dialect.
// CHECK-LABEL: llvm.func @kernel
// CHECK:       llvm.fadd
// CHECK:       llvm.intr.maximum
// CHECK-NOT:   nnsimple.
// CHECK-NOT:   linalg.
// CHECK-NOT:   tensor.empty
