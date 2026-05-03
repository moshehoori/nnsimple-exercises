// RUN: nnsimple-opt %s -canonicalize | FileCheck %s

// mul(const 2.0, const 3.0) -> const 6.0
// CHECK-LABEL: func @fold_mul_constants
// CHECK-NEXT:  %[[C:.*]] = nnsimple.const dense<6.000000e+00>
// CHECK-NEXT:  return %[[C]]
func.func @fold_mul_constants() -> !nnsimple.tensor<f32, [2, 2], NCHW> {
    %0 = nnsimple.const dense<2.0> : tensor<2x2xf32> : !nnsimple.tensor<f32, [2, 2], NCHW>
    %1 = nnsimple.const dense<3.0> : tensor<2x2xf32> : !nnsimple.tensor<f32, [2, 2], NCHW>
    %2 = nnsimple.mul %0, %1 : (!nnsimple.tensor<f32, [2, 2], NCHW>, !nnsimple.tensor<f32, [2, 2], NCHW>) -> !nnsimple.tensor<f32, [2, 2], NCHW>
    return %2 : !nnsimple.tensor<f32, [2, 2], NCHW>
}

// With a non-constant operand, the mul should NOT fold.
// CHECK-LABEL: func @no_fold_with_arg
// CHECK:       nnsimple.mul
func.func @no_fold_with_arg(%arg0: !nnsimple.tensor<f32, [2, 2], NCHW>)
    -> !nnsimple.tensor<f32, [2, 2], NCHW> {
    %0 = nnsimple.const dense<3.0> : tensor<2x2xf32> : !nnsimple.tensor<f32, [2, 2], NCHW>
    %1 = nnsimple.mul %arg0, %0 : (!nnsimple.tensor<f32, [2, 2], NCHW>, !nnsimple.tensor<f32, [2, 2], NCHW>) -> !nnsimple.tensor<f32, [2, 2], NCHW>
    return %1 : !nnsimple.tensor<f32, [2, 2], NCHW>
}
