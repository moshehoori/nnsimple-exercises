// RUN: nnsimple-opt %s -canonicalize | FileCheck %s

// mul(x, 1) -> x
// CHECK-LABEL: func @mul_by_one
// CHECK-SAME:  (%[[ARG:.*]]: !nnsimple.tensor<f32, [2, 2], NCHW>)
// CHECK-NEXT:  return %[[ARG]]
func.func @mul_by_one(%arg0: !nnsimple.tensor<f32, [2, 2], NCHW>)
    -> !nnsimple.tensor<f32, [2, 2], NCHW> {
    %one = nnsimple.const dense<1.0> : tensor<2x2xf32> : !nnsimple.tensor<f32, [2, 2], NCHW>
    %0 = nnsimple.mul %arg0, %one : (!nnsimple.tensor<f32, [2, 2], NCHW>, !nnsimple.tensor<f32, [2, 2], NCHW>) -> !nnsimple.tensor<f32, [2, 2], NCHW>
    return %0 : !nnsimple.tensor<f32, [2, 2], NCHW>
}

// Commutative swap: mul(1, x) also folds (canonicalizer sorts const to rhs).
// CHECK-LABEL: func @one_mul
// CHECK-SAME:  (%[[ARG:.*]]: !nnsimple.tensor<f32, [2, 2], NCHW>)
// CHECK-NEXT:  return %[[ARG]]
func.func @one_mul(%arg0: !nnsimple.tensor<f32, [2, 2], NCHW>)
    -> !nnsimple.tensor<f32, [2, 2], NCHW> {
    %one = nnsimple.const dense<1.0> : tensor<2x2xf32> : !nnsimple.tensor<f32, [2, 2], NCHW>
    %0 = nnsimple.mul %one, %arg0 : (!nnsimple.tensor<f32, [2, 2], NCHW>, !nnsimple.tensor<f32, [2, 2], NCHW>) -> !nnsimple.tensor<f32, [2, 2], NCHW>
    return %0 : !nnsimple.tensor<f32, [2, 2], NCHW>
}

// mul(x, 0) -> 0
// CHECK-LABEL: func @mul_by_zero
// CHECK-NEXT:  %[[C:.*]] = nnsimple.const dense<0.000000e+00>
// CHECK-NEXT:  return %[[C]]
func.func @mul_by_zero(%arg0: !nnsimple.tensor<f32, [2, 2], NCHW>)
    -> !nnsimple.tensor<f32, [2, 2], NCHW> {
    %zero = nnsimple.const dense<0.0> : tensor<2x2xf32> : !nnsimple.tensor<f32, [2, 2], NCHW>
    %0 = nnsimple.mul %arg0, %zero : (!nnsimple.tensor<f32, [2, 2], NCHW>, !nnsimple.tensor<f32, [2, 2], NCHW>) -> !nnsimple.tensor<f32, [2, 2], NCHW>
    return %0 : !nnsimple.tensor<f32, [2, 2], NCHW>
}
