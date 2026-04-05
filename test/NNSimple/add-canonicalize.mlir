// RUN: nnsimple-opt -canonicalize %s | FileCheck %s

// Test: constant folding — add(const, const) folds to a single constant.
// CHECK-LABEL: func @fold_add_constants
// CHECK-NEXT: %[[C:.*]] = nnsimple.const dense<3.000000e+00> : tensor<2x2xf32> : !nnsimple.tensor<f32, [2, 2], NCHW>
// CHECK-NEXT: return %[[C]]
func.func @fold_add_constants() -> !nnsimple.tensor<f32, [2, 2], NCHW> {
    %0 = nnsimple.const dense<1.0> : tensor<2x2xf32> : !nnsimple.tensor<f32, [2, 2], NCHW>
    %1 = nnsimple.const dense<2.0> : tensor<2x2xf32> : !nnsimple.tensor<f32, [2, 2], NCHW>
    %2 = nnsimple.add %0, %1 : (!nnsimple.tensor<f32, [2, 2], NCHW>, !nnsimple.tensor<f32, [2, 2], NCHW>) -> !nnsimple.tensor<f32, [2, 2], NCHW>
    return %2 : !nnsimple.tensor<f32, [2, 2], NCHW>
}

// Test: canonicalization — add(x, 0) is eliminated, result is x.
// CHECK-LABEL: func @canonicalize_add_zero
// CHECK-SAME: (%[[ARG:.*]]: !nnsimple.tensor<f32, [2, 2], NCHW>)
// CHECK-NEXT: return %[[ARG]]
func.func @canonicalize_add_zero(%arg0: !nnsimple.tensor<f32, [2, 2], NCHW>) -> !nnsimple.tensor<f32, [2, 2], NCHW> {
    %zero = nnsimple.const dense<0.0> : tensor<2x2xf32> : !nnsimple.tensor<f32, [2, 2], NCHW>
    %0 = nnsimple.add %arg0, %zero : (!nnsimple.tensor<f32, [2, 2], NCHW>, !nnsimple.tensor<f32, [2, 2], NCHW>) -> !nnsimple.tensor<f32, [2, 2], NCHW>
    return %0 : !nnsimple.tensor<f32, [2, 2], NCHW>
}

// Test: canonicalization — add(0, x) also works (commutative sorting moves 0 to rhs).
// CHECK-LABEL: func @canonicalize_zero_add
// CHECK-SAME: (%[[ARG:.*]]: !nnsimple.tensor<f32, [2, 2], NCHW>)
// CHECK-NEXT: return %[[ARG]]
func.func @canonicalize_zero_add(%arg0: !nnsimple.tensor<f32, [2, 2], NCHW>) -> !nnsimple.tensor<f32, [2, 2], NCHW> {
    %zero = nnsimple.const dense<0.0> : tensor<2x2xf32> : !nnsimple.tensor<f32, [2, 2], NCHW>
    %0 = nnsimple.add %zero, %arg0 : (!nnsimple.tensor<f32, [2, 2], NCHW>, !nnsimple.tensor<f32, [2, 2], NCHW>) -> !nnsimple.tensor<f32, [2, 2], NCHW>
    return %0 : !nnsimple.tensor<f32, [2, 2], NCHW>
}
