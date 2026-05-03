// RUN: nnsimple-opt %s -canonicalize | FileCheck %s

// Positive test: neg(neg(x)) collapses to x.
// CHECK-LABEL: func @neg_neg_elim
// CHECK-SAME:  (%[[X:.*]]: !nnsimple.tensor<f32, [2, 2], NCHW>)
// CHECK-NEXT:  return %[[X]]
func.func @neg_neg_elim(%x: !nnsimple.tensor<f32, [2, 2], NCHW>)
    -> !nnsimple.tensor<f32, [2, 2], NCHW> {
    %0 = nnsimple.neg %x : !nnsimple.tensor<f32, [2, 2], NCHW> -> !nnsimple.tensor<f32, [2, 2], NCHW>
    %1 = nnsimple.neg %0 : !nnsimple.tensor<f32, [2, 2], NCHW> -> !nnsimple.tensor<f32, [2, 2], NCHW>
    return %1 : !nnsimple.tensor<f32, [2, 2], NCHW>
}

// Negative (no-op) test: a single neg is left alone.
// CHECK-LABEL: func @single_neg_survives
// CHECK:       nnsimple.neg
func.func @single_neg_survives(%x: !nnsimple.tensor<f32, [2, 2], NCHW>)
    -> !nnsimple.tensor<f32, [2, 2], NCHW> {
    %0 = nnsimple.neg %x : !nnsimple.tensor<f32, [2, 2], NCHW> -> !nnsimple.tensor<f32, [2, 2], NCHW>
    return %0 : !nnsimple.tensor<f32, [2, 2], NCHW>
}
