// RUN: nnsimple-opt %s -nnsimple-lower-to-linalg | FileCheck %s

// CHECK-LABEL: func @lower_sub
// CHECK-SAME:  (%[[A:.*]]: tensor<4x4xf32>, %[[B:.*]]: tensor<4x4xf32>)
// CHECK:       %[[INIT:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK:       %[[R:.*]] = linalg.sub ins(%[[A]], %[[B]] : {{.*}}) outs(%[[INIT]]
// CHECK:       return %[[R]]
func.func @lower_sub(%a: !nnsimple.tensor<f32, [4, 4], NCHW>,
                     %b: !nnsimple.tensor<f32, [4, 4], NCHW>)
    -> !nnsimple.tensor<f32, [4, 4], NCHW> {
    %0 = nnsimple.sub %a, %b : (!nnsimple.tensor<f32, [4, 4], NCHW>, !nnsimple.tensor<f32, [4, 4], NCHW>) -> !nnsimple.tensor<f32, [4, 4], NCHW>
    return %0 : !nnsimple.tensor<f32, [4, 4], NCHW>
}
