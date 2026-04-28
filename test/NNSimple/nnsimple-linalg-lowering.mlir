
// RUN: nnsimple-opt --nnsimple-lower-to-linalg %s | FileCheck %s

// CHECK-LABEL: func.func @test_add
// CHECK-SAME: (%[[A:.*]]: tensor<4x4xf32>, %[[B:.*]]: tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK: %[[RES:.*]] = linalg.add ins(%[[A]], %[[B]] : tensor<4x4xf32>, tensor<4x4xf32>) outs(%[[INIT]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: return %[[RES]]
func.func @test_add(
    %a: !nnsimple.tensor<f32, [4, 4], NHWC>,
    %b: !nnsimple.tensor<f32, [4, 4], NHWC>
) -> !nnsimple.tensor<f32, [4, 4], NHWC> {
    %0 = nnsimple.add %a, %b : (!nnsimple.tensor<f32, [4, 4], NHWC>,
                               !nnsimple.tensor<f32, [4, 4], NHWC>)
                           -> !nnsimple.tensor<f32, [4, 4], NHWC>
    return %0 : !nnsimple.tensor<f32, [4, 4], NHWC>
}

// CHECK-LABEL: func.func @test_relu
// CHECK-SAME: (%[[IN:.*]]: tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK:   arith.constant 0.000000e+00
// CHECK:   arith.maximumf
// CHECK:   linalg.yield
func.func @test_relu(
    %input: !nnsimple.tensor<f32, [2, 3], RowMajor>
) -> !nnsimple.tensor<f32, [2, 3], RowMajor> {
    %0 = nnsimple.relu %input : !nnsimple.tensor<f32, [2, 3], RowMajor>
                             -> !nnsimple.tensor<f32, [2, 3], RowMajor>
    return %0 : !nnsimple.tensor<f32, [2, 3], RowMajor>
}
