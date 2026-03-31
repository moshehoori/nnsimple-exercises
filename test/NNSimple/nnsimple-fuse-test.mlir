// RUN: nnsimple-opt -nnsimple-fuse-add-relu %s | FileCheck %s



func.func @check_add(%arg0: !nnsimple.tensor<f64, [1, 128], NCHW>, %arg1: !nnsimple.tensor<f64, [1, 128], NCHW>) -> !nnsimple.tensor<f64, [1, 128], NCHW> {
    %0 = nnsimple.add %arg0, %arg1 : (!nnsimple.tensor<f64, [1, 128], NCHW>, !nnsimple.tensor<f64, [1, 128], NCHW>) -> !nnsimple.tensor<f64, [1, 128], NCHW>
    %1 = nnsimple.relu %0 : !nnsimple.tensor<f64, [1, 128], NCHW> -> !nnsimple.tensor<f64, [1, 128], NCHW>
    return %1 : !nnsimple.tensor<f64, [1, 128], NCHW>
}
// CHECK-LABEL: func @check_add(%arg0: !nnsimple.tensor<f64, [1, 128], NCHW>, %arg1: !nnsimple.tensor<f64, [1, 128], NCHW>) -> !nnsimple.tensor<f64, [1, 128], NCHW>
// CHECK: %{{.*}} = nnsimple.fused_add_relu %{{.*}}, %{{.*}} : (!nnsimple.tensor<f64, [1, 128], NCHW>, !nnsimple.tensor<f64, [1, 128], NCHW>) -> !nnsimple.tensor<f64, [1, 128], NCHW>
