// RUN: nnsimple-opt %s | FileCheck %s



func.func @check_add(%arg0: !nnsimple.tensor<f64, [1, 128], NCHW>) -> !nnsimple.tensor<f64, [1, 128], NCHW> {
    %0 = nnsimple.add %arg0, %arg0 : (!nnsimple.tensor<f64, [1, 128], NCHW>, !nnsimple.tensor<f64, [1, 128], NCHW>) -> !nnsimple.tensor<f64, [1, 128], NCHW>
    return %0 : !nnsimple.tensor<f64, [1, 128], NCHW>
}
// CHECK-LABEL: func @check_add(%arg0: !nnsimple.tensor<f64, [1, 128], NCHW>) -> !nnsimple.tensor<f64, [1, 128], NCHW>
// CHECK: %{{.*}} = nnsimple.add %{{.*}}, %{{.*}} : (!nnsimple.tensor<f64, [1, 128], NCHW>, !nnsimple.tensor<f64, [1, 128], NCHW>) -> !nnsimple.tensor<f64, [1, 128], NCHW>

func.func @check_mul(%arg0: !nnsimple.tensor<f64, [1, 128], NCHW>) -> !nnsimple.tensor<f64, [1, 128], NCHW> {
    %0 = nnsimple.mul %arg0, %arg0 : (!nnsimple.tensor<f64, [1, 128], NCHW>, !nnsimple.tensor<f64, [1, 128], NCHW>) -> !nnsimple.tensor<f64, [1, 128], NCHW>
    return %0 : !nnsimple.tensor<f64, [1, 128], NCHW>
}

// CHECK-LABEL: func @check_mul(%arg0: !nnsimple.tensor<f64, [1, 128], NCHW>) -> !nnsimple.tensor<f64, [1, 128], NCHW>
// CHECK: %{{.*}} = nnsimple.mul %{{.*}}, %{{.*}} : (!nnsimple.tensor<f64, [1, 128], NCHW>, !nnsimple.tensor<f64, [1, 128], NCHW>) -> !
