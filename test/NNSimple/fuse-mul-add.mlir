// RUN: nnsimple-opt %s -nnsimple-fuse-mul-add | FileCheck %s

// add(mul(a, b), c) -> fused_mul_add(a, b, c)
// CHECK-LABEL: func @fma
// CHECK-NOT:   nnsimple.mul
// CHECK-NOT:   nnsimple.add
// CHECK:       nnsimple.fused_mul_add
func.func @fma(%a: !nnsimple.tensor<f32, [4], NCHW>,
               %b: !nnsimple.tensor<f32, [4], NCHW>,
               %c: !nnsimple.tensor<f32, [4], NCHW>)
    -> !nnsimple.tensor<f32, [4], NCHW> {
    %0 = nnsimple.mul %a, %b : (!nnsimple.tensor<f32, [4], NCHW>, !nnsimple.tensor<f32, [4], NCHW>) -> !nnsimple.tensor<f32, [4], NCHW>
    %1 = nnsimple.add %0, %c : (!nnsimple.tensor<f32, [4], NCHW>, !nnsimple.tensor<f32, [4], NCHW>) -> !nnsimple.tensor<f32, [4], NCHW>
    return %1 : !nnsimple.tensor<f32, [4], NCHW>
}

// If the mul result has more than one use, the fusion must NOT fire
// (we'd duplicate work).
// CHECK-LABEL: func @no_fuse_multi_use
// CHECK:       nnsimple.mul
// CHECK:       nnsimple.add
// CHECK-NOT:   nnsimple.fused_mul_add
func.func @no_fuse_multi_use(%a: !nnsimple.tensor<f32, [4], NCHW>,
                             %b: !nnsimple.tensor<f32, [4], NCHW>,
                             %c: !nnsimple.tensor<f32, [4], NCHW>)
    -> (!nnsimple.tensor<f32, [4], NCHW>, !nnsimple.tensor<f32, [4], NCHW>) {
    %0 = nnsimple.mul %a, %b : (!nnsimple.tensor<f32, [4], NCHW>, !nnsimple.tensor<f32, [4], NCHW>) -> !nnsimple.tensor<f32, [4], NCHW>
    %1 = nnsimple.add %0, %c : (!nnsimple.tensor<f32, [4], NCHW>, !nnsimple.tensor<f32, [4], NCHW>) -> !nnsimple.tensor<f32, [4], NCHW>
    return %1, %0 : !nnsimple.tensor<f32, [4], NCHW>, !nnsimple.tensor<f32, [4], NCHW>
}
