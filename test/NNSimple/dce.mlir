// RUN: nnsimple-opt %s -nnsimple-dce | FileCheck %s

// Dead Pure op (its result is unused) should be erased.
// CHECK-LABEL: func @drop_dead_add
// CHECK-NOT:   nnsimple.add
// CHECK:       return
func.func @drop_dead_add(%a: !nnsimple.tensor<f32, [4], NCHW>,
                         %b: !nnsimple.tensor<f32, [4], NCHW>) {
    %0 = nnsimple.add %a, %b : (!nnsimple.tensor<f32, [4], NCHW>, !nnsimple.tensor<f32, [4], NCHW>) -> !nnsimple.tensor<f32, [4], NCHW>
    return
}

// A used op is preserved.
// CHECK-LABEL: func @keep_used_add
// CHECK:       nnsimple.add
// CHECK:       return
func.func @keep_used_add(%a: !nnsimple.tensor<f32, [4], NCHW>,
                         %b: !nnsimple.tensor<f32, [4], NCHW>)
    -> !nnsimple.tensor<f32, [4], NCHW> {
    %0 = nnsimple.add %a, %b : (!nnsimple.tensor<f32, [4], NCHW>, !nnsimple.tensor<f32, [4], NCHW>) -> !nnsimple.tensor<f32, [4], NCHW>
    return %0 : !nnsimple.tensor<f32, [4], NCHW>
}

// Chain of dead ops: dead add, dead relu on top — both should go.
// CHECK-LABEL: func @drop_dead_chain
// CHECK-NOT:   nnsimple.add
// CHECK-NOT:   nnsimple.relu
// CHECK:       return
func.func @drop_dead_chain(%a: !nnsimple.tensor<f32, [4], NCHW>,
                           %b: !nnsimple.tensor<f32, [4], NCHW>) {
    %0 = nnsimple.add %a, %b : (!nnsimple.tensor<f32, [4], NCHW>, !nnsimple.tensor<f32, [4], NCHW>) -> !nnsimple.tensor<f32, [4], NCHW>
    %1 = nnsimple.relu %0 : !nnsimple.tensor<f32, [4], NCHW> -> !nnsimple.tensor<f32, [4], NCHW>
    return
}
