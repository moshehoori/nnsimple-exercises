// RUN: nnsimple-opt %s | FileCheck %s
// RUN: nnsimple-opt -split-input-file -verify-diagnostics %s

// Positive test: parse+print roundtrip of the new op.
// CHECK-LABEL: func @check_sub
// CHECK: nnsimple.sub %{{.*}}, %{{.*}} : (!nnsimple.tensor<f32, [1, 128], NCHW>, !nnsimple.tensor<f32, [1, 128], NCHW>) -> !nnsimple.tensor<f32, [1, 128], NCHW>
func.func @check_sub(%a: !nnsimple.tensor<f32, [1, 128], NCHW>,
                     %b: !nnsimple.tensor<f32, [1, 128], NCHW>)
    -> !nnsimple.tensor<f32, [1, 128], NCHW> {
    %0 = nnsimple.sub %a, %b : (!nnsimple.tensor<f32, [1, 128], NCHW>, !nnsimple.tensor<f32, [1, 128], NCHW>) -> !nnsimple.tensor<f32, [1, 128], NCHW>
    return %0 : !nnsimple.tensor<f32, [1, 128], NCHW>
}

// -----

// Negative test: operand types must match — verifier rejects mismatched operands.
func.func @bad_sub_operand_mismatch(%a: !nnsimple.tensor<f32, [1, 128], NCHW>,
                                    %b: !nnsimple.tensor<f32, [1, 64], NCHW>)
    -> !nnsimple.tensor<f32, [1, 128], NCHW> {
    // expected-error @+1 {{operand types don't match}}
    %0 = nnsimple.sub %a, %b : (!nnsimple.tensor<f32, [1, 128], NCHW>, !nnsimple.tensor<f32, [1, 64], NCHW>) -> !nnsimple.tensor<f32, [1, 128], NCHW>
    return %0 : !nnsimple.tensor<f32, [1, 128], NCHW>
}

// -----

// Negative test: result type must match operand type.
func.func @bad_sub_result_mismatch(%a: !nnsimple.tensor<f32, [1, 128], NCHW>,
                                   %b: !nnsimple.tensor<f32, [1, 128], NCHW>)
    -> !nnsimple.tensor<f32, [1, 64], NCHW> {
    // expected-error @+1 {{operand type doesn't match result type}}
    %0 = nnsimple.sub %a, %b : (!nnsimple.tensor<f32, [1, 128], NCHW>, !nnsimple.tensor<f32, [1, 128], NCHW>) -> !nnsimple.tensor<f32, [1, 64], NCHW>
    return %0 : !nnsimple.tensor<f32, [1, 64], NCHW>
}
