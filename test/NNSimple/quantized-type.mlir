// RUN: nnsimple-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// Unquantized tensor still roundtrips through the new hand-written parse/print.
// CHECK-LABEL: func @roundtrip_plain
// CHECK-SAME:  !nnsimple.tensor<f32, [4, 4], NHWC>
func.func @roundtrip_plain(%arg0: !nnsimple.tensor<f32, [4, 4], NHWC>)
    -> !nnsimple.tensor<f32, [4, 4], NHWC> {
    return %arg0 : !nnsimple.tensor<f32, [4, 4], NHWC>
}

// -----

// Quantized tensor type roundtrips.
// CHECK-LABEL: func @roundtrip_quantized
// CHECK-SAME:  !nnsimple.tensor<f32, [4], NCHW, scale 1.000000e-01 : f64, zero_point 128 : i64>
func.func @roundtrip_quantized(%arg0: !nnsimple.tensor<f32, [4], NCHW, scale 0.1, zero_point 128>)
    -> !nnsimple.tensor<f32, [4], NCHW, scale 0.1, zero_point 128> {
    return %arg0 : !nnsimple.tensor<f32, [4], NCHW, scale 0.1, zero_point 128>
}

// -----

// nnsimple.quantize attaches the quant info.
// CHECK-LABEL: func @do_quantize
// CHECK:       nnsimple.quantize
func.func @do_quantize(%arg0: !nnsimple.tensor<f32, [4], NCHW>)
    -> !nnsimple.tensor<f32, [4], NCHW, scale 0.1, zero_point 128> {
    %0 = nnsimple.quantize %arg0
        : !nnsimple.tensor<f32, [4], NCHW>
       -> !nnsimple.tensor<f32, [4], NCHW, scale 0.1, zero_point 128>
    return %0 : !nnsimple.tensor<f32, [4], NCHW, scale 0.1, zero_point 128>
}

// -----

// Verifier rejects double-quantize.
func.func @double_q(%arg0: !nnsimple.tensor<f32, [4], NCHW, scale 0.1, zero_point 0>)
    -> !nnsimple.tensor<f32, [4], NCHW, scale 0.2, zero_point 128> {
    // expected-error @+1 {{input must not already be quantized}}
    %0 = nnsimple.quantize %arg0
        : !nnsimple.tensor<f32, [4], NCHW, scale 0.1, zero_point 0>
       -> !nnsimple.tensor<f32, [4], NCHW, scale 0.2, zero_point 128>
    return %0 : !nnsimple.tensor<f32, [4], NCHW, scale 0.2, zero_point 128>
}

// -----

// Verifier rejects unquantized output.
func.func @no_q_out(%arg0: !nnsimple.tensor<f32, [4], NCHW>)
    -> !nnsimple.tensor<f32, [4], NCHW> {
    // expected-error @+1 {{output must be quantized}}
    %0 = nnsimple.quantize %arg0
        : !nnsimple.tensor<f32, [4], NCHW>
       -> !nnsimple.tensor<f32, [4], NCHW>
    return %0 : !nnsimple.tensor<f32, [4], NCHW>
}

// -----

// Verifier rejects shape/layout mismatches.
func.func @shape_mismatch(%arg0: !nnsimple.tensor<f32, [4], NCHW>)
    -> !nnsimple.tensor<f32, [8], NCHW, scale 0.1, zero_point 0> {
    // expected-error @+1 {{input and output must share element type, shape, and layout}}
    %0 = nnsimple.quantize %arg0
        : !nnsimple.tensor<f32, [4], NCHW>
       -> !nnsimple.tensor<f32, [8], NCHW, scale 0.1, zero_point 0>
    return %0 : !nnsimple.tensor<f32, [8], NCHW, scale 0.1, zero_point 0>
}
