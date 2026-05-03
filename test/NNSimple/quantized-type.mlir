// RUN: nnsimple-opt %s | nnsimple-opt | FileCheck %s

// Roundtrip test: quantized type parses, re-prints, parses again.
// CHECK-LABEL: func @roundtrip_quantized
// CHECK-SAME:  !nnsimple.quantized<f32, [4], 1.000000e-01 : f64, 128>
func.func @roundtrip_quantized(%arg0: !nnsimple.quantized<f32, [4], 0.1, 128>)
    -> !nnsimple.quantized<f32, [4], 0.1, 128> {
    return %arg0 : !nnsimple.quantized<f32, [4], 0.1, 128>
}

// CHECK-LABEL: func @roundtrip_quantized_2d
// CHECK-SAME:  !nnsimple.quantized<f32, [2, 3], 5.000000e-01 : f64, 0>
func.func @roundtrip_quantized_2d(%arg0: !nnsimple.quantized<f32, [2, 3], 0.5, 0>)
    -> !nnsimple.quantized<f32, [2, 3], 0.5, 0> {
    return %arg0 : !nnsimple.quantized<f32, [2, 3], 0.5, 0>
}
