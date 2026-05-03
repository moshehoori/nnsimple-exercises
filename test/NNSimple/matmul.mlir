// RUN: nnsimple-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// Positive test: 2x3 * 3x4 -> 2x4 parses and verifies.
// CHECK-LABEL: func @matmul_ok
// CHECK: nnsimple.matmul
func.func @matmul_ok(%a: !nnsimple.tensor<f32, [2, 3], RowMajor>,
                     %b: !nnsimple.tensor<f32, [3, 4], RowMajor>)
    -> !nnsimple.tensor<f32, [2, 4], RowMajor> {
    %0 = nnsimple.matmul %a, %b : (!nnsimple.tensor<f32, [2, 3], RowMajor>, !nnsimple.tensor<f32, [3, 4], RowMajor>) -> !nnsimple.tensor<f32, [2, 4], RowMajor>
    return %0 : !nnsimple.tensor<f32, [2, 4], RowMajor>
}

// -----

// Rank-2 check: rank-3 operand is rejected.
func.func @matmul_bad_rank(%a: !nnsimple.tensor<f32, [1, 2, 3], NCHW>,
                           %b: !nnsimple.tensor<f32, [3, 4], RowMajor>)
    -> !nnsimple.tensor<f32, [2, 4], RowMajor> {
    // expected-error @+1 {{operands must be rank-2}}
    %0 = nnsimple.matmul %a, %b : (!nnsimple.tensor<f32, [1, 2, 3], NCHW>, !nnsimple.tensor<f32, [3, 4], RowMajor>) -> !nnsimple.tensor<f32, [2, 4], RowMajor>
    return %0 : !nnsimple.tensor<f32, [2, 4], RowMajor>
}

// -----

// Inner-dim check: lhs.shape[1]=3 must match rhs.shape[0]=5.
func.func @matmul_bad_inner(%a: !nnsimple.tensor<f32, [2, 3], RowMajor>,
                            %b: !nnsimple.tensor<f32, [5, 4], RowMajor>)
    -> !nnsimple.tensor<f32, [2, 4], RowMajor> {
    // expected-error @+1 {{inner dimensions must match}}
    %0 = nnsimple.matmul %a, %b : (!nnsimple.tensor<f32, [2, 3], RowMajor>, !nnsimple.tensor<f32, [5, 4], RowMajor>) -> !nnsimple.tensor<f32, [2, 4], RowMajor>
    return %0 : !nnsimple.tensor<f32, [2, 4], RowMajor>
}

// -----

// Result-shape check: result must be [lhs.shape[0], rhs.shape[1]].
func.func @matmul_bad_result(%a: !nnsimple.tensor<f32, [2, 3], RowMajor>,
                             %b: !nnsimple.tensor<f32, [3, 4], RowMajor>)
    -> !nnsimple.tensor<f32, [2, 5], RowMajor> {
    // expected-error @+1 {{result shape must be [lhs.shape[0], rhs.shape[1]]}}
    %0 = nnsimple.matmul %a, %b : (!nnsimple.tensor<f32, [2, 3], RowMajor>, !nnsimple.tensor<f32, [3, 4], RowMajor>) -> !nnsimple.tensor<f32, [2, 5], RowMajor>
    return %0 : !nnsimple.tensor<f32, [2, 5], RowMajor>
}
