//===- NNSimpleOps.td - NNSimple dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NNSIMPLE_OPS
#define NNSIMPLE_OPS

include "NNSimple/NNSimpleTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/PatternBase.td"

def NNSimple_AddOp : NNSimple_Op<"add", [Pure, Commutative]> {
    let summary = "Add two tensors element-wise.";
    let description = [{
        The `nnsimple.add` operation performs element-wise addition of two tensors.
        It takes two tensors as input and produces a single tensor as output.
    }];

    let arguments = (ins NNSimple_TensorType:$lhs, NNSimple_TensorType:$rhs);
    let results = (outs NNSimple_TensorType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

def NNSimple_MulOp : NNSimple_Op<"mul", [Pure, Commutative]> {
    let summary = "Multiply two tensors element-wise.";
    let description = [{
        The `nnsimple.mul` operation performs element-wise multiplication of two tensors.
        It takes two tensors as input and produces a single tensor as output.
    }];

    let arguments = (ins NNSimple_TensorType:$lhs, NNSimple_TensorType:$rhs);
    let results = (outs NNSimple_TensorType:$result);
    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];
}

def NNSimple_ReluOp : NNSimple_Op<"relu", [Pure]> {
    let summary = "Apply the ReLU activation function element-wise.";
    let description = [{
        The `nnsimple.relu` operation applies the ReLU activation function to each
        element of the input tensor. It takes a single tensor as input and produces
        a single tensor as output.
    }];

    let arguments = (ins NNSimple_TensorType:$input);
    let results = (outs NNSimple_TensorType:$output);
    let assemblyFormat = [{
        $input attr-dict `:` type($input) `->` type($output)
    }];
    let hasCanonicalizer = 1;
}

def NNSimple_ConstOp : NNSimple_Op<"const", [Pure, ConstantLike]> {
    let summary = "Create a constant tensor.";
    let description = [{
        The `nnsimple.const` operation creates a constant tensor with the specified
        values. It takes an attribute containing the tensor values and produces a
        single tensor as output.
    }];

    let arguments = (ins ElementsAttr:$value);
    let results = (outs NNSimple_TensorType:$output);
    let assemblyFormat = [{
        $value attr-dict `:` type($output)
    }];
    let hasVerifier = 1;
    let hasFolder = 1;
}

// ============================================================================
// EXERCISE A-01: Add the `nnsimple.matmul` op here, with a shape-compatibility
// verifier.
//
// The op takes two rank-2 tensors and produces a rank-2 tensor:
//   matmul(%a: [M, K], %b: [K, N]) -> [M, N]
//
// Verifier must emit:
//   - "operands must be rank-2"   when any operand shape has len != 2
//   - "inner dimensions must match"  when lhs.shape[1] != rhs.shape[0]
//   - "result shape must be [lhs.shape[0], rhs.shape[1]]"  on result mismatch
//
// Mirror NNSimple_AddOp (lines 17-34) for the op skeleton; implement
// MatMulOp::verify in lib/NNSimple/NNSimpleOps.cpp — cast operand types to
// `nnsimple::TensorType` and call `.getShape()` to inspect dimensions.
//
// See EXERCISE.md at the repo root.
// ============================================================================
// def NNSimple_MatMulOp : NNSimple_Op<"matmul", [Pure]> {
//     let summary = "Matrix multiplication.";
//     // TODO: arguments, results, assemblyFormat, hasVerifier
// }

def NNSimple_FusedAddReluOp : NNSimple_Op<"fused_add_relu", [Pure, Commutative]> {
    let summary = "Perform fused addition followed by ReLU activation.";
    let description = [{
        The `nnsimple.fused_add_relu` operation performs element-wise addition of two
        tensors followed by the application of the ReLU activation function. It takes
        two tensors as input and produces a single tensor as output.
    }];

    let arguments = (ins NNSimple_TensorType:$lhs, NNSimple_TensorType:$rhs);
    let results = (outs NNSimple_TensorType:$result);
    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];
}

// Relu(Relu(x)) -> Relu(x)
def ReluReluElimination : Pat<
  (NNSimple_ReluOp (NNSimple_ReluOp $input)),
  (NNSimple_ReluOp $input)>;

#endif // NNSIMPLE_OPS
