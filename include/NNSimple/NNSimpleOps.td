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

// ============================================================================
// EXERCISE A-02: Define the `nnsimple.quantize` op.
//
// The op takes a plain `!nnsimple.tensor` (no quant info) and produces a
// `!nnsimple.tensor` of the same element type, shape, and layout, but with
// scale and zero_point attached. It's a pure type coercion — no arithmetic.
//
// Keep Pure. Same operand and result "tensor shape/type/layout"; verifier
// (in NNSimpleOps.cpp) enforces that the input must NOT be quantized and the
// output MUST be quantized. Mirror NNSimple_ReluOp for the op skeleton.
//
// After uncommenting, also implement QuantizeOp::verify in
// lib/NNSimple/NNSimpleOps.cpp.
// ============================================================================
// def NNSimple_QuantizeOp : NNSimple_Op<"quantize", [Pure]> {
//     let summary = "Attach quantization params (scale, zero_point) to a tensor.";
//     let arguments = (ins NNSimple_TensorType:$input);
//     let results = (outs NNSimple_TensorType:$output);
//     let assemblyFormat = [{
//         $input attr-dict `:` type($input) `->` type($output)
//     }];
//     let hasVerifier = 1;
// }

#endif // NNSIMPLE_OPS
