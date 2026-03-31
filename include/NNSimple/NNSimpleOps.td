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
}

def NNSimple_ConstOp : NNSimple_Op<"const", [Pure]> {
    let summary = "Create a constant tensor.";
    let description = [{
        The `nnsimple.const` operation creates a constant tensor with the specified
        values. It takes an attribute containing the tensor values and produces a
        single tensor as output.
    }];

    let arguments = (ins);
    let results = (outs AnyTensor:$output);
    let assemblyFormat = [{
        attr-dict `:` type($output)
    }];
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


#endif // NNSIMPLE_OPS
