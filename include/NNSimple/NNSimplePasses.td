//===- NNSimplePasses.td - NNSimple dialect passes ----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NNSIMPLE_PASS
#define NNSIMPLE_PASS

include "mlir/Pass/PassBase.td"

def NNSimpleFuseAddRelu: Pass<"nnsimple-fuse-add-relu", "::mlir::func::FuncOp"> {
  let summary = "Fuses an add followed by a relu into a single fused op.";
  let description = [{
    Fuses an add followed by a relu into a single fused op.
    ```
    %1 = nnsimple.add %a, %b : tensor<4xf32>
    %2 = nnsimple.relu %1 : tensor<4xf32>
    // Gets transformed to:
    %2 = nnsimple.fused_add_relu %a, %b : tensor<4xf32>
    ```
  }];
}

def NNSimpleLowerToLinalg : Pass<"nnsimple-lower-to-linalg", "::mlir::ModuleOp"> {
  let summary = "Lower NNSimple dialect to Linalg on standard tensors.";
  let description = [{
    Converts nnsimple.tensor types to standard ranked tensor types (dropping
    the layout) and lowers NNSimple ops to linalg.generic + arith operations.

    !nnsimple.tensor<f32, [4, 4], NHWC> -> tensor<4x4xf32>
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

#endif // NNSIMPLE_PASS
