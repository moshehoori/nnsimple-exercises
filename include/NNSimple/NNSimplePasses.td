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

def NNSimpleSwitchBarFoo: Pass<"nnsimple-switch-bar-foo", "::mlir::ModuleOp"> {
  let summary = "Switches the name of a FuncOp named `bar` to `foo` and folds.";
  let description = [{
    Switches the name of a FuncOp named `bar` to `foo` and folds.
    ```
    func.func @bar() {
      return
    }
    // Gets transformed to:
    func.func @foo() {
      return
    }
    ```
  }];
}

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

#endif // NNSIMPLE_PASS
