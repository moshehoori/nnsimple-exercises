//===- NNSimpleTypes.td - NNSimple dialect types -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NNSIMPLE_TYPES
#define NNSIMPLE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "NNSimple/NNSimpleDialect.td"

//===----------------------------------------------------------------------===//
// NNSimple enum definitions
//===----------------------------------------------------------------------===//

def NNSimple_NCHW : I32EnumAttrCase<"NCHW", 0>;
def NNSimple_NHWC : I32EnumAttrCase<"NHWC", 1>;
def NNSimple_NCDHW : I32EnumAttrCase<"NCDHW", 2>;
def NNSimple_NDHWC : I32EnumAttrCase<"NDHWC", 3>;
def NNSimple_RowMajor : I32EnumAttrCase<"RowMajor", 4>;
def NNSimple_ColMajor : I32EnumAttrCase<"ColMajor", 5>;

def NNSimple_DataLayout : I32EnumAttr<"DataLayout",
    "Data layout for tensor types", [
      NNSimple_NCHW,
      NNSimple_NHWC,
      NNSimple_NCDHW,
      NNSimple_NDHWC,
      NNSimple_RowMajor,
      NNSimple_ColMajor
    ]> {
  let cppNamespace = "::mlir::nnsimple";
}

//===----------------------------------------------------------------------===//
// NNSimple type definitions
//===----------------------------------------------------------------------===//

class NNSimple_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<NNSimple_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def NNSimple_TensorType : NNSimple_Type<"Tensor", "tensor"> {
    let summary = "A tensor type with explicit layout information.";
    let description = [{
        The `nnsimple.tensor` type represents a tensor with explicit layout
        information. It includes the element type, shape, and a data layout enum.

        Example:

        ```mlir
        !nnsimple.tensor<f32, [4, 4], NHWC>
        ```
    }];

    let parameters = (ins TypeParameter<"::mlir::Type", "element type">:$elementType,
                         ArrayRefParameter<"int64_t", "shape">:$shape,
                         EnumParameter<NNSimple_DataLayout>:$layout);
    let assemblyFormat = "`<` $elementType `,` `[` $shape `]` `,` $layout `>`";
}

def NNSimple_CustomType : NNSimple_Type<"Custom", "custom"> {
    let summary = "NNSimple custom type";
    let description = "Custom type in nnsimple dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // NNSIMPLE_TYPES
