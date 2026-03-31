//===- NNSimpleOps.cpp - NNSimple dialect ops -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NNSimple/NNSimpleOps.h"
#include "NNSimple/NNSimpleDialect.h"
#include "NNSimple/NNSimpleTypes.h"

using namespace mlir;
using namespace mlir::nnsimple;

LogicalResult ConstOp::verify() {
  auto attrType = getValue().getShapedType();
  auto resultType = llvm::cast<TensorType>(getOutput().getType());

  if (attrType.getShape() != resultType.getShape())
    return emitOpError("attribute shape doesn't match result shape");

  if (attrType.getElementType() != resultType.getElementType())
    return emitOpError("attribute element type doesn't match result type");

  return success();
}

#define GET_OP_CLASSES
#include "NNSimple/NNSimpleOps.cpp.inc"
