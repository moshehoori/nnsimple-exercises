//===- NNSimpleTypes.cpp - NNSimple dialect types ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NNSimple/NNSimpleTypes.h"

#include "NNSimple/NNSimpleDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::nnsimple;

#include "NNSimple/NNSimpleOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "NNSimple/NNSimpleOpsTypes.cpp.inc"

void NNSimpleDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "NNSimple/NNSimpleOpsTypes.cpp.inc"
      >();
}
