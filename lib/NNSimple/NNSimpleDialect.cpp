//===- NNSimpleDialect.cpp - NNSimple dialect -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NNSimple/NNSimpleDialect.h"
#include "NNSimple/NNSimpleOps.h"
#include "NNSimple/NNSimpleTypes.h"

using namespace mlir;
using namespace mlir::nnsimple;

#include "NNSimple/NNSimpleOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// NNSimple dialect.
//===----------------------------------------------------------------------===//

void NNSimpleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NNSimple/NNSimpleOps.cpp.inc"
      >();
  registerTypes();
}
