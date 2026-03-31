//===- NNSimpleTypes.h - NNSimple dialect types -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NNSIMPLE_NNSIMPLETYPES_H
#define NNSIMPLE_NNSIMPLETYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "NNSimple/NNSimpleOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "NNSimple/NNSimpleOpsTypes.h.inc"

#endif // NNSIMPLE_NNSIMPLETYPES_H
