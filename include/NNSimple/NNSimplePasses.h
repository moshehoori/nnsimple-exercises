//===- NNSimplePasses.h - NNSimple passes ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef NNSIMPLE_NNSIMPLEPASSES_H
#define NNSIMPLE_NNSIMPLEPASSES_H

#include "NNSimple/NNSimpleDialect.h"
#include "NNSimple/NNSimpleOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace nnsimple {
#define GEN_PASS_DECL
#include "NNSimple/NNSimplePasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "NNSimple/NNSimplePasses.h.inc"

void registerNNSimplePipeline();

} // namespace nnsimple
} // namespace mlir

#endif
