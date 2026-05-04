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

// ============================================================================
// EXERCISE A-02: Implement TensorType::parse and TensorType::print here.
// (These functions only exist once you set `hasCustomAssemblyFormat = 1` in
// NNSimpleTypes.td. Until then this file builds fine with assemblyFormat.)
//
// Grammar:
//   `<` type `,` `[` shape `]` `,` layout-keyword
//       ( `,` `scale` float-attr `,` `zero_point` int-attr )? `>`
//
// API reference: mlir::AsmParser exposes parseLess, parseType, parseComma,
// parseLSquare, parseRSquare, parseInteger, parseKeyword, parseAttribute,
// parseOptionalComma, parseGreater. mlir::AsmPrinter supports operator<<
// and llvm::interleaveComma for arrays.
//
// Use `symbolizeDataLayout(keyword)` to turn the layout keyword into the enum.
// ============================================================================
// ::mlir::Type TensorType::parse(::mlir::AsmParser &parser) {
//   // TODO
//   return {};
// }
// void TensorType::print(::mlir::AsmPrinter &printer) const {
//   // TODO
// }

#define GET_TYPEDEF_CLASSES
#include "NNSimple/NNSimpleOpsTypes.cpp.inc"

void NNSimpleDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "NNSimple/NNSimpleOpsTypes.cpp.inc"
      >();
}
