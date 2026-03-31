//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NNSimple-c/Dialects.h"

#include "NNSimple/NNSimpleDialect.h"
#include "NNSimple/NNSimpleTypes.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(NNSimple, nnsimple,
                                      mlir::nnsimple::NNSimpleDialect)

MlirType mlirNNSimpleCustomTypeGet(MlirContext ctx, MlirStringRef value) {
  return wrap(mlir::nnsimple::CustomType::get(unwrap(ctx), unwrap(value)));
}

bool mlirNNSimpleTypeIsACustomType(MlirType t) {
  return llvm::isa<mlir::nnsimple::CustomType>(unwrap(t));
}

MlirTypeID mlirNNSimpleCustomTypeGetTypeID() {
  return wrap(mlir::nnsimple::CustomType::getTypeID());
}
