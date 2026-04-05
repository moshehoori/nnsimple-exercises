//===- NNSimpleDialect.td - NNSimple dialect ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NNSIMPLE_DIALECT
#define NNSIMPLE_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// NNSimple dialect definition.
//===----------------------------------------------------------------------===//

def NNSimple_Dialect : Dialect {
    let name = "nnsimple";
    let summary = "A nnsimple out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::nnsimple";

    let useDefaultTypePrinterParser = 1;
    let hasConstantMaterializer = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base nnsimple operation definition.
//===----------------------------------------------------------------------===//

class NNSimple_Op<string mnemonic, list<Trait> traits = []> :
        Op<NNSimple_Dialect, mnemonic, traits>;

#endif // NNSIMPLE_DIALECT
