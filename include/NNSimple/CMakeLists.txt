set(LLVM_TARGET_DEFINITIONS NNSimpleOps.td)
mlir_tablegen(NNSimpleOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(NNSimpleOpsEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect(NNSimpleOps nnsimple)
add_mlir_doc(NNSimpleDialect NNSimpleDialect NNSimple/ -gen-dialect-doc)
add_mlir_doc(NNSimpleOps NNSimpleOps NNSimple/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS NNSimpleOps.td)
mlir_tablegen(NNSimpleCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(MLIRNNSimpleCanonicalizationIncGen)

set(LLVM_TARGET_DEFINITIONS NNSimplePasses.td)
mlir_tablegen(NNSimplePasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRNNSimplePassesIncGen)
