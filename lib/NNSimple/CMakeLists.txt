add_mlir_dialect_library(MLIRNNSimple
        NNSimpleTypes.cpp
        NNSimpleDialect.cpp
        NNSimpleOps.cpp
        NNSimplePasses.cpp
        NNSimpleLinalgLowering.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/NNSimple

        DEPENDS
        MLIRNNSimpleOpsIncGen
        MLIRNNSimplePassesIncGen
        MLIRNNSimpleCanonicalizationIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        MLIRArithDialect
        MLIRLinalgDialect
        MLIRTensorDialect
        MLIRTransformUtils
        MLIRFuncTransforms
        )
