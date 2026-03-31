add_mlir_dialect_library(MLIRNNSimple
        NNSimpleTypes.cpp
        NNSimpleDialect.cpp
        NNSimpleOps.cpp
        NNSimplePasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/NNSimple

        DEPENDS
        MLIRNNSimpleOpsIncGen
        MLIRNNSimplePassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
