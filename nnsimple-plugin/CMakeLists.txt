add_llvm_library(NNSimplePlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        nnsimple-plugin.cpp

        DEPENDS
        MLIRNNSimple
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRNNSimple
        )
target_include_directories(
        NNSimplePlugin
        PRIVATE
        "${NNSIMPLE_BINARY_DIR}/include"
)
