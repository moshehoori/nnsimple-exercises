add_mlir_public_c_api_library(NNSimpleCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRNNSimple
)
