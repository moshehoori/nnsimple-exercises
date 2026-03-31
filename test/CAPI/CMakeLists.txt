# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own nnsimple dialect.
add_mlir_aggregate(NNSimpleCAPITestLib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  MLIRCAPIArith
  NNSimpleCAPI
)

add_llvm_executable(nnsimple-capi-test
  nnsimple-capi-test.c
)
llvm_update_compile_flags(nnsimple-capi-test)
target_link_libraries(nnsimple-capi-test
  PRIVATE NNSimpleCAPITestLib)
