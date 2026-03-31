set(LIBS
    MLIRArithDialect
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRNNSimple
  )
add_llvm_executable(nnsimple-opt nnsimple-opt.cpp)

llvm_update_compile_flags(nnsimple-opt)
target_link_libraries(nnsimple-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(nnsimple-opt)
