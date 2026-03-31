llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(NNSIMPLE_TEST_DEPENDS
        FileCheck count not
        nnsimple-capi-test
        nnsimple-opt
        nnsimple-translate
        )
if(NOT WIN32 AND EXTERNAL_PROJECT_BUILD)
  list(APPEND NNSIMPLE_TEST_DEPENDS NNSimplePlugin)
endif()
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND NNSIMPLE_TEST_DEPENDS NNSimplePythonModules)
endif()

add_lit_testsuite(check-nnsimple "Running the nnsimple regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${NNSIMPLE_TEST_DEPENDS}
        )
set_target_properties(check-nnsimple PROPERTIES FOLDER "Tests")

add_lit_testsuites(NNSIMPLE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${NNSIMPLE_TEST_DEPENDS})

add_subdirectory(CAPI)
