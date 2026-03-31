include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_nnsimple`
# top level package (the API has been embedded in a relocatable way).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(NNSimplePythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT NNSimplePythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_nnsimple"
  TD_FILE dialects/NNSimpleOps.td
  SOURCES
    dialects/nnsimple_nanobind.py
    _mlir_libs/_nnsimpleDialectsNanobind/py.typed
  DIALECT_NAME nnsimple)

declare_mlir_python_extension(NNSimplePythonSources.NanobindExtension
  MODULE_NAME _nnsimpleDialectsNanobind
  ADD_TO_PARENT NNSimplePythonSources
  SOURCES
    NNSimpleExtensionNanobind.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIArith
    MLIRCAPITransforms
    NNSimpleCAPI
)


################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(NNSimplePythonCAPI
  INSTALL_COMPONENT NNSimplePythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    NNSimplePythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
)

################################################################################
# Instantiation of all Python modules
################################################################################

set(NNSimplePythonModules_ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

if(MLIR_PYTHON_STUBGEN_ENABLED)
  # Everything here is very tightly coupled. See the ample descriptions at the bottom of
  # mlir/python/CMakeLists.txt.

  # For a non-external projects build (e.g., installed distro) the type gen targets for the core _mlir module
  # need to be re-declared. On the contrary, for an external projects build, the MLIRPythonExtension.Core.type_stub_gen
  # target already exists and can just be added to DECLARED_SOURCES (see below).
  if(NOT EXTERNAL_PROJECT_BUILD)
    set(_core_type_stub_sources
      _mlir/__init__.pyi
      _mlir/ir.pyi
      _mlir/passmanager.pyi
      _mlir/rewrite.pyi
    )
    get_target_property(_core_extension_srcs MLIRPythonExtension.Core INTERFACE_SOURCES)
    mlir_generate_type_stubs(
      MODULE_NAME _mlir
      DEPENDS_TARGETS NNSimplePythonModules.extension._mlir.dso
      OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_mlir_libs"
      OUTPUTS "${_core_type_stub_sources}"
      DEPENDS_TARGET_SRC_DEPS "${_core_extension_srcs}"
      IMPORT_PATHS "${NNSimplePythonModules_ROOT_PREFIX}/_mlir_libs"
      VERBOSE
    )
    set(_mlir_typestub_gen_target "${NB_STUBGEN_CUSTOM_TARGET}")

    list(TRANSFORM _core_type_stub_sources PREPEND "_mlir_libs/")
    declare_mlir_python_sources(
      NNSimplePythonExtension.Core.type_stub_gen
      ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
      ADD_TO_PARENT NNSimplePythonSources
      SOURCES "${_core_type_stub_sources}"
    )
  endif()

  get_target_property(_nnsimple_extension_srcs NNSimplePythonSources.NanobindExtension INTERFACE_SOURCES)
  mlir_generate_type_stubs(
    MODULE_NAME mlir_nnsimple._mlir_libs._nnsimpleDialectsNanobind
    DEPENDS_TARGETS
      NNSimplePythonModules.extension._mlir.dso
      NNSimplePythonModules.extension._nnsimpleDialectsNanobind.dso
    OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_mlir_libs"
    OUTPUTS
      _nnsimpleDialectsNanobind/__init__.pyi
      _nnsimpleDialectsNanobind/nnsimple.pyi
    DEPENDS_TARGET_SRC_DEPS "${_nnsimple_extension_srcs}"
    IMPORT_PATHS "${NNSimplePythonModules_ROOT_PREFIX}/.."
  )
  set(_nnsimpleDialectsNanobind_typestub_gen_target "${NB_STUBGEN_CUSTOM_TARGET}")

  declare_mlir_python_sources(
    NNSimplePythonSources.type_stub_gen
    ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
    ADD_TO_PARENT NNSimplePythonSources
    SOURCES
      _mlir_libs/_nnsimpleDialectsNanobind/__init__.pyi
      _mlir_libs/_nnsimpleDialectsNanobind/nnsimple.pyi
  )
endif()

set(_declared_sources
  NNSimplePythonSources
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.builtin
)
# For an external projects build, the MLIRPythonExtension.Core.type_stub_gen
# target already exists and can just be added to DECLARED_SOURCES.
if(EXTERNAL_PROJECT_BUILD AND MLIR_PYTHON_STUBGEN_ENABLED)
  list(APPEND _declared_sources MLIRPythonExtension.Core.type_stub_gen)
endif()

add_mlir_python_modules(NNSimplePythonModules
  ROOT_PREFIX "${NNSimplePythonModules_ROOT_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES "${_declared_sources}"
  COMMON_CAPI_LINK_LIBS
    NNSimplePythonCAPI
)

if(MLIR_PYTHON_STUBGEN_ENABLED)
  if(NOT EXTERNAL_PROJECT_BUILD)
    add_dependencies(NNSimplePythonModules "${_mlir_typestub_gen_target}")
  endif()
  add_dependencies(NNSimplePythonModules "${_nnsimpleDialectsNanobind_typestub_gen_target}")
endif()
