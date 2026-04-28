# NNSimple MLIR Dialect

NNSimple is an out-of-tree MLIR dialect for neural network operations, demonstrating best practices for building custom MLIR dialects outside the LLVM source tree.

## Features

- **Custom Type System**: Tensors with explicit data layout information (NCHW, NHWC, NCDHW, NDHWC, RowMajor, ColMajor)
- **Neural Network Operations**: Add, multiply, ReLU, constant, and fused operations
- **Optimization Passes**: Operation fusion (add+relu) and lowering to Linalg dialect
- **Python Bindings**: Nanobind-based Python API for dialect integration
- **Testing Infrastructure**: Comprehensive lit-based tests with FileCheck
- **Plugin Support**: Loadable plugin for upstream mlir-opt (in-tree builds)

## Type System

The dialect introduces `!nnsimple.tensor` with explicit layout annotations:

```mlir
!nnsimple.tensor<f32, [4, 4], NHWC>   // 4x4 float32 tensor in NHWC layout
!nnsimple.tensor<f64, [1, 128], NCHW> // 1x128 float64 tensor in NCHW layout
```

Layout information is preserved through optimization passes and dropped during lowering to standard tensor types.

## Operations

### Basic Operations

```mlir
// Element-wise addition (commutative, with folder and canonicalizer)
%result = nnsimple.add %lhs, %rhs : (!nnsimple.tensor<f32, [4, 4], NHWC>,
                                     !nnsimple.tensor<f32, [4, 4], NHWC>)
                                 -> !nnsimple.tensor<f32, [4, 4], NHWC>

// Element-wise multiplication (commutative)
%result = nnsimple.mul %lhs, %rhs : (!nnsimple.tensor<f32, [4, 4], NHWC>,
                                     !nnsimple.tensor<f32, [4, 4], NHWC>)
                                 -> !nnsimple.tensor<f32, [4, 4], NHWC>

// ReLU activation (with canonicalizer: relu(relu(x)) -> relu(x))
%output = nnsimple.relu %input : !nnsimple.tensor<f32, [2, 3], RowMajor>
                              -> !nnsimple.tensor<f32, [2, 3], RowMajor>

// Constant tensor (with folder and verifier)
%const = nnsimple.const dense<[[1.0, 2.0], [3.0, 4.0]]> : !nnsimple.tensor<f32, [2, 2], RowMajor>

// Fused add+relu operation
%result = nnsimple.fused_add_relu %lhs, %rhs : (!nnsimple.tensor<f32, [4, 4], NHWC>,
                                                 !nnsimple.tensor<f32, [4, 4], NHWC>)
                                             -> !nnsimple.tensor<f32, [4, 4], NHWC>
```

## Transformation Passes

### Operation Fusion: `-nnsimple-fuse-add-relu`

Fuses sequential add and relu operations into a single fused operation:

```mlir
// Before:
%0 = nnsimple.add %a, %b : (tensor, tensor) -> tensor
%1 = nnsimple.relu %0 : tensor -> tensor

// After:
%1 = nnsimple.fused_add_relu %a, %b : (tensor, tensor) -> tensor
```

### Lowering to Linalg: `-nnsimple-lower-to-linalg`

Converts NNSimple operations to Linalg + Arith operations on standard tensors:

```mlir
// Before:
func.func @example(%a: !nnsimple.tensor<f32, [4, 4], NHWC>) -> !nnsimple.tensor<f32, [4, 4], NHWC> {
  %0 = nnsimple.relu %a : !nnsimple.tensor<f32, [4, 4], NHWC> -> !nnsimple.tensor<f32, [4, 4], NHWC>
  return %0 : !nnsimple.tensor<f32, [4, 4], NHWC>
}

// After:
func.func @example(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %empty = tensor.empty() : tensor<4x4xf32>
  %relu = linalg.generic {...} ins(%a : tensor<4x4xf32>) outs(%empty : tensor<4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %zero = arith.constant 0.0 : f32
      %max = arith.maximumf %in, %zero : f32
      linalg.yield %max : f32
  } -> tensor<4x4xf32>
  return %relu : tensor<4x4xf32>
}
```

## Building

### Prerequisites

- CMake 3.20+
- Ninja build system
- Clang/LLVM toolchain
- Pre-built MLIR and LLVM (from llvm-project)

### Configuration

```bash
mkdir build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON \
  -DMLIR_INCLUDE_TESTS=ON
```

### Build Commands

```bash
# Build all targets
ninja

# Build specific tools
ninja nnsimple-opt        # Optimizer tool
ninja nnsimple-translate  # Translation tool

# Run tests
ninja check-nnsimple
```

## Usage Examples

### Running Transformations

```bash
# Parse and print (verifies syntax)
./bin/nnsimple-opt input.mlir

# Apply fusion pass
./bin/nnsimple-opt input.mlir -nnsimple-fuse-add-relu

# Lower to Linalg dialect
./bin/nnsimple-opt input.mlir -nnsimple-lower-to-linalg

# Chain multiple passes
./bin/nnsimple-opt input.mlir -nnsimple-fuse-add-relu -nnsimple-lower-to-linalg
```

### Example Input File

```mlir
func.func @simple_network(
    %input: !nnsimple.tensor<f32, [1, 128], NCHW>,
    %weights: !nnsimple.tensor<f32, [1, 128], NCHW>
) -> !nnsimple.tensor<f32, [1, 128], NCHW> {
    %0 = nnsimple.add %input, %weights : 
        (!nnsimple.tensor<f32, [1, 128], NCHW>, !nnsimple.tensor<f32, [1, 128], NCHW>) 
        -> !nnsimple.tensor<f32, [1, 128], NCHW>
    %1 = nnsimple.relu %0 : 
        !nnsimple.tensor<f32, [1, 128], NCHW> -> !nnsimple.tensor<f32, [1, 128], NCHW>
    return %1 : !nnsimple.tensor<f32, [1, 128], NCHW>
}
```

## Project Structure

```
nnsimple/
├── include/NNSimple/          # Public headers and TableGen definitions
│   ├── NNSimpleDialect.td     # Dialect definition
│   ├── NNSimpleTypes.td       # Type definitions (tensor with layout)
│   ├── NNSimpleOps.td         # Operation definitions
│   └── NNSimplePasses.td      # Pass declarations
├── lib/NNSimple/              # C++ implementation
│   ├── NNSimpleDialect.cpp    # Dialect registration
│   ├── NNSimpleTypes.cpp      # Type implementation
│   ├── NNSimpleOps.cpp        # Operation verification, folders, canonicalizers
│   ├── NNSimplePasses.cpp     # Fusion pass implementation
│   └── NNSimpleLinalgLowering.cpp  # Linalg lowering pass
├── nnsimple-opt/              # Standalone optimizer tool
├── nnsimple-translate/        # Translation tool
├── nnsimple-plugin/           # Plugin for mlir-opt (in-tree builds)
├── python/                    # Python bindings (nanobind)
└── test/                      # Lit-based tests
    └── NNSimple/              # Dialect tests (.mlir files)
```

## Testing

The project uses MLIR's `lit` testing framework:

```bash
# Run all tests
ninja check-nnsimple

# Run a specific test
./bin/nnsimple-opt ../test/NNSimple/nnsimple-fuse-test.mlir -nnsimple-fuse-add-relu | FileCheck ../test/NNSimple/nnsimple-fuse-test.mlir
```

Test files use RUN and CHECK directives:

```mlir
// RUN: nnsimple-opt -nnsimple-fuse-add-relu %s | FileCheck %s

func.func @test(%a: !nnsimple.tensor<f32, [4, 4], NHWC>) {
  // CHECK: nnsimple.fused_add_relu
  %0 = nnsimple.add %a, %a
  %1 = nnsimple.relu %0
  return
}
```

## Python Bindings

Build with Python support:

```bash
cmake ... -DMLIR_ENABLE_BINDINGS_PYTHON=ON
```

The Python package (`mlir_nnsimple`) provides access to the dialect from Python, enabling integration with Python-based ML frameworks.

## Key Features for Dialect Developers

This project demonstrates:

- **TableGen-driven development**: Operations, types, and passes defined declaratively
- **Type system extension**: Custom types with parameters (element type, shape, layout)
- **Pattern rewriting**: Both declarative (in .td) and imperative (C++) patterns
- **Constant folding**: Implementation of folders for compile-time evaluation
- **Canonicalization**: Simplification patterns (e.g., `relu(relu(x))` → `relu(x)`)
- **Dialect lowering**: Type conversion and operation lowering to standard dialects
- **Out-of-tree build**: Complete project buildable outside LLVM source tree
- **Testing best practices**: Comprehensive lit tests with FileCheck

## License

This project is licensed under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.

## Resources

- [MLIR Documentation](https://mlir.llvm.org/)
- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)
- [TableGen Documentation](https://llvm.org/docs/TableGen/)
- [LLVM Discourse](https://discourse.llvm.org/)
