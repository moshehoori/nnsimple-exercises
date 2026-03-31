## How to compile

```bash
cmake -G Ninja .. \
  -DMLIR_DIR=/home/ybuchble/projects/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/home/ybuchble/projects/llvm-project/build/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON \
  -DMLIR_INCLUDE_TESTS=ON
```
