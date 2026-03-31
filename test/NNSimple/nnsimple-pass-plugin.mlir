// UNSUPPORTED: system-windows
// RUN: mlir-opt %s --load-pass-plugin=%nnsimple_libs/NNSimplePlugin%shlibext --pass-pipeline="builtin.module(nnsimple-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @abar()
  func.func @abar() {
    return
  }
}
