// UNSUPPORTED: system-windows
// RUN: mlir-opt %s --load-dialect-plugin=%nnsimple_libs/NNSimplePlugin%shlibext --pass-pipeline="builtin.module(nnsimple-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @nnsimple_types(%arg0: !nnsimple.custom<"10">)
  func.func @nnsimple_types(%arg0: !nnsimple.custom<"10">) {
    return
  }
}
