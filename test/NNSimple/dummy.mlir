// RUN: nnsimple-opt %s | nnsimple-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = nnsimple.foo %{{.*}} : i32
        %res = nnsimple.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @nnsimple_types(%arg0: !nnsimple.custom<"10">)
    func.func @nnsimple_types(%arg0: !nnsimple.custom<"10">) {
        return
    }
}
