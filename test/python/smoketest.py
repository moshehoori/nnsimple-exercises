# RUN: %python %s 2>&1 | FileCheck %s
import sys

# CHECK: Testing mlir_nnsimple package
print("Testing mlir_nnsimple package", file=sys.stderr)

import mlir_nnsimple.ir
from mlir_nnsimple.dialects import nnsimple_nanobind as nnsimple_d

with mlir_nnsimple.ir.Context():
    nnsimple_d.register_dialects()
    nnsimple_module = mlir_nnsimple.ir.Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = nnsimple.foo %0 : i32
    """
    )
    # CHECK: %[[C2:.*]] = arith.constant 2 : i32
    # CHECK: nnsimple.foo %[[C2]] : i32
    print(str(nnsimple_module), file=sys.stderr)

    custom_type = nnsimple_d.CustomType.get("foo")
    # CHECK: !nnsimple.custom<"foo">
    print(custom_type, file=sys.stderr)

    # CHECK: this is a fp16 type
    nnsimple_d.print_fp_type(mlir_nnsimple.ir.F16Type.get(), sys.stderr)
    # CHECK: this is a fp32 type
    nnsimple_d.print_fp_type(mlir_nnsimple.ir.F32Type.get(), sys.stderr)
    # CHECK: this is a fp64 type
    nnsimple_d.print_fp_type(mlir_nnsimple.ir.F64Type.get(), sys.stderr)


# CHECK: Testing mlir package
print("Testing mlir package", file=sys.stderr)

from mlir.ir import *

# CHECK-NOT: RuntimeWarning: nanobind: type '{{.*}}' was already registered!
