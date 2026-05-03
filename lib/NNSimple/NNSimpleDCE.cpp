//===- NNSimpleDCE.cpp - Dead code elimination ------------------*- C++ -*-===//
//
// EXERCISE C-02: Implement a dead-op-elimination pass here.
// See EXERCISE.md at the repo root.
//
// Erase every op that is:
//   - Pure (no side effects)        — query with `mlir::isPure(op)`
//   - Has no uses                    — op->use_empty()
//   - Has at least one result        — so we don't erase terminators
//
// Use a fixed-point loop (collect-then-erase) so chains of dead ops collapse.
//===----------------------------------------------------------------------===//

#include "NNSimple/NNSimpleOps.h"
#include "NNSimple/NNSimplePasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::nnsimple {
#define GEN_PASS_DEF_NNSIMPLEDCE
#include "NNSimple/NNSimplePasses.h.inc"

namespace {
class NNSimpleDCE : public impl::NNSimpleDCEBase<NNSimpleDCE> {
public:
  using impl::NNSimpleDCEBase<NNSimpleDCE>::NNSimpleDCEBase;

  void runOnOperation() final {
    // TODO: fixed-point loop — walk, collect dead ops (Pure + use_empty +
    // has results), erase them, repeat until nothing changes.
    // For now this does nothing (starter).
  }
};
} // namespace
} // namespace mlir::nnsimple
