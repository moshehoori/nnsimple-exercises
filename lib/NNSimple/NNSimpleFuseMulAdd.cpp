//===- NNSimpleFuseMulAdd.cpp - Fuse mul+add --------------------*- C++ -*-===//
//
// EXERCISE C-01: Implement the fuse-mul-add pass here.
// See EXERCISE.md at the repo root.
//
// Pattern: add(mul(a, b), c) -> fused_mul_add(a, b, c)
//   - Require the mul to have exactly one use (so we don't leave it behind).
//   - Either operand of the add could be the mul; check both.
//
// Mirror NNSimplePasses.cpp (NNSimpleFuseAddReluRewriter + the pass class).
//===----------------------------------------------------------------------===//

#include "NNSimple/NNSimpleOps.h"
#include "NNSimple/NNSimplePasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::nnsimple {
#define GEN_PASS_DEF_NNSIMPLEFUSEMULADD
#include "NNSimple/NNSimplePasses.h.inc"

namespace {
// TODO: define FuseMulAddRewriter : public OpRewritePattern<AddOp>

class NNSimpleFuseMulAdd
    : public impl::NNSimpleFuseMulAddBase<NNSimpleFuseMulAdd> {
public:
  using impl::NNSimpleFuseMulAddBase<NNSimpleFuseMulAdd>::NNSimpleFuseMulAddBase;

  void runOnOperation() final {
    // TODO: set up a RewritePatternSet, add your rewriter, run
    // applyPatternsGreedily(getOperation(), std::move(patterns)).
    // For now this does nothing (starter).
  }
};
} // namespace
} // namespace mlir::nnsimple
