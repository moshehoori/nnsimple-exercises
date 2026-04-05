//===- NNSimplePasses.cpp - NNSimple passes ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "NNSimple/NNSimpleOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "NNSimple/NNSimplePasses.h"

namespace mlir::nnsimple {
#define GEN_PASS_DEF_NNSIMPLESWITCHBARFOO
#define GEN_PASS_DEF_NNSIMPLEFUSEADDRELU
#include "NNSimple/NNSimplePasses.h.inc"

namespace {
class NNSimpleFuseAddReluRewriter : public OpRewritePattern<nnsimple::ReluOp> {
public:
  using OpRewritePattern<nnsimple::ReluOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(nnsimple::ReluOp reluOp,
                                PatternRewriter &rewriter) const final {
    if (!reluOp.getResult().hasOneUse())
      return failure();
    auto addOp = dyn_cast<AddOp>(*reluOp.getOperand().getDefiningOp());
    if (!addOp)
      return failure();
    rewriter.setInsertionPoint(reluOp);
    auto fusedOp = FusedAddReluOp::create(rewriter, reluOp.getLoc(),
                                          reluOp.getResult().getType(),
                                          addOp.getLhs(), addOp.getRhs());
    rewriter.replaceOp(reluOp, fusedOp.getResult());
    rewriter.eraseOp(addOp);
    return success();
  }
};

class NNSimpleFuseAddRelu
    : public impl::NNSimpleFuseAddReluBase<NNSimpleFuseAddRelu> {
public:
  using impl::NNSimpleFuseAddReluBase<
      NNSimpleFuseAddRelu>::NNSimpleFuseAddReluBase;

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<NNSimpleFuseAddReluRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(func, patternSet)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::nnsimple
