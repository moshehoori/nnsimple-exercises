//===- NNSimpleOps.cpp - NNSimple dialect ops -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NNSimple/NNSimpleOps.h"
#include "NNSimple/NNSimpleDialect.h"
#include "NNSimple/NNSimpleTypes.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::nnsimple;

LogicalResult ConstOp::verify() {
  auto attrType = getValue().getShapedType();
  auto resultType = llvm::cast<TensorType>(getOutput().getType());

  if (attrType.getShape() != resultType.getShape())
    return emitOpError("attribute shape doesn't match result shape");

  if (attrType.getElementType() != resultType.getElementType())
    return emitOpError("attribute element type doesn't match result type");

  return success();
}

OpFoldResult ConstOp::fold(FoldAdaptor adaptor) { return getValue(); }

LogicalResult AddOp::verify() {
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();
  auto resultType = getResult().getType();

  if (lhsType != rhsType)
    return emitOpError("operand types don't match");

  if (lhsType != resultType)
    return emitOpError("operand type doesn't match result type");

  return success();
}

// AddOp folder: both operands are constant → fold to a new constant.
OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = llvm::dyn_cast_or_null<DenseElementsAttr>(adaptor.getLhs());
  auto rhsAttr = llvm::dyn_cast_or_null<DenseElementsAttr>(adaptor.getRhs());
  if (!lhsAttr || !rhsAttr)
    return {};

  // Both must be dense float and have the same type.
  auto lhsType = lhsAttr.getType();
  if (lhsType != rhsAttr.getType())
    return {};

  SmallVector<APFloat> results;
  for (auto [a, b] :
       llvm::zip(lhsAttr.getValues<APFloat>(), rhsAttr.getValues<APFloat>())) {
    APFloat sum = a;
    sum.add(b, APFloat::rmNearestTiesToEven);
    results.push_back(sum);
  }
  return DenseElementsAttr::get(lhsType, results);
}

/// Canonicalization: x + 0 → x
struct AddZeroElimination : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    // The Commutative trait causes the canonicalizer to sort operands,
    // placing constants on the rhs, so we only need to check rhs.
    Attribute rhs;
    if (matchPattern(op.getRhs(), m_Constant(&rhs))) {
      if (auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(rhs)) {
        if (denseAttr.isSplat()) {
          if (auto floatVal = denseAttr.getSplatValue<APFloat>();
              floatVal.isZero()) {
            rewriter.replaceOp(op, op.getLhs());
            return success();
          }
        }
      }
    }
    return failure();
  }
};

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<AddZeroElimination>(context);
}

#include "NNSimple/NNSimpleCanonicalization.inc"

void ReluOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<ReluReluElimination>(context);
}

// ============================================================================
// EXERCISE A-02: Implement QuantizeOp::verify here.
//
// Error messages the tests expect:
//   "input and output must share element type, shape, and layout"
//   "input must not already be quantized"
//   "output must be quantized (carry scale and zero_point)"
//
// Use llvm::cast<nnsimple::TensorType>(value.getType()) to get at the type,
// then call .getElementType(), .getShape(), .getLayout(), .isQuantized().
// ============================================================================
// LogicalResult QuantizeOp::verify() {
//   // TODO
//   return success();
// }

#define GET_OP_CLASSES
#include "NNSimple/NNSimpleOps.cpp.inc"
