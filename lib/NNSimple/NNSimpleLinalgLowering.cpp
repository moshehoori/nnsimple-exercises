//===- NNSimpleLinalgLowering.cpp - Lower NNSimple to Linalg ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NNSimple/NNSimpleOps.h"
#include "NNSimple/NNSimplePasses.h"
#include "NNSimple/NNSimpleTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::nnsimple {
#define GEN_PASS_DEF_NNSIMPLELOWERTOLINALG
#include "NNSimple/NNSimplePasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class NNSimpleTypeConverter : public TypeConverter {
public:
  NNSimpleTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](nnsimple::TensorType type) -> RankedTensorType {
      return RankedTensorType::get(type.getShape(), type.getElementType());
    });
  }
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static SmallVector<AffineMap> getElemwiseMaps(unsigned rank, unsigned count,
                                              MLIRContext *ctx) {
  auto id = AffineMap::getMultiDimIdentityMap(rank, ctx);
  return SmallVector<AffineMap>(count, id);
}

static SmallVector<utils::IteratorType> getParallelIters(unsigned rank) {
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//

// nnsimple.add -> linalg.add
struct AddOpLowering : public OpConversionPattern<AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto resultType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    auto init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                        resultType.getElementType());
    auto add = linalg::AddOp::create(
        rewriter, loc, resultType,
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{init});
    rewriter.replaceOp(op, add.getResults());
    return success();
  }
};

// nnsimple.mul -> linalg.mul
struct MulOpLowering : public OpConversionPattern<MulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto resultType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    auto init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                        resultType.getElementType());
    auto mul = linalg::MulOp::create(
        rewriter, loc, resultType,
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{init});
    rewriter.replaceOp(op, mul.getResults());
    return success();
  }
};

// nnsimple.relu -> linalg.generic { arith.maximumf(x, 0) }
struct ReluOpLowering : public OpConversionPattern<ReluOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto resultType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getOutput().getType()));
    unsigned rank = resultType.getRank();
    auto elemType = resultType.getElementType();

    auto init =
        tensor::EmptyOp::create(rewriter, loc, resultType.getShape(), elemType);
    auto generic = linalg::GenericOp::create(
        rewriter, loc, resultType, ValueRange{adaptor.getInput()},
        ValueRange{init}, getElemwiseMaps(rank, 2, rewriter.getContext()),
        getParallelIters(rank),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto zero =
              arith::ConstantOp::create(b, loc, b.getFloatAttr(elemType, 0.0));
          auto relu = arith::MaximumFOp::create(b, loc, args[0], zero);
          linalg::YieldOp::create(b, loc, ValueRange{relu});
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

// nnsimple.const -> arith.constant
struct ConstOpLowering : public OpConversionPattern<ConstOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, arith::ConstantOp::create(rewriter, op.getLoc(),
                                                     adaptor.getValue()));
    return success();
  }
};

// nnsimple.fused_add_relu -> linalg.generic { arith.addf + arith.maximumf }
struct FusedAddReluOpLowering : public OpConversionPattern<FusedAddReluOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FusedAddReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto resultType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    unsigned rank = resultType.getRank();
    auto elemType = resultType.getElementType();

    auto init =
        tensor::EmptyOp::create(rewriter, loc, resultType.getShape(), elemType);
    auto generic = linalg::GenericOp::create(
        rewriter, loc, resultType,
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{init},
        getElemwiseMaps(rank, 3, rewriter.getContext()), getParallelIters(rank),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto sum = arith::AddFOp::create(b, loc, args[0], args[1]);
          auto zero =
              arith::ConstantOp::create(b, loc, b.getFloatAttr(elemType, 0.0));
          auto relu = arith::MaximumFOp::create(b, loc, sum, zero);
          linalg::YieldOp::create(b, loc, ValueRange{relu});
        });

    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class NNSimpleLowerToLinalg
    : public impl::NNSimpleLowerToLinalgBase<NNSimpleLowerToLinalg> {
public:
  using impl::NNSimpleLowerToLinalgBase<
      NNSimpleLowerToLinalg>::NNSimpleLowerToLinalgBase;

  void runOnOperation() final {
    NNSimpleTypeConverter typeConverter;

    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect,
                           tensor::TensorDialect, func::FuncDialect>();
    target.addIllegalDialect<NNSimpleDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return typeConverter.isLegal(op.getOperandTypes());
    });
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    RewritePatternSet patterns(&getContext());
    patterns.add<AddOpLowering, MulOpLowering, ReluOpLowering, ConstOpLowering,
                 FusedAddReluOpLowering>(typeConverter, &getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::nnsimple
