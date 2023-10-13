#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/include/mlir/IR/OperationSupport.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/transforms/infer_memref_layout.h"
#include "jaxlib/mosaic/dialect/tpu/util.h"
#include "xla/array.h"
#include "xla/layout.h"
#include "xla/util.h"

// TODO(tlongeri): Prefer returning failure over CHECKs. In particular, be more
// consistent about this for layout null checks in rules.

#define NYI(msg)                            \
  op->emitOpError("not implemented: " msg); \
  return failure();

namespace mlir::tpu {
// TODO(tlongeri): Maybe just roll our own multi-dimensional array instead of
// using XLA's? There's too much glue for going from/to ArrayRef.

#define GEN_PASS_DECL_APPLYVECTORLAYOUTPASS
#define GEN_PASS_DEF_APPLYVECTORLAYOUTPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

struct RewriteContext {
  func::FuncOp func;
  // TODO(tlongeri): target_shape should be determined from hardware_generation
  const int hardware_generation;
  const std::array<int64_t, 2> target_shape;

  MLIRContext *getMLIRContext() { return func.getContext(); }
};

LogicalResult applyLayoutBlock(RewriteContext &ctx, Block &block);
RollVectorsOp assemble(RewriteContext &ctx, OpBuilder &builder, VectorType vty,
                       const VectorLayout &layout,
                       const xla::Array<Value> &vals);
FailureOr<xla::Array<Value>> disassemble(RewriteContext &ctx,
                                         OpBuilder &builder,
                                         const VectorLayout &layout, Value val);
namespace {

void moveAllRegions(Operation &src, Operation &dst) {
  for (auto [src_region, dst_region] :
       llvm::zip_equal(src.getRegions(), dst.getRegions())) {
    dst_region.takeBody(src_region);
  }
}
// Masks all values outside of bounds.
//
// Arguments:
//   value: A rank 2 MLIR vector to be masked.
//   bounds: A TargetTuple of slices specifying a rectangular subregion of value
//     that should be preserved during masking.
//   neutral: A scalar attribute specifying the value that will be inserted
//     for all values outside of specified bounds.
//
// Returns:
//   An MLIR value of the same type as the value argument, with all entries
//   outside of bounds replaced by neutral.
FailureOr<Value> maskOOB(RewriteContext &ctx, OpBuilder &builder,
                         TypedValue<VectorType> value,
                         const VRegDataBounds &bounds,
                         const TypedAttr neutral) {
  CHECK(llvm::equal(value.getType().getShape(), ctx.target_shape));
  if (bounds.isComplete(ctx.target_shape)) {
    return value;
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      TypedValue<VectorType> mask,
      bounds.getVectorMask(builder, value.getLoc(), ctx.hardware_generation,
                           ctx.target_shape));
  if (cast<IntegerType>(mask.getType().getElementType()).getWidth() != 1) {
    return emitError(value.getLoc(),
                     "Not implemented: Unsupported mask bitwidth");
  }
  auto neutral_vec_ty = VectorType::get(ctx.target_shape, neutral.getType());
  auto neutral_vec = builder.create<arith::ConstantOp>(
      value.getLoc(), neutral_vec_ty,
      DenseElementsAttr::get(neutral_vec_ty, neutral));
  return builder
      .create<arith::SelectOp>(value.getLoc(), mask, value, neutral_vec)
      .getResult();
}

// Models Numpy's np.repeat, repeating each element `repeats` times along the
// specified axis. For example, if `src` is [1, 2], `axis` is 0 and `repeats` is
// 3, this will return [1, 1, 1, 2, 2, 2].
xla::Array<Value> repeat(const xla::Array<Value> &src, const int repeats,
                         const int64_t axis) {
  SmallVector<int64_t> dims(toArrayRef(src.dimensions()));
  dims[axis] *= repeats;
  xla::Array<Value> res(dims);
  src.Each([&](absl::Span<const int64_t> idx, const Value v) {
    SmallVector<int64_t> res_idx(toArrayRef(idx));
    res_idx[axis] *= repeats;
    for (int i = 0; i < repeats; ++i) {
      res(res_idx) = v;
      ++res_idx[axis];
    }
  });
  return res;
}

template <typename T>
ArrayRef<T> XlaArrayToFlatArrayRef(xla::Array<T> xla_array) {
  return ArrayRef<T>(xla_array.data(), xla_array.num_elements());
}

template <typename T, typename Range>
xla::Array<T> XlaArrayFromShapeAndValues(ArrayRef<int64_t> sizes, Range vals) {
  // TODO(tlongeri): is there no way to avoid default initialization in the
  // constructor?
  xla::Array<T> arr(sizes);
  arr.SetValues(vals);
  return arr;
}

bool incrementSliceIndex(const MutableArrayRef<int64_t> idx,
                         const absl::Span<const int64_t> starts,
                         const absl::Span<const int64_t> limits) {
  const int64_t nd = idx.size();
  CHECK_EQ(nd, starts.size());
  CHECK_EQ(nd, limits.size());
  for (int64_t i = nd - 1; i >= 0; --i) {
    ++idx[i];
    if (idx[i] < limits[i]) {
      return true;
    }
    idx[i] = starts[i];
  }
  return false;
}

bool incrementIndex(const MutableArrayRef<int64_t> idx,
                    const absl::Span<const int64_t> limits) {
  const int64_t nd = idx.size();
  CHECK_EQ(nd, limits.size());
  for (int64_t i = nd - 1; i >= 0; --i) {
    ++idx[i];
    if (idx[i] < limits[i]) {
      return true;
    }
    idx[i] = 0;
  }
  return false;
}

bool sliceIsEmpty(const absl::Span<const int64_t> starts,
                 const absl::Span<const int64_t> limits) {
  for (auto [s, l] : llvm::zip_equal(starts, limits)) {
    CHECK_LE(s, l);
    if (s == l) {
      return true;
    }
  }
  return false;
}

// An alternative to xla::Array::UpdateSlice that takes a single value
template <typename T>
void updateSlice(xla::Array<T> &arr, const T &value,
                 const absl::Span<const int64_t> starts,
                 const absl::Span<const int64_t> limits) {
  if (sliceIsEmpty(starts, limits)) {
    return;
  }
  SmallVector<int64_t> idx(toArrayRef(starts));
  do {
    arr(idx) = value;
  } while (incrementSliceIndex(idx, starts, limits));
}

// An alternative to xla::Array::UpdateSlice that takes a range of data
template <typename T, typename Range>
void updateSliceFromRange(xla::Array<T> &arr, Range data,
                          const absl::Span<const int64_t> starts,
                          const absl::Span<const int64_t> limits) {
  if (sliceIsEmpty(starts, limits)) {
    return;
  }
  SmallVector<int64_t> idx(toArrayRef(starts));
  auto data_it = data.begin();
  do {
    arr(idx) = *data_it;
    ++data_it;
  } while (incrementSliceIndex(idx, starts, limits));
  CHECK(data_it == data.end());
}

FailureOr<TypedAttr> getZeroIntOrFloatAttr(Type ty) {
  if (isa<FloatType>(ty)) {
    return TypedAttr(FloatAttr::get(ty, 0));
  }
  if (isa<IntegerType>(ty)) {
    return TypedAttr(IntegerAttr::get(ty, 0));
  }
  return emitError(UnknownLoc::get(ty.getContext()), "Not implemented: ") << ty;
}

FailureOr<int64_t> getIntConst(Value v) {
  if (auto constant_op = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto integer_attr = dyn_cast<IntegerAttr>(constant_op.getValue())) {
      return integer_attr.getValue().getSExtValue();
    }
  }
  return emitError(v.getLoc(), "Expected an integer constant");
}

FailureOr<SmallVector<int64_t>> getIntConstsFromOperandRange(
    OperandRange vals) {
  SmallVector<int64_t> res(vals.size());
  for (int i = 0; i < vals.size(); ++i) {
    FAILUREOR_ASSIGN_OR_RETURN(res[i], getIntConst(vals[i]));
  }
  return res;
}

// Returns the first-level tiling of a (packed and tiled) memref value.
FailureOr<std::array<int64_t, 2>> getMemRefTiling(
    TypedValue<MemRefType> value, const std::array<int64_t, 2> target_shape) {
  if (auto erase_layout_op =
          dyn_cast_if_present<EraseLayoutOp>(value.getDefiningOp())) {
    value = erase_layout_op.getOperand();
  }
  const MemRefType memref_ty = value.getType();
  const auto mem_layout = dyn_cast<TiledLayoutAttr>(memref_ty.getLayout());
  if (mem_layout == nullptr) {
    return emitError(value.getLoc(), "Expected a tiled memref");
  }
  FAILUREOR_ASSIGN_OR_RETURN(int8_t bitwidth,
                             getTypeBitwidth(memref_ty.getElementType()));
  const int packing = 32 / bitwidth;
  const ArrayRef<xla::Tile> tiles = mem_layout.getTiles();
  const xla::Tile &first_tile = tiles.front();
  if (first_tile.dimensions().size() == 1) {
    const int64_t tile_size = first_tile.dimension(0);
    if (tile_size % (target_shape[1] * packing) != 0) {
      return emitError(value.getLoc(), "Not implemented");
    }
    if (bitwidth == 32) {
      if (tiles.size() > 1) {
        return emitError(value.getLoc(), "Not implemented");
      }
    } else if (bitwidth < 32) {
      if (tiles.drop_front() !=
          ArrayRef<xla::Tile>{xla::Tile({target_shape[1]}),
                              xla::Tile({packing, 1})}) {
        return emitError(value.getLoc(), "Not implemented");
      }
    }
    return std::array<int64_t, 2>{1, tile_size};
  }
  if (first_tile.dimensions().size() == 2) {
    if (bitwidth == 32) {
      if (tiles.size() > 1) {
        return emitError(value.getLoc(), "Not implemented");
      }
      return std::array<int64_t, 2>{first_tile.dimension(0),
                                    first_tile.dimension(1)};
    }
    if (bitwidth < 32) {
      if (tiles.size() != 2 || tiles[1] != xla::Tile({packing, 1})) {
        return emitError(value.getLoc(), "Not implemented");
      }
      return std::array<int64_t, 2>{first_tile.dimension(0),
                                    first_tile.dimension(1)};
    }
  }
  return emitError(value.getLoc(), "Not implemented");
}

// Hoist a vector constant as an additional argument of the function.
FailureOr<BlockArgument> appendConstant(RewriteContext &ctx,
                                        DenseElementsAttr value) {
  MLIRContext *mlir_ctx = ctx.func.getContext();
  Block &entry_block = ctx.func.getBody().front();
  auto value_ty = cast<VectorType>(value.getType());
  if (value_ty.getElementType().getIntOrFloatBitWidth() != 32) {
    return ctx.func.emitOpError("Only 32-bit constants supported");
  }
  if (ctx.func->getAttr("scratch_operands")) {
    return ctx.func.emitOpError(
        "Not implemented: function has scratch_operands");
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      MemRefType arg_type,
      inferMemref(
          MemRefType::get(value_ty.getShape(), value_ty.getElementType()),
          ctx.hardware_generation));
  const BlockArgument argument =
      entry_block.insertArgument(entry_block.getNumArguments() - 1, arg_type,
                                 UnknownLoc::get(ctx.getMLIRContext()));
  const FunctionType func_ty = ctx.func.getFunctionType();
  // Adjust the function type.
  SmallVector<Type> new_arg_tys(func_ty.getInputs());
  new_arg_tys.insert(new_arg_tys.begin() + (new_arg_tys.size() - 1), arg_type);
  const auto new_func_ty =
      FunctionType::get(mlir_ctx, new_arg_tys, func_ty.getResults());
  ctx.func.setFunctionType(new_func_ty);
  // Adjust the constants attribute.
  if (auto prev_cst = ctx.func->getAttrOfType<ArrayAttr>("vector_constants")) {
    SmallVector<Attribute> vector_constants(prev_cst.getValue());
    vector_constants.push_back(value);
    ctx.func->setAttr("vector_constants",
                      ArrayAttr::get(ctx.func.getContext(), vector_constants));
  } else {
    ctx.func->setAttr("vector_constants",
                      ArrayAttr::get(ctx.func.getContext(), value));
  }
  // Adjust window params for the extra operand.
  if (auto window_params =
          ctx.func->getAttrOfType<ArrayAttr>("window_params")) {
    const auto iteration_bounds =
        ctx.func->getAttrOfType<DenseI64ArrayAttr>("iteration_bounds");
    CHECK(iteration_bounds);
    const int64_t iteration_rank = iteration_bounds.getSize();
    const SmallVector<AffineExpr> zeros(
        iteration_rank, getAffineConstantExpr(0, ctx.func.getContext()));
    const auto transform_indices =
        AffineMap::get(iteration_rank, 0, zeros, ctx.func.getContext());
    const auto new_param = DictionaryAttr::get(
        ctx.func.getContext(),
        NamedAttribute(
            StringAttr::get(ctx.func.getContext(), "transform_indices"),
            AffineMapAttr::get(transform_indices)));
    SmallVector<Attribute> window_params_values(window_params.getValue());
    window_params_values.insert(window_params_values.end() - 1, new_param);
    ctx.func->setAttr("window_params", ArrayAttr::get(ctx.func.getContext(),
                                                      window_params_values));
  }
  return argument;
}

FailureOr<VectorType> getNativeVregType(
    Type elem_ty, const std::array<int64_t, 2> target_shape) {
  FAILUREOR_ASSIGN_OR_RETURN(const int8_t bitwidth,
                             getTypeBitwidth<true>(elem_ty));
  if (bitwidth == 32) {
    return VectorType::get(target_shape, elem_ty);
  }
  // bitwidth != 32
  return VectorType::get({target_shape[0], target_shape[1], 32 / bitwidth},
                         elem_ty);
}

// Get the layout from a VectorLayoutAttr or StringAttr.
mlir::FailureOr<Layout> getLayoutFromAttr(Attribute attr) {
  if (attr == nullptr) {
    return failure();
  }

  if (auto layout_attr = dyn_cast<VectorLayoutAttr>(attr)) {
    return layout_attr.getLayout();
  }

  // TODO(tlongeri): StringAttr support was only added temporarily to avoid
  // having Python bindings for VectorLayoutAttr. Remove this once we get rid
  // of the Python implementation
  if (auto string_attr = dyn_cast<StringAttr>(attr)) {
    StringRef str = string_attr.getValue();
    if (!str.consume_front("#tpu.vpad<\"")) {
      return failure();
    }
    if (str.consume_front("none")) {
      return kNoLayout;
    }
    if (auto layout = VectorLayout::parse(&str)) {
      return layout;
    }
    return failure();
  }

  return failure();
}

// Returns empty vector on null attribute
FailureOr<SmallVector<Layout>> getLayoutArrayFromAttr(const Attribute attr) {
  if (const auto array_attr = dyn_cast_if_present<ArrayAttr>(attr)) {
    SmallVector<Layout> out_layouts;
    out_layouts.reserve(array_attr.size());
    for (const Attribute a : array_attr) {
      FAILUREOR_ASSIGN_OR_RETURN(const Layout layout, getLayoutFromAttr(a));
      out_layouts.push_back(layout);
    }
    return out_layouts;
  }
  return SmallVector<Layout>{};
}

// TODO(tlongeri): Unify with infer_vector_layout.cc's getOutLayout.
FailureOr<SmallVector<Layout>> getOutLayout(Operation &op) {
  // TODO(tlongeri): non-array attribute path should be removed after tests are
  // updated
  FailureOr<Layout> failure_or_layout =
      getLayoutFromAttr(op.getAttr("out_layout"));
  if (succeeded(failure_or_layout)) {
    return SmallVector<Layout>{failure_or_layout.value()};
  }
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> out_layout,
                             getLayoutArrayFromAttr(op.getAttr("out_layout")));
  if (out_layout.size() != op.getNumResults()) {
    return failure();
  }
  return out_layout;
}

FailureOr<SmallVector<Layout>> getInLayout(Operation &op) {
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> in_layout,
                             getLayoutArrayFromAttr(op.getAttr("in_layout")));
  if (in_layout.size() != op.getNumOperands()) {
    return failure();
  }
  return in_layout;
}

LogicalResult elementwise_op_rule(RewriteContext &ctx, Operation &op,
                                  const ArrayRef<Layout> layouts_in,
                                  const ArrayRef<Layout> layouts_out) {
  CHECK(OpTrait::hasElementwiseMappableTraits(&op));
  if (op.getNumResults() != 1) {
    return op.emitError("Not implemented: Only ops with one result supported");
  }
  CHECK_EQ(layouts_in.size(), op.getNumOperands());
  CHECK_GT(layouts_in.size(), 0);
  CHECK_EQ(layouts_out.size(), 1);
  OpBuilder builder(&op);
  if (!(layouts_out.front().has_value() &&
        llvm::all_of(layouts_in,
                     [&](const Layout &l) { return l.has_value(); }))) {
    return op.emitOpError(
        "Not implemented: Null layout / non-vector operand in elementwise "
        "operation");
  }
  const auto out_ty = cast<VectorType>(op.getResult(0).getType());
  const VectorLayout &layout_out = *layouts_out.front();
  if (!llvm::all_of(layouts_in, [&](const Layout &l) {
        return l->generalizes(layout_out, out_ty.getShape(), ctx.target_shape);
      })) {
    return op.emitOpError("Incompatible layouts in elementwise operation");
  }
  const unsigned num_operands = op.getNumOperands();
  SmallVector<xla::Array<Value>> in_vreg_arrays;
  in_vreg_arrays.reserve(num_operands);
  for (unsigned i = 0; i < num_operands; ++i) {
    FAILUREOR_ASSIGN_OR_RETURN(
        xla::Array<Value> tile_array,
        disassemble(ctx, builder, *layouts_in[i], op.getOperand(i)));
    in_vreg_arrays.emplace_back(std::move(tile_array));
  }

  FAILUREOR_ASSIGN_OR_RETURN(
      const VectorType out_vreg_ty,
      getNativeVregType(out_ty.getElementType(), ctx.target_shape));

  NamedAttrList attributes(op.getAttrDictionary());
  attributes.erase("in_layout");
  attributes.erase("out_layout");

  // Note that we have to broadcast to handle replicate dimensions.
  SmallVector<int64_t> broadcasted_shape(
      toArrayRef(in_vreg_arrays[0].dimensions()));
  for (size_t i = 1; i < num_operands; ++i) {
    SmallVector<int64_t> new_broadcasted_shape;
    CHECK(OpTrait::util::getBroadcastedShape(
        broadcasted_shape, toArrayRef(in_vreg_arrays[i].dimensions()),
        new_broadcasted_shape));
    broadcasted_shape = std::move(new_broadcasted_shape);
  }
  CHECK(broadcasted_shape ==
        layout_out.tileArrayShape(out_ty.getShape(), ctx.target_shape));

  // TODO(tlongeri): Can we avoid initializing the array before filling values?
  xla::Array<Value> out_vreg_array(broadcasted_shape);
  out_vreg_array.Each([&](absl::Span<const int64_t> idx, Value *out_vreg) {
    SmallVector<Value> operands(num_operands);

    for (unsigned i = 0; i < num_operands; ++i) {
      // Handle indices for broadcasted dimensions
      SmallVector<int64_t> operand_idx(toArrayRef(idx));
      for (unsigned j = 0; j < idx.size(); ++j) {
        if (in_vreg_arrays[i].dim(j) == 1) {
          operand_idx[j] = 0;
        }
      }
      operands[i] = in_vreg_arrays[i](operand_idx);
    }
    Operation *vreg_op =
        builder.create(op.getLoc(), op.getName().getIdentifier(), operands,
                       out_vreg_ty, attributes.getAttrs());
    CHECK(vreg_op);
    CHECK_EQ(vreg_op->getNumResults(), 1);
    *out_vreg = vreg_op->getResult(0);
  });
  op.replaceAllUsesWith(
      assemble(ctx, builder, out_ty, layout_out, std::move(out_vreg_array)));
  op.erase();
  return success();
}

using rule_type = std::function<LogicalResult(
    RewriteContext &, Operation &, ArrayRef<Layout>, ArrayRef<Layout>)>;

template <typename OpTy>
LogicalResult ext_op_rule_impl(RewriteContext &ctx, OpTy op,
                               const VectorLayout &layout_in,
                               const VectorLayout &layout_out) {
  ImplicitLocOpBuilder builder(op.getLoc(), op.getOperation());
  auto result_ty = cast<VectorType>(op.getResult().getType());
  if (layout_out.bitwidth() != 32) {
    return op.emitOpError("Only extensions to 32-bit supported");
  }
  FAILUREOR_ASSIGN_OR_RETURN(const xla::Array<Value> input_vregs,
                             disassemble(ctx, builder, layout_in, op.getIn()));
  xla::Array<Value> output_vregs(
      layout_out.tileArrayShape(result_ty.getShape(), ctx.target_shape));
  FAILUREOR_ASSIGN_OR_RETURN(
      const VectorType res_vreg_ty,
      getNativeVregType(result_ty.getElementType(), ctx.target_shape));
  if (layout_in.implicit_dim() != layout_out.implicit_dim()) {
    return op.emitOpError("Not implemented: Change of layout during the cast");
  }
  switch (layout_in.implicit_dim()) {
    case VectorLayout::ImplicitDim::kNone: {
      if (layout_in.tiling() != layout_out.tiling()) {
        return op.emitOpError("Changing tiling during extension");
      }
      auto tiling = layout_in.tiling();
      if (ctx.target_shape[0] % tiling[0] != 0 ||
          ctx.target_shape[1] != tiling[1]) {
        return op.emitOpError("Not implemented: tiling not supported");
      }
      const int packing = layout_in.packing();
      output_vregs.Each([&](absl::Span<const int64_t> idxs, Value *v) {
        SmallVector<int64_t> input_vreg_idxs(toArrayRef(idxs));
        input_vreg_idxs.back() /= packing;
        const int64_t vreg_part = idxs.back() % packing;
        *v = builder.create<UnpackSubelementsOp>(
            res_vreg_ty, input_vregs(input_vreg_idxs), vreg_part);
      });
    } break;
    case VectorLayout::ImplicitDim::kMinor:
      return op.emitOpError(
          "Not implemented: Only casts of lane-oriented values supported");
    case VectorLayout::ImplicitDim::kSecondMinor: {
      if (input_vregs.dimensions() != absl::Span<const int64_t>{1} ||
          output_vregs.dimensions() != absl::Span<const int64_t>{1}) {
        return op.emitOpError("Not implemented");
      }
      if (layout_in.offsets()[0] >= ctx.target_shape[0]) {
        return op.emitOpError("Not implemented");
      }
      auto unpack_subelements_op = builder.create<UnpackSubelementsOp>(
          res_vreg_ty, *input_vregs.begin(), 0);
      output_vregs.Fill(unpack_subelements_op.getResult());
    }
  }
  op.replaceAllUsesWith(
      assemble(ctx, builder, result_ty, layout_out, std::move(output_vregs))
          .getResult());
  op.erase();
  return success();
}

LogicalResult arith_extf_rule(RewriteContext &ctx, Operation &op,
                              const ArrayRef<Layout> layouts_in,
                              const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 1);
  CHECK(layouts_in.front().has_value());
  CHECK(layouts_out.front().has_value());
  auto extf_op = cast<arith::ExtFOp>(op);
  if (layouts_in.front()->bitwidth() != 16 ||
      layouts_out.front()->bitwidth() != 32) {
    return op.emitOpError("Only 16-bit to 32-bit conversion supported");
  }
  return ext_op_rule_impl(ctx, extf_op, *layouts_in.front(),
                          *layouts_out.front());
}

LogicalResult arith_extsi_rule(RewriteContext &ctx, Operation &op,
                               const ArrayRef<Layout> layouts_in,
                               const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 1);
  CHECK(layouts_in.front().has_value());
  CHECK_EQ(layouts_out.size(), 1);
  CHECK(layouts_out.front().has_value());
  auto extsi_op = cast<arith::ExtSIOp>(op);
  return ext_op_rule_impl(ctx, extsi_op, *layouts_in.front(),
                          *layouts_out.front());
}

template <typename OpTy>
LogicalResult trunc_op_rule_impl(RewriteContext &ctx, OpTy op,
                                 const VectorLayout &layout_in,
                                 const VectorLayout &layout_out) {
  ImplicitLocOpBuilder builder(op.getLoc(), op.getOperation());
  auto result_ty = cast<VectorType>(op.getResult().getType());
  FAILUREOR_ASSIGN_OR_RETURN(const xla::Array<Value> input_vregs,
                             disassemble(ctx, builder, layout_in, op.getIn()));
  xla::Array<Value> output_vregs(
      layout_out.tileArrayShape(result_ty.getShape(), ctx.target_shape));
  if (layout_in.bitwidth() != 32) {
    return op.emitOpError("Only 32-bit truncation supported");
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      VectorType res_vreg_ty,
      getNativeVregType(result_ty.getElementType(), ctx.target_shape));
  if (layout_in.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      layout_out.implicit_dim() == VectorLayout::ImplicitDim::kNone) {
    if (layout_in.tiling() != ctx.target_shape) {
      return op.emitOpError("Not implemented: Only (8,128) tiling supported");
    }
    if (layout_out.tiling() == ctx.target_shape) {
      const int packing = layout_out.packing();
      output_vregs.Each([&](absl::Span<const int64_t> idxs, Value *v) {
        SmallVector<Value> parts;
        SmallVector<int64_t> idxs_local(toArrayRef(idxs));
        idxs_local.back() *= packing;
        for (int64_t i = 0; i < packing; ++i) {
          parts.push_back(input_vregs(idxs_local));
          // Pack any data lying around if OOB
          if (idxs_local.back() < input_vregs.dimensions().back() - 1) {
            ++idxs_local.back();
          }
        }
        *v = builder.create<PackSubelementsOp>(res_vreg_ty, parts);
      });

    } else if (layout_out.bitwidth() == 16 &&
               layout_out.tiling() ==
                   std::array<int64_t, 2>{2 * ctx.target_shape[0],
                                          ctx.target_shape[1]}) {
      output_vregs.Each([&](absl::Span<const int64_t> idxs, Value *v) {
        // TODO(tlongeri): should probably express as a multiple of target_shape
        // instead of (16, 128)
        CHECK_GE(idxs.size(), 2);
        SmallVector<int64_t> idxs_local(toArrayRef(idxs));
        idxs_local[idxs.size() - 2] *= 2;
        const Value first = input_vregs(idxs_local);
        Value second;
        if (idxs[idxs.size() - 2] * 2 + 1 ==
            input_vregs.dim(input_vregs.num_dimensions() - 2)) {
          second = first;
        } else {
          idxs_local[idxs.size() - 2] += 1;
          second = input_vregs(idxs_local);
        }
        *v = builder.create<PackSubelementsOp>(res_vreg_ty,
                                               ArrayRef<Value>{first, second});
      });
    } else {
      return op.emitOpError("Not implemented");
    }
    op.replaceAllUsesWith(
        assemble(ctx, builder, result_ty, layout_out, std::move(output_vregs))
            .getResult());
    op.erase();
    return success();
  }
  // TODO(tlongeri): why wasn't this part of the original code?
  return op.emitOpError("Not implemented");
}

LogicalResult arith_truncf_rule(RewriteContext &ctx, Operation &op,
                                const ArrayRef<Layout> layouts_in,
                                const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 1);
  CHECK(layouts_in.front().has_value());
  CHECK_EQ(layouts_out.size(), 1);
  CHECK(layouts_out.front().has_value());
  auto truncf_op = cast<arith::TruncFOp>(op);
  if (layouts_in.front()->bitwidth() != 32 ||
      layouts_out.front()->bitwidth() != 16) {
    return op.emitOpError(
        "Not implemented: Only 32-bit to 16-bit conversion supported");
  }
  return trunc_op_rule_impl(ctx, truncf_op, *layouts_in.front(),
                            *layouts_out.front());
}

LogicalResult arith_trunci_rule(RewriteContext &ctx, Operation &op,
                                const ArrayRef<Layout> layouts_in,
                                const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 1);
  CHECK(layouts_in.front().has_value());
  CHECK_EQ(layouts_out.size(), 1);
  CHECK(layouts_out.front().has_value());
  auto trunci_op = cast<arith::TruncIOp>(op);
  return trunc_op_rule_impl(ctx, trunci_op, *layouts_in.front(),
                            *layouts_out.front());
}

LogicalResult func_return_rule(RewriteContext &ctx, Operation &op,
                               const ArrayRef<Layout> layouts_in,
                               const ArrayRef<Layout> layouts_out) {
  CHECK(layouts_out.empty());
  for (const Layout &layout_in : layouts_in) {
    if (layout_in.has_value()) {
      return op.emitOpError("Vector-typed return values are not supported");
    }
  }
  return success();
}

LogicalResult scf_for_rule(RewriteContext &ctx, Operation &op,
                           const ArrayRef<Layout> layouts_in,
                           const ArrayRef<Layout> layouts_out) {
  scf::ForOp for_op = cast<scf::ForOp>(op);
  CHECK_EQ(layouts_in.size(), 3 + for_op.getInitArgs().size());
  CHECK_EQ(layouts_out.size(), for_op.getResults().size());
  if (!for_op.getInitArgs().empty() || !for_op.getResults().empty()) {
    return for_op.emitOpError("Not implemented: inputs and outputs in scf.for");
  }
  // It is an invariant that scf::ForOp should have a single region with a
  // single block (checked by MLIR verifier).
  return applyLayoutBlock(ctx, for_op.getRegion().getBlocks().front());
}

LogicalResult scf_if_rule(RewriteContext &ctx, Operation &op,
                          const ArrayRef<Layout> layouts_in,
                          const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 1);
  CHECK(!layouts_in.front().has_value());
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  scf::IfOp if_op = cast<scf::IfOp>(op);
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> then_yield_in_layouts,
                             getInLayout(*if_op.thenYield()));
  // TODO(tlongeri): ArrayRef<Layout> conversion should not be necessary, fix
  //                 after LLVM adds const qualifiers to ==/!= operators. Also
  //                 applies to else_yield_in_layouts comparison below.
  if (!layouts_out.empty() &&
      ArrayRef<Layout>(then_yield_in_layouts) != layouts_out) {
    return op.emitOpError(
        "Not implemented: different layouts in then yield's operands and if's "
        "results");
  }
  if (failed(applyLayoutBlock(ctx, *if_op.thenBlock()))) {
    return failure();
  }
  if (if_op.getElseRegion().empty()) {
    CHECK_EQ(if_op->getNumResults(), 0)
        << "Expected no results if op does not have an else block";
    CHECK_EQ(layouts_out.size(), 0);
    return success();
  }
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> else_yield_in_layouts,
                             getInLayout(*if_op.elseYield()));
  if (!layouts_out.empty() &&
      ArrayRef<Layout>(else_yield_in_layouts) != layouts_out) {
    return op.emitOpError(
        "Not implemented: different layouts in else yield's operands and if's "
        "results");
  }
  if (failed(applyLayoutBlock(ctx, *if_op.elseBlock()))) {
    return failure();
  }

  // Apply layout to results after applying layout in the true and false
  // regions.
  if (if_op.getNumResults() == 0) {
    CHECK_EQ(layouts_out.size(), 0);
    return success();
  }
  CHECK_EQ(if_op.getNumResults(), layouts_out.size());
  // If scf.if has results, it should have both non-empty true and false
  // regions.
  CHECK(!if_op.getThenRegion().empty() && !if_op.getElseRegion().empty());

  // Move true and false regions to the new if op whose result has same type and
  // layout as yield operand's.
  auto new_op = builder.create<scf::IfOp>(
      TypeRange(if_op.thenYield().getResults()), if_op.getCondition(),
      /*withElseRegion =*/true);
  moveAllRegions(*if_op, *new_op);

  int64_t index = 0;
  SmallVector<Value> rolled_results;
  for (auto [result, layout] :
       llvm::zip_equal(if_op.getResults(), layouts_out)) {
    if (const auto vty = dyn_cast<VectorType>(result.getType())) {
      // When the result has a vector type, assemble the result.
      CHECK(layout.has_value());
      const SmallVector<int64_t> tiles_shape =
          layout->tileArrayShape(vty.getShape(), ctx.target_shape);
      const int64_t num_vectors = ShapedType::getNumElements(tiles_shape);
      xla::Array<Value> tiles(tiles_shape);
      CHECK_LE(index + num_vectors, new_op.getResults().size());
      tiles.SetValues(
          llvm::make_range(new_op.getResults().begin() + index,
                           new_op.getResults().begin() + index + num_vectors));
      index += num_vectors;
      RollVectorsOp rolled_op = assemble(ctx, builder, vty, *layout, tiles);
      rolled_results.push_back(rolled_op);
    } else {
      CHECK(!layout.has_value());
      rolled_results.push_back(new_op.getResult(index));
      ++index;
    }
  }
  if_op.replaceAllUsesWith(rolled_results);
  if_op.erase();
  return success();
}

LogicalResult scf_yield_rule(RewriteContext &ctx, Operation &op,
                             const ArrayRef<Layout> layouts_in,
                             const ArrayRef<Layout> layouts_out) {
  OpBuilder builder(&op);
  auto yield_op = cast<scf::YieldOp>(op);
  CHECK_EQ(layouts_in.size(), yield_op.getNumOperands());
  CHECK_EQ(layouts_out.size(), 0);
  if (yield_op.getNumOperands() == 0) {
    return success();
  }
  SmallVector<Value> unrolled;
  for (auto [operand, layout] :
       llvm::zip_equal(yield_op.getOperands(), layouts_in)) {
    if (auto vty = dyn_cast<VectorType>(operand.getType())) {
      // When the operand has vector type, disassemble the operand.
      CHECK(layout.has_value());
      FAILUREOR_ASSIGN_OR_RETURN(const xla::Array<Value> tiles,
                                 disassemble(ctx, builder, *layout, operand));
      unrolled.append(tiles.begin(), tiles.end());
    } else {
      CHECK(!layout.has_value());
      unrolled.push_back(operand);
    }
  }

  // Replace the old operands with unrolled operands.
  yield_op->setOperands(unrolled);
  return success();
}

LogicalResult tpu_load_rule(RewriteContext &ctx, Operation &op,
                            const ArrayRef<Layout> layouts_in,
                            const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_out.size(), 1);
  if (llvm::any_of(layouts_in,
                   [&](const Layout &l) { return l.has_value(); })) {
    return op.emitOpError("Expected null input layouts");
  }
  if (!layouts_out.front().has_value()) {
    return op.emitOpError("Expected non-null output layout");
  }
  const VectorLayout &layout_out = *layouts_out.front();
  // We expect the result is already a native-sized vreg.
  // TODO(b/300493694): Support other bitwidths
  if (layout_out.bitwidth() != 32) {
    return op.emitOpError("Not implemented: Only 32-bit loads supported");
  }
  tpu::LoadOp load_op = cast<tpu::LoadOp>(op);
  if (layout_out != VectorLayout(32, {0, 0}, ctx.target_shape,
                                   VectorLayout::ImplicitDim::kNone)) {
    return op.emitOpError("Invalid output layout for ") << load_op->getName();
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      const SmallVector<int64_t> indices,
      getIntConstsFromOperandRange(load_op.getIndices()));
  CHECK_EQ(indices.size(), 2);
  if (indices[1] % ctx.target_shape[1] != 0) {
    return op.emitOpError("Not implemented: Lane index is not a multiple of ")
           << ctx.target_shape[1];
  }

  OpBuilder builder(op.getContext());
  builder.setInsertionPointAfter(&op);
  const RollVectorsOp roll_vectors_op =
      assemble(ctx, builder, load_op.getResult().getType(), layout_out,
               {{load_op.getResult()}});
  load_op->replaceUsesWithIf(roll_vectors_op, [&](OpOperand &operand) {
    return operand.getOwner() != roll_vectors_op;
  });
  return success();
}

LogicalResult matmul_rule_impl(RewriteContext &ctx, Operation &op,
                               const bool transpose_lhs,
                               const bool transpose_rhs,
                               const VectorLayout &layout_lhs,
                               const VectorLayout &layout_rhs,
                               const VectorLayout &layout_acc,
                               const VectorLayout &layout_out) {
  if (transpose_lhs) {
    return op.emitOpError("Not implemented: Transposed LHS");
  }
  const std::array<std::reference_wrapper<const VectorLayout>, 3> layouts_in = {
      layout_lhs, layout_rhs, layout_acc};
  const std::array<std::reference_wrapper<const VectorLayout>, 4> all_layouts =
      {layout_lhs, layout_rhs, layout_acc, layout_out};
  for (const VectorLayout &layout : all_layouts) {
    for (const LayoutOffset offset : layout.offsets()) {
      if (offset.value_or(0) != 0) {
        return op.emitOpError("Not implemented: Unaligned layout in matmul");
      }
    }
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  TypedValue<VectorType> lhs, rhs, acc, res;
  if (auto tpu_matmul_op = dyn_cast<tpu::MatmulOp>(op)) {
    lhs = tpu_matmul_op.getLhs();
    rhs = tpu_matmul_op.getRhs();
    acc = tpu_matmul_op.getAcc();
    res = tpu_matmul_op.getResult();
  } else if (auto vector_contraction_op = dyn_cast<vector::ContractionOp>(op)) {
    lhs = vector_contraction_op.getLhs();
    rhs = vector_contraction_op.getRhs();
    acc = cast<TypedValue<VectorType>>(vector_contraction_op.getAcc());
    res = cast<TypedValue<VectorType>>(vector_contraction_op.getResult());
  } else {
    LOG(FATAL) << "Unexpected op type";
  }

  for (const VectorLayout &layout : layouts_in) {
    if (layout.implicit_dim() != VectorLayout::ImplicitDim::kNone) {
      return op.emitOpError(
          "Not implemented: Unsupported matmul operand layout");
    }
    if (!layout.hasNativeTiling(ctx.target_shape)) {
      return op.emitOpError(
          "Not implemented: Unsupported matmul operand tiling");
    }
  }
  if (acc.getType().getElementType().getIntOrFloatBitWidth() != 32) {
    return op.emitOpError("Not implemented: Non-32-bit matmul result");
  }
  const ArrayRef<int64_t> lhs_shape = lhs.getType().getShape();
  const ArrayRef<int64_t> rhs_shape = rhs.getType().getShape();
  // TODO(tlongeri): This should be part of the tpu::MatmulOp verifier
  CHECK_EQ(lhs_shape.size(), 2);
  CHECK_EQ(rhs_shape.size(), 2);
  // The code below puts no constraints on the second dimension of both lhs and
  // rhs. However, leading axis of lhs needs to be a multiple of native tiling
  // for packed types, while leading axis of rhs needs to be a multiple of 128
  // (no matter the type and transpose mode).
  if (layout_lhs.packing() != 1 && lhs_shape[0] % layout_lhs.tiling()[0] != 0) {
    return op.emitOpError("Not implemented: Unsupported LHS shape");
  }
  if (rhs_shape[0] % 128 != 0) {
    return op.emitOpError("Not implemented: Unsupported RHS shape");
  }
  const int64_t padded_lhs_rows =
      llvm::alignTo(lhs_shape[0], layout_lhs.tiling()[0]);
  const auto lhs_col_ty =
      VectorType::get({padded_lhs_rows, 128}, lhs.getType().getElementType());
  if (llvm::alignTo(lhs_shape[0], layout_acc.tiling()[0]) != padded_lhs_rows) {
    return op.emitOpError(
        "Not implemented: Matmul acc requires less padding than lhs");
  }
  const auto acc_col_ty =
      VectorType::get({padded_lhs_rows, 128}, acc.getType().getElementType());
  FAILUREOR_ASSIGN_OR_RETURN(xla::Array<Value> lhs_vregs,
                             disassemble(ctx, builder, layout_lhs, lhs));
  FAILUREOR_ASSIGN_OR_RETURN(xla::Array<Value> acc_vregs,
                             disassemble(ctx, builder, layout_acc, acc));
  CHECK_EQ(padded_lhs_rows, lhs_vregs.dim(0) * layout_lhs.tiling()[0]);
  CHECK_EQ(padded_lhs_rows, acc_vregs.dim(0) * layout_acc.tiling()[0]);
  SmallVector<tpu::RollVectorsOp> lhs_cols(lhs_vregs.dim(1));

  TypedValue<VectorType> contraction_lane_mask;
  auto maskLastLaneContractionVreg = [&](TypedValue<VectorType> zeros,
                                         TypedValue<VectorType> vreg) {
    CHECK(contraction_lane_mask != nullptr);
    TypedValue<VectorType> mask = contraction_lane_mask;
    if (vreg.getType().getShape() != mask.getType().getShape()) {
      mask = builder.create<tpu::MaskCastOp>(
          VectorType::get(vreg.getType().getShape(), builder.getI1Type()),
          mask);
    }
    return builder.create<arith::SelectOp>(mask, vreg, zeros);
  };
  if (const int64_t contraction_rem = lhs_shape[1] % 128) {
    FAILUREOR_ASSIGN_OR_RETURN(
        const VectorType i32_vreg,
        getNativeVregType(builder.getI32Type(), ctx.target_shape));
    contraction_lane_mask = cast<TypedValue<VectorType>>(
        builder
            .create<arith::CmpIOp>(
                arith::CmpIPredicate::slt,
                builder.create<tpu::IotaOp>(
                    i32_vreg,
                    /*dimension=*/builder.getI32IntegerAttr(1)),
                builder.create<arith::ConstantOp>(
                    i32_vreg, builder.getI32IntegerAttr(contraction_rem)))
            .getResult());
    const VectorType lhs_vreg_type =
        cast<VectorType>(lhs_vregs.begin()->getType());
    FAILUREOR_ASSIGN_OR_RETURN(
        const Attribute zero_attr,
        getZeroIntOrFloatAttr(lhs_vreg_type.getElementType()));
    auto lhs_zeros = cast<TypedValue<VectorType>>(
        builder
            .create<arith::ConstantOp>(
                op.getLoc(), DenseElementsAttr::get(lhs_vreg_type, zero_attr))
            .getResult());
    for (int64_t i = 0; i < lhs_vregs.dim(0); ++i) {
      Value &vreg = lhs_vregs({i, lhs_vregs.dim(1) - 1});
      vreg = maskLastLaneContractionVreg(lhs_zeros,
                                         cast<TypedValue<VectorType>>(vreg));
    }
  }
  const ArrayAttr lhs_layout_attr =
      builder.getArrayAttr({builder.getAttr<VectorLayoutAttr>(layout_lhs)});
  const ArrayAttr rhs_layout_attr =
      builder.getArrayAttr({builder.getAttr<VectorLayoutAttr>(layout_rhs)});
  const ArrayAttr acc_layout_attr =
      builder.getArrayAttr({builder.getAttr<VectorLayoutAttr>(layout_acc)});
  for (int64_t i = 0; i < lhs_vregs.dim(1); ++i) {
    const xla::Array<Value> col_vregs =
        lhs_vregs.Slice({0, i}, {lhs_vregs.dim(0), i + 1});
    lhs_cols[i] = builder.create<tpu::RollVectorsOp>(
        op.getLoc(), lhs_col_ty, XlaArrayToFlatArrayRef(col_vregs));
    lhs_cols[i]->setAttr("out_layout", lhs_layout_attr);
  }
  // Here, "tile" is used as in the context of the MXU, a 128x128 operand to a
  // matmul computation (NOT as in the context of tiled layouts).
  const auto rhs_tile_ty =
      VectorType::get({128, 128}, rhs.getType().getElementType());
  FAILUREOR_ASSIGN_OR_RETURN(xla::Array<Value> rhs_vregs,
                             disassemble(ctx, builder, layout_rhs, rhs));
  const int64_t rhs_vregs_per_tile = 16 / layout_rhs.packing();
  int64_t nj, nk;
  if (transpose_rhs) {
    nj = llvm::divideCeil(rhs_shape[0], 128);
    nk = llvm::divideCeil(rhs_shape[1], 128);
    rhs_vregs.Reshape({nj, rhs_vregs_per_tile, nk});
    rhs_vregs.TransposeDimensions({2, 0, 1});
  } else {
    nj = llvm::divideCeil(rhs_shape[1], 128);
    nk = llvm::divideCeil(rhs_shape[0], 128);
    rhs_vregs.Reshape({nk, rhs_vregs_per_tile, nj});
    rhs_vregs.TransposeDimensions({0, 2, 1});
  }
  const tpu::ContractPrecisionAttr precision_attr =  // May be null
      op.getAttrOfType<tpu::ContractPrecisionAttr>("precision");
  const auto rhs_vreg_type = cast<VectorType>(rhs_vregs.begin()->getType());
  FAILUREOR_ASSIGN_OR_RETURN(
      const Attribute zero_attr,
      getZeroIntOrFloatAttr(rhs_vreg_type.getElementType()));
  auto rhs_zeros = cast<TypedValue<VectorType>>(
      builder
          .create<arith::ConstantOp>(
              op.getLoc(), DenseElementsAttr::get(rhs_vreg_type, zero_attr))
          .getResult());
  for (int64_t j = 0; j < nj; ++j) {
    for (int64_t k = 0; k < nk; ++k) {
      // TODO(tlongeri): there should be a way to slice without copying
      xla::Array<Value> rhs_tile =
          rhs_vregs.Slice({k, j, 0}, {k + 1, j + 1, rhs_vregs_per_tile});
      if (contraction_lane_mask != nullptr && k == nk - 1) {
        rhs_tile.Each(
            [&](const absl::Span<const int64_t> idx, Value *const vreg) {
              *vreg = maskLastLaneContractionVreg(
                  rhs_zeros, cast<TypedValue<VectorType>>(*vreg));
            });
      }
      auto rhs_rolled_tile = builder.create<tpu::RollVectorsOp>(
          op.getLoc(), rhs_tile_ty, XlaArrayToFlatArrayRef(rhs_tile));
      rhs_rolled_tile->setAttr("out_layout", rhs_layout_attr);
      const xla::Array<Value> acc_col_vregs =
          acc_vregs.Slice({0, j}, {acc_vregs.dim(0), j + 1});
      auto acc_col = builder.create<tpu::RollVectorsOp>(
          op.getLoc(), acc_col_ty, XlaArrayToFlatArrayRef(acc_col_vregs));
      acc_col->setAttr("out_layout", acc_layout_attr);
      auto new_acc_col = builder.create<tpu::MatmulOp>(
          op.getLoc(), acc_col_ty, lhs_cols[k], rhs_rolled_tile, acc_col,
          transpose_lhs, transpose_rhs, precision_attr);
      new_acc_col->setAttr("out_layout", acc_layout_attr);
      auto new_acc_vregs = builder.create<tpu::UnrollVectorsOp>(
          op.getLoc(),
          TypeRange(ValueRange(XlaArrayToFlatArrayRef(acc_col_vregs))),
          new_acc_col);
      new_acc_vregs->setAttr("in_layout", acc_layout_attr);
      updateSliceFromRange(acc_vregs, new_acc_vregs->getResults(), {0, j},
                           {acc_vregs.dim(0), j + 1});
    }
  }
  op.replaceAllUsesWith(
      assemble(ctx, builder, res.getType(), layout_out, acc_vregs)
          .getOperation());
  op.erase();
  return success();
}

LogicalResult tpu_matmul_rule(RewriteContext &ctx, Operation &op,
                              const ArrayRef<Layout> layouts_in,
                              const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 3);
  CHECK_EQ(layouts_out.size(), 1);
  for (const Layout &layout : layouts_in) {
    if (!layout.has_value()) {
      return op.emitOpError("Expected non-null input layouts");
    }
  }
  if (!layouts_out.front().has_value()) {
    return op.emitOpError("Expected non-null output layout");
  }
  auto matmul_op = cast<tpu::MatmulOp>(op);
  return matmul_rule_impl(ctx, *matmul_op, matmul_op.getTransposeLhs(),
                          matmul_op.getTransposeRhs(), *layouts_in[0],
                          *layouts_in[1], *layouts_in[2], *layouts_out[0]);
}

LogicalResult tpu_store_rule(RewriteContext &ctx, Operation &op,
                             const ArrayRef<Layout> layouts_in,
                             const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_out.size(), 0);
  if (llvm::any_of(layouts_in.drop_front(),
                   [&](const Layout &l) { return l.has_value(); })) {
    return op.emitOpError("Expected null layouts for tpu.store indices");
  }
  if (!layouts_in.front().has_value()) {
    return op.emitOpError("Expected non-null layout for tpu.store base");
  }
  OpBuilder builder(&op);
  const VectorLayout &to_store_layout = *layouts_in.front();
  // We expect the value to store is already a native-sized vreg.
  if (to_store_layout.bitwidth() != 32) {
    return op.emitOpError("Not implemented: Only 32-bit loads supported");
  }
  CHECK(to_store_layout == VectorLayout(32, {0, 0}, ctx.target_shape,
                                        VectorLayout::ImplicitDim::kNone));
  tpu::StoreOp store_op = cast<tpu::StoreOp>(op);
  FAILUREOR_ASSIGN_OR_RETURN(
      const SmallVector<int64_t> indices,
      getIntConstsFromOperandRange(store_op.getIndices()));
  CHECK_EQ(indices.size(), 2);
  if (indices[1] % ctx.target_shape[1] != 0) {
    return op.emitOpError("Not implemented: Lane index is not a multiple of ")
           << ctx.target_shape[1];
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> tiles,
      disassemble(ctx, builder, to_store_layout, store_op.getValueToStore()));
  CHECK((tiles.dimensions() == xla::DimensionVector{1, 1}));
  store_op.getValueToStoreMutable().assign(tiles({0, 0}));
  return success();
}

LogicalResult tpu_trace_rule(RewriteContext &ctx, Operation &op,
                             const ArrayRef<Layout> layouts_in,
                             const ArrayRef<Layout> layouts_out) {
  if (op.getNumOperands() != 0 || op.getNumResults() != 0) {
    return op.emitOpError(
        "Not implemented: tpu.traced_block with inputs or outputs");
  }
  CHECK_EQ(layouts_in.size(), 0);
  CHECK_EQ(layouts_out.size(), 0);
  // We don't modify the op, but we do rewrite the branch bodies.
  CHECK_EQ(op.getNumRegions(), 1);
  Region &region = op.getRegion(0);
  CHECK(region.hasOneBlock());
  Block &block = region.front();
  return applyLayoutBlock(ctx, block);
}

LogicalResult tpu_iota_rule(RewriteContext &ctx, Operation &op,
                            const ArrayRef<Layout> layouts_in,
                            const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 0);
  CHECK_EQ(layouts_out.size(), 1);
  if (!layouts_out.front().has_value()) {
    return op.emitOpError("Expected non-null output layout");
  }
  const VectorLayout &layout_out = *layouts_out.front();
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  tpu::IotaOp iota_op = cast<tpu::IotaOp>(op);
  VectorType vty = iota_op.getResult().getType();
  if (const auto int_ty = dyn_cast<IntegerType>(vty.getElementType());
      int_ty == nullptr || int_ty.getWidth() != 32) {
    return iota_op.emitOpError("Not implemented: Only 32-bit Iota supported");
  }
  if (!layout_out.hasNativeTiling(ctx.target_shape)) {
    return iota_op.emitOpError("Not implemented: Only native tiling supported");
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      const auto native_vreg_ty,
      getNativeVregType(vty.getElementType(), ctx.target_shape));
  if (layout_out.implicit_dim() != VectorLayout::ImplicitDim::kNone) {
    return op.emitOpError("Not implemented: Only 2D layouts supported");
  }
  const SmallVector<int64_t> tile_array_shape =
      layout_out.tileArrayShape(vty.getShape(), ctx.target_shape);
  const std::optional<int32_t> dimension = iota_op.getDimension();
  if (!dimension.has_value()) {
    return op.emitOpError("Not implemented: null dimension");
  }
  if (*dimension == vty.getRank() - 1) {
    if (layout_out.offsets()[1] != 0) {
      return op.emitOpError("Not implemented: Unsupported offset");
    }
    const int64_t num_tiles = tile_array_shape[tile_array_shape.size() - 1];
    SmallVector<Value> tiles(num_tiles);
    auto vreg_iota = builder.create<tpu::IotaOp>(
        native_vreg_ty,
        /*dimension =*/builder.getI32IntegerAttr(1));
    for (int64_t i = 0; i < num_tiles; ++i) {
      auto offset = builder.create<arith::ConstantOp>(
          native_vreg_ty,
          DenseElementsAttr::get(
              native_vreg_ty,
              IntegerAttr::get(vty.getElementType(),
                               i * *(native_vreg_ty.getShape().end() - 1))));
      tiles[i] = builder.create<arith::AddIOp>(vreg_iota, offset);
    }
    xla::Array<Value> broadcasted_tiles(tile_array_shape);
    broadcasted_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      *v = tiles[*(idxs.end() - 1)];
    });
    op.replaceAllUsesWith(
        assemble(ctx, builder, vty, layout_out, broadcasted_tiles));
    op.erase();
    return success();
  }
  if (*dimension == vty.getRank() - 2) {
    if (layout_out.offsets()[0] != 0) {
      return op.emitOpError("Not implemented: Unsupported offset");
    }
    const int64_t num_tiles = tile_array_shape[tile_array_shape.size() - 2];
    SmallVector<Value> tiles(num_tiles);
    auto vreg_iota = builder.create<tpu::IotaOp>(
        native_vreg_ty,
        /*dimension =*/builder.getI32IntegerAttr(0));
    for (int64_t i = 0; i < num_tiles; ++i) {
      auto offset = builder.create<arith::ConstantOp>(
          native_vreg_ty,
          DenseElementsAttr::get(
              native_vreg_ty,
              IntegerAttr::get(vty.getElementType(),
                               i * *(native_vreg_ty.getShape().end() - 2))));
      tiles[i] = builder.create<arith::AddIOp>(vreg_iota, offset);
    }
    xla::Array<Value> broadcasted_tiles(tile_array_shape);
    broadcasted_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      *v = tiles[*(idxs.end() - 2)];
    });
    op.replaceAllUsesWith(
        assemble(ctx, builder, vty, layout_out, broadcasted_tiles));
    op.erase();
    return success();
  }
  return op.emitOpError("Not implemented: Unsupported dimension");
}

LogicalResult tpu_gather_rule(RewriteContext &ctx, Operation &op,
                              const ArrayRef<Layout> layouts_in,
                              const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 1);
  CHECK_EQ(layouts_out.size(), 1);
  if (!layouts_in.front().has_value()) {
    return op.emitOpError("Expected non-null input layout");
  }
  if (!layouts_out.front().has_value()) {
    return op.emitOpError("Expected non-null output layout");
  }
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  if (layout_in.implicit_dim() != VectorLayout::ImplicitDim::kNone ||
      layout_out.implicit_dim() != VectorLayout::ImplicitDim::kNone ||
      layout_in.offsets() != layout_out.offsets() ||
      llvm::any_of(layout_in.offsets(), [&](const LayoutOffset o) {
        return o.has_value() && o != 0;
      })) {
    return op.emitOpError("Not implemented: Only 2D layouts supported");
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto gather_op = cast<tpu::GatherOp>(op);
  const VectorType vty = gather_op.getResult().getType();
  const uint32_t dimension = gather_op.getDimension();
  if (dimension + 2 < vty.getRank()) {
    return op.emitOpError("Not implemented: Unsupported dimension");
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      const xla::Array<Value> in_tiles,
      disassemble(ctx, builder, layout_in, gather_op.getSource()));
  const int64_t width = ctx.target_shape[2 - (vty.getRank() - dimension)];
  const ArrayRef<int32_t> indices(gather_op.getIndices());
  auto [num_sections, rem] = std::div(indices.size(), width);
  SmallVector<int32_t> segment_indices;
  if (rem == 0) {
    for (int64_t i = 0; i < width; ++i) {
      const int64_t offset = i - i % width;
      if (!(offset <= indices[i] && indices[i] < offset + width)) {
        return op.emitOpError("Not implemented: Cross-segment gather");
      }
    }
    for (int64_t i = width; i < indices.size(); ++i) {
      const int64_t offset = i - i % width;
      if (indices[i] != indices[i % width] + offset) {
        return op.emitOpError(
            "Not implemented: Indices varying between segments");
      }
    }
    segment_indices.assign(indices.begin(), indices.begin() + width);
  } else if (num_sections == 0) {  // Only one vreg.
    segment_indices.assign(indices.begin(), indices.end());
    segment_indices.append(width - indices.size(), 0);
  } else {
    return op.emitOpError("Not implemented: Not a multiple of target length");
  }
  xla::Array<Value> out_tiles(in_tiles.dimensions());
  if (dimension == vty.getRank() - 1) {
    // TODO(b/265133497): Remove the broadcast once 2nd minor works.
    const auto dyn_ix_ty =
        VectorType::get(ctx.target_shape, builder.getI32Type());
    // Broadcast indices to target_shape
    SmallVector<int32_t> dyn_ix_val;
    for (int64_t i = 0; i < ctx.target_shape[0]; ++i) {  // Broadcast
      dyn_ix_val.append(segment_indices);
    }
    FAILUREOR_ASSIGN_OR_RETURN(
        const BlockArgument dyn_ix_ref,
        appendConstant(ctx, DenseIntElementsAttr::get(dyn_ix_ty, dyn_ix_val)));
    auto all_sublanes = builder.getAttr<DenseBoolArrayAttr>(
        SmallVector<bool>(ctx.target_shape[1], true));
    auto dyn_ix = builder.create<tpu::LoadOp>(
        dyn_ix_ty, dyn_ix_ref,
        SmallVector<Value>(2, IdxConst(0, builder, op.getLoc())),
        /*sublane_mask=*/all_sublanes, /*sublane_stride=*/nullptr);
    out_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      const Value in_tile = in_tiles(idxs);
      *v = builder.create<tpu::DynamicGatherOp>(in_tile.getType(), in_tile,
                                                dyn_ix, 1);
    });
  } else {
    CHECK_EQ(dimension, vty.getRank() - 2);
    const auto segment_indices_attr =
        builder.getAttr<DenseI32ArrayAttr>(segment_indices);
    out_tiles.Each([&](absl::Span<const int64_t> idxs, Value *v) {
      const Value in_tile = in_tiles(idxs);
      *v = builder.create<tpu::GatherOp>(in_tile.getType(), in_tile,
                                         segment_indices_attr, 0);
    });
  }
  gather_op.replaceAllUsesWith(
      assemble(ctx, builder, vty, layout_out, out_tiles).getOperation());
  gather_op.erase();
  return success();
}

LogicalResult tpu_repeat_rule(RewriteContext &ctx, Operation &op,
                              const ArrayRef<Layout> layouts_in,
                              const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 1);
  CHECK_EQ(layouts_out.size(), 1);
  if (!layouts_in.front().has_value()) {
    return op.emitOpError("Expected non-null input layout");
  }
  if (!layouts_out.front().has_value()) {
    return op.emitOpError("Expected non-null output layout");
  }
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  if (layout_in.implicit_dim() != VectorLayout::ImplicitDim::kNone) {
    return op.emitOpError("Not implemented: Only 2D layouts supported");
  }
  if (layout_in != layout_out) {
    return op.emitOpError("Not implemented: Changing layout mid-repeat");
  }
  if (!layout_in.hasNaturalTopology(ctx.target_shape) ||
      layout_in.offsets() != LayoutOffsets{0, 0}) {
    return op.emitOpError("Not implemented: Non-trivial layouts unsupported");
  }
  OpBuilder builder(&op);
  tpu::RepeatOp repeat_op = cast<tpu::RepeatOp>(op);
  VectorType src_ty = repeat_op.getSource().getType();
  const uint32_t dim = repeat_op.getDimension();
  if (dim != src_ty.getRank() - 1) {
    return op.emitOpError(
        "Not implemented: Only repeats along the last dim supported");
  }
  if (src_ty.getShape().back() % ctx.target_shape.back() != 0) {
    return op.emitOpError("Not implemented: Only free repeats are suppported");
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      const xla::Array<Value> &in_vregs,
      disassemble(ctx, builder, layout_in, repeat_op.getSource()));
  xla::Array<Value> out_vregs = repeat(in_vregs, repeat_op.getTimes(), dim);
  repeat_op->replaceAllUsesWith(assemble(
      ctx, builder, repeat_op.getResult().getType(), layout_out, out_vregs));
  repeat_op->erase();
  return success();
}

LogicalResult vector_load_rule(RewriteContext &ctx, Operation &op,
                               const ArrayRef<Layout> layouts_in,
                               const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_out.size(), 1);
  MLIRContext *const mlir_ctx = op.getContext();
  if (llvm::any_of(layouts_in,
                   [&](const Layout &l) { return l.has_value(); })) {
    return op.emitOpError("Expected null input layouts");
  }
  if (!layouts_out.front().has_value()) {
    return op.emitOpError("Expected non-null output layout");
  }
  const VectorLayout &layout_out = *layouts_out.front();
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto load_op = cast<vector::LoadOp>(op);
  const auto memref_ty = cast<MemRefType>(load_op.getBase().getType());
  const auto vty = cast<VectorType>(load_op.getResult().getType());
  FAILUREOR_ASSIGN_OR_RETURN(
      VectorType target_ty,
      getNativeVregType(vty.getElementType(), ctx.target_shape));
  if (layout_out.implicit_dim() == VectorLayout::ImplicitDim::kMinor) {
    return op.emitOpError("Not implemented");
  }
  const bool is_1d =
      layout_out.implicit_dim() != VectorLayout::ImplicitDim::kNone;
  using Tiling = std::array<int64_t, 2>;  // To avoid comma in macro
  FAILUREOR_ASSIGN_OR_RETURN(
      Tiling memref_tiling,
      getMemRefTiling(load_op.getBase(), ctx.target_shape));
  if (layout_out.tiling() != memref_tiling) {
    // Now we can handle the case when tiling is (1, TARGET_SHAPE.lanes).
    // TODO(b/295393167): need to support strided load for bitwidth < 32.
    if (layout_out.bitwidth() != 32 ||
        layout_out.tiling() != std::array<int64_t, 2>{1, ctx.target_shape[1]}) {
      return op.emitOpError("Not implemented");
    }
  }
  // TODO(apaszke): Check that loads are from vmem!
  FAILUREOR_ASSIGN_OR_RETURN(
      const SmallVector<int64_t> indices,
      getIntConstsFromOperandRange(load_op.getIndices()));
  if (llvm::any_of(
          llvm::zip_equal(indices, vty.getShape(), memref_ty.getShape()),
          [](auto tup) {
            auto [idx, n, extent] = tup;
            return idx + n > extent;
          })) {
    return op.emitOpError("Reading out of bounds");
  }
  const SmallVector<int64_t> implicit_shape =
      layout_out.implicitShape(vty.getShape());
  const int64_t ss = implicit_shape[implicit_shape.size() - 2];
  int64_t sublane_stride = 1;
  if (layout_out.bitwidth() == 32 &&
      layout_out.tiling() == std::array<int64_t, 2>{1, ctx.target_shape[1]} &&
      ss == 1) {
    sublane_stride = memref_tiling[0];
  }
  const LayoutOffsets offsets = layout_out.offsets();
  AffineMap load_map;
  arith::ConstantOp padding;
  if (offsets[1] == std::nullopt) {
    return op.emitOpError("Load replicated along lanes is unsupported");
  }
  if (offsets[0] == std::nullopt) {
    if (ss != 1) {
      return op.emitOpError(
          "Sublane-replicated load with size > 1 is unsupported");
    }
    if (!layout_out.hasNativeTiling(ctx.target_shape)) {
      return op.emitOpError("Not implemented");
    }
    // affine_map<(..., j) -> (0, j)
    load_map =
        AffineMap::get(memref_ty.getRank(), 0,
                       {getAffineConstantExpr(0, mlir_ctx),
                        getAffineDimExpr(memref_ty.getRank() - 1, mlir_ctx)},
                       mlir_ctx);
    FAILUREOR_ASSIGN_OR_RETURN(const TypedAttr zero_attr,
                               getZeroIntOrFloatAttr(vty.getElementType()));
    padding =
        builder.create<arith::ConstantOp>(vty.getElementType(), zero_attr);
  }
  xla::Array<Value> tiles(
      layout_out.tileArrayShape(vty.getShape(), ctx.target_shape));
  const std::array<int64_t, 2> vreg_slice =
      layout_out.vregSlice(ctx.target_shape);
  const int64_t num_dims = indices.size();
  const int64_t num_batch_dims = num_dims - (is_1d ? 1 : 2);
  const absl::Status status =
      tiles.EachStatus([&](absl::Span<const int64_t> tile_idxs, Value * /*v*/) {
        CHECK_EQ(num_dims, tile_idxs.size());
        SmallVector<int64_t> idxs(tile_idxs.size());
        for (int64_t i = 0; i < num_batch_dims; ++i) {
          idxs[i] = tile_idxs[i] + indices[i];
        }
        const int64_t base_l = indices[num_dims - 1];
        const int64_t lidx = tile_idxs[num_dims - 1];
        idxs[num_dims - 1] = base_l + lidx * vreg_slice[1] - *offsets[1];
        if (!is_1d) {
          const int64_t base_s = indices[num_dims - 2];
          const int64_t sidx = tile_idxs[num_dims - 2];
          idxs[num_dims - 2] =
              base_s + sidx * vreg_slice[0] - offsets[0].value_or(0);
        }
        CHECK(tile_idxs[num_dims - 1] + ctx.target_shape[1] <=
              memref_ty.getShape()[num_dims - 1]);
        std::unique_ptr<VRegDataBounds> bounds = layout_out.tileDataBounds(
            mlir_ctx, vty.getShape(), toArrayRef(tile_idxs), ctx.target_shape,
            /*allow_replicated =*/{true, false});
        SmallVector<Value> idxs_vs(idxs.size());
        for (int64_t i = 0; i < idxs.size(); ++i) {
          idxs_vs[i] = IdxConst(idxs[i], builder, load_op->getLoc());
        }
        Operation *tile;
        if (bounds->maskVariesAlong(Direction::kSublanes, ctx.target_shape)) {
          CHECK(offsets[0].has_value());
          tile = builder.create<tpu::LoadOp>(
              target_ty, load_op.getBase(), idxs_vs,
              bounds->getSublaneMask(mlir_ctx, ctx.target_shape),
              builder.getI32IntegerAttr(sublane_stride));
        } else {
          if (load_map) {
            CHECK(padding);
            if (layout_out.bitwidth() != 32) {
              load_op.emitOpError("Not implemented");
              return absl::UnimplementedError("");
            }
            tile = builder.create<vector::TransferReadOp>(
                target_ty, load_op.getBase(), idxs_vs, load_map, padding,
                nullptr, nullptr);
          } else {
            const SmallVector<bool> sublane_mask(ctx.target_shape[0], true);
            const auto sublane_mask_attr =
                DenseBoolArrayAttr::get(mlir_ctx, sublane_mask);
            tile = builder.create<tpu::LoadOp>(
                target_ty, load_op.getBase(), idxs_vs, sublane_mask_attr,
                builder.getI32IntegerAttr(sublane_stride));
          }
        }
        tiles(tile_idxs) = tile->getResult(0);
        return absl::OkStatus();
      });
  if (!status.ok()) {
    return failure();
  }
  load_op->replaceAllUsesWith(
      assemble(ctx, builder, vty, layout_out, std::move(tiles)));
  load_op->erase();
  return success();
}

LogicalResult arith_constant_rule(RewriteContext &ctx, Operation &op,
                                  const ArrayRef<Layout> layouts_in,
                                  const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 0);
  CHECK_EQ(layouts_out.size(), 1);
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto constant_op = cast<arith::ConstantOp>(op);
  auto vty = dyn_cast<VectorType>(op.getResult(0).getType());
  if (vty) {
    if (!layouts_out.front().has_value()) {
      return op.emitOpError(
          "Expected non-null output layout for vector constant");
    }
    const VectorLayout &layout_out = *layouts_out.front();
    DenseElementsAttr value = cast<DenseElementsAttr>(constant_op.getValue());
    FAILUREOR_ASSIGN_OR_RETURN(
        const VectorType target_vty,
        getNativeVregType(vty.getElementType(), ctx.target_shape));
    if (value.isSplat()) {
      if (layout_out.offsets() != LayoutOffsets{std::nullopt, std::nullopt}) {
        return op.emitOpError("Non-replicated splat constants");
      }
      auto new_value =
          DenseElementsAttr::get(target_vty, value.getSplatValue<Attribute>());
      const auto tile =
          builder.create<arith::ConstantOp>(target_vty, new_value);
      const xla::Array<Value> tiles(
          layout_out.tileArrayShape(vty.getShape(), ctx.target_shape),
          tile->getResult(0));
      op.replaceAllUsesWith(
          assemble(ctx, builder, vty, layout_out, std::move(tiles)));
      op.erase();
      return success();
    }
    // !value.isSplat()
    if (getTypeBitwidth<true>(vty.getElementType()) != 32) {
      return op.emitOpError("Only 32-bit non-splat constants are supported");
    }
    FAILUREOR_ASSIGN_OR_RETURN(const BlockArgument ref,
                               appendConstant(ctx, value));
    auto load_op = builder.create<vector::LoadOp>(
        vty, ref,
        SmallVector<Value>(vty.getRank(), IdxConst(0, builder, op.getLoc())));
    op.replaceAllUsesWith(ArrayRef<Value>{load_op.getResult()});
    op.erase();
    const SmallVector<Layout> vector_load_in_layouts(vty.getRank() + 1);
    return vector_load_rule(ctx, *load_op, vector_load_in_layouts,
                            {VectorLayout(/*bitwidth=*/32, /*offsets=*/{0, 0},
                                          /*tiling=*/ctx.target_shape)});
  }
  return op.emitOpError("Unsupported arith.const type: ")
         << op.getResult(0).getType();
}

LogicalResult vector_broadcast_rule(RewriteContext &ctx, Operation &op,
                                    const ArrayRef<Layout> layouts_in,
                                    const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 1);
  CHECK_EQ(layouts_out.size(), 1);
  if (!layouts_out.front().has_value()) {
    return op.emitOpError("Expected non-null output layout");
  }
  const Layout &maybe_layout_in = layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  vector::BroadcastOp broadcast_op = cast<vector::BroadcastOp>(op);
  const VectorType dst_ty = broadcast_op.getResult().getType();
  const SmallVector<int64_t> dst_tiles_shape =
      layout_out.tileArrayShape(dst_ty.getShape(), ctx.target_shape);
  if (auto src_ty = dyn_cast<VectorType>(broadcast_op.getSourceType())) {
    CHECK(maybe_layout_in.has_value());
    const VectorLayout &layout_in = *maybe_layout_in;
    if (layout_in.implicit_dim() != layout_out.implicit_dim()) {
      return op.emitOpError(
          "Not implemented: Changing implicit dims mid-broadcast");
    }
    const VectorLayout::ImplicitDim implicit_dim = layout_in.implicit_dim();
    const LayoutOffsets offsets_in = layout_in.offsets();
    const LayoutOffsets offsets_out = layout_out.offsets();
    if (layout_in.tiling() != layout_out.tiling()) {
      return op.emitOpError(
          "Not implemented: Changing tiling mid-broadcast");
    }
    auto tiling = layout_in.tiling();

    const int64_t expand_rank = dst_ty.getRank() - src_ty.getRank();
    SmallVector<int64_t> src_shape_padded(expand_rank, -1);
    const ArrayRef<int64_t> src_shape = src_ty.getShape();
    src_shape_padded.append(src_shape.begin(), src_shape.end());
    const SmallVector<bool> dim_eq = llvm::map_to_vector(
        llvm::zip(src_shape_padded, dst_ty.getShape()), [](auto tup) {
          auto [i, o] = tup;
          return i == o;
        });

    bool no_op = false;
    switch (implicit_dim) {
      case VectorLayout::ImplicitDim::kNone: {
        const ArrayRef<bool> tiled_dim_eq = ArrayRef<bool>(dim_eq).take_back(2);
        for (auto [in_off, out_off, eq] :
             llvm::zip(offsets_in, offsets_out, tiled_dim_eq)) {
          if (eq && in_off != out_off) {
            return op.emitOpError(
                "Not implemented: Changing offsets mid-broadcast");
          }
        }
        no_op = layout_in.hasNaturalTopology(ctx.target_shape) &&
                layout_out.hasNaturalTopology(ctx.target_shape) &&
                llvm::all_of(llvm::zip_equal(offsets_in, tiled_dim_eq),
                             [](auto tup) {
                               auto [o, eq] = tup;
                               return eq || !o.has_value();
                             });
      } break;
      case VectorLayout::ImplicitDim::kMinor:
      case VectorLayout::ImplicitDim::kSecondMinor:
        if (dim_eq.back()) {
          if (offsets_in != offsets_out) {
            return op.emitOpError(
                "Not implemented: Changing offsets mid-broadcast");
          }
          no_op = true;
        } else if (implicit_dim == VectorLayout::ImplicitDim::kSecondMinor &&
                   !offsets_in[1].has_value()) {
          no_op = true;
        } else if (implicit_dim == VectorLayout::ImplicitDim::kMinor &&
                   !offsets_in[0].has_value()) {
          no_op = true;
        }
        break;
    }

    FAILUREOR_ASSIGN_OR_RETURN(
        xla::Array<Value> src_tiles,
        disassemble(ctx, builder, layout_in, broadcast_op.getSource()));
    xla::Array<Value> dst_tiles(dst_tiles_shape);
    if (no_op) {
      SmallVector<int64_t> reshape_dims(expand_rank, 1);
      const absl::Span<const int64_t> src_tiles_dims = src_tiles.dimensions();
      reshape_dims.append(src_tiles_dims.begin(), src_tiles_dims.end());
      src_tiles.Reshape(reshape_dims);
      dst_tiles.Each([&](const absl::Span<const int64_t> dst_idx, Value *tile) {
        const SmallVector<int64_t> src_idx =
            llvm::map_to_vector(llvm::zip_equal(dst_idx, dim_eq), [](auto tup) {
              auto [i, eq] = tup;
              return eq ? i : 0;
            });
        *tile = src_tiles(src_idx);
      });
    } else if (implicit_dim == VectorLayout::ImplicitDim::kNone) {
      if (layout_in.bitwidth() != 32) {
        return op.emitOpError(
            "Not implemented: Only 32-bit broadcast supported");
      }
      if (tiling[1] != ctx.target_shape[1]) {
        return op.emitOpError("Not implemented: unsupported tiling");
      }
      int64_t num_tiles = layout_in.tilesPerVreg(ctx.target_shape);
      if (*(dim_eq.end() - 1)) {  // Sublane broadcast
        if (num_tiles != 1) {
          return op.emitOpError(
              "Not implemented: Only native tiling supported");
        }
        CHECK_EQ(*(src_tiles.dimensions().end() - 2), 1);
        CHECK(offsets_in[0].has_value());
        const int64_t offset = *offsets_in[0];
        const DenseI32ArrayAttr indices = builder.getDenseI32ArrayAttr(
            SmallVector<int32_t>(ctx.target_shape[0], offset));
        src_tiles.Each([&](const absl::Span<const int64_t> src_idx,
                           Value *const src_tile) {
          SmallVector<int64_t> dst_starts(dst_tiles_shape.size());
          SmallVector<int64_t> dst_limits(dst_tiles_shape.size());
          for (int64_t i = 0; i < dst_tiles.num_dimensions(); ++i) {
            if (i < expand_rank || !dim_eq[i]) {
              dst_starts[i] = 0;
              dst_limits[i] = dst_tiles_shape[i];
            } else {
              dst_starts[i] = src_idx[i - expand_rank];
              dst_limits[i] = dst_starts[i] + 1;
            }
          }
          updateSlice<Value>(dst_tiles,
                             builder.create<tpu::GatherOp>(
                                 src_tile->getType(), *src_tile, indices, 0),
                             dst_starts, dst_limits);
        });
      } else if (*(dim_eq.end() - 2)) {  // Lane broadcast
        CHECK_EQ(*(src_tiles.dimensions().end() - 1), 1);
        CHECK(offsets_in[1].has_value());
        const int64_t offset = *offsets_in[1];
        const auto idx_ty =
            VectorType::get(ctx.target_shape, builder.getI32Type());
        auto idx_const = builder.create<arith::ConstantOp>(
            broadcast_op.getLoc(), idx_ty,
            DenseElementsAttr::get(idx_ty,
                                   builder.getI32IntegerAttr(offset)));
        int64_t sublanes_per_tile = layout_in.sublanesPerTile(ctx.target_shape);
        DenseI32ArrayAttr sublane_pattern;
        if (num_tiles != 1) {
          SmallVector<int32_t> pattern;
          pattern.reserve(ctx.target_shape[0]);
          for (int32_t t = 0; t < num_tiles; ++t) {
            for (int32_t i = 0; i < sublanes_per_tile; ++i) {
              pattern.push_back(i);
            }
          }
          sublane_pattern = builder.getDenseI32ArrayAttr(pattern);
        }
        src_tiles.Each([&](const absl::Span<const int64_t> src_idx,
                           Value *const src_tile) {
          SmallVector<int64_t> dst_starts(dst_tiles_shape.size());
          SmallVector<int64_t> dst_limits(dst_tiles_shape.size());
          for (int64_t i = 0; i < dst_tiles.num_dimensions(); ++i) {
            if (i < expand_rank || !dim_eq[i]) {
              dst_starts[i] = 0;
              dst_limits[i] = dst_tiles_shape[i];
            } else {
              dst_starts[i] = src_idx[i - expand_rank];
              dst_limits[i] = dst_starts[i] + 1;
            }
          }
          Value res_vreg = builder.create<tpu::DynamicGatherOp>(
              broadcast_op.getLoc(), src_tile->getType(), *src_tile, idx_const,
              /*dimension=*/1);
          if (num_tiles != 1) {
            res_vreg = builder.create<tpu::GatherOp>(
                broadcast_op.getLoc(), res_vreg.getType(), res_vreg,
                sublane_pattern, 0);
          }
          updateSlice<Value>(dst_tiles, res_vreg, dst_starts, dst_limits);
        });
      } else {
        return op.emitOpError("Not implemented");
      }
    } else {
      return op.emitOpError("Not implemented");
    }
    broadcast_op.replaceAllUsesWith(
        assemble(ctx, builder, dst_ty, layout_out, dst_tiles).getOperation());
    broadcast_op.erase();
    return success();
  } else {
    FAILUREOR_ASSIGN_OR_RETURN(
        const VectorType native_vreg_ty,
        getNativeVregType(broadcast_op.getSourceType(), ctx.target_shape));
    auto tile = builder.create<vector::BroadcastOp>(native_vreg_ty,
                                                    broadcast_op.getSource());
    const xla::Array<Value> dst_tiles(dst_tiles_shape, tile);
    broadcast_op.replaceAllUsesWith(
        assemble(ctx, builder, dst_ty, layout_out, dst_tiles).getOperation());
    broadcast_op.erase();
    return success();
  }
}

LogicalResult vector_contract_rule(RewriteContext &ctx, Operation &op,
                                   const ArrayRef<Layout> layouts_in,
                                   const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 3);
  CHECK_EQ(layouts_out.size(), 1);
  for (const Layout &layout : layouts_in) {
    if (!layout.has_value()) {
      return op.emitOpError("Expected non-null input layouts");
    }
  }
  if (!layouts_out.front().has_value()) {
    return op.emitOpError("Expected non-null output layout");
  }
  MLIRContext *const mlir_ctx = ctx.getMLIRContext();
  Builder builder(mlir_ctx);
  auto vector_contract_op = cast<vector::ContractionOp>(op);
  // TODO(tlongeri): There is some unnecessary uniquing happening but not sure
  // if it can be avoided without sacrificing readability rather severely.
  auto getMapAttr = [&](const unsigned first, const unsigned second) {
    return AffineMapAttr::get(AffineMap::get(
        3, 0,
        {getAffineDimExpr(first, mlir_ctx), getAffineDimExpr(second, mlir_ctx)},
        mlir_ctx));
  };
  const ArrayAttr matmul_indexing_maps = builder.getArrayAttr(
      {getMapAttr(0, 2), getMapAttr(2, 1), getMapAttr(0, 1)});
  const ArrayAttr matmul_indexing_maps_transposed = builder.getArrayAttr(
      {getMapAttr(0, 2), getMapAttr(1, 2), getMapAttr(0, 1)});
  const auto indexing_maps = vector_contract_op->getAttr("indexing_maps");
  if (indexing_maps != matmul_indexing_maps &&
      indexing_maps != matmul_indexing_maps_transposed) {
    return vector_contract_op->emitOpError(
        "Non-matmul or unsupported indexing_maps");
  }
  const bool transpose_rhs = indexing_maps == matmul_indexing_maps_transposed;
  const ArrayAttr matmul_iterator_types =
      builder.getArrayAttr({builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::parallel),
                            builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::parallel),
                            builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::reduction)});
  if (vector_contract_op->getAttr("iterator_types") != matmul_iterator_types) {
    return vector_contract_op->emitOpError(
        "Not implemented: Non-matmul iterator_types");
  }
  const bool transpose_lhs =
      false;  // TODO(apaszke): Support that in the affine maps
  return matmul_rule_impl(ctx, *vector_contract_op, transpose_lhs,
                          transpose_rhs, *layouts_in[0], *layouts_in[1],
                          *layouts_in[2], *layouts_out[0]);
}

LogicalResult vector_extract_strided_slice_rule(
    RewriteContext &ctx, Operation &op, const ArrayRef<Layout> layouts_in,
    const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 1);
  CHECK_EQ(layouts_out.size(), 1);
  if (!layouts_in.front().has_value()) {
    return op.emitOpError("Expected non-null input layout");
  }
  if (!layouts_out.front().has_value()) {
    return op.emitOpError("Expected non-null output layout");
  }
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  if (!layout_in.hasNaturalTopology(ctx.target_shape)) {
    return op.emitOpError("Not implemented: Unsupported input layout");
  }
  if (layout_out != layout_in) {
    return op.emitOpError("Not implemented: Unsupported output layout");
  }
  OpBuilder builder(&op);
  vector::ExtractStridedSliceOp extract_strided_slice_op =
      cast<vector::ExtractStridedSliceOp>(op);
  const ArrayRef<int64_t> tiled_dims =
      extract_strided_slice_op.getVector().getType().getShape().take_back(2);
  if (tiled_dims[0] % layout_in.tiling()[0] != 0 ||
      tiled_dims[1] % layout_in.tiling()[1] != 0) {
    return op.emitOpError(
        "Not implemented: Extract strides slices only works with operands with "
        "sizes that are multiples of the native tiling");
  }

  auto I64ArrayToSmallVector = [&](const ArrayAttr array_attr) {
    return llvm::map_to_vector(array_attr, [](Attribute attr) {
      return cast<IntegerAttr>(attr).getValue().getSExtValue();
    });
  };

  // We currently only support zero-offset, tile-aligned slices. This implies
  // the output layout is merely a slice of the input layout, without needing to
  // modify physical any of the vregs' layouts.
  const SmallVector<int64_t> offsets =
      I64ArrayToSmallVector(extract_strided_slice_op.getOffsets());
  for (const int64_t offset : ArrayRef<int64_t>(offsets).take_back(2)) {
    if (offset != 0) {
      return extract_strided_slice_op.emitOpError(
          "Not implemented: Only tile-aligned slices supported");
    }
  }

  const SmallVector<int64_t> slice_sizes =
      I64ArrayToSmallVector(extract_strided_slice_op.getSizes());
  const SmallVector<int64_t> slice_tiled_shape =
      layout_in.tileArrayShape(slice_sizes, ctx.target_shape);
  FAILUREOR_ASSIGN_OR_RETURN(const xla::Array<Value> input_tiles,
                             disassemble(ctx, builder, layout_in,
                                         extract_strided_slice_op.getVector()));
  const xla::Array<Value> dst_tiles =
      input_tiles.Slice(offsets, slice_tiled_shape);
  const VectorType dst_ty = extract_strided_slice_op.getResult().getType();
  extract_strided_slice_op.replaceAllUsesWith(
      assemble(ctx, builder, dst_ty, layout_out, dst_tiles).getOperation());
  extract_strided_slice_op.erase();
  return success();
}

LogicalResult vector_multi_reduction_rule(RewriteContext &ctx, Operation &op,
                                          const ArrayRef<Layout> layouts_in,
                                          const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 2);
  CHECK_EQ(layouts_out.size(), 1);
  for (const Layout &layout : layouts_in) {
    if (!layout.has_value()) {
      return op.emitOpError("Expected non-null input layout");
    }
  }
  const VectorLayout &src_layout = *layouts_in[0];
  const VectorLayout &acc_layout = *layouts_in[1];
  const VectorLayout &layout_out = *layouts_out[0];
  OpBuilder builder(&op);
  auto multi_reduction_op = cast<vector::MultiDimReductionOp>(op);
  const ArrayAttr dims = multi_reduction_op.getReductionDims();
  if (dims.size() != 1) {
    return multi_reduction_op.emitOpError(
        "Not implemented: Only 1D reductions supported");
  }
  const int64_t dim =
      cast<IntegerAttr>(*dims.begin()).getValue().getSExtValue();
  const VectorType src_ty = multi_reduction_op.getSourceVectorType();
  const auto res_ty = dyn_cast<VectorType>(multi_reduction_op.getDestType());
  if (res_ty == nullptr) {
    return multi_reduction_op.emitOpError(
        "Not implemented: Can only reduce into vectors");
  }
  if (!layouts_out.front().has_value()) {
    // Shouldn't be empty since result is a vector
    return op.emitOpError("Expected non-null output layout");
  }

  // Make sure that the accumulator is a splat of the neutral value
  if (acc_layout.offsets() != LayoutOffsets{std::nullopt, std::nullopt}) {
    return multi_reduction_op.emitOpError(
        "Not implemented: Only replicated accumulator supported");
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      const xla::Array<Value> acc_vregs,
      disassemble(ctx, builder, acc_layout, multi_reduction_op.getAcc()));
  const Value acc_vreg = *acc_vregs.begin();
  auto acc_def =
      dyn_cast_if_present<arith::ConstantOp>(acc_vreg.getDefiningOp());
  if (acc_def == nullptr) {
    return multi_reduction_op.emitOpError(
        "Not implemented: Only constant accumulator supported");
  }
  if (!src_ty.getElementType().isF32()) {
    return multi_reduction_op.emitOpError(
               "Not implemented: Only FP32 reductions supported, but got ")
           << src_ty;
  }
  // Element types of source, dest, acc match (by multi_dim reduction's
  // definition), so we expect an f32 constant
  const auto acc_def_value = dyn_cast<DenseFPElementsAttr>(acc_def.getValue());
  if (acc_def_value == nullptr || !acc_def_value.isSplat()) {
    return multi_reduction_op.emitOpError("Expected a splat constant");
  }
  CHECK(acc_def_value.getElementType().isF32());
  const auto val = acc_def_value.getSplatValue<float>();
  FloatAttr neutral;
  switch (multi_reduction_op.getKind()) {
    case vector::CombiningKind::ADD:
      neutral = builder.getF32FloatAttr(0);
      break;
    case vector::CombiningKind::MAXF: {
      neutral = builder.getFloatAttr(
          builder.getF32Type(),
          APFloat::getInf(APFloat::IEEEsingle(), /*Negative=*/true));
    } break;
    default:
      return multi_reduction_op.emitOpError(
          "Not implemented: unsupported kind");
  }
  if (val != neutral.getValueAsDouble()) {
    return multi_reduction_op.emitOpError("Only neutral accumulator supported");
  }

  if (src_layout.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      src_layout.hasNaturalTopology(ctx.target_shape)) {
    auto [sublane_offset, lane_offset] = src_layout.offsets();
    if (dim < 0) {
      return multi_reduction_op.emitOpError(
          "Negative reduction dimension unsupported");
    }
    int64_t vdim;
    Direction reduce_over;
    std::array<bool, 2> allow_replicated;
    if (dim < src_ty.getRank() - 2) {
      return multi_reduction_op.emitOpError(
          "Not implemented: Reductions over non-layout dims");
    } else if (dim == src_ty.getRank() - 2) {
      reduce_over = Direction::kSublanes;
      allow_replicated = {false, true};
      vdim = 0;
      if (!sublane_offset.has_value()) {
        // TODO(apaszke): Note that it is just scaling!
        return multi_reduction_op.emitOpError(
            "Not implemented: Reductions over replicated axes");
      }
      if (layout_out != VectorLayout(32, {std::nullopt, lane_offset},
                                     ctx.target_shape,
                                     VectorLayout::ImplicitDim::kSecondMinor)) {
        return multi_reduction_op.emitOpError(
            "Not implemented: Unexpected destination layout");
      }
    } else if (dim == src_ty.getRank() - 1) {
      reduce_over = Direction::kLanes;
      allow_replicated = {true, false};
      vdim = 1;
      if (!lane_offset.has_value()) {
        // TODO(apaszke): Note that it is just scaling!
        return multi_reduction_op.emitOpError(
            "Not implemented: Reductions over replicated axes");
      }
      if (layout_out != VectorLayout(32, {sublane_offset, std::nullopt},
                                     ctx.target_shape,
                                     VectorLayout::ImplicitDim::kMinor)) {
        return multi_reduction_op.emitOpError(
                   "Not implemented: Unexpected destination layout: ")
               << layout_out;
      }
    } else {
      // Never should reach, this should be checked by MLIR verifier
      LOG(FATAL) << "Invalid reduction dimension: " << dim;
    }
    FAILUREOR_ASSIGN_OR_RETURN(
        xla::Array<Value> src_vregs,
        disassemble(ctx, builder, src_layout, multi_reduction_op.getSource()));
    xla::Array<Value> result_vregs(
        layout_out.tileArrayShape(res_ty.getShape(), ctx.target_shape));
    tpu::ReductionKind tpu_kind;
    switch (multi_reduction_op.getKind()) {
      case vector::CombiningKind::ADD:
        tpu_kind = tpu::ReductionKind::SUM;
        break;
      case vector::CombiningKind::MAXF:
        tpu_kind = tpu::ReductionKind::MAX;
        break;
      default:
        LOG(FATAL) << "Unreachable";
    }
    const ArrayRef<int64_t> src_shape = src_ty.getShape();
    absl::Status status = src_vregs.EachStatus(
        [&](const absl::Span<const int64_t> idx, Value *const src_vreg) {
          const std::unique_ptr<VRegDataBounds> data_bounds =
              src_layout.tileDataBounds(builder.getContext(), src_shape,
                                        toArrayRef(idx), ctx.target_shape,
                                        allow_replicated);
          // TODO(tlongeri): Maybe assemble/disassemble should take
          // TypedValue<VectorType> and we could save casts here and elsewhere
          FailureOr<Value> failure_or_vreg =
              maskOOB(ctx, builder, cast<TypedValue<VectorType>>(*src_vreg),
                      *data_bounds, neutral);
          if (failed(failure_or_vreg)) {
            return absl::UnknownError("");
          }
          Value vreg = failure_or_vreg.value();
          const int64_t lix = *(idx.end() - 1);
          const int64_t six = *(idx.end() - 2);
          SmallVector<int64_t> outer(toArrayRef(idx));
          int64_t reduced_ix, last_reduced_ix;
          if (reduce_over == Direction::kLanes) {
            outer.erase(outer.end() - 1);
            reduced_ix = lix;
            last_reduced_ix = *(src_vregs.dimensions().end() - 1) - 1;
          } else {
            CHECK(reduce_over == Direction::kSublanes);
            outer.erase(outer.end() - 2);
            reduced_ix = six;
            last_reduced_ix = *(src_vregs.dimensions().end() - 2) - 1;
          }
          Value new_acc;
          if (reduced_ix == 0) {
            new_acc = vreg;
          } else {
            switch (tpu_kind) {
              case tpu::ReductionKind::SUM:
                new_acc = builder.create<arith::AddFOp>(
                    vreg.getLoc(), result_vregs(outer), vreg);
                break;
              case tpu::ReductionKind::MAX:
                new_acc = builder.create<arith::MaximumFOp>(
                    vreg.getLoc(), result_vregs(outer), vreg);
                break;
            }
          }
          if (reduced_ix == last_reduced_ix) {
            new_acc = builder.create<tpu::AllReduceOp>(new_acc.getLoc(),
                                                       new_acc, vdim, tpu_kind);
          }
          result_vregs(outer) = new_acc;
          return absl::OkStatus();
        });
    if (!status.ok()) {
      return failure();
    }
    multi_reduction_op->replaceAllUsesWith(
        assemble(ctx, builder, res_ty, layout_out, result_vregs));
    multi_reduction_op->erase();
    return success();
  }
  return multi_reduction_op->emitOpError("Unsupported layout: ") << src_layout;
}

LogicalResult vector_shape_cast_rule(RewriteContext &ctx, Operation &op,
                                     const ArrayRef<Layout> layouts_in,
                                     const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 1);
  CHECK_EQ(layouts_out.size(), 1);
  if (!layouts_in.front().has_value()) {
    return op.emitOpError("Expected non-null input layout");
  }
  if (!layouts_out.front().has_value()) {
    return op.emitOpError("Expected non-null output layout");
  }
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto shape_cast_op = cast<vector::ShapeCastOp>(op);
  const VectorType src_ty = shape_cast_op.getSourceVectorType();
  const ArrayRef<int64_t> src_shape = src_ty.getShape();
  const VectorType dst_ty = shape_cast_op.getResultVectorType();
  const ArrayRef<int64_t> dst_shape = dst_ty.getShape();
  const int layout_rank = layout_in.layout_rank();
  bool no_op = false;
  // TODO(tlongeri): It looks like this could probably be simplified by using
  // VectorLayout::implicitShape()
  if (layout_in == layout_out && src_ty.getShape().take_back(layout_rank) ==
                                     dst_ty.getShape().take_back(layout_rank)) {
    no_op = true;
  } else if (layout_in.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
             layout_out.implicit_dim() ==
                 VectorLayout::ImplicitDim::kSecondMinor &&
             layout_in.hasNativeTiling(ctx.target_shape) &&
             layout_in.tiling() == layout_out.tiling() &&
             layout_in.offsets() == layout_out.offsets() &&
             *(src_shape.end() - 1) == *(dst_shape.end() - 1) &&
             *(src_shape.end() - 2) == 1) {
    no_op = true;
  } else if (layout_in.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
             layout_out.implicit_dim() == VectorLayout::ImplicitDim::kMinor &&
             layout_in.hasNaturalTopology(ctx.target_shape) &&
             layout_in.tiling() == layout_out.tiling() &&
             layout_in.offsets() == layout_out.offsets() &&
             src_shape ==
                 ArrayRef<int64_t>(layout_out.implicitShape(dst_shape))) {
    no_op = true;
  } else if (layout_in.implicit_dim() == VectorLayout::ImplicitDim::kMinor &&
             layout_out.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
             layout_out.hasNaturalTopology(ctx.target_shape) &&
             layout_in.tiling() == layout_out.tiling() &&
             layout_in.offsets() == layout_out.offsets() &&
             dst_shape ==
                 ArrayRef<int64_t>(layout_in.implicitShape(src_shape))) {
    no_op = true;
  } else if (  // Fold or unfold sublane dim, but keeping a whole number of
               // vregs.
      layout_in.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      layout_out.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      layout_in.offsets() == layout_out.offsets() &&
      layout_in.offsets() == LayoutOffsets{0, 0} &&
      layout_in.tiling() == layout_out.tiling() &&
      layout_in.tiling()[1] == ctx.target_shape[1] &&
      *(dst_shape.end() - 1) == *(src_shape.end() - 1) &&
      *(dst_shape.end() - 2) % layout_in.tiling()[0] == 0 &&
      *(src_shape.end() - 2) % layout_in.tiling()[0] == 0) {
    no_op = true;
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> src_vregs,
      disassemble(ctx, builder, layout_in, shape_cast_op.getSource()));
  auto getDstVregs = [&]() -> FailureOr<xla::Array<Value>> {
    if (no_op) {
      xla::Array<Value> dst_vregs_local = src_vregs;
      dst_vregs_local.Reshape(
          layout_out.tileArrayShape(dst_shape, ctx.target_shape));
      return dst_vregs_local;
    } else if (dst_shape.take_back(2) ==
                   ArrayRef<int64_t>{src_shape.back(), 1} &&
               layout_in.bitwidth() == 32 &&
               (layout_in.implicit_dim() == VectorLayout::ImplicitDim::kNone ||
                layout_in.implicit_dim() ==
                    VectorLayout::ImplicitDim::kSecondMinor) &&
               layout_out.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
               layout_in.hasNativeTiling(ctx.target_shape) &&
               layout_in.tiling() == layout_out.tiling() &&
               layout_in.offsets()[0].value_or(0) == 0 &&
               layout_in.offsets()[1] == 0 && layout_out.offsets()[0] == 0
               // layout_out.offsets[1] can be anything, as we produce a
               // replicated result
    ) {
      // First, insert the new singleton lane dimension.
      llvm::SmallVector<int64_t> s(src_shape);
      s.push_back(1);
      xla::Array<Value> dst_vregs_local(
          layout_out.tileArrayShape(s, ctx.target_shape));
      if (layout_in.implicit_dim() == VectorLayout::ImplicitDim::kSecondMinor) {
        // Make the sublane dimension explicit.
        SmallVector<int64_t> new_src_vregs_shape(
            toArrayRef(src_vregs.dimensions()));
        new_src_vregs_shape.insert(new_src_vregs_shape.end() - 1, 1);
        src_vregs.Reshape(new_src_vregs_shape);
        SmallVector<int64_t> new_dst_vregs_shape(
            toArrayRef(dst_vregs_local.dimensions()));
        new_dst_vregs_shape.insert(new_dst_vregs_shape.end() - 2, 1);
        dst_vregs_local.Reshape(new_dst_vregs_shape);
      }
      CHECK_EQ(dst_vregs_local.dimensions().back(),
               1);  // We're inserting a singleton dimension
      dst_vregs_local.Each(
          [&](const absl::Span<const int64_t> dst_idx, Value *const dst_vreg) {
            const int64_t col_idx = *(dst_idx.end() - 2);
            const int64_t row_idx = *(dst_idx.end() - 3);
            auto [sublanes_in_lane, rem] =
                std::div(ctx.target_shape[1], ctx.target_shape[0]);
            CHECK_EQ(rem, 0);
            if (!layout_in.offsets()[0].has_value() && row_idx != 0) {
              return;  // All vregs along that dimension are the same.
            }
            SmallVector<int64_t> src_idx(toArrayRef(dst_idx));
            src_idx.pop_back();
            *(src_idx.end() - 2) /= ctx.target_shape[0];
            *(src_idx.end() - 1) /= sublanes_in_lane;
            Value col_vreg = src_vregs(src_idx);
            // BroadcastInSublanesOp requires the sublanes to be replicated.
            if (layout_in.offsets()[0].has_value()) {
              const int32_t sublane = row_idx % ctx.target_shape[0];
              col_vreg = builder.create<tpu::GatherOp>(
                  col_vreg.getType(), col_vreg,
                  /*indices=*/
                  SmallVector<int32_t>(ctx.target_shape[0], sublane),
                  /*dimension=*/0);
            }
            *dst_vreg = builder.create<BroadcastInSublanesOp>(
                col_vreg.getType(), col_vreg,
                /*lane=*/(col_idx % sublanes_in_lane) * ctx.target_shape[0]);
          });
      if (!layout_in.offsets()[0].has_value()) {
        // Broadcast the sublane vregs.
        // TODO(tlongeri): This could be done more efficiently
        dst_vregs_local.Each([&](const absl::Span<const int64_t> dst_idx,
                                 Value *const dst_vreg) {
          SmallVector<int64_t> first_row_idx(toArrayRef(dst_idx));
          *(first_row_idx.end() - 3) = 0;
          *dst_vreg = dst_vregs_local(first_row_idx);
        });
      }
      // Now, permute the major axes of the vreg array.
      dst_vregs_local.Reshape(
          layout_out.tileArrayShape(dst_shape, ctx.target_shape));
      return dst_vregs_local;
    } else {
      return shape_cast_op.emitOpError(
                 "Not implemented: Unsupported vector.shape_cast: ")
             << *shape_cast_op;
    }
  };
  FAILUREOR_ASSIGN_OR_RETURN(const xla::Array<Value> dst_vregs, getDstVregs());
  shape_cast_op->replaceAllUsesWith(
      assemble(ctx, builder, dst_ty, layout_out, dst_vregs));
  shape_cast_op->erase();
  return success();
}
LogicalResult vector_store_rule(RewriteContext &ctx, Operation &op,
                                const ArrayRef<Layout> layouts_in,
                                const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_out.size(), 0);
  MLIRContext *const mlir_ctx = op.getContext();
  if (!layouts_in.front().has_value() ||
      llvm::any_of(layouts_in.drop_front(),
                   [&](const Layout &l) { return l.has_value(); })) {
    return op.emitOpError(
        "Expected null input layouts for vector.store indices");
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  vector::StoreOp store_op = cast<vector::StoreOp>(op);
  const VectorType ty = store_op.getValueToStore().getType();
  const VectorLayout &to_store_layout = *layouts_in.front();
  if (to_store_layout.implicit_dim() == VectorLayout::ImplicitDim::kMinor) {
    return op.emitOpError("Not implemented");
  }
  const bool is_1d =
      to_store_layout.implicit_dim() != VectorLayout::ImplicitDim::kNone;
  using Tiling = std::array<int64_t, 2>;
  FAILUREOR_ASSIGN_OR_RETURN(
      const Tiling memref_tiling,
      getMemRefTiling(store_op.getBase(), ctx.target_shape));
  if (to_store_layout.tiling() != memref_tiling) {
    // Now we can handle the case when tiling is (1, TARGET_SHAPE.lanes).
    // TODO(b/295393167): need to support strided store for bitwidth < 32.
    if (to_store_layout.bitwidth() != 32 ||
        to_store_layout.tiling() != Tiling{1, ctx.target_shape[1]}) {
      return op.emitOpError("Not implemented");
    }
  }
  FAILUREOR_ASSIGN_OR_RETURN(
      const SmallVector<int64_t> base_indices,
      getIntConstsFromOperandRange(store_op.getIndices()));
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> tiles,
      disassemble(ctx, builder, to_store_layout, store_op.getValueToStore()));
  const int64_t ndims = base_indices.size();
  const int64_t nbatchdims = is_1d ? ndims - 1 : ndims - 2;
  const int64_t base_s = is_1d ? 0 : base_indices[ndims - 2];
  const int64_t base_l = base_indices[ndims - 1];
  if (is_1d) {
    tiles.Reshape(
        to_store_layout.implicitShape(toArrayRef(tiles.dimensions())));
  }
  const LayoutOffset sublane_offset = to_store_layout.offsets()[0];
  const LayoutOffset lane_offset = to_store_layout.offsets()[1];
  if (!sublane_offset.has_value() || !lane_offset.has_value()) {
    return store_op.emitOpError(
        "Not implemented: Replicated layout disallowed in vector store");
  }
  const SmallVector<int64_t> stored_shape =
      to_store_layout.implicitShape(ty.getShape());
  int64_t sublane_stride = 1;
  // The stride of store should be the number of sublanes in memref tile when
  // store a single sublane.
  if (to_store_layout.bitwidth() == 32 &&
      to_store_layout.tiling() == Tiling{1, ctx.target_shape[1]}) {
    sublane_stride = memref_tiling[0];
  }
  const std::array<int64_t, 2> vreg_slice =
      to_store_layout.vregSlice(ctx.target_shape);
  const absl::Status status =
      tiles.EachStatus([&](const absl::Span<const int64_t> idx,
                           const Value tile) -> absl::Status {
        const std::unique_ptr<VRegDataBounds> bounds =
            to_store_layout.tileDataBounds(mlir_ctx, stored_shape,
                                           toArrayRef(idx), ctx.target_shape);
        const int64_t sidx = *(idx.end() - 2);
        const int64_t lidx = *(idx.end() - 1);
        SmallVector<Value> indices(ndims);
        auto boundIdxConst = std::bind(IdxConst, std::placeholders::_1, builder,
                                       store_op->getLoc());
        for (int64_t i = 0; i < nbatchdims; ++i) {
          indices[i] = boundIdxConst(idx[i] + base_indices[i]);
        }
        if (!is_1d) {
          *(indices.end() - 2) =
              boundIdxConst(base_s + sidx * vreg_slice[0] - *sublane_offset);
        }
        *(indices.end() - 1) =
            boundIdxConst(base_l + lidx * vreg_slice[1] - *lane_offset);
        const DenseBoolArrayAttr sublane_mask =
            bounds->getSublaneMask(store_op->getContext(), ctx.target_shape);
        const bool masks_subelements =
            bounds->maskVariesAlong(Direction::kSubelements, ctx.target_shape);
        if (bounds->maskVariesAlong(Direction::kLanes, ctx.target_shape) ||
            masks_subelements) {
          auto failure_or_mask =
              bounds->getVectorMask(builder, store_op.getLoc(),
                                    ctx.hardware_generation, ctx.target_shape);
          if (failed(failure_or_mask)) {
            return absl::UnimplementedError("Failed to get vector mask");
          }
          TypedValue<VectorType> mask = failure_or_mask.value();
          // Vmem stores don't support masking below 32-bit granularity, so we
          // need to load and blend explicitly if needed.
          if (masks_subelements) {
            auto data = builder.create<tpu::LoadOp>(
                tile.getType(), store_op.getBase(), indices, sublane_mask,
                /*sublane_stride=*/nullptr);
            const bool mask_is_a_bitmask =
                cast<IntegerType>(mask.getType().getElementType()).getWidth() ==
                32;
            Value updated;
            if (mask_is_a_bitmask) {
              auto ones = builder.create<arith::ConstantOp>(
                  mask.getType(),
                  DenseElementsAttr::get(
                      mask.getType(),
                      builder.getIntegerAttr(builder.getI32Type(),
                                             APInt(32, 0xFFFFFFFF))));
              auto masked_tile = builder.create<arith::AndIOp>(
                  store_op.getLoc(), mask,
                  builder.create<tpu::BitcastOp>(mask.getType(), tile));
              auto mask_neg = builder.create<arith::XOrIOp>(ones, mask);
              auto masked_data = builder.create<arith::AndIOp>(
                  mask_neg,
                  builder.create<tpu::BitcastOp>(mask.getType(), data));
              updated = builder.create<tpu::BitcastOp>(
                  tile.getType(),
                  builder.create<arith::OrIOp>(masked_data, masked_tile));
            } else {
              updated = builder.create<arith::SelectOp>(mask, tile, data);
            }
            builder.create<tpu::StoreOp>(
                updated, store_op.getBase(), indices, sublane_mask,
                /*mask=*/nullptr,
                /*sublane_stride=*/builder.getI32IntegerAttr(sublane_stride));
          } else {
            builder.create<tpu::StoreOp>(
                tile, store_op.getBase(), indices, sublane_mask,
                /*mask=*/mask,
                /*sublane_stride=*/builder.getI32IntegerAttr(sublane_stride));
          }
        } else {
          builder.create<tpu::StoreOp>(
              tile, store_op.getBase(), indices, sublane_mask,
              /*mask=*/nullptr,
              /*sublane_stride=*/builder.getI32IntegerAttr(sublane_stride));
        }
        return absl::OkStatus();
      });
  if (!status.ok()) {
    return failure();
  }
  store_op->erase();
  return success();
}

LogicalResult vector_transpose_rule(RewriteContext &ctx, Operation &op,
                                    const ArrayRef<Layout> layouts_in,
                                    const ArrayRef<Layout> layouts_out) {
  CHECK_EQ(layouts_in.size(), 1);
  CHECK_EQ(layouts_out.size(), 1);
  if (!layouts_in.front().has_value()) {
    return op.emitOpError("Expected non-null input layout");
  }
  if (!layouts_out.front().has_value()) {
    return op.emitOpError("Expected non-null output layout");
  }
  const VectorLayout &layout_in = *layouts_in.front();
  const VectorLayout &layout_out = *layouts_out.front();
  if (layout_in.implicit_dim() != VectorLayout::ImplicitDim::kNone ||
      layout_in != layout_out) {
    return op.emitOpError("Not implemented: Unsupported 2D layouts");
  }
  ImplicitLocOpBuilder builder(op.getLoc(), &op);
  auto transpose_op = cast<vector::TransposeOp>(op);
  VectorType src_ty = transpose_op.getSourceVectorType();
  VectorType dst_ty = transpose_op.getResultVectorType();
  const int64_t rank = src_ty.getRank();
  FAILUREOR_ASSIGN_OR_RETURN(
      xla::Array<Value> src_vregs,
      disassemble(ctx, builder, layout_in, transpose_op.getVector()));
  const SmallVector<int64_t> permutation =
      llvm::map_to_vector(transpose_op.getTransp(), [&](const Attribute attr) {
        return cast<IntegerAttr>(attr).getValue().getSExtValue();
      });
  const auto tile_perm = ArrayRef<int64_t>(permutation).take_back(2);
  if (tile_perm != ArrayRef<int64_t>{rank - 2, rank - 1} &&
      tile_perm != ArrayRef<int64_t>{rank - 1, rank - 2}) {
    return transpose_op->emitOpError(
        "Not implemented: Unsupported permutation");
  }
  {
    SmallVector<int64_t> p(permutation);
    p[rank - 2] = rank - 2;
    p[rank - 1] = rank - 1;
    src_vregs.TransposeDimensions(p);
  }
  if (tile_perm == ArrayRef<int64_t>{rank - 2, rank - 1}) {
    transpose_op->replaceAllUsesWith(
        assemble(ctx, builder, dst_ty, layout_out, src_vregs));
    transpose_op.erase();
    return success();
  }
  if (layout_in.offsets() != LayoutOffsets{0, 0} ||
      !layout_in.hasNativeTiling(ctx.target_shape)) {
    return transpose_op->emitOpError(
        "Not implemented: Non-native or offset layout unsupported");
  }
  const int64_t transpose_unit_size = ctx.target_shape[1];
  for (const int64_t s : src_ty.getShape().take_back(2)) {
    if (s % transpose_unit_size != 0) {
      return transpose_op->emitOpError("Not implemented: Padded transpose");
    }
  }
  if (ctx.hardware_generation < 4 && layout_in.bitwidth() != 32) {
    return transpose_op->emitOpError(
        "Not implemented: TPUs before v4 only support 32-bit transposes");
  }
  xla::Array<Value> dst_vregs(
      layout_out.tileArrayShape(dst_ty.getShape(), ctx.target_shape));
  const int packing = layout_in.packing();
  // Note that we checked for native tiling above.
  const int64_t vregs_per_tile = transpose_unit_size / layout_in.tiling()[0];
  const ArrayAttr minor_perm = builder.getArrayAttr(
      {builder.getI64IntegerAttr(1), builder.getI64IntegerAttr(0)});
  const auto tile_ty = VectorType::get(
      {transpose_unit_size, transpose_unit_size}, src_ty.getElementType());
  const auto batch_tile_ty_in =
      VectorType::get({transpose_unit_size, transpose_unit_size * packing},
                      src_ty.getElementType());
  const auto batch_tile_ty_out =
      VectorType::get({transpose_unit_size * packing, transpose_unit_size},
                      src_ty.getElementType());
  // For packed types, we can increase the XLU throughput by batching together
  // multiple tiles. At the moment we always batch along columns, with the
  // reasoning being that if all the tiles are fed into the MXU, then it's
  // better if we end up with results that contribute to the same contraction.
  const bool can_batch = layout_in.bitwidth() == 16;
  auto doTranspose = [&](const ArrayRef<int64_t> batch_idx,
                         const int64_t src_row, const int64_t src_col,
                         const int64_t src_col_end, const VectorType tile_ty_in,
                         const VectorType tile_ty_out) {
    SmallVector<int64_t> src_slice_starts;
    src_slice_starts.reserve(rank);
    src_slice_starts.append(batch_idx.begin(), batch_idx.end());
    src_slice_starts.append({src_row * vregs_per_tile, src_col});
    SmallVector<int64_t> src_slice_ends;
    src_slice_ends.reserve(rank);
    auto incremented_batch_idx =
        map_range(batch_idx, [](int64_t i) { return i + 1; });
    src_slice_ends.append(incremented_batch_idx.begin(),
                          incremented_batch_idx.end());
    src_slice_ends.append({(src_row + 1) * vregs_per_tile, src_col_end});
    xla::Array<Value> src_tile_vregs =
        src_vregs.Slice(src_slice_starts, src_slice_ends);
    // Drop leading singleton (batch) dimensions to have a shape that conforms
    // with the vreg array shape specified by layout_in, as expected by assemble
    src_tile_vregs.Reshape(
        ArrayRef<int64_t>{vregs_per_tile, src_col_end - src_col});
    const Value src_tile =
        assemble(ctx, builder, tile_ty_in, layout_in, src_tile_vregs);
    auto new_transpose_op =
        builder.create<vector::TransposeOp>(tile_ty_out, src_tile, minor_perm);
    new_transpose_op->setAttr("out_layout",
                              builder.getAttr<VectorLayoutAttr>(layout_out));
    auto unroll_vectors_op = builder.create<tpu::UnrollVectorsOp>(
        llvm::map_to_vector(src_tile_vregs,
                            [](Value v) { return v.getType(); }),
        new_transpose_op);
    SmallVector<int64_t> dst_slice_starts;
    dst_slice_starts.reserve(rank);
    dst_slice_starts.append(batch_idx.begin(), batch_idx.end());
    dst_slice_starts.append({src_col * vregs_per_tile, src_row});
    SmallVector<int64_t> dst_slice_ends;
    dst_slice_ends.reserve(rank);
    dst_slice_ends.append(incremented_batch_idx.begin(),
                          incremented_batch_idx.end());
    dst_slice_ends.append({src_col_end * vregs_per_tile, src_row + 1});
    updateSliceFromRange(dst_vregs, unroll_vectors_op.getResults(),
                         dst_slice_starts, dst_slice_ends);
  };
  const int num_batch_dims = rank - 2;
  const ArrayRef<int64_t> batch_sizes =
      dst_ty.getShape().take_front(num_batch_dims);
  SmallVector<int64_t> batch_idx(num_batch_dims);
  do {
    const int64_t tile_rows =
        *(src_ty.getShape().end() - 2) / transpose_unit_size;
    for (int64_t src_row = 0; src_row < tile_rows; ++src_row) {
      const int64_t num_col_tiles =
          *(src_ty.getShape().end() - 1) / transpose_unit_size;
      if (can_batch) {
        const int64_t num_batch_tiles = num_col_tiles / 2;
        for (int64_t src_col = 0; src_col < num_batch_tiles; ++src_col) {
          doTranspose(batch_idx, src_row, src_col * 2, (src_col + 1) * 2,
                      batch_tile_ty_in, batch_tile_ty_out);
        }
        if (num_col_tiles % 2 == 1) {
          doTranspose(batch_idx, src_row, num_col_tiles - 1, num_col_tiles,
                      tile_ty, tile_ty);
        }
      } else {
        for (int64_t src_col = 0; src_col < num_col_tiles; ++src_col) {
          doTranspose(batch_idx, src_row, src_col, src_col + 1, tile_ty,
                      tile_ty);
        }
      }
    }
  } while (incrementIndex(batch_idx, batch_sizes));
  for (const Value v : dst_vregs) {
    CHECK(v != nullptr);
  }
  transpose_op->replaceAllUsesWith(
      assemble(ctx, builder, dst_ty, layout_out, dst_vregs));
  transpose_op->erase();
  return success();
}
const llvm::StringMap<rule_type> &rules() {
  static auto rules = new llvm::StringMap<rule_type>{
      {arith::ConstantOp::getOperationName(), arith_constant_rule},
      {arith::ExtFOp::getOperationName(), arith_extf_rule},
      {arith::ExtSIOp::getOperationName(), arith_extsi_rule},
      {arith::TruncFOp::getOperationName(), arith_truncf_rule},
      {arith::TruncIOp::getOperationName(), arith_trunci_rule},
      {func::ReturnOp::getOperationName(), func_return_rule},
      {scf::ForOp::getOperationName(), scf_for_rule},
      {scf::IfOp::getOperationName(), scf_if_rule},
      {scf::YieldOp::getOperationName(), scf_yield_rule},
      {tpu::IotaOp::getOperationName(), tpu_iota_rule},
      {tpu::GatherOp::getOperationName(), tpu_gather_rule},
      {tpu::LoadOp::getOperationName(), tpu_load_rule},
      {tpu::MatmulOp::getOperationName(), tpu_matmul_rule},
      {tpu::RepeatOp::getOperationName(), tpu_repeat_rule},
      {tpu::StoreOp::getOperationName(), tpu_store_rule},
      {tpu::TraceOp::getOperationName(), tpu_trace_rule},
      {vector::BroadcastOp::getOperationName(), vector_broadcast_rule},
      {vector::ContractionOp::getOperationName(), vector_contract_rule},
      {vector::LoadOp::getOperationName(), vector_load_rule},
      {vector::MultiDimReductionOp::getOperationName(),
       vector_multi_reduction_rule},
      {vector::ExtractStridedSliceOp::getOperationName(),
       vector_extract_strided_slice_rule},
      {vector::ShapeCastOp::getOperationName(), vector_shape_cast_rule},
      {vector::StoreOp::getOperationName(), vector_store_rule},
      {vector::TransposeOp::getOperationName(), vector_transpose_rule}};
  return *rules;
}
}  // namespace

RollVectorsOp assemble(RewriteContext &ctx, OpBuilder &builder, VectorType vty,
                       const VectorLayout &layout,
                       const xla::Array<Value> &vals) {
  CHECK(vals.dimensions() ==
        layout.tileArrayShape(vty.getShape(), ctx.target_shape));
  CHECK_GT(vals.num_elements(), 0);
  Location loc = vals.begin()->getLoc();
  auto op =
      builder.create<RollVectorsOp>(loc, vty, XlaArrayToFlatArrayRef(vals));
  op->setAttr("out_layout", builder.getAttr<ArrayAttr>(ArrayRef<Attribute>{
                                builder.getAttr<VectorLayoutAttr>(layout)}));
  return op;
}

// Disassemble an MLIR vector into an ndarray of native vectors.
//
// Args:
//   layout: The layout of val. Used to determine the unrolling into
//     native-shaped vectors.
//   val: Value to disassemble. Must be of type VectorType.
//
// Returns:
//   An ndarray of MLIR values representing the tiling of val given by layout.
FailureOr<xla::Array<Value>> disassemble(RewriteContext &ctx,
                                         OpBuilder &builder,
                                         const VectorLayout &layout,
                                         const Value val) {
  const auto vty = cast<VectorType>(val.getType());
  const auto op_result = dyn_cast<OpResult>(val);
  if (op_result == nullptr) {
    return failure();
  }
  Operation *const op = op_result.getOwner();
  const unsigned res_idx = op_result.getResultNumber();
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> def_layouts,
                             getOutLayout(*op));
  const Layout def_layout = def_layouts[res_idx];
  CHECK(def_layout.has_value());
  CHECK(def_layout->generalizes(layout, vty.getShape(), ctx.target_shape));
  SmallVector<int64_t> layout_shape =
      layout.tileArrayShape(vty.getShape(), ctx.target_shape);
  if (auto roll_vectors_op = dyn_cast<RollVectorsOp>(op)) {
    return XlaArrayFromShapeAndValues<Value>(layout_shape,
                                             roll_vectors_op->getOperands());
  }
  if (auto contraction_op = dyn_cast<vector::ContractionOp>(op)) {
    const int64_t num_vectors = ShapedType::getNumElements(layout_shape);
    FAILUREOR_ASSIGN_OR_RETURN(
        VectorType vreg_ty,
        getNativeVregType(vty.getElementType(), ctx.target_shape));
    // TODO(tlongeri): nicer way of doing ValueTypeRange?
    Operation *const u = builder.create<UnrollVectorsOp>(
        val.getLoc(), SmallVector<Type>(num_vectors, vreg_ty), val);
    return XlaArrayFromShapeAndValues<Value>(layout_shape, u->getResults());
  }
  return op->emitOpError("unimplemented: ") << val;
}

// Assembles a destination tile using partial data from rotated vregs using a
// divide-and-conquer strategy.
//
// Arguments:
//   rotated_row_vregs: A row of rotated vregs, from which destination tile(s)
//     is/are to be selected to assemble a new vreg.
//   src_layout: The source layout.
//   start_src_col: The first rotated vreg in the row of rotated vregs to
//     process.
//   end_src_col: The last rotated vreg in the row of rotated vreg to process.
//   first_dst_tile_sublane_offset: Sublane offset where the first dst tile to
//   be
//     selected starts.
//   dst_layout: Destination layout, based on which retiling is being performed.
//   hw_generation: The generation of a target hardware.
//
// Returns:
//   A new vreg assembled from dst tiles stored in given rotated vregs.
Value selectTilesFromRotatedRowVregs(
    RewriteContext &ctx, OpBuilder &builder,
    const ArrayRef<Value> &rotated_row_vregs, const int64_t start_src_col,
    const int64_t end_src_col, const int64_t first_dst_tile_sublane_offset,
    const VectorLayout &dst_layout) {
  CHECK_LE(start_src_col, end_src_col);
  CHECK_LE(start_src_col, end_src_col);
  if (start_src_col == end_src_col) {
    return rotated_row_vregs[start_src_col];
  }
  const int64_t mid_src_col = start_src_col + (end_src_col - start_src_col) / 2;

  Value left_partial_vreg = selectTilesFromRotatedRowVregs(
      ctx, builder, rotated_row_vregs, start_src_col, mid_src_col,
      first_dst_tile_sublane_offset, dst_layout);

  const int64_t left_tiles_count = mid_src_col - start_src_col + 1;
  const int64_t right_first_dst_tile_sublane_offset =
      (first_dst_tile_sublane_offset +
       left_tiles_count * dst_layout.sublanesPerTile(ctx.target_shape)) %
      ctx.target_shape[0];

  Value right_partial_vreg = selectTilesFromRotatedRowVregs(
      ctx, builder, rotated_row_vregs, mid_src_col + 1, end_src_col,
      right_first_dst_tile_sublane_offset, dst_layout);

  const IntegerType i1 = builder.getI1Type();
  const auto mask_vreg_ty =
      dst_layout.packing() == 2
          ? VectorType::get(
                ArrayRef<int64_t>{ctx.target_shape[0], ctx.target_shape[1], 2},
                i1)
          : VectorType::get(ctx.target_shape, i1);

  auto boundIdxConst = std::bind(IdxConst, std::placeholders::_1, builder,
                                 left_partial_vreg.getLoc());
  if (first_dst_tile_sublane_offset < right_first_dst_tile_sublane_offset) {
    // The useful data sublanes in left vregs do not wrap around in vreg.
    // For e.g. consider (2,128) destination tiling and we are trying to merge
    // two vregs as follows:
    //
    //   vreg 0:        vreg 1:
    //   x x x x x     dst_tile_2
    //   x x x x x     dst_tile_3
    //   dst_tile_4    x x x x x
    //   dst_tile_5    x x x x x
    //   dst_tile_6    x x x x x
    //   dst_tile_7    x x x x x
    //   x x x x x     dst_tile_0
    //   x x x x x     dst_tile_1
    //
    // In the above case, the data we want to select from vreg 1 wraps around,
    // whereas vreg 0 useful data is contiguous. It is easier to create '1' mask
    // for vreg 0.
    auto sublanes_mask = builder.create<tpu::CreateMaskOp>(
        left_partial_vreg.getLoc(), mask_vreg_ty,
        ArrayRef<Value>{boundIdxConst(first_dst_tile_sublane_offset),
                        boundIdxConst(0)},
        ArrayRef<Value>{boundIdxConst(right_first_dst_tile_sublane_offset),
                        boundIdxConst(ctx.target_shape[1])});
    return builder.create<arith::SelectOp>(left_partial_vreg.getLoc(),
                                           sublanes_mask, left_partial_vreg,
                                           right_partial_vreg);
  }

  auto sublanes_mask = builder.create<tpu::CreateMaskOp>(
      left_partial_vreg.getLoc(), mask_vreg_ty,
      ArrayRef<Value>{boundIdxConst(right_first_dst_tile_sublane_offset),
                      boundIdxConst(0)},
      ArrayRef<Value>{boundIdxConst(first_dst_tile_sublane_offset),
                      boundIdxConst(ctx.target_shape[1])});
  return builder.create<arith::SelectOp>(left_partial_vreg.getLoc(),
                                         sublanes_mask, right_partial_vreg,
                                         left_partial_vreg);
}

// Retiles across vregs to match the destination layout when the sublane tiling
// dimension is reduced.
//
// Arguments:
//   value_shape: The shape of the value which needs to be retiled in vregs.
//   src: The source layout.
//   src_vreg_array: An array of vregs storing source tiles.
//   dst_layout: The destination layout, with reduced sublane dimension, based
//   on
//     which the retiling will be performed.
//   hw_generation: The generation of a target hardware.
//
// Returns:
//   A new array of vregs that store tiles based on the destination layout.
xla::Array<Value> retileToReducedSublanes(
    RewriteContext &ctx, OpBuilder &builder,
    const ArrayRef<int64_t> value_shape, const VectorLayout &src_layout,
    const xla::Array<Value> &src_vreg_array, const VectorLayout &dst_layout) {
  const int64_t dst_tiling_sublane = dst_layout.tiling()[0];
  CHECK_LT(0, dst_tiling_sublane);
  CHECK_LT(dst_tiling_sublane, src_layout.tiling()[0]);
  CHECK(llvm::isPowerOf2_64(dst_tiling_sublane));

  xla::Array<Value> dst_vreg_array(
      dst_layout.tileArrayShape(value_shape, ctx.target_shape));

  // We need to rotate each src tile in each src vreg once so that that they can
  // be merged to form new vregs. If a src vreg contains more than one src tile,
  // it will be rotated once per src tile. Consider (8,512) tensor stored with
  // layout (8,128) in a vreg array of shape (1, 4). Each src vreg
  // contains one src tile in this case. Given, the destination layout is
  // (2,128), each src tile is divided into 4 destination tiles as shown below:
  //
  //   src_vreg_0_0:     src_vreg_0_1:    src_vreg_0_2:   src_vreg_0_3:
  // dst_tile_0_0_0    dst_tile_0_0_1   dst_tile_0_0_2  dst_tile_0_0_3
  // dst_tile_1_0_0    dst_tile_1_0_1   dst_tile_1_0_2  dst_tile_1_0_3
  // dst_tile_2_0_0    dst_tile_2_0_1   dst_tile_2_0_2  dst_tile_2_0_3
  // dst_tile_3_0_0    dst_tile_3_0_1   dst_tile_3_0_2  dst_tile_3_0_3
  //
  // In this example, each src tile in the src vreg is rotated by
  // col *  sublanes_per_tile to produce the following rotated src vregs:
  //
  // rot_src_vreg_0_0: rot_src_vreg_0_1: rot_src_vreg_0_2: rot_src_vreg_0_3:
  //     dst_tile_0_0_0    dst_tile_3_0_1    dst_tile_2_0_2    dst_tile_1_0_3
  //     dst_tile_1_0_0    dst_tile_0_0_1    dst_tile_3_0_2    dst_tile_2_0_3
  //     dst_tile_2_0_0    dst_tile_1_0_1    dst_tile_0_0_2    dst_tile_3_0_3
  //     dst_tile_3_0_0    dst_tile_2_0_1    dst_tile_1_0_2    dst_tile_0_0_3

  // If there were 2 src tiles in the src vreg, we would have rotated each src
  // vreg twice, producing 2 rotated src vreg per src vreg. The rotation amount
  // is calculated from the src and the dest tiling.

  const int64_t src_tiles_per_vreg = src_layout.tilesPerVreg(ctx.target_shape);
  const int64_t dst_tiles_per_vreg = dst_layout.tilesPerVreg(ctx.target_shape);
  const int64_t src_sublanes_per_tile =
      src_layout.sublanesPerTile(ctx.target_shape);
  const int64_t dst_sublanes_per_tile =
      dst_layout.sublanesPerTile(ctx.target_shape);
  // Each vreg may store more than one src tile. We may have to rotate a vreg,
  // once for every src tile in the vreg.
  SmallVector<int64_t> rotated_src_vreg_array_shape(
      toArrayRef(src_vreg_array.dimensions()));
  rotated_src_vreg_array_shape.back() *= src_tiles_per_vreg;
  xla::Array<Value> rotated_src_vreg_array(rotated_src_vreg_array_shape);

  rotated_src_vreg_array.Each([&](const absl::Span<const int64_t> rotated_idx,
                                  Value *const rotated_src_vreg) {
    const int64_t idx = rotated_idx.back();
    const int64_t tile_idx = idx % dst_tiles_per_vreg;
    const int64_t dst_sublane = tile_idx * dst_sublanes_per_tile;
    auto [src_col, src_tile_offset] = std::div(idx, src_tiles_per_vreg);
    SmallVector<int64_t> src_vreg_idx(toArrayRef(rotated_idx));
    src_vreg_idx.back() = src_col;
    Value src_vreg = src_vreg_array(src_vreg_idx);
    const int64_t src_sublane = src_tile_offset * src_sublanes_per_tile;
    int64_t rotate_amt = dst_sublane - src_sublane;
    if (rotate_amt == 0) {
      *rotated_src_vreg = src_vreg;
      return;
    }
    if (rotate_amt < 0) {
      rotate_amt += ctx.target_shape[0];
    }
    *rotated_src_vreg = builder.create<tpu::RotateOp>(
        src_vreg.getLoc(), src_vreg, rotate_amt, /*dimension=*/0);
  });
  // Assemble output vregs using tiles from rotated vregs using select.
  // Given, above example, destination vregs are then assembled as follows:
  //  dst_vreg_0_0:
  // dst_tile_0_0_0
  // dst_tile_0_0_1
  // dst_tile_0_0_2
  // dst_tile_0_0_3

  //  dst_vreg_1_0: (Notice dst tiles are not in correct offset!)
  // dst_tile_1_0_3
  // dst_tile_1_0_0
  // dst_tile_1_0_1
  // dst_tile_1_0_2

  //  dst_vreg_2_0: (Notice dst tiles are not in correct offset!)
  // dst_tile_2_0_2
  // dst_tile_2_0_3
  // dst_tile_2_0_0
  // dst_tile_2_0_1

  //  dst_vreg_3_0: (Notice dst tiles are not in correct offset!)
  // dst_tile_3_0_1
  // dst_tile_3_0_2
  // dst_tile_3_0_3
  // dst_tile_3_0_0

  // Each destination vreg is assembled from destination tiles in multiple
  // rotated src vregs. In the above example, if we wanted each destination tile
  // to be in correct sublane offset in a rotated vreg, say rot_src_vreg_0_1,
  // before assembling the destination tiles, we would have had to rotate
  // src_vreg_0_1 four times, creating 4 rotated vregs (instead of 1) for each
  // src vreg. In the above example, we instead rotated a src vreg src_vreg_0_1
  // only once to obtain rot_src_vreg_0_1 where the dst_tile_0_0_1 is in correct
  // final sublane offset, i.e. 2. But notice the sublane offset of
  // dst_tile_1_0_1 in the same rotated vreg. Its correct final destination
  // sublane offset is 2, but in rot_src_vreg_0_1, its offset is 4. Its sublane
  // offset is off by 2. We need to correct these sublane offsets in the final
  // assembled dst vregs. A single rotation of each assembled dst vreg is needed
  // to correct such sublane offsets. This strategy reduces the number of
  // sublane rotations required. See comments below.
  const int64_t tile_sublane_change_factor =
      src_layout.tiling()[0] / dst_layout.tiling()[0];

  dst_vreg_array.Each([&](absl::Span<const int64_t> idx,
                          Value *const dst_vreg) {
    const int64_t row = *(idx.end() - 2);
    const int64_t col = *(idx.end() - 1);
    auto [rotated_vreg_row, first_dst_tile_offset] =
        std::div(row, tile_sublane_change_factor);
    const int64_t first_dst_tile_sublane_offset =
        first_dst_tile_offset * dst_sublanes_per_tile;
    const int64_t src_vreg_array_col_start = col * dst_tiles_per_vreg;
    const int64_t src_vreg_array_col_end =
        std::min((col + 1) * dst_tiles_per_vreg,
                 rotated_src_vreg_array.dimensions().back()) -
        1;

    // TODO(tlongeri): Find a better way to slice that doesn't involve so
    // copying so many index vectors and hopefully is more concise. Probably
    // by expanding xla::Array (maybe could just expose calculate_index?).
    SmallVector<int64_t> rotated_row_starts(toArrayRef(idx));
    *(rotated_row_starts.end() - 2) = rotated_vreg_row;
    *(rotated_row_starts.end() - 1) = 0;
    SmallVector<int64_t> rotated_row_ends(idx.size());
    for (size_t i = 0; i + 1 < rotated_row_ends.size(); ++i) {
      rotated_row_ends[i] = rotated_row_starts[i] + 1;
    }
    *(rotated_row_ends.end() - 1) = rotated_src_vreg_array.dimensions().back();
    const xla::Array<Value> rotated_row_slice =
        rotated_src_vreg_array.Slice(rotated_row_starts, rotated_row_ends);
    const Value dst_tile = selectTilesFromRotatedRowVregs(
        ctx, builder, /*rotated_row_vregs=*/
        ArrayRef(rotated_row_slice.begin(), rotated_row_slice.end()),
        src_vreg_array_col_start, src_vreg_array_col_end,
        first_dst_tile_sublane_offset, dst_layout);
    if (first_dst_tile_sublane_offset == 0) {
      // No need to rotate. First dst tile is already at offset 0, which means
      // rest of the dst tiles are also at correct sublane offset.
      *dst_vreg = dst_tile;
    } else {
      // Fix the destination tile sublane offset by rotating assembled dest vreg
      // once (See comments above). The dst vregs are fixed as follows:
      // No rotation needed.
      // dst_tile_0_0_0
      // dst_tile_0_0_1
      // dst_tile_0_0_2
      // dst_tile_0_0_3

      // Rotated by -1 * (sublanes_per_tile=2) * (row=1):
      // dst_tile_1_0_0
      // dst_tile_1_0_1
      // dst_tile_1_0_2
      // dst_tile_1_0_3

      // Rotated by -1 * (sublanes_per_tile=2) * (row=2):
      // dst_tile_2_0_0
      // dst_tile_2_0_1
      // dst_tile_2_0_2
      // dst_tile_2_0_3

      // Rotated by -1 * (sublanes_per_tile=2) * (row=3):
      // dst_tile_3_0_0
      // dst_tile_3_0_1
      // dst_tile_3_0_2
      // dst_tile_3_0_3
      *dst_vreg = builder.create<tpu::RotateOp>(
          dst_tile.getLoc(), dst_tile,
          ctx.target_shape[0] - first_dst_tile_sublane_offset, /*dimension=*/0);
    }
  });
  return dst_vreg_array;
}

// Returns true iff the layout changes involve reduced sublanes per tile.
//
// Arguments:
//  src: The existing layout.
//  dst: The new layout based on which the retiling is to be carried out.
bool isSupportedReducedSublanesRetile(RewriteContext &ctx,
                                      const VectorLayout &src,
                                      const VectorLayout &dst) {
  return src.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
         dst.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
         llvm::all_of(llvm::zip_equal(src.offsets(), dst.offsets()),
                      [](auto tup) {
                        auto [lhs, rhs] = tup;
                        return lhs.value_or(0) == rhs.value_or(0);
                      })
         // TODO (kumudbhandari): We have not tested any tile size where
         // tile[-1] != TARGET_SHAPE.lanes. It should work but needs to be
         // tested.
         && src.tiling()[1] == ctx.target_shape[1] &&
         dst.tiling()[1] == ctx.target_shape[1] &&
         dst.tiling()[0] < src.tiling()[0] &&
         src.bitwidth() == dst.bitwidth() &&
         llvm::isPowerOf2_64(src.tiling()[0]) &&
         llvm::isPowerOf2_64(dst.tiling()[0]);
}

// Changes the layout of a vector value.
//
// Arguments:
//   v: The value to relayout. Must be of type VectorType.
//   src: The current layout of v.
//   dst: The target layout of v.
//
// Returns:
//   A new MLIR vector value, laid out as requested by dst.
// TODO(apaszke): Test this function properly
FailureOr<Value> relayout(RewriteContext &ctx, OpBuilder &builder, Value v,
                          VectorLayout src, const VectorLayout &dst) {
  const int8_t bitwidth = src.bitwidth();
  if (bitwidth != dst.bitwidth()) {
    return emitError(v.getLoc(), "Can't change bitwidth during a relayout");
  }
  const int packing = src.packing();
  VectorType vty = cast<VectorType>(v.getType());
  FAILUREOR_ASSIGN_OR_RETURN(xla::Array<Value> src_tiles,
                             disassemble(ctx, builder, src, v));
  SmallVector<int64_t> dst_tiles_shape =
      dst.tileArrayShape(vty.getShape(), ctx.target_shape);
  if (src.generalizes(dst, vty.getShape(), ctx.target_shape)) {
    return assemble(ctx, builder, vty, dst, std::move(src_tiles)).getResult();
  }
  if (!src.offsets()[0].has_value() && !src.offsets()[1].has_value() &&
      src.tilesPerVreg(ctx.target_shape) == 1) {
    // A fully replicated value is always easy to relayout
    // It would be nice to be able to assert this here, but given replicated
    // values our rules can introduce equivalent expressions.
    // assert all(t is src_tiles_list[0] for t in src_tiles_list)
    xla::Array<Value> dst_tiles(
        /*sizes=*/dst.tileArrayShape(vty.getShape(), ctx.target_shape),
        /*value=*/src_tiles.data()[0]);
    return assemble(ctx, builder, vty, dst, std::move(dst_tiles)).getResult();
  }
  // Try to reconcile differences in implicit dim.
  if (src.implicit_dim() != dst.implicit_dim()) {
    VectorLayout candidate(src.bitwidth(), src.offsets(), src.tiling(),
                           dst.implicit_dim());
    if (candidate.equivalentTo(src, vty.getShape(), ctx.target_shape)) {
      src = candidate;
    }
  }

  // Handle retiling from (1, 128) to (8, 128) for 32-bit data.
  if (src.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      dst.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      src.bitwidth() == 32 && src.offsets() == LayoutOffsets{0, 0} &&
      (dst.offsets() == LayoutOffsets{std::nullopt, 0} ||
       dst.offsets() == LayoutOffsets{0, 0}) &&
      src.tiling() == std::array<int64_t, 2>{1, 128} &&
      dst.tiling() == std::array<int64_t, 2>{8, 128} &&
      *(src_tiles.dimensions().end() - 2) == 1) {
    xla::Array<Value> src_tiles_retiled(
        dst.tileArrayShape(vty.getShape(), ctx.target_shape));
    src_tiles_retiled.Each(
        [&](const absl::Span<const int64_t> idx, Value *const new_src_tile) {
          const int64_t dst_col = idx.back();
          const int64_t src_col = dst_col / 8;
          const int64_t slane_idx = dst_col % 8;
          const DenseI32ArrayAttr gather_indices =
              builder.getDenseI32ArrayAttr(SmallVector<int32_t>(8, slane_idx));
          SmallVector<int64_t> src_idx(toArrayRef(idx));
          src_idx.back() = src_col;
          Value src_tile = src_tiles(src_idx);
          *new_src_tile = builder.create<tpu::GatherOp>(
              v.getLoc(), src_tile.getType(), src_tile, gather_indices,
              /*dimension=*/0);
        });
    src = dst;
    src_tiles = std::move(src_tiles_retiled);
  }
  // Handle retiling from (2, 128) to (8, 128) for 32-bit data.
  if (src.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      dst.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      src.bitwidth() == 32 && src.offsets() == LayoutOffsets{0, 0} &&
      dst.offsets() == LayoutOffsets{0, 0} &&
      src.tiling() == std::array<int64_t, 2>{2, 128} &&
      dst.tiling() == std::array<int64_t, 2>{8, 128} &&
      *(src_tiles.dimensions().end() - 2) <= 2) {
    xla::Array<Value> src_tiles_retiled(
        dst.tileArrayShape(vty.getShape(), ctx.target_shape));
    src_tiles_retiled.Each([&](const absl::Span<const int64_t> idx,
                               Value *const new_src_tile) {
      const int64_t dst_col = idx.back();
      const int64_t src_col = dst_col / 4;
      const int64_t start_slane_idx = 2 * (dst_col % 4);
      SmallVector<int64_t> src_idx(toArrayRef(idx));
      src_idx.back() = src_col;
      Value src_tile = src_tiles(src_idx);
      if (start_slane_idx) {
        SmallVector<int32_t> slane_idxs;
        slane_idxs.reserve(8);
        for (int i = 0; i < 8; ++i) {
          slane_idxs.push_back(start_slane_idx + (i % 2));
        }
        const DenseI32ArrayAttr gather_indices =
            builder.getDenseI32ArrayAttr(slane_idxs);
        *new_src_tile = builder.create<tpu::GatherOp>(
            v.getLoc(), src_tile.getType(), src_tile, gather_indices,
            /*dimension=*/0);
      } else {
        *new_src_tile = src_tile;
      }
    });
    src = dst;
    src_tiles = std::move(src_tiles_retiled);
  }
  // TODO(b/265133506): Generalize retiling to general 16-bit types (might
  // need to use a different unpacking op).
  VectorType vreg_f32 = VectorType::get(ctx.target_shape, builder.getF32Type());
  // (8,128) -> (16,128) tiling change for packed 16-bit types.
  if (src.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      dst.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      vty.getElementType() == builder.getBF16Type() &&
      src.offsets() == dst.offsets() &&
      src.tiling() == std::array<int64_t, 2>{8, 128} &&
      dst.tiling() == std::array<int64_t, 2>{16, 128}) {
    const VectorLayout new_src(src.bitwidth(), src.offsets(), dst.tiling());
    xla::Array<Value> src_tiles_retiled(
        new_src.tileArrayShape(vty.getShape(), ctx.target_shape));
    src_tiles_retiled.Each([&](absl::Span<const int64_t> idx, Value *tile) {
      SmallVector<int64_t> src_idx(idx.begin(), idx.end());
      src_idx[src_idx.size() - 2] *= 2;
      src_idx[src_idx.size() - 1] /= 2;
      Value src_row1 = src_tiles(src_idx);
      if (src_idx[src_idx.size() - 2] + 1 <
          src_tiles.dim(src_tiles.num_dimensions() - 2)) {
        ++src_idx[src_idx.size() - 2];
      }
      Value src_row2 = src_tiles(src_idx);
      const int vreg_part = idx[idx.size() - 1] % 2;
      auto half_row1 = builder.create<tpu::UnpackSubelementsOp>(
          v.getLoc(), vreg_f32, src_row1, vreg_part);
      auto half_row2 = builder.create<tpu::UnpackSubelementsOp>(
          v.getLoc(), vreg_f32, src_row2, vreg_part);
      *tile = builder.create<tpu::PackSubelementsOp>(
          v.getLoc(), src_row1.getType(), ValueRange{half_row1, half_row2});
    });
    src = new_src;
    src_tiles = std::move(src_tiles_retiled);
  }

  // (8,128) -> (32,128) tiling change for packed 8-bit integers.
  if (src.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      dst.implicit_dim() == VectorLayout::ImplicitDim::kNone &&
      vty.getElementType() == builder.getI8Type() &&
      src.offsets() == dst.offsets() &&
      src.tiling() == std::array<int64_t, 2>{8, 128} &&
      dst.tiling() == std::array<int64_t, 2>{32, 128}) {
    const VectorLayout new_src(src.bitwidth(), src.offsets(), dst.tiling());
    xla::Array<Value> src_tiles_retiled(
        new_src.tileArrayShape(vty.getShape(), ctx.target_shape));
    VectorType vreg_i32 =
        getNativeVregType(builder.getI32Type(), ctx.target_shape).value();
    src_tiles_retiled.Each([&](absl::Span<const int64_t> idx, Value *tile) {
      const int vreg_part = idx.back() % 4;
      std::array<Value, 4> parts;
      SmallVector<int64_t> src_idx(idx.begin(), idx.end());
      src_idx[src_idx.size() - 2] *= 4;
      src_idx[src_idx.size() - 1] /= 4;
      for (int i = 0; i < 4; ++i) {
        parts[i] = builder.create<tpu::UnpackSubelementsOp>(
            v.getLoc(), vreg_i32, src_tiles(src_idx), vreg_part);
        if (src_idx[src_idx.size() - 2] <
            src_tiles.dim(src_tiles.num_dimensions() - 2) - 1) {
          ++src_idx[src_idx.size() - 2];
        }
      }
      *tile = builder.create<tpu::PackSubelementsOp>(
          v.getLoc(), src_tiles.begin()->getType(), parts);
    });
    src = new_src;
    src_tiles = std::move(src_tiles_retiled);
  }

  if (isSupportedReducedSublanesRetile(ctx, src, dst)) {
    src_tiles = retileToReducedSublanes(ctx, builder, vty.getShape(), src,
                                        src_tiles, dst);
    src = dst;
  }

  // Fix up the offsets, assuming everything else matches between src and dst.
  if (src.tiling() == dst.tiling() &&
      src.implicit_dim() == dst.implicit_dim()) {
    const auto &tiling = src.tiling();
    // TODO(apaszke): Changing an offset might add or remove one vreg.
    if (dst_tiles_shape != src_tiles.dimensions()) {
      return emitError(v.getLoc(), "Offsets changing the vreg array shape");
    }
    xla::Array<Value> dst_tiles = src_tiles;

    // Shifting rows
    int row_diff;
    if (!src.offsets()[0].has_value()) {
      row_diff = 0;
    } else if (!dst.offsets()[0].has_value()) {
      return emitError(v.getLoc(), "Sublane broadcast not implemented");
    } else {
      row_diff = *dst.offsets()[0] - *src.offsets()[0];
    }

    if (row_diff != 0) {  // This is an easy case, because we never need to
                          // combine multiple vregs.
      const SmallVector<int64_t> implicit_shape =
          src.implicitShape(vty.getShape());
      if (implicit_shape[implicit_shape.size() - 2] != 1) {
        return emitError(v.getLoc(), "Row shifts for multi-row values");
      }
      const int64_t src_sublane = *src.offsets()[0] / packing;
      const int64_t dst_sublane = *dst.offsets()[0] / packing;
      if (int64_t sublane_diff = dst_sublane - src_sublane) {
        if (sublane_diff < 0) {
          sublane_diff += ctx.target_shape[0];
        }
        src_tiles.Each([&](absl::Span<const int64_t> idx, Value tile) {
          dst_tiles(idx) = builder
                               .create<tpu::RotateOp>(v.getLoc(), tile,
                                                      /*amount=*/sublane_diff,
                                                      /*dimension=*/0)
                               .getResult();
        });
      }
      const int src_subelem = *src.offsets()[0] % packing;
      const int dst_subelem = *dst.offsets()[0] % packing;
      if (src_subelem != dst_subelem) {
        const int subelem_diff = dst_subelem - src_subelem;
        const int shift_bits = bitwidth * std::abs(subelem_diff);
        VectorType bits_vreg_ty =
            VectorType::get(ctx.target_shape, builder.getI32Type());
        auto shift_vreg = builder.create<arith::ConstantOp>(
            v.getLoc(), bits_vreg_ty,
            DenseElementsAttr::get(bits_vreg_ty, shift_bits));
        dst_tiles.Each([&](absl::Span<const int64_t> /*idx*/, Value *tile) {
          auto bit_tile =
              builder.create<tpu::BitcastOp>(v.getLoc(), bits_vreg_ty, *tile);
          Operation *shift_tile;
          if (subelem_diff > 0) {
            shift_tile =
                builder.create<arith::ShLIOp>(v.getLoc(), bit_tile, shift_vreg);
          } else {  // subelem_diff < 0
            CHECK_LT(subelem_diff, 0);
            shift_tile = builder.create<arith::ShRUIOp>(v.getLoc(), bit_tile,
                                                        shift_vreg);
          }
          *tile = builder
                      .create<tpu::BitcastOp>(v.getLoc(), tile->getType(),
                                              shift_tile->getResult(0))
                      .getResult();
          return absl::OkStatus();
        });
      }
    }
    int64_t col_diff;
    if (!src.offsets()[1].has_value()) {
      col_diff = 0;
    } else if (!dst.offsets()[1].has_value()) {
      return emitError(v.getLoc(), "Not implemented: Lane broadcast");
    } else {
      col_diff = *dst.offsets()[1] - *src.offsets()[1];
    }
    if (col_diff != 0) {
      if (row_diff != 0) {
        return emitError(v.getLoc(),
                         "Not implemented: Both columns and rows are shifted");
      }
      if (col_diff < 0) {
        return emitError(v.getLoc(), "Not implemented: Shifts to the left");
      }
      if (bitwidth != 32 || tiling != ctx.target_shape) {
        return emitError(v.getLoc(),
                         "Not implemented: Only 32-bit column shifts for "
                         "native layouts supported");
      }
      const int64_t sublane_diff = col_diff;
      CHECK_GE(src_tiles.num_dimensions(), 1);
      std::optional<tpu::CreateMaskOp> maybe_create_mask;
      if (src_tiles.dimensions()[src_tiles.num_dimensions() - 1] > 1) {
        auto boundIdxConst =
            std::bind(IdxConst, std::placeholders::_1, builder, v.getLoc());
        maybe_create_mask = builder.create<tpu::CreateMaskOp>(
            v.getLoc(), VectorType::get(ctx.target_shape, builder.getI1Type()),
            ValueRange{boundIdxConst(0), boundIdxConst(0)},
            ValueRange{boundIdxConst(ctx.target_shape[0]),
                       boundIdxConst(col_diff)});
      }
      src_tiles.Each([&](absl::Span<const int64_t> idx, Value tile) {
        Value rot_tile = builder
                             .create<tpu::RotateOp>(v.getLoc(), tile,
                                                    /*amount=*/sublane_diff,
                                                    /*dimension=*/1)
                             .getResult();
        if (idx[idx.size() - 1] != 0) {
          SmallVector<int64_t> prev_idx(idx.begin(), idx.end());
          --prev_idx[idx.size() - 1];
          Value prev_rot_tile = dst_tiles(prev_idx);
          rot_tile = builder.create<arith::SelectOp>(
              v.getLoc(), maybe_create_mask->getResult(), prev_rot_tile, tile);
        }
        dst_tiles(idx) = rot_tile;
      });
    }
    return assemble(ctx, builder, vty, dst, std::move(dst_tiles)).getResult();
  }
  // TODO(apaszke): Implement general relayout
  return emitError(v.getLoc(), "unsupported layout change for ")
         << vty << ": " << src << " -> " << dst;
}

// Rewrites the operation according to its layout annotations.
//
// Args:
//   ctx: The context used for rewriting.
//   op: An MLIR operation to be rewritten.
//
// A valid op is expected to have a layout_in attribute unless it has no
// operands. The layout_in attribute must fulfill the following:
//   - All vector operands originate from an operation (not a BlockArgument)
//   and
//     have a valid layout (Layout1D or Layout2D)
//   - All non-vector operands must have NoLayout.
// TODO(apaszke): Implement a debug mode that inserts additional assertions.
// For example, we should verify that ops that were supposed to generate
// replicated outputs satisfy that requirement.
LogicalResult applyLayoutOp(RewriteContext &ctx, Operation &op) {
  // TODO(tlongeri): Once we support all ops, return failure instead.
  if (!rules().contains(op.getName().getStringRef()) &&
      !OpTrait::hasElementwiseMappableTraits(&op)) {
    return success();
  }

  // When an operation does not have any operands, the layout_in tuple is empty.
  // If one of the operands is not of vector type, the corresponding entry in
  // the layout_in tuple will be None. The same applies to the results of the
  // operation and the layout_out tuple.
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> layout_out,
                             getOutLayout(op));
  FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> layout_in,
                             getInLayout(op));
  if (!layout_in.empty()) {
    // Relayout the operands, if their requested input layouts don't match the
    // layouts in which they were produced.
    for (auto [idx, tup] :
         llvm::enumerate(llvm::zip(op.getOperands(), layout_in))) {
      auto [operand, li] = tup;
      auto vty = dyn_cast<VectorType>(operand.getType());
      if ((vty == nullptr) == li.has_value()) {
        return op.emitError(
            "layout should be none iff operand is not a vector");
      }
      if (vty == nullptr) {
        continue;
      }

      // The operand should always be an Operation (and not a BlockArgument)
      // since we expect the FuncOp to have only memrefs and semaphores as
      // arguments.
      auto op_result = dyn_cast<OpResult>(operand);
      if (op_result == nullptr) {
        return op.emitError("expected operand to be an operation result");
      }
      Operation *const def_op = op_result.getOwner();
      CHECK(def_op);
      const unsigned res_idx = op_result.getResultNumber();
      FAILUREOR_ASSIGN_OR_RETURN(const SmallVector<Layout> def_layouts,
                                 getOutLayout(*def_op));
      const Layout lo = def_layouts[res_idx];
      if (!lo.has_value()) {
        return op.emitError() << "vector result should have a defined layout";
      }
      if (lo->generalizes(*li, vty.getShape(), ctx.target_shape)) {
        continue;
      }
      OpBuilder builder(&op);
      FAILUREOR_ASSIGN_OR_RETURN(Value new_v,
                                 relayout(ctx, builder, operand, /*src=*/*lo,
                                          /*dst=*/*li));
      op.setOperand(idx, new_v);
    }
  }

  const bool no_vector_args =
      llvm::none_of(layout_out,
                    [](Layout layout) { return layout.has_value(); }) &&
      llvm::none_of(layout_in,
                    [](Layout layout) { return layout.has_value(); });
  if (no_vector_args && op.getRegions().empty()) {
    // We don't need to do anything for scalar operations.
    if (!op.getOperands().empty()) {
      op.removeAttr("in_layout");
    }
    if (!op.getResults().empty()) {
      op.removeAttr("out_layout");
    }
    return success();
  }
  if (auto rule_it = rules().find(op.getName().getStringRef());
      rule_it != rules().end()) {
    const rule_type &rule = rule_it->getValue();
    return rule(ctx, op, layout_in, layout_out);
  }
  if (OpTrait::hasElementwiseMappableTraits(&op)) {
    return elementwise_op_rule(ctx, op, layout_in, layout_out);
  }
  return op.emitError("Unsupported operation: ") << op.getName();
}

LogicalResult applyLayoutBlock(RewriteContext &ctx, Block &block) {
  // We'll be modifying the block, so use early increment.
  for (Operation &op : make_early_inc_range(block)) {
    if (failed(applyLayoutOp(ctx, op))) {
      return failure();
    }
  }
  return success();
}

// Rewrites the function according to layout annotations of its operations.
//
//   Args:
//     ctx: The context used for rewriting.
//     f: An MLIR function to be rewritten.
LogicalResult applyLayoutFunc(RewriteContext &ctx, func::FuncOp f) {
  if (f->getNumRegions() != 1) {
    return f.emitError("Expected FuncOp to have a single region");
  }
  if (!f.getBody().hasOneBlock()) {
    return f.emitError("Expected FuncOp to have a single block");
  }
  return applyLayoutBlock(ctx, f.getBody().front());
}

struct ApplyVectorLayoutPass
    : public impl::ApplyVectorLayoutPassBase<ApplyVectorLayoutPass> {
  ApplyVectorLayoutPass(int hardware_generation_, int lane_count_,
                        int sublane_count_) {
    hardware_generation = hardware_generation_;
    sublane_count = sublane_count_;
    lane_count = lane_count_;
  }
  void runOnOperation() override {
    // Fail if hardware_generation has not been set from the default value.
    if (hardware_generation < 0) {
      signalPassFailure();
      return;
    }
    func::FuncOp func = getOperation();
    RewriteContext ctx{func, hardware_generation, {sublane_count, lane_count}};
    if (failed(applyLayoutFunc(ctx, func))) {
      signalPassFailure();
      return;
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createApplyVectorLayoutPass(
    int hardware_generation, int lane_count, int sublane_count) {
  return std::make_unique<ApplyVectorLayoutPass>(hardware_generation,
                                                 lane_count, sublane_count);
}

}  // namespace mlir::tpu
