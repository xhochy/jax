/* Copyright 2023 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "mlir-c/AffineMap.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "pybind11/detail/common.h"
#include "pybind11/pytypes.h"
#include "absl/log/check.h"
#include "third_party/llvm/llvm-project/mlir/lib/Bindings/Python/IRModule.h"
#include "jaxlib/mosaic/dialect/tpu/integrations/c/tpu_dialect.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"
#include "third_party/pybind11/include/pybind11/attr.h"
#include "third_party/pybind11/include/pybind11/cast.h"
#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "third_party/pybind11/include/pybind11/typing.h"

// TODO(tlongeri): Can I add my own return type annotations to functions?
// TODO(tlongeri): I don't understand why MLIR uses the C API to implement
// Python bindings. Do we have a reason to do that?

namespace {
constexpr const char LAYOUT_DEFS[] = "jax.jaxlib.mosaic.python.layout_defs";
// TODO(tlongeri): get rid of this somehow
constexpr std::array<int64_t, 2> TARGET_SHAPE{8, 128};

// Add sequence type with annotation
// TODO(tlongeri): Can I enforce this for this and other types?
template <typename Type>
class Sequence : public py::sequence {
  using sequence::sequence;
};
}  // namespace
template <typename Type>
struct py::detail::handle_type_name<Sequence<Type>> {
  static constexpr auto name =
      const_name("sequence[") + make_caster<Type>::name + const_name("]");
};

template <>
struct py::detail::type_caster<mlir::tpu::VectorLayout::ImplicitDim> {
  PYBIND11_TYPE_CASTER(mlir::tpu::VectorLayout::ImplicitDim,
                       const_name("optional[ImplicitDim]"));

  bool load(handle src, bool) {
    if (src.is_none()) {
      value = mlir::tpu::VectorLayout::ImplicitDim::kNone;
      return true;
    }
    if (!py::isinstance(src,
                        py::module_::import(LAYOUT_DEFS).attr("ImplicitDim"))) {
      return false;
    }
    auto implicit_dim_cls =
        py::module_::import(LAYOUT_DEFS).attr("ImplicitDim");
    if (src.is(implicit_dim_cls.attr("MINOR"))) {
      value = mlir::tpu::VectorLayout::ImplicitDim::kMinor;
    } else if (src.is(implicit_dim_cls.attr("SECOND_MINOR"))) {
      value = mlir::tpu::VectorLayout::ImplicitDim::kSecondMinor;
    } else {
      throw py::value_error();
    }
    return true;
  }

  static handle cast(mlir::tpu::VectorLayout::ImplicitDim implicit_dim,
                     return_value_policy /* policy */, handle /* parent */) {
    auto implicit_dim_cls =
        py::module_::import(LAYOUT_DEFS).attr("ImplicitDim");
    switch (implicit_dim) {
      case mlir::tpu::VectorLayout::ImplicitDim::kNone:
        return py::none().release();
      case mlir::tpu::VectorLayout::ImplicitDim::kMinor:
        return static_cast<py::object>(implicit_dim_cls.attr("MINOR"))
            .release();
      case mlir::tpu::VectorLayout::ImplicitDim::kSecondMinor:
        return static_cast<py::object>(implicit_dim_cls.attr("SECOND_MINOR"))
            .release();
    }
  }
};

template <>
struct py::detail::type_caster<mlir::tpu::Direction> {
  PYBIND11_TYPE_CASTER(mlir::tpu::Direction, const_name("Direction"));

  bool load(handle src, bool) {
    if (!py::isinstance(src,
                        py::module_::import(LAYOUT_DEFS).attr("Direction"))) {
      return false;
    }
    auto direction_cls = py::module_::import(LAYOUT_DEFS).attr("Direction");
    if (src.is(direction_cls.attr("LANES"))) {
      value = mlir::tpu::Direction::kLanes;
    } else if (src.is(direction_cls.attr("SUBLANES"))) {
      value = mlir::tpu::Direction::kSublanes;
    } else if (src.is(direction_cls.attr("SUBELEMENTS"))) {
      value = mlir::tpu::Direction::kSubelements;
    } else {
      throw py::value_error();
    }
    return true;
  }

  static handle cast(mlir::tpu::Direction direction,
                     return_value_policy /* policy */, handle /* parent */) {
    auto direction_cls = py::module_::import(LAYOUT_DEFS).attr("ImplicitDim");
    switch (direction) {
      case mlir::tpu::Direction::kLanes:
        return static_cast<py::object>(direction_cls.attr("LANES")).release();
      case mlir::tpu::Direction::kSublanes:
        return static_cast<py::object>(direction_cls.attr("SUBLANES"))
            .release();
      case mlir::tpu::Direction::kSubelements:
        return static_cast<py::object>(direction_cls.attr("SUBELEMENTS"))
            .release();
    }
  }
};

namespace {
py::object toPyLayoutOffset(mlir::tpu::LayoutOffset offset) {
  if (offset) {
    return py::int_(*offset);
  } else {
    return py::module_::import(LAYOUT_DEFS).attr("REPLICATED");
  }
}

// TODO(tlongeri): Would `type_caster`s let me avoid defining all of these
// to/from functions?
mlir::tpu::LayoutOffset fromPyLayoutOffset(py::object offset) {
  if (py::isinstance<py::int_>(offset)) {
    return py::cast<py::int_>(offset);
  } else if (offset.equal(
                 py::module_::import(LAYOUT_DEFS).attr("REPLICATED"))) {
    return std::nullopt;
  } else {
    throw py::type_error("Invalid layout offset type");
  }
}

template <typename T>
mlir::SmallVector<T> sequenceToSmallVector(py::sequence seq) {
  return llvm::map_to_vector(
      seq, [](py::handle handle) { return py::cast<T>(handle); });
}

template <typename Container>
py::tuple containerToTuple(const Container& cont) {
  py::tuple tuple(cont.size());
  for (size_t i = 0; i < cont.size(); ++i) {
    tuple[i] = cont[i];
  }
  return tuple;
}

mlir::OpBuilder getOpBuilderFromPyThreadContext() {
  mlir::python::PyInsertionPoint* insertion_point =
      mlir::python::PyThreadContextEntry::getDefaultInsertionPoint();
  if (insertion_point == nullptr) {
    throw std::runtime_error("Insertion point has not been set");
  }
  if (std::optional<mlir::python::PyOperationRef> maybe_ref_operation =
          insertion_point->getRefOperation()) {
    return mlir::OpBuilder(unwrap(maybe_ref_operation->get()->get()));
  }
  return mlir::OpBuilder::atBlockEnd(unwrap(insertion_point->getBlock().get()));
}

mlir::Location getLocationFromPyThreadContext() {
  mlir::python::PyLocation* location =
      mlir::python::PyThreadContextEntry::getDefaultLocation();
  if (location == nullptr) {
    throw std::runtime_error("Location has not been set");
  }
  return unwrap(location->get());
}

mlir::MLIRContext* getContextFromPyThreadContext() {
  mlir::python::PyMlirContext* ctx =
      mlir::python::PyThreadContextEntry::getDefaultContext();
  if (ctx == nullptr) {
    throw std::runtime_error("Context has not been set");
  }
  return unwrap(ctx->get());
}

}  // namespace

PYBIND11_MODULE(_tpu_ext, m) {
  mlirRegisterTPUPasses();  // Register all passes on load.

  py::class_<mlir::tpu::VRegDataBounds>(m, "VRegDataBounds", py::module_local())
      .def("mask_varies_along",
           [](mlir::tpu::VRegDataBounds& self, mlir::tpu::Direction direction) {
             return self.maskVariesAlong(direction, TARGET_SHAPE);
           })
      .def_property_readonly("complete",
                             [](mlir::tpu::VRegDataBounds& self) {
                               return self.isComplete(TARGET_SHAPE);
                             })
      .def("get_vector_mask",
           [](mlir::tpu::VRegDataBounds& self, int generation) {
             // TODO: Does this work? Test in Python
             mlir::OpBuilder builder = getOpBuilderFromPyThreadContext();
             auto failure_or_mask =
                 self.getVectorMask(builder, getLocationFromPyThreadContext(),
                                    generation, TARGET_SHAPE);
             if (mlir::failed(failure_or_mask)) {
               throw std::runtime_error("getVectorMask failed");
             }
             return wrap(*failure_or_mask);
           })
      .def("get_sublane_mask", [](mlir::tpu::VRegDataBounds& self) {
        return wrap(
            self.getSublaneMask(getContextFromPyThreadContext(), TARGET_SHAPE));
      });

  // TODO(tlongeri): More precise argument type annotations. There currently
  // seems to be no way to define your own?
  py::class_<mlir::tpu::VectorLayout>(m, "VectorLayout", py::module_local())
      .def(py::init([](int bitwidth, py::tuple offsets,
                       py::typing::Tuple<int, int> tiling,
                       mlir::tpu::VectorLayout::ImplicitDim implicit_dim) {
             CHECK_EQ(offsets.size(), 2);
             return mlir::tpu::VectorLayout(
                 bitwidth,
                 {fromPyLayoutOffset(offsets[0]),
                  fromPyLayoutOffset(offsets[1])},
                 {tiling[0].cast<int64_t>(), tiling[1].cast<int64_t>()},
                 implicit_dim);
           }),
           py::arg("bitwidth"), py::arg("offsets"), py::arg("tiling"),
           py::arg("implicit_dim"))
      .def_property_readonly("bitwidth", &mlir::tpu::VectorLayout::bitwidth,
                             "The bitwidth of the stored values.")
      .def_property_readonly(
          "offsets",
          [](mlir::tpu::VectorLayout& self) {
            return py::make_tuple(toPyLayoutOffset(self.offsets()[0]),
                                  toPyLayoutOffset(self.offsets()[1]));
          },
          "The coordinates of the first valid element. If an offset is "
          "REPLICATED, then any offset is valid as the value does not vary "
          "across sublanes or lanes respectively.")
      .def_property_readonly(
          "tiling",
          [](mlir::tpu::VectorLayout& self) {
            return containerToTuple(self.tiling());
          },
          "The tiling used to lay out values (see the XLA docs). For values of "
          "bitwidth < 32, an implicit (32 // bitwidth, 1) tiling is appended "
          "to the one specified as an attribute.")
      .def_property_readonly(
          "implicit_dim", &mlir::tpu::VectorLayout::implicit_dim,
          "If specified, the value has an implicit dim inserted in either "
          "minormost or second minormost position.")
      .def_property_readonly(
          "packing", &mlir::tpu::VectorLayout::packing,
          "Returns the number of values stored in a vreg entry.")
      .def_property_readonly(
          "layout_rank", &mlir::tpu::VectorLayout::layout_rank,
          "The number of minormost dimensions tiled by this layout.")
      .def_property_readonly(
          "has_natural_topology",
          [](mlir::tpu::VectorLayout& self) {
            return self.hasNaturalTopology(TARGET_SHAPE);
          },
          "True, if every vector register has a layout without jumps.\n"
          "\n"
          "By without jumps we mean that traversing vregs over (sub)lanes "
          "always leads to a contiguous traversal of the (second) minormost "
          "dimension of data. This is only true for 32-bit types, since "
          "narrower types use two level tiling.")
      .def_property_readonly(
          "has_native_tiling",
          [](mlir::tpu::VectorLayout& self) {
            return self.hasNativeTiling(TARGET_SHAPE);
          },
          "True, if every vector register has a natural \"packed\" topology.\n"
          "\n"
          "This is equivalent to has_natural_topology for 32-bit types, but "
          "generalizes it to narrower values with packed layouts too.")
      .def_property_readonly(
          "tiles_per_vreg",
          [](mlir::tpu::VectorLayout& self) {
            return self.tilesPerVreg(TARGET_SHAPE);
          },
          "How many tiles fit in each vector register.")
      .def_property_readonly(
          "sublanes_per_tile",
          [](mlir::tpu::VectorLayout& self) {
            return self.sublanesPerTile(TARGET_SHAPE);
          },
          "The number of sublanes necessary to store each tile.")
      .def_property_readonly(
          "vreg_slice",
          [](mlir::tpu::VectorLayout& self) {
            std::array<int64_t, 2> vreg_slice = self.vregSlice(TARGET_SHAPE);
            return py::module_::import(LAYOUT_DEFS)
                .attr("TargetTuple")(vreg_slice[0], vreg_slice[1]);
          },
          "Returns the size of a window contained in a single vreg.\n"
          "\n"
          "We never reuse the same vector register to store data of multiple "
          "rows, so only the minormost dimension can increase.")
      .def(
          "implicit_shape",
          [](mlir::tpu::VectorLayout& self, Sequence<int64_t> shape) {
            return containerToTuple(
                self.implicitShape(sequenceToSmallVector<int64_t>(shape)));
          },
          py::arg("shape"))
      .def(
          "tile_array_shape",
          [](mlir::tpu::VectorLayout& self, Sequence<int64_t> shape) {
            return containerToTuple(self.tileArrayShape(
                sequenceToSmallVector<int64_t>(shape), TARGET_SHAPE));
          },
          py::arg("shape"),
          "Returns the shape of an ndarray of vregs needed to represent a "
          "value.\n"
          "\n"
          "All but the last two dimensions are unrolled over vregs. In the "
          "last two dims we need as many vregs as indicated by dividing the "
          "point at which the value ends (given by the start offset plus the "
          "dim size) divided by the respective vreg capacity in that dim (and "
          "a ceiling if non-integral). If a value is replicated, then any "
          "offset is valid and we pick 0 to minimize the number of vregs.\n"
          "\n"
          "Args:\n"
          "  shape: The shape of the ndarray to tile.")
      .def(
          "generalizes",
          [](mlir::tpu::VectorLayout& self, mlir::tpu::VectorLayout& other,
             std::optional<Sequence<int64_t>> shape) {
            if (shape) {
              return self.generalizes(
                  other, sequenceToSmallVector<int64_t>(*shape), TARGET_SHAPE);
            }
            return self.generalizes(other, std::nullopt, TARGET_SHAPE);
          },
          py::arg("other"), py::arg("shape") = std::nullopt,
          "Returns True if the other layout is a special case of this one.\n"
          "\n"
          "In here, other is considered \"a special case\" when the set of "
          "vector register entries that represent a value in that layout is "
          "also the set of entries in which self stores the value. This is of "
          "course true for layouts that are equivalent, but it does not need "
          "to hold both ways. For example, a layout that implies the value "
          "does not change along an axis of the vector register is more "
          "general than the layout that picks a fixed starting point for the "
          "value and does not encode that assumption.\n"
          "\n"
          "The generalization relation is a non-strict partial order. You can "
          "think of it as a partial <= on vector layouts, but we don't "
          "overload Python operators since there's no clear way to decide "
          "where the bottom and top should be.\n"
          "\n"
          "Args:\n"
          "  other: The layout compared against self.\n"
          "  shape: An optional shape of the vector to which both layouts "
          "apply.\n"
          "    The generalization relation is larger than usual for some "
          "shapes. That is, if self.generalizes(other) then also "
          "self.generalizes(other, shape) for any shape, but that implication "
          "does not hold the other way around for some shapes.")
      .def(
          "equivalent_to",
          [](mlir::tpu::VectorLayout& self, mlir::tpu::VectorLayout& other,
             std::optional<Sequence<int64_t>> shape) {
            if (shape) {
              return self.equivalentTo(
                  other, sequenceToSmallVector<int64_t>(*shape), TARGET_SHAPE);
            }
            return self.equivalentTo(other, std::nullopt, TARGET_SHAPE);
          },
          py::arg("other"), py::arg("shape") = std::nullopt,
          "Returns True if the two layouts are equivalent.\n"
          "\n"
          "That is, when all potential vector entries where the value can be "
          "stored (there might be multiple choices for some layouts!) are "
          "equal in both self and other.\n"
          "\n"
          "Args:\n"
          "  other: The layout compared against self.\n"
          "  shape: An optional shape of the vector to which both layouts "
          "apply. More layouts are considered equivalent when the shape is "
          "specified. Also see the docstring of the generalizes method.")
      .def(
          "tile_data_bounds",
          [](mlir::tpu::VectorLayout& self, Sequence<int64_t> shape,
             Sequence<int64_t> ixs,
             std::variant<bool, py::typing::Tuple<bool, bool>>
                 allow_replicated) {
            std::visit(
                [&](auto ar) {
                  if constexpr (std::is_same_v<decltype(ar), bool>) {
                    return self.tileDataBounds(
                        getContextFromPyThreadContext(),
                        sequenceToSmallVector<int64_t>(shape),
                        sequenceToSmallVector<int64_t>(ixs), TARGET_SHAPE, ar);
                  } else {
                    return self.tileDataBounds(
                        getContextFromPyThreadContext(),
                        sequenceToSmallVector<int64_t>(shape),
                        sequenceToSmallVector<int64_t>(ixs), TARGET_SHAPE,
                        {ar[0].template cast<bool>(),
                         ar[1].template cast<bool>()});
                  }
                },
                allow_replicated);
          },
          py::arg("shape"), py::arg("ixs"), py::arg("allow_replicated") = false,
          "Returns the bounds of the given tile that hold useful data.\n"
          "\n"
          "Arguments:\n"
          "  full_shape: The shape of the full vector this layout applies to.\n"
          "  ixs: The indices into an array of tiles representing the full "
          "vector (see tile_array_shape for bounds) selecting the tile for "
          "which the bounds are queried.\n"
          "  allow_replicated: If False, no offset is allowed to be "
          "REPLICATED. If True, offsets are allowed to be REPLICATED, but the "
          "bounds will span the full dimension of the tile (i.e. potentially "
          "multiple repeats of the actual data).\n"
          "\n"
          "Returns:\n"
          "  A TargetTuple of slices, indicating the span of useful data "
          "within the tile selected by idx.")
      .def("__eq__", &mlir::tpu::VectorLayout::operator==);

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__tpu__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

  m.def("private_is_tiled_layout", [](MlirAttribute attr) {
    return mlirTPUAttributeIsATiledLayoutAttr(attr);
  });
  m.def("private_get_tiles", [](MlirAttribute attr) -> py::object {
    MlirAttribute encoded_tiles = mlirTPUTiledLayoutAttrGetTiles(attr);
    py::tuple py_tiles(mlirArrayAttrGetNumElements(encoded_tiles));
    for (intptr_t i = 0; i < mlirArrayAttrGetNumElements(encoded_tiles); ++i) {
      MlirAttribute tile = mlirArrayAttrGetElement(encoded_tiles, i);
      py::tuple py_tile(mlirDenseArrayGetNumElements(tile));
      for (intptr_t j = 0; j < mlirDenseArrayGetNumElements(tile); ++j) {
        py_tile[j] = mlirDenseI64ArrayGetElement(tile, j);
      }
      py_tiles[i] = py_tile;
    }
    return py_tiles;
  });
  m.def("private_has_communication", [](MlirOperation op) {
    bool has_communication;
    bool has_custom_barrier;
    mlirTPUAnalyzePotentialCommunication(op, &has_communication,
                                         &has_custom_barrier);
    return std::make_pair(has_communication, has_custom_barrier);
  });

  // TODO(apaszke): All of those should be upstreamed to MLIR Python bindings.
  m.def("private_replace_all_uses_with", [](MlirOperation op,
                                            std::vector<MlirValue> vals) {
    if (vals.size() != mlirOperationGetNumResults(op)) {
      throw py::value_error("length mismatch in replace_all_uses_with");
    }
    for (int i = 0; i < vals.size(); ++i) {
      mlirValueReplaceAllUsesOfWith(mlirOperationGetResult(op, i), vals[i]);
    }
  });
  m.def("private_replace_all_uses_except",
        [](MlirValue old, MlirValue new_val, MlirOperation except) {
          for (intptr_t i = 0; i < mlirOperationGetNumOperands(except); ++i) {
            if (mlirValueEqual(mlirOperationGetOperand(except, i), new_val)) {
              throw py::value_error("new val already used in except");
            }
          }
          mlirValueReplaceAllUsesOfWith(old, new_val);
          // Undo the replacement in the except op.
          for (intptr_t i = 0; i < mlirOperationGetNumOperands(except); ++i) {
            if (mlirValueEqual(mlirOperationGetOperand(except, i), new_val)) {
              mlirOperationSetOperand(except, i, old);
            }
          }
        });
  m.def("private_set_operand",
        [](MlirOperation op, int idx, MlirValue new_operand) {
          mlirOperationSetOperand(op, idx, new_operand);
        });
  m.def("private_set_operands", [](MlirOperation op,
                                   std::vector<MlirValue> new_operands) {
    mlirOperationSetOperands(op, new_operands.size(), new_operands.data());
  });
  m.def("private_has_no_memory_space", [](MlirType ty) {
    return mlirAttributeIsNull(mlirMemRefTypeGetMemorySpace(ty));
  });
  m.def("private_is_identity", [](MlirAttribute attr) {
    return mlirAffineMapIsIdentity(mlirAffineMapAttrGetValue(attr));
  });
  m.def("private_insert_argument",
        [](int index, MlirBlock block, MlirType type) -> MlirValue {
          return mlirBlockInsertArgument(
              block, index, type,
              mlirLocationUnknownGet(mlirTypeGetContext(type)));
        });
  m.def("private_set_arg_attr",
        [](MlirOperation op, unsigned i, std::string name, MlirAttribute attr) {
          mlirFuncSetArgAttr(
              op, i, mlirStringRefCreateFromCString(name.c_str()), attr);
        });
  m.def("private_move_all_regions", [](MlirOperation src, MlirOperation dst) {
    if (mlirOperationGetNumRegions(src) != mlirOperationGetNumRegions(dst)) {
      throw py::value_error(
          "Region counts do not match in src operation and dst operations");
    }
    for (intptr_t i = 0; i < mlirOperationGetNumRegions(src); ++i) {
      mlirRegionTakeBody(mlirOperationGetRegion(dst, i),
                         mlirOperationGetRegion(src, i));
    }
  });
}
