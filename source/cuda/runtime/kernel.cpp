// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cuda/runtime/kernel.hpp"

#include "toolchain/memory/lifetime.hpp"

#include "perimortem/core/view/null_terminated.hpp"
#include "perimortem/memory/allocator/global.hpp"

#include "ttx/graph/policies/none.h"
#include "ttx/graph/policies/unknown.h"

#include "cuda/runtime/current_context.hpp"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Toolchain::Memory;
using namespace Cuda::Runtime;
using namespace Ttx::Graph;
using namespace Ttx::Data;

Kernel::Kernel(
    Program& program,
    CUfunction function,
    const ttx_representation& frame,
    Global::Vector<Count> offsets)
    : program(program),
      function(function),
      form(Form::Representation(frame).get_bytes()),
      offsets(Data::take(offsets)) {
  program.retain();
  arguments.resize(this->offsets.get_size());
}

Kernel::~Kernel() {
  program.release();
}

auto Kernel::prepare(
    Program& program,
    perimortem_view_bytes entry,
    const ttx_representation& frame,
    const cuda_argument* parameters,
    Count count,
    cuda_diagnostics diagnostics,
    ttx_retained* output) -> cuda_status {
  if (!output || !entry.data || !entry.size || entry.size == Count(-1) ||
      !frame.data || !frame.size || (count && !parameters)) {
    return CUDA_INVALID;
  }

  for (Count i = 0; i < entry.size; ++i) {
    if (entry.data[i] == 0) {
      return CUDA_INVALID;
    }
  }

  for (Count i = 0; i < count; ++i) {
    if (!parameters[i].representation || !parameters[i].representation->data ||
        !parameters[i].representation->size) {
      return CUDA_INVALID;
    }
  }

  CurrentContext current(program.get_context());
  if (current.status != CUDA_SUCCESS) {
    return CUDA_DRIVER_FAILED;
  }

  Global::Bytes name;
  name.forgetful_resize(entry.size + 1);
  Data::copy(name.get_access().get_data(), entry.data, entry.size);
  name.get_access().get_data()[entry.size] = 0;

  CUfunction function = nullptr;
  if (cuModuleGetFunction(
          &function, program.get_module(),
          Data::cast<const char>(name.get_data())) != CUDA_SUCCESS) {
    if (diagnostics.callback) {
      diagnostics.callback(
          diagnostics.callback_context, "CUDA entry was not found."_view);
    }

    return CUDA_UNAVAILABLE;
  }

  CppSize actual_count = 0;
  if (cuFuncGetParamCount(function, &actual_count) != CUDA_SUCCESS) {
    return CUDA_DRIVER_FAILED;
  }

  if (actual_count != count) {
    return CUDA_INCOMPATIBLE;
  }

  // CUDA reports occupied parameter sizes. Canonical forms establish host
  // placement and frame agreement, while the source author establishes the
  // corresponding device types. Equal parameter sizes alone cannot prove them.
  const Form::Representation host_frame(frame);
  Global::Vector<Count> offsets;
  Global::Vector<Form::Representation::Position> expected;
  Count end = 0;
  for (Count i = 0; i < count; ++i) {
    const auto& argument = parameters[i];
    const Form::Representation representation(*argument.representation);
    const auto extent = representation.get_extent();
    CppSize device_offset = 0, size = 0;
    if (cuFuncGetParamInfo(function, i, &device_offset, &size) !=
        CUDA_SUCCESS) {
      return CUDA_DRIVER_FAILED;
    }

    if (size != extent || argument.offset < end ||
        argument.offset > host_frame.get_extent() ||
        extent > host_frame.get_extent() - argument.offset ||
        argument.offset % representation.get_alignment()) {
      return CUDA_INCOMPATIBLE;
    }

    const auto status =
        representation.visit([&](Form::Representation::Position position) {
          auto value = position.get_abi();
          value.offset += argument.offset;
          expected.insert(Form::Representation::Position(value));
          return Status::Success;
        });

    if (status != Status::Success) {
      return CUDA_INCOMPATIBLE;
    }

    end = argument.offset + extent;
    offsets.insert(argument.offset);
  }

  // Both walks advance in byte order, covering every occupied frame position
  // exactly once. Padding remains outside the formal argument list.
  Count next = 0;
  const auto agreement =
      host_frame.visit([&](Form::Representation::Position position) {
        if (next == expected.get_size()) {
          return Status::Incompatible;
        }

        const auto& wanted = expected.get_data()[next];
        ++next;
        return position.get_offset() == wanted.get_offset() &&
                       position.compatible(wanted)
                   ? Status::Success
                   : Status::Incompatible;
      });

  if (agreement != Status::Success || next != expected.get_size()) {
    return CUDA_INCOMPATIBLE;
  }

  static_assert(alignof(Kernel) <= Allocator::Global::allocation_alignment);
  auto memory = Allocator::Global::allocate(sizeof(Kernel));
  auto& kernel = Lifetime::construct<Kernel>(memory.get_data(), [&] {
    return Kernel(program, function, frame, Data::take(offsets));
  });

  *output = {&kernel, &retained_operations};
  return CUDA_SUCCESS_STATUS;
}

auto Kernel::launch(cuda_launch geometry, ttx_storage supplied) -> cuda_status {
  if (!supplied.representation || !supplied.representation->data) {
    return CUDA_INVALID;
  }

  const Form::Representation required(form.get_view());
  if (!required.compatible(*supplied.representation)) {
    return CUDA_INCOMPATIBLE;
  }

  const auto status = ttx_storage_check(supplied);
  if (status != TTX_DATA_SUCCESS) {
    return status == TTX_DATA_BOUNDS ? CUDA_BOUNDS : CUDA_INVALID;
  }

  for (Count i = 0; i < offsets.get_size(); ++i) {
    arguments.get_data()[i] = supplied.data + offsets.get_data()[i];
  }

  CurrentContext current(program.get_context());
  if (current.status != CUDA_SUCCESS) {
    return CUDA_DRIVER_FAILED;
  }

  auto driver = cuLaunchKernel(
      function, geometry.grid_x, geometry.grid_y, geometry.grid_z,
      geometry.block_x, geometry.block_y, geometry.block_z,
      geometry.shared_bytes, nullptr, arguments.get_data(), nullptr);
  // Frame and allocation lifetimes are guaranteed only through this call.
  // Complete submitted work before returning control to their consumer.
  if (driver == CUDA_SUCCESS) {
    driver = cuCtxSynchronize();
  }

  return driver == CUDA_SUCCESS ? CUDA_SUCCESS_STATUS : CUDA_DRIVER_FAILED;
}

auto Kernel::launch_context(
    void* context,
    cuda_launch geometry,
    ttx_storage arguments) -> cuda_status {
  return Data::cast<Kernel>(context)->launch(geometry, arguments);
}

const ttx_abstract_ops Kernel::graph_operations = {
  supports, bind, resolve_route, visit_routes, get_state};

const ttx_retained_ops Kernel::retained_operations = {graph, release_context};

const ttx_retain_ops Kernel::retain_operations = {graph, acquire};
const cuda_kernel_ops Kernel::operations = {graph, launch_context};

auto Kernel::graph(void* context) -> ttx_abstract {
  return {context, &graph_operations};
}

auto Kernel::supports(void* context, perimortem_uuid id)
    -> ttx_negotiation_status {
  const Identity::Uuid contract(id);
  if (contract == Identity::Uuid(TTX_NONE_ID_HIGH, TTX_NONE_ID_LOW) ||
      contract == Identity::Uuid(TTX_UNKNOWN_ID_HIGH, TTX_UNKNOWN_ID_LOW)) {
    return TTX_NEGOTIATION_REJECTED;
  }

  return contract == Abstract::contract_id ||
                 contract ==
                     Identity::Uuid(TTX_RETAIN_ID_HIGH, TTX_RETAIN_ID_LOW) ||
                 contract == Identity::Uuid(
                                 TTX_RETAINED_ID_HIGH, TTX_RETAINED_ID_LOW) ||
                 contract ==
                     Identity::Uuid(TTX_CUDA_KERNEL_HIGH, TTX_CUDA_KERNEL_LOW)
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

auto Kernel::bind(void* context, perimortem_uuid id, void* output)
    -> ttx_negotiation_status {
  const auto status = supports(context, id);
  if (status != TTX_NEGOTIATION_SATISFIED) {
    return status;
  }

  const Identity::Uuid contract(id);
  if (contract == Abstract::contract_id) {
    *Data::cast<ttx_abstract>(output) = graph(context);
  } else if (
      contract == Identity::Uuid(TTX_RETAIN_ID_HIGH, TTX_RETAIN_ID_LOW)) {
    *Data::cast<ttx_retain>(output) = {context, &retain_operations};
  } else if (
      contract == Identity::Uuid(TTX_RETAINED_ID_HIGH, TTX_RETAINED_ID_LOW)) {
    *Data::cast<ttx_retained>(output) = {context, &retained_operations};
  } else {
    *Data::cast<cuda_kernel>(output) = {context, &operations};
  }

  return TTX_NEGOTIATION_SATISFIED;
}

auto Kernel::resolve_route(void* context, perimortem_view_bytes route)
    -> ttx_abstract {
  return ttx_none();
}

auto Kernel::visit_routes(void* context, ttx_graph_visitor visitor) -> void {}

auto Kernel::get_state(void* context) -> ttx_abstract_state {
  return TTX_ABSTRACT_PRESENT;
}

// Binding preserves the supplying release obligation. Only acquisition adds
// a reference and grants an independently released duration of use.
auto Kernel::acquire(void* context, ttx_retained* output)
    -> ttx_negotiation_status {
  auto& value = *Data::cast<Kernel>(context);
  if (value.references == Count(-1)) {
    return TTX_NEGOTIATION_REJECTED;
  }

  ++value.references;
  *output = {context, &retained_operations};
  return TTX_NEGOTIATION_SATISFIED;
}

auto Kernel::release_context(void* context) -> void {
  auto* value = Data::cast<Kernel>(context);
  if (--value->references == 0) {
    value->~Kernel();
    Allocator::Global::release(Data::cast<U8>(value));
  }
}
