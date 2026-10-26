// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cuda/runtime/buffer.hpp"

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

auto Buffer::allocate(Program& program, Count size, ttx_retained* output)
    -> cuda_status {
  if (!size || !output) {
    return CUDA_INVALID;
  }

  CurrentContext current(program.get_context());
  CUdeviceptr address = 0;
  auto status = current.status;
  if (status == CUDA_SUCCESS) {
    status = cuMemAlloc(&address, size);
  }

  if (status != CUDA_SUCCESS) {
    return CUDA_DRIVER_FAILED;
  }

  static_assert(alignof(Buffer) <= Allocator::Global::allocation_alignment);
  auto memory = Allocator::Global::allocate(sizeof(Buffer));
  auto& buffer = Lifetime::construct<Buffer>(
      memory.get_data(), [&] { return Buffer(program, address, size); });
  *output = {&buffer, &retained_operations};
  return CUDA_SUCCESS_STATUS;
}

auto Buffer::get_address(void* context) -> U64 {
  return Data::cast<Buffer>(context)->address;
}

auto Buffer::get_size(void* context) -> Count {
  return Data::cast<Buffer>(context)->size;
}

auto Buffer::read(void* context, Count offset, U8* output, Count size)
    -> cuda_status {
  const auto& buffer = *Data::cast<Buffer>(context);
  if (offset > buffer.size || size > buffer.size - offset) {
    return CUDA_BOUNDS;
  }

  if (size && !output) {
    return CUDA_INVALID;
  }

  if (!size) {
    return CUDA_SUCCESS_STATUS;
  }

  CurrentContext current(buffer.program.get_context());
  if (current.status != CUDA_SUCCESS) {
    return CUDA_DRIVER_FAILED;
  }

  return cuMemcpyDtoH(output, buffer.address + offset, size) == CUDA_SUCCESS
             ? CUDA_SUCCESS_STATUS
             : CUDA_DRIVER_FAILED;
}

auto Buffer::write(void* context, Count offset, perimortem_view_bytes input)
    -> cuda_status {
  const auto& buffer = *Data::cast<Buffer>(context);
  if (offset > buffer.size || input.size > buffer.size - offset) {
    return CUDA_BOUNDS;
  }

  if (input.size && !input.data) {
    return CUDA_INVALID;
  }

  if (!input.size) {
    return CUDA_SUCCESS_STATUS;
  }

  CurrentContext current(buffer.program.get_context());
  if (current.status != CUDA_SUCCESS) {
    return CUDA_DRIVER_FAILED;
  }

  return cuMemcpyHtoD(buffer.address + offset, input.data, input.size) ==
                 CUDA_SUCCESS
             ? CUDA_SUCCESS_STATUS
             : CUDA_DRIVER_FAILED;
}

const ttx_abstract_ops Buffer::graph_operations = {
  supports, bind, resolve_route, visit_routes, get_state};

const ttx_retained_ops Buffer::retained_operations = {graph, release_context};

const ttx_retain_ops Buffer::retain_operations = {graph, acquire};
const cuda_buffer_ops Buffer::operations = {
  graph, get_address, get_size, read, write};

auto Buffer::graph(void* context) -> ttx_abstract {
  return {context, &graph_operations};
}

auto Buffer::supports(void* context, perimortem_uuid id)
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
                     Identity::Uuid(TTX_CUDA_BUFFER_HIGH, TTX_CUDA_BUFFER_LOW)
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

auto Buffer::bind(void* context, perimortem_uuid id, void* output)
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
    *Data::cast<cuda_buffer>(output) = {context, &operations};
  }

  return TTX_NEGOTIATION_SATISFIED;
}

auto Buffer::resolve_route(void* context, perimortem_view_bytes route)
    -> ttx_abstract {
  return ttx_none();
}

auto Buffer::visit_routes(void* context, ttx_graph_visitor visitor) -> void {}

auto Buffer::get_state(void* context) -> ttx_abstract_state {
  return TTX_ABSTRACT_PRESENT;
}

// Binding preserves the supplying release obligation. Only acquisition adds
// a reference and grants an independently released duration of use.
auto Buffer::acquire(void* context, ttx_retained* output)
    -> ttx_negotiation_status {
  auto& value = *Data::cast<Buffer>(context);
  if (value.references == Count(-1)) {
    return TTX_NEGOTIATION_REJECTED;
  }

  ++value.references;
  *output = {context, &retained_operations};
  return TTX_NEGOTIATION_SATISFIED;
}

auto Buffer::release_context(void* context) -> ttx_release_status {
  auto* value = Data::cast<Buffer>(context);
  if (--value->references) {
    return TTX_RELEASE_SUCCESS;
  }

  auto outcome = TTX_RELEASE_SUCCESS;
  if (cuCtxPushCurrent(value->program.get_context()) == CUDA_SUCCESS) {
    if (cuMemFree(value->address) != CUDA_SUCCESS) {
      outcome = TTX_RELEASE_CLEANUP_FAILED;
    }

    CUcontext previous;
    if (cuCtxPopCurrent(&previous) != CUDA_SUCCESS) {
      outcome = TTX_RELEASE_CLEANUP_FAILED;
    }
  } else {
    outcome = TTX_RELEASE_CLEANUP_FAILED;
  }

  if (value->program.release() != Policies::Retained::Status::Success) {
    outcome = TTX_RELEASE_CLEANUP_FAILED;
  }

  value->~Buffer();
  Allocator::Global::release(Data::cast<U8>(value));
  return outcome;
}
