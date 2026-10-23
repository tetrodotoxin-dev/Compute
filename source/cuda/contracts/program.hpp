// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "perimortem/memory/global/bytes.hpp"

#include "ttx/data/form/representation.hpp"
#include "ttx/graph/abstract.hpp"
#include "ttx/graph/policies/retained.hpp"

#include "cuda/contracts/execution_status.hpp"
#include "cuda/contracts/program.h"

namespace Cuda::Contracts {

// Program creates retained Kernels and Buffers. Their resource lifetimes remain
// independent of the result that supplied this Program binding. The caller
// serializes operations on related resources and releases each result once.
class Program {
 public:
  using Api = cuda_program;
  static constexpr auto contract_id = Perimortem::Core::Identity::Uuid(
      TTX_CUDA_PROGRAM_HIGH,
      TTX_CUDA_PROGRAM_LOW);

  // Explicit construction accepts the complete C record established by its
  // supplying contract. Copies keep that context and table paired and share
  // the original access duration.
  explicit constexpr Program(Api api) : api(api) {}
  constexpr auto get_abi() const -> Api { return api; }

  // The returned Abstract represents the graph location that supplied this
  // Program API. The provider supplies its complete context and operations.
  auto get_abstract() const -> Ttx::Graph::Abstract {
    return Ttx::Graph::Abstract(api.operations->get_abstract(api.context));
  }

  auto prepare(
      Perimortem::Core::View::Bytes entry,
      const Ttx::Data::Form::Representation& frame,
      Perimortem::Core::View::Vector<cuda_argument> arguments,
      Perimortem::Memory::Global::Bytes& errors) const -> Perimortem::Core::
      Storage::Result<Ttx::Graph::Policies::Retained, ExecutionStatus> {
    errors.clear();

    ttx_retained output{};
    const auto status = api.operations->prepare(
        api.context, entry, &frame.get_abi(), arguments.get_data(),
        arguments.get_size(), {&errors, diagnostic}, &output);
    if (status != CUDA_SUCCESS_STATUS) {
      return ExecutionStatus(status);
    }

    return Ttx::Graph::Policies::Retained(output);
  }

  auto allocate(Count size) const -> Perimortem::Core::Storage::
      Result<Ttx::Graph::Policies::Retained, ExecutionStatus> {
    ttx_retained output{};
    const auto status = api.operations->allocate(api.context, size, &output);
    if (status != CUDA_SUCCESS_STATUS) {
      return ExecutionStatus(status);
    }

    return Ttx::Graph::Policies::Retained(output);
  }

 private:
  static auto diagnostic(void* callback_context, perimortem_view_bytes text)
      -> void {
    auto& errors =
        *Perimortem::Core::Data::cast<Perimortem::Memory::Global::Bytes>(
            callback_context);
    errors = Perimortem::Core::View::Bytes(text);
  }

  Api api;
};

}  // namespace Cuda::Contracts
