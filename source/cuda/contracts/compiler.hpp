// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "perimortem/memory/global/bytes.hpp"

#include "ttx/graph/abstract.hpp"
#include "ttx/graph/policies/retained.hpp"

#include "cuda/contracts/compiler.h"
#include "cuda/contracts/execution_status.hpp"

namespace Cuda::Contracts {

// Compiler builds a retained Program from caller supplied source. The caller
// releases each successful result after its final use. Diagnostics replace the
// supplied Bytes during the synchronous compilation call.
class Compiler {
 public:
  using Api = cuda_compiler;
  static constexpr auto contract_id = Perimortem::Core::Identity::Uuid(
      TTX_CUDA_COMPILER_HIGH,
      TTX_CUDA_COMPILER_LOW);

  // Explicit construction accepts the complete C record established by its
  // supplying contract. Copies keep that context and table paired and share
  // the original access duration.
  explicit constexpr Compiler(Api api) : api(api) {}
  constexpr auto get_abi() const -> Api { return api; }

  // The returned Abstract represents the graph location that supplied this
  // Compiler API. The provider supplies its complete context and operations.
  auto get_abstract() const -> Ttx::Graph::Abstract {
    return Ttx::Graph::Abstract(api.operations->get_abstract(api.context));
  }

  auto compile(
      cuda_compile_request request,
      Perimortem::Memory::Global::Bytes& errors) const -> Perimortem::Core::
      Storage::Result<Ttx::Graph::Policies::Retained, ExecutionStatus> {
    errors.clear();

    ttx_retained output{};
    const auto status = api.operations->compile(
        api.context, request, {&errors, diagnostic}, &output);
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
