// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "ttx/data/form/storage.hpp"
#include "ttx/graph/abstract.hpp"

#include "cuda/contracts/execution_status.hpp"
#include "cuda/contracts/kernel.h"

namespace Cuda::Contracts {

// Kernel launches a prepared entry and waits for completion. Argument storage
// and referenced device allocations must remain accessible until launch
// returns. The consumer serializes launches on the same Kernel.
class Kernel {
 public:
  using Api = cuda_kernel;
  static constexpr auto contract_id = Perimortem::Core::Identity::Uuid(
      TTX_CUDA_KERNEL_HIGH,
      TTX_CUDA_KERNEL_LOW);

  // Explicit construction accepts the complete C record established by its
  // supplying contract. Copies keep that context and table paired and share
  // the original access duration.
  explicit constexpr Kernel(Api api) : api(api) {}
  constexpr auto get_abi() const -> Api { return api; }

  // The returned Abstract represents the graph location that supplied this
  // Kernel API. The provider supplies its complete context and operations.
  auto get_abstract() const -> Ttx::Graph::Abstract {
    return Ttx::Graph::Abstract(api.operations->get_abstract(api.context));
  }

  auto launch(cuda_launch geometry, Ttx::Data::Form::Storage arguments) const
      -> ExecutionStatus {
    return ExecutionStatus(
        api.operations->launch(api.context, geometry, arguments.get_abi()));
  }

 private:
  Api api;
};

}  // namespace Cuda::Contracts
