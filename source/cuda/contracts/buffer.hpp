// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "perimortem/core/access/bytes.hpp"

#include "ttx/graph/abstract.hpp"

#include "cuda/contracts/buffer.h"
#include "cuda/contracts/execution_status.hpp"

namespace Cuda::Contracts {

// Buffer supplies a device address and bounded host transfers. The address
// remains usable while the supplying Retained result remains accessible.
class Buffer {
 public:
  using Api = cuda_buffer;
  static constexpr auto contract_id = Perimortem::Core::Identity::Uuid(
      TTX_CUDA_BUFFER_HIGH,
      TTX_CUDA_BUFFER_LOW);

  // Explicit construction accepts the complete C record established by its
  // supplying contract. Copies keep that context and table paired and share
  // the original access duration.
  explicit constexpr Buffer(Api api) : api(api) {}
  constexpr auto get_abi() const -> Api { return api; }

  // The returned Abstract represents the graph location that supplied this
  // Buffer API. The provider supplies its complete context and operations.
  auto get_abstract() const -> Ttx::Graph::Abstract {
    return Ttx::Graph::Abstract(api.operations->get_abstract(api.context));
  }

  auto get_address() const -> U64 {
    return api.operations->address(api.context);
  }

  auto get_size() const -> Count { return api.operations->size(api.context); }

  auto read(Count offset, Perimortem::Core::Access::Bytes output) const
      -> ExecutionStatus {
    return ExecutionStatus(api.operations->read(
        api.context, offset, output.get_data(), output.get_size()));
  }

  auto write(Count offset, Perimortem::Core::View::Bytes input) const
      -> ExecutionStatus {
    return ExecutionStatus(api.operations->write(api.context, offset, input));
  }

 private:
  Api api;
};

}  // namespace Cuda::Contracts
