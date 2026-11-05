// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "perimortem/core/storage/option.hpp"

#include "cuda/contracts/buffer_extension.h"
#include "ttx/graph/abstract.hpp"

namespace Cuda::Contracts {

// BufferExtension requests a device address for one CUDA driver context. The
// public Compute Buffer continues to own storage access and final release.
// Construction stores the complete extension record supplied by its binding.
class BufferExtension {
 public:
  using Api = cuda_buffer_extension;
  static constexpr auto contract_id = Perimortem::Core::Identity::Uuid(
      CUDA_BUFFER_EXTENSION_ID_HIGH,
      CUDA_BUFFER_EXTENSION_ID_LOW);

  explicit constexpr BufferExtension(Api api) : api(api) {}

  auto address(void* driver_context) const
      -> Perimortem::Core::Storage::Option<U64> {
    U64 result = 0;
    if (api.operations->address(api.context, driver_context, &result) !=
        TTX_NEGOTIATION_SATISFIED) {
      return {};
    }

    return result;
  }

 private:
  Api api;
};

}  // namespace Cuda::Contracts
