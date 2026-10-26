// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "perimortem/core/storage/result.hpp"

#include "ttx/graph/abstract.hpp"

#include "cuda/platform/device_image.h"

namespace Cuda::Platform {

// DeviceImage requests ordered read access through the image provider. Its
// device address is usable only under the context, stream and retention
// obligations of device_image.h.
class DeviceImage {
 public:
  using Api = cuda_device_image;
  static constexpr auto contract_id = Perimortem::Core::Identity::Uuid(
      CUDA_DEVICE_IMAGE_ID_HIGH,
      CUDA_DEVICE_IMAGE_ID_LOW);
  explicit constexpr DeviceImage(Api api) : api(api) {}
  auto get_abstract() const -> Ttx::Graph::Abstract {
    return Ttx::Graph::Abstract(api.operations->get_abstract(api.context));
  }

  auto address(CUcontext context, CUstream stream) const
      -> Perimortem::Core::Storage::Result<U64, compute_image_status> {
    U64 output = 0;
    const auto status =
        api.operations->address(api.context, context, stream, &output);
    if (status != COMPUTE_IMAGE_SUCCESS) {
      return status;
    }

    return output;
  }

  constexpr auto get_abi() const -> Api { return api; }
  auto get_context() const -> CUcontext {
    return api.operations->get_context(api.context);
  }

 private:
  Api api;
};

}  // namespace Cuda::Platform
