// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "perimortem/core/access/bytes.hpp"

#include "ttx/graph/abstract.hpp"

#include "compute/image/pixels.h"

namespace Compute::Image {

// Pixels reads an RGBA8 snapshot through its supplied operations. The provider
// decides where pixels live and how the read is implemented. This interface
// preserves the API record's existing access duration.
class Pixels {
 public:
  using Api = compute_pixels;
  static constexpr auto contract_id = Perimortem::Core::Identity::Uuid(
      COMPUTE_PIXELS_ID_HIGH,
      COMPUTE_PIXELS_ID_LOW);

  explicit constexpr Pixels(Api api) : api(api) {}
  constexpr auto get_abi() const -> Api { return api; }
  auto get_abstract() const -> Ttx::Graph::Abstract {
    return Ttx::Graph::Abstract(api.operations->get_abstract(api.context));
  }

  auto dimensions() const -> compute_image_dimensions {
    return api.operations->dimensions(api.context);
  }

  auto read(Perimortem::Core::Access::Bytes output) const
      -> compute_image_status {
    return api.operations->read(
        api.context, output.get_data(), output.get_size());
  }

 private:
  Api api;
};

}  // namespace Compute::Image
