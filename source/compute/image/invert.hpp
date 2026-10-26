// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "ttx/graph/policies/retained.hpp"

#include "compute/image/invert.h"

namespace Compute::Image {

// Invert accepts an image Abstract and returns a separately retained snapshot
// with complemented RGB and preserved alpha. The complete contract and input
// lifetime are specified by the C operation in invert.h.
class Invert {
 public:
  using Api = compute_invert;
  static constexpr auto contract_id = Perimortem::Core::Identity::Uuid(
      COMPUTE_INVERT_ID_HIGH,
      COMPUTE_INVERT_ID_LOW);

  explicit constexpr Invert(Api api) : api(api) {}
  auto get_abstract() const -> Ttx::Graph::Abstract {
    return Ttx::Graph::Abstract(api.operations->get_abstract(api.context));
  }

  auto apply(Ttx::Graph::Abstract image) const -> Perimortem::Core::Storage::
      Result<Ttx::Graph::Policies::Retained, compute_image_status> {
    ttx_retained output{};
    const auto status =
        api.operations->apply(api.context, image.get_abi(), &output);
    if (status != COMPUTE_IMAGE_SUCCESS) {
      return status;
    }

    return Ttx::Graph::Policies::Retained(output);
  }

 private:
  Api api;
};

}  // namespace Compute::Image
