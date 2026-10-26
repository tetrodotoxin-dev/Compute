// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "toolchain/memory/lifetime.hpp"

#include "ttx/graph/capabilities/retain.hpp"

#include "compute/image/pixels.hpp"
#include "cuda/contracts/buffer.hpp"

namespace Cuda::Platform {

// Image publishes CUDA device storage through Compute's Pixels contract.
// Creation consumes the supplied Buffer acquisition on every outcome. A
// successful creation returns a separate image acquisition that keeps the
// buffer available until final release. Consumers read an RGBA8 snapshot
// through Pixels while the buffer's address remains private.
// The caller completes all writes before creation and keeps the buffer bytes
// unchanged through every image acquisition, including through other Buffer
// acquisitions that refer to the same storage.
class Image {
 public:
  static EXPORTED(CUDA) auto create(
      compute_image_dimensions dimensions,
      Ttx::Graph::Policies::Retained buffer) -> Perimortem::Core::Storage::
      Result<Ttx::Graph::Policies::Retained, compute_image_status>;

 private:
  friend Toolchain::Memory::Lifetime;
  Image(
      compute_image_dimensions dimensions,
      Ttx::Graph::Policies::Retained buffer,
      Contracts::Buffer api)
      : size(dimensions), buffer(buffer), api(api) {}
  static auto graph(void* context) -> ttx_abstract;
  static auto supports(void* context, perimortem_uuid id)
      -> ttx_negotiation_status;
  static auto bind(void* context, perimortem_uuid id, void* output)
      -> ttx_negotiation_status;
  static auto route(void* context, perimortem_view_bytes route) -> ttx_abstract;
  static auto visit(void* context, ttx_graph_visitor visitor) -> void;
  static auto state(void* context) -> ttx_abstract_state;
  static auto dimensions(void* context) -> compute_image_dimensions;
  static auto read(void* context, U8* output, Count capacity)
      -> compute_image_status;
  static auto retain(void* context, ttx_retained* output)
      -> ttx_negotiation_status;
  static auto release(void* context) -> ttx_release_status;
  static const ttx_abstract_ops graph_operations;
  static const compute_pixels_ops pixel_operations;
  static const ttx_retain_ops retain_operations;
  static const ttx_retained_ops retained_operations;
  Count references = 1;
  compute_image_dimensions size;
  Ttx::Graph::Policies::Retained buffer;
  Contracts::Buffer api;
};

}  // namespace Cuda::Platform
