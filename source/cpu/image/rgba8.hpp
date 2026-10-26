// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "toolchain/memory/lifetime.hpp"

#include "perimortem/memory/global/bytes.hpp"

#include "ttx/graph/capabilities/retain.hpp"

#include "compute/image/pixels.hpp"

namespace Cpu::Image {

// Rgba8 publishes a host image snapshot. Creation copies the supplied pixels
// into independent storage and returns one Retained acquisition. Every later
// acquisition shares that snapshot until its own release. The final release
// destroys its storage. Dimensions and byte count must agree exactly.
// The consumer serializes operations and acquisition changes on each image
// across all bindings and independent acquisitions.
class Rgba8 {
 public:
  static EXPORTED(CPU) auto
      create(U32 width, U32 height, Perimortem::Core::View::Bytes pixels)
          -> Perimortem::Core::Storage::
              Result<Ttx::Graph::Policies::Retained, compute_image_status>;

 private:
  friend Toolchain::Memory::Lifetime;
  Rgba8(U32 width, U32 height, Perimortem::Core::View::Bytes pixels)
      : size{width, height}, pixels(pixels) {}
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
  Perimortem::Memory::Global::Bytes pixels;
};

}  // namespace Cpu::Image
