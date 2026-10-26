// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cpu/image/rgba8.hpp"

#include "toolchain/memory/lifetime.hpp"

#include "perimortem/memory/allocator/global.hpp"

#include "ttx/graph/policies/none.h"
#include "ttx/graph/policies/unknown.h"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Toolchain::Memory;
using namespace Ttx::Graph;
using namespace Compute::Image;
using namespace Cpu::Image;

auto Rgba8::create(U32 width, U32 height, View::Bytes pixels)
    -> Storage::Result<Policies::Retained, compute_image_status> {
  const U64 count = U64(width) * height;
  if (!width || !height || count > Count(-1) / 4 ||
      pixels.get_size() != count * 4) {
    return COMPUTE_IMAGE_INVALID;
  }

  static_assert(alignof(Rgba8) <= Allocator::Global::allocation_alignment);
  auto memory = Allocator::Global::allocate(sizeof(Rgba8));
  auto& image = Lifetime::construct<Rgba8>(
      memory.get_data(), [&] { return Rgba8(width, height, pixels); });
  return Policies::Retained({&image, &retained_operations});
}

auto Rgba8::graph(void* context) -> ttx_abstract {
  return {context, &graph_operations};
}

auto Rgba8::supports(void* context, perimortem_uuid id)
    -> ttx_negotiation_status {
  const Identity::Uuid contract(id);
  if (!id.high && id.low <= 1) {
    return TTX_NEGOTIATION_REJECTED;
  }

  return contract == Abstract::contract_id || contract == Pixels::contract_id ||
                 contract == Capabilities::Retain::contract_id ||
                 contract == Policies::Retained::contract_id
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

auto Rgba8::bind(void* context, perimortem_uuid id, void* output)
    -> ttx_negotiation_status {
  const auto status = supports(context, id);
  if (status != TTX_NEGOTIATION_SATISFIED) {
    return status;
  }

  const Identity::Uuid contract(id);
  if (contract == Pixels::contract_id) {
    *Data::cast<compute_pixels>(output) = {context, &pixel_operations};
  } else if (contract == Capabilities::Retain::contract_id) {
    *Data::cast<ttx_retain>(output) = {context, &retain_operations};
  } else if (contract == Policies::Retained::contract_id) {
    *Data::cast<ttx_retained>(output) = {context, &retained_operations};
  } else {
    *Data::cast<ttx_abstract>(output) = graph(context);
  }

  return TTX_NEGOTIATION_SATISFIED;
}

auto Rgba8::route(void* context, perimortem_view_bytes route) -> ttx_abstract {
  return ttx_none();
}

auto Rgba8::visit(void* context, ttx_graph_visitor visitor) -> void {}
auto Rgba8::state(void* context) -> ttx_abstract_state {
  return TTX_ABSTRACT_PRESENT;
}

auto Rgba8::dimensions(void* context) -> compute_image_dimensions {
  return Data::cast<Rgba8>(context)->size;
}

auto Rgba8::read(void* context, U8* output, Count capacity)
    -> compute_image_status {
  const auto& image = *Data::cast<Rgba8>(context);
  if (!output || capacity < image.pixels.get_size()) {
    return COMPUTE_IMAGE_INVALID;
  }

  Data::copy(output, image.pixels.get_data(), image.pixels.get_size());
  return COMPUTE_IMAGE_SUCCESS;
}

auto Rgba8::retain(void* context, ttx_retained* output)
    -> ttx_negotiation_status {
  auto& image = *Data::cast<Rgba8>(context);
  if (image.references == Count(-1)) {
    return TTX_NEGOTIATION_REJECTED;
  }

  ++image.references;
  *output = {context, &retained_operations};
  return TTX_NEGOTIATION_SATISFIED;
}

auto Rgba8::release(void* context) -> ttx_release_status {
  auto* image = Data::cast<Rgba8>(context);
  if (--image->references == 0) {
    image->~Rgba8();
    Allocator::Global::release(Data::cast<U8>(image));
  }

  return TTX_RELEASE_SUCCESS;
}

const ttx_abstract_ops Rgba8::graph_operations = {
  supports, bind, route, visit, state};

const compute_pixels_ops Rgba8::pixel_operations = {graph, dimensions, read};
const ttx_retain_ops Rgba8::retain_operations = {graph, retain};
const ttx_retained_ops Rgba8::retained_operations = {graph, release};
