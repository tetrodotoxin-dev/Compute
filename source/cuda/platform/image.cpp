// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cuda/platform/image.hpp"

#include "perimortem/memory/allocator/global.hpp"

#include "ttx/graph/policies/none.h"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Toolchain::Memory;
using namespace Ttx::Graph;
using namespace Compute::Image;
using namespace Cuda::Platform;
using namespace Cuda::Contracts;

auto Image::create(
    compute_image_dimensions dimensions,
    Policies::Retained buffer)
    -> Storage::Result<Policies::Retained, compute_image_status> {
  return buffer.get_abstract().bind<Buffer>().visit(
      [&](Buffer api)
          -> Storage::Result<Policies::Retained, compute_image_status> {
        const U64 count = U64(dimensions.width) * dimensions.height;
        if (!count || count > Count(-1) / 4 || api.get_size() != count * 4) {
          buffer.release();
          return COMPUTE_IMAGE_INVALID;
        }

        static_assert(
            alignof(Image) <= Allocator::Global::allocation_alignment);
        auto memory = Allocator::Global::allocate(sizeof(Image));
        auto& image = Lifetime::construct<Image>(
            memory.get_data(), [&] { return Image(dimensions, buffer, api); });
        return Policies::Retained({&image, &retained_operations});
      },

      [&]() -> Storage::Result<Policies::Retained, compute_image_status> {
        buffer.release();
        return COMPUTE_IMAGE_INVALID;
      });
}

auto Image::graph(void* context) -> ttx_abstract {
  return {context, &graph_operations};
}

auto Image::supports(void* context, perimortem_uuid id)
    -> ttx_negotiation_status {
  if (!id.high && id.low <= 1) {
    return TTX_NEGOTIATION_REJECTED;
  }

  const Identity::Uuid contract(id);
  return contract == Abstract::contract_id || contract == Pixels::contract_id ||
                 contract == Capabilities::Retain::contract_id ||
                 contract == Policies::Retained::contract_id
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

auto Image::bind(void* context, perimortem_uuid id, void* output)
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

auto Image::route(void* context, perimortem_view_bytes route) -> ttx_abstract {
  return ttx_none();
}

auto Image::visit(void* context, ttx_graph_visitor visitor) -> void {}
auto Image::state(void* context) -> ttx_abstract_state {
  return TTX_ABSTRACT_PRESENT;
}

auto Image::dimensions(void* context) -> compute_image_dimensions {
  return Data::cast<Image>(context)->size;
}

auto Image::read(void* context, U8* output, Count capacity)
    -> compute_image_status {
  const auto& image = *Data::cast<Image>(context);
  const Count count = Count(image.size.width) * image.size.height * 4;
  if (!output || capacity < count) {
    return COMPUTE_IMAGE_INVALID;
  }

  return image.api.read(0, Access::Bytes(*output, count)) ==
                 ExecutionStatus::Success
             ? COMPUTE_IMAGE_SUCCESS
             : COMPUTE_IMAGE_FAILED;
}

auto Image::retain(void* context, ttx_retained* output)
    -> ttx_negotiation_status {
  auto& image = *Data::cast<Image>(context);
  if (image.references == Count(-1)) {
    return TTX_NEGOTIATION_REJECTED;
  }

  ++image.references;
  *output = {context, &retained_operations};
  return TTX_NEGOTIATION_SATISFIED;
}

auto Image::release(void* context) -> ttx_release_status {
  auto* image = Data::cast<Image>(context);
  if (--image->references) {
    return TTX_RELEASE_SUCCESS;
  }

  const auto outcome = image->buffer.release();
  image->~Image();
  Allocator::Global::release(Data::cast<U8>(image));
  return ttx_release_status(outcome);
}

const ttx_abstract_ops Image::graph_operations = {
  supports, bind, route, visit, state};

const compute_pixels_ops Image::pixel_operations = {graph, dimensions, read};
const ttx_retain_ops Image::retain_operations = {graph, retain};
const ttx_retained_ops Image::retained_operations = {graph, release};
