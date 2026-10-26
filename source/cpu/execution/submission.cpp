// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cpu/execution/submission.hpp"

#include "perimortem/core/view/null_terminated.hpp"
#include "perimortem/memory/allocator/global.hpp"
#include "perimortem/memory/global/bytes.hpp"

#include "ttx/graph/policies/none.h"

#include "compute/execution/commands.hpp"
#include "compute/execution/dependency.hpp"
#include "compute/image/invert.hpp"
#include "compute/image/pixels.hpp"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Toolchain::Memory;
using namespace Ttx::Graph;
using namespace Compute::Image;

using namespace Cpu::Execution;
using namespace Compute::Execution;
using namespace Ttx::Semantic;

auto Submission::create() -> Policies::Retained {
  static_assert(alignof(Submission) <= Allocator::Global::allocation_alignment);
  auto memory = Allocator::Global::allocate(sizeof(Submission));
  auto& commands = Lifetime::construct<Submission>(
      memory.get_data(), [] { return Submission(); });
  return Policies::Retained({&commands, &retained_operations});
}

auto Submission::graph(void* context) -> ttx_abstract {
  return {context, &graph_operations};
}

auto Submission::supports(void* context, perimortem_uuid id)
    -> ttx_negotiation_status {
  if (!id.high && id.low <= 1) {
    return TTX_NEGOTIATION_REJECTED;
  }

  const Identity::Uuid contract(id);
  return contract == Abstract::contract_id ||
                 contract == Commands::contract_id ||
                 contract == Invert::contract_id ||
                 contract == Capabilities::Retain::contract_id
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

auto Submission::bind(void* context, perimortem_uuid id, void* output)
    -> ttx_negotiation_status {
  const auto status = supports(context, id);
  if (status != TTX_NEGOTIATION_SATISFIED) {
    return status;
  }

  const Identity::Uuid contract(id);
  if (contract == Commands::contract_id) {
    *Data::cast<compute_commands>(output) = {context, &command_operations};
  } else if (contract == Invert::contract_id) {
    *Data::cast<compute_invert>(output) = {context, &invert_operations};
  } else if (contract == Capabilities::Retain::contract_id) {
    *Data::cast<ttx_retain>(output) = {context, &retain_operations};
  } else {
    *Data::cast<ttx_abstract>(output) = graph(context);
  }

  return TTX_NEGOTIATION_SATISFIED;
}

auto Submission::route(void* context, perimortem_view_bytes route)
    -> ttx_abstract {
  return ttx_none();
}

auto Submission::visit(void* context, ttx_graph_visitor visitor) -> void {}
auto Submission::state(void* context) -> ttx_abstract_state {
  return TTX_ABSTRACT_PRESENT;
}

auto Submission::retain(void* context, ttx_retained* output)
    -> ttx_negotiation_status {
  auto& commands = *Data::cast<Submission>(context);
  if (commands.references == Count(-1)) {
    return TTX_NEGOTIATION_REJECTED;
  }

  ++commands.references;
  *output = {context, &retained_operations};
  return TTX_NEGOTIATION_SATISFIED;
}

auto Submission::release(void* context) -> ttx_release_status {
  auto* commands = Data::cast<Submission>(context);
  if (--commands->references) {
    return TTX_RELEASE_SUCCESS;
  }

  auto outcome = TTX_RELEASE_SUCCESS;
  for (Count i = 0; i < commands->images.get_size(); ++i) {
    auto* image = commands->images.get_data()[i];
    image->input.visit(
        [&](ttx_retained input) {
          if (input.operations->release(input.context) != TTX_RELEASE_SUCCESS) {
            outcome = TTX_RELEASE_CLEANUP_FAILED;
          }
        },

        [] {});
    image->~Image();
    Allocator::Global::release(Data::cast<U8>(image));
  }

  commands->~Submission();
  Allocator::Global::release(Data::cast<U8>(commands));
  return outcome;
}

auto Submission::record(void* context, ttx_abstract input, ttx_retained* output)
    -> compute_image_status {
  auto& commands = *Data::cast<Submission>(context);
  if (!output || commands.phase != Phase::Recording ||
      commands.references == Count(-1)) {
    return COMPUTE_IMAGE_INVALID;
  }

  return Abstract(input).bind<Pixels>().visit(
      [&](Pixels supplied) -> compute_image_status {
        const auto size = supplied.dimensions();
        const U64 count = U64(size.width) * size.height;
        if (!count || count > Count(-1) / 4) {
          return COMPUTE_IMAGE_INVALID;
        }

        static_assert(
            alignof(Image) <= Allocator::Global::allocation_alignment);
        auto memory = Allocator::Global::allocate(sizeof(Image));
        auto& image = Lifetime::construct<Image>(memory.get_data(), [&] {
          return Image(commands, commands.images.get_size(), size);
        });

        auto status = COMPUTE_IMAGE_SUCCESS;
        Bool deferred = False;
        Abstract(input).bind<Dependency>().visit(
            [&](auto dependency) {
              image.previous =
                  dependency.locate(Commands({context, &command_operations}));
              if (!image.previous) {
                deferred = dependency.submitted();
                if (!deferred) {
                  status = COMPUTE_IMAGE_INVALID;
                }
              }
            },

            [] {});
        if (image.previous) {
          image.previous.visit(
              [&](Count position) {
                if (position >= commands.images.get_size()) {
                  status = COMPUTE_IMAGE_INVALID;
                } else {
                  const auto before =
                      commands.images.get_data()[position]->size;
                  if (before.width != size.width ||
                      before.height != size.height) {
                    status = COMPUTE_IMAGE_INVALID;
                  }
                }
              },

              [] {});
        } else {
          // Retain may substitute both the Abstract and its typed contexts.
          // Bind Pixels again on that returned Abstract before storing a
          // deferred read.
          if (deferred && status == COMPUTE_IMAGE_SUCCESS) {
            Abstract(input).bind<Capabilities::Retain>().visit(
                [&](auto retention) {
                  retention.retain().visit(
                      [&](Policies::Retained acquired) {
                        acquired.get_abstract().bind<Pixels>().visit(
                            [&](Pixels retained) {
                              const auto actual = retained.dimensions();
                              if (actual.width == size.width &&
                                  actual.height == size.height) {
                                image.input = acquired.get_abi();
                                image.pixels = retained.get_abi();
                              }
                            },

                            [] {});
                        if (!image.input &&
                            acquired.release() !=
                                Policies::Retained::Status::Success) {
                          status = COMPUTE_IMAGE_FAILED;
                        }
                      },

                      [](Negotiation::Status) {});
                },

                [] {});
          }

          if (!image.input && status == COMPUTE_IMAGE_SUCCESS) {
            image.data.forgetful_resize(count * 4);
            status = supplied.read(image.data.get_access());
          }
        }

        if (status != COMPUTE_IMAGE_SUCCESS) {
          image.~Image();
          Allocator::Global::release(memory.get_data());
          return status;
        }

        commands.images.insert(&image);
        ++commands.references;
        *output = {&image, &image_retained};
        return COMPUTE_IMAGE_SUCCESS;
      },

      []() -> compute_image_status { return COMPUTE_IMAGE_INVALID; });
}

auto Submission::submit(void* context) -> compute_execution_status {
  auto& commands = *Data::cast<Submission>(context);
  if (commands.phase != Phase::Recording) {
    return COMPUTE_EXECUTION_INVALID;
  }

  // End recording before calling an input provider. A callback cannot append
  // commands to the sequence that is already executing.
  commands.phase = Phase::Failed;
  for (Count i = 0; i < commands.images.get_size(); ++i) {
    auto& image = *commands.images.get_data()[i];
    auto status = COMPUTE_IMAGE_SUCCESS;
    image.previous.visit(
        [&](Count position) {
          image.data = commands.images.get_data()[position]->data.get_view();
        },

        [&] {
          image.pixels.visit(
              [&](compute_pixels pixels) {
                image.data.forgetful_resize(
                    Count(image.size.width) * image.size.height * 4);
                status = Pixels(pixels).read(image.data.get_access());
              },

              [] {});
        });

    if (status != COMPUTE_IMAGE_SUCCESS) {
      return COMPUTE_EXECUTION_FAILED;
    }

    auto* bytes = image.data.get_access().get_data();
    for (Count offset = 0; offset < image.data.get_size(); offset += 4) {
      bytes[offset] ^= 255;
      bytes[offset + 1] ^= 255;
      bytes[offset + 2] ^= 255;
    }
  }

  commands.phase = Phase::Complete;
  return COMPUTE_EXECUTION_SUCCESS;
}

auto Submission::wait(void* context) -> compute_execution_status {
  const auto& commands = *Data::cast<Submission>(context);
  return commands.phase == Phase::Recording  ? COMPUTE_EXECUTION_INVALID
         : commands.phase == Phase::Complete ? COMPUTE_EXECUTION_SUCCESS
                                             : COMPUTE_EXECUTION_FAILED;
}

auto Submission::image_graph(void* context) -> ttx_abstract {
  return {context, &image_operations};
}

auto Submission::image_supports(void* context, perimortem_uuid id)
    -> ttx_negotiation_status {
  if (!id.high && id.low <= 1) {
    return TTX_NEGOTIATION_REJECTED;
  }

  const Identity::Uuid contract(id);
  return contract == Abstract::contract_id || contract == Pixels::contract_id ||
                 contract == Dependency::contract_id ||
                 contract == Capabilities::Retain::contract_id
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

auto Submission::image_bind(void* context, perimortem_uuid id, void* output)
    -> ttx_negotiation_status {
  const auto status = image_supports(context, id);
  if (status != TTX_NEGOTIATION_SATISFIED) {
    return status;
  }

  const Identity::Uuid contract(id);
  if (contract == Pixels::contract_id) {
    *Data::cast<compute_pixels>(output) = {context, &pixel_operations};
  } else if (contract == Dependency::contract_id) {
    *Data::cast<compute_dependency>(output) = {context, &dependency_operations};
  } else if (contract == Capabilities::Retain::contract_id) {
    *Data::cast<ttx_retain>(output) = {context, &image_retention};
  } else {
    *Data::cast<ttx_abstract>(output) = image_graph(context);
  }

  return TTX_NEGOTIATION_SATISFIED;
}

auto Submission::image_retain(void* context, ttx_retained* output)
    -> ttx_negotiation_status {
  auto& image = *Data::cast<Image>(context);
  ttx_retained acquired{};
  const auto status = retain(&image.commands, &acquired);
  if (status == TTX_NEGOTIATION_SATISFIED) {
    *output = {context, &image_retained};
  }

  return status;
}

auto Submission::image_release(void* context) -> ttx_release_status {
  return release(&Data::cast<Image>(context)->commands);
}

auto Submission::dimensions(void* context) -> compute_image_dimensions {
  return Data::cast<Image>(context)->size;
}

auto Submission::read(void* context, U8* output, Count capacity)
    -> compute_image_status {
  const auto& image = *Data::cast<Image>(context);
  const Count count = Count(image.size.width) * image.size.height * 4;
  if (!output || capacity < count || image.commands.phase == Phase::Recording) {
    return COMPUTE_IMAGE_INVALID;
  }

  if (wait(&image.commands) != COMPUTE_EXECUTION_SUCCESS) {
    return COMPUTE_IMAGE_FAILED;
  }

  Data::copy(output, image.data.get_data(), count);
  return COMPUTE_IMAGE_SUCCESS;
}

auto Submission::submitted(void* context) -> U8 {
  return Data::cast<Image>(context)->commands.phase != Phase::Recording;
}

auto Submission::locate(
    void* context,
    compute_commands requested,
    Count* position) -> ttx_negotiation_status {
  const auto& image = *Data::cast<Image>(context);
  if (requested.context != &image.commands ||
      requested.operations != &command_operations) {
    return TTX_NEGOTIATION_REJECTED;
  }

  *position = image.position;
  return TTX_NEGOTIATION_SATISFIED;
}

const ttx_abstract_ops Submission::graph_operations = {
  supports, bind, route, visit, state};

const ttx_abstract_ops Submission::image_operations = {
  image_supports, image_bind, route, visit, state};

const ttx_retain_ops Submission::retain_operations = {graph, retain};
const ttx_retained_ops Submission::retained_operations = {graph, release};
const ttx_retain_ops Submission::image_retention = {image_graph, image_retain};
const ttx_retained_ops Submission::image_retained = {
  image_graph, image_release};

const compute_commands_ops Submission::command_operations = {
  graph, submit, wait};

const compute_invert_ops Submission::invert_operations = {graph, record};
const compute_pixels_ops Submission::pixel_operations = {
  image_graph, dimensions, read};

const compute_dependency_ops Submission::dependency_operations = {
  image_graph, submitted, locate};
