// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "perimortem/core/view/null_terminated.hpp"

#include "ttx/graph/capabilities/create.hpp"
#include "ttx/graph/capabilities/retain.hpp"
#include "ttx/graph/policies/none.hpp"

#include "compute/execution/commands.hpp"
#include "compute/image/invert.hpp"
#include "cpu/image/rgba8.hpp"
#include "cpu/platform/engine.hpp"
#include "cuda/platform/engine.hpp"

using namespace Perimortem::Core;
using namespace Ttx::Graph;
using namespace Ttx::Semantic;
using namespace Compute::Image;
using namespace Compute::Execution;
using namespace Cpu::Image;

static auto execute(Abstract factory, Abstract input)
    -> Storage::Option<Policies::Retained> {
  Storage::Option<Policies::Retained> output;
  auto callback = [&](Abstract engine) {
    auto record = [&](Abstract buffer) {
      buffer.bind<Invert>().visit(
          [&](Invert operation) {
            operation.record(input).visit(
                [&](Policies::Retained image) {
                  buffer.bind<Commands>().visit(
                      [&](auto commands) {
                        if (commands.submit() == COMPUTE_EXECUTION_SUCCESS) {
                          output = image;
                        }
                      },

                      [] {});
                  if (!output) {
                    image.release();
                  }
                },

                [](compute_image_status) {});
          },

          [] {});
    };

    engine.resolve_route("commands"_view)
        .bind<Capabilities::Create>()
        .visit(
            [&](auto create) {
              create.create(Policies::None::get_none().get_abstract(), record);
            },

            [] {});
  };

  factory.bind<Capabilities::Create>().visit(
      [&](Capabilities::Create create) {
        create.create(Policies::None::get_none().get_abstract(), callback);
      },

      [] {});
  return output;
}

// Both engines supply the same domain contracts. CUDA consumes a CPU-produced
// image, and its result stays readable after its engine callback has ended.
S32 main() {
  const U8 pixels[] = {0, 53, 255, 91};
  S32 result = 1;
  Rgba8::create(1, 1, View::Bytes(pixels))
      .visit(
          [&](Policies::Retained input) {
            execute(Cpu::Platform::Engine::factory(), input.get_abstract())
                .visit(
                    [&](Policies::Retained cpu) {
                      execute(
                          Cuda::Platform::Engine::factory(), cpu.get_abstract())
                          .visit(
                              [&](Policies::Retained gpu) {
                                U8 output[4]{};
                                gpu.get_abstract().bind<Pixels>().visit(
                                    [&](Pixels supplied) {
                                      if (supplied.read(
                                              Access::Bytes(output[0], 4)) ==
                                              COMPUTE_IMAGE_SUCCESS &&
                                          View::Bytes(output) ==
                                              View::Bytes(pixels)) {
                                        result = 0;
                                      }
                                    },

                                    [] {});
                                if (gpu.release() !=
                                    Policies::Retained::Status::Success) {
                                  result = 2;
                                }
                              },

                              [] {});
                      if (cpu.release() !=
                          Policies::Retained::Status::Success) {
                        result = 2;
                      }
                    },

                    [] {});
            if (input.release() != Policies::Retained::Status::Success) {
              result = 2;
            }
          },

          [](compute_image_status) {});
  return result;
}
