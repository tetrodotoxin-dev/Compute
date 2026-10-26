// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "toolchain/validation/test.hpp"

#include "perimortem/core/diagnostics/log.hpp"
#include "perimortem/core/view/null_terminated.hpp"

#include "ttx/graph/capabilities/create.hpp"
#include "ttx/graph/capabilities/retain.hpp"
#include "ttx/graph/policies/none.hpp"

#include "compute/execution/commands.hpp"
#include "compute/execution/dependency.hpp"
#include "compute/execution/selection.hpp"
#include "compute/image/invert.hpp"
#include "compute/policies/accelerated.h"
#include "cpu/image/rgba8.hpp"
#include "cpu/platform/engine.hpp"
#include "validation/tests/deferred.h"
#include "validation/tests/generated.h"

using namespace Perimortem::Core;
using namespace Toolchain::Validation;
using namespace Ttx::Graph;
using namespace Ttx::Semantic;
using namespace Compute::Image;
using namespace Cpu::Image;
using namespace Cpu::Platform;
using namespace Compute::Execution;

static Harness ComputeTests = {.name = "Compute::Cpu"};

template <typename Contract>
static auto bound(Abstract abstract) -> Contract {
  return abstract.bind<Contract>().visit(
      [](Contract api) -> Contract { return api; },
      []() -> Contract {
        Diagnostics::Log::fatal(
            "Required compute fixture contract is unavailable."_view);
      });
}

static auto retained(Abstract abstract) -> Policies::Retained {
  return bound<Capabilities::Retain>(abstract).retain().visit(
      [](Policies::Retained acquired) -> Policies::Retained {
        return acquired;
      },

      [](Negotiation::Status) -> Policies::Retained {
        Diagnostics::Log::fatal("Compute fixture retention failed."_view);
      });
}

static auto create(Abstract factory) -> Policies::Retained {
  Storage::Option<Policies::Retained> result;
  auto callback = [&](Abstract created) { result = retained(created); };
  bound<Capabilities::Create>(factory).create(
      Policies::None::get_none().get_abstract(), callback);
  return result.visit(
      [](Policies::Retained acquired) -> Policies::Retained {
        return acquired;
      },

      []() -> Policies::Retained {
        Diagnostics::Log::fatal("Compute fixture creation failed."_view);
      });
}

static auto record(Abstract commands, Abstract image) -> Policies::Retained {
  return bound<Invert>(commands).record(image).visit(
      [](Policies::Retained acquired) -> Policies::Retained {
        return acquired;
      },

      [](compute_image_status) -> Policies::Retained {
        Diagnostics::Log::fatal("Compute fixture recording failed."_view);
      });
}

// Payload reads and wait cannot submit work. Two recorded operations consume
// the same authored chain after its input and engine acquisitions are gone.
VALIDATION_TEST(ComputeTests, create_invert_release) {
  const auto engine = create(Engine::factory());
  const auto buffer =
      create(engine.get_abstract().resolve_route("commands"_view));
  const auto commands = bound<Commands>(buffer.get_abstract());
  Storage::Option<Policies::Retained> first;
  auto callback = [&](Abstract image) {
    first = record(buffer.get_abstract(), image);
  };

  compute_generated_image(&callback, [](void* context, ttx_abstract image) {
    (*Data::cast<decltype(callback)>(context))(Abstract(image));
  });

  ASSERT(first);
  first.visit(
      [&](Policies::Retained input) {
        const auto output = record(buffer.get_abstract(), input.get_abstract());
        const auto second_consumer = retained(output.get_abstract());
        const auto pixels = bound<Pixels>(output.get_abstract());
        EXPECT_EQ(pixels.dimensions().width, U32(2));
        EXPECT_EQ(pixels.dimensions().height, U32(2));

        U8 observed[16];
        for (auto& byte : observed) {
          byte = 73;
        }

        EXPECT_EQ(
            pixels.read(Access::Bytes(observed[0], 16)), COMPUTE_IMAGE_INVALID);
        for (auto byte : observed) {
          EXPECT_EQ(byte, U8(73));
        }

        EXPECT_EQ(commands.wait(), COMPUTE_EXECUTION_INVALID);
        EXPECT(input.release() == Policies::Retained::Status::Success);
        EXPECT(engine.release() == Policies::Retained::Status::Success);
        EXPECT_EQ(commands.submit(), COMPUTE_EXECUTION_SUCCESS);
        EXPECT_EQ(commands.submit(), COMPUTE_EXECUTION_INVALID);
        EXPECT_EQ(commands.wait(), COMPUTE_EXECUTION_SUCCESS);
        EXPECT_EQ(commands.wait(), COMPUTE_EXECUTION_SUCCESS);
        bound<Invert>(buffer.get_abstract())
            .record(output.get_abstract())
            .visit(
                [&](Policies::Retained unexpected) {
                  EXPECT(False);
                  unexpected.release();
                },

                [&](compute_image_status status) {
                  EXPECT_EQ(status, COMPUTE_IMAGE_INVALID);
                });

        EXPECT(buffer.release() == Policies::Retained::Status::Success);
        EXPECT(output.release() == Policies::Retained::Status::Success);
        EXPECT_EQ(
            bound<Pixels>(second_consumer.get_abstract())
                .read(Access::Bytes(observed[0], 16)),
            COMPUTE_IMAGE_SUCCESS);
        for (Count i = 0; i < 16; ++i) {
          EXPECT_EQ(observed[i], U8(17 + i * 13));
        }

        EXPECT(
            second_consumer.release() == Policies::Retained::Status::Success);
      },

      [] {});
}

VALIDATION_TEST(ComputeTests, refused_creation) {
  auto unexpected = [&](Abstract unexpected) { EXPECT(False); };
  Engine::factory().bind<Capabilities::Create>().visit(
      [&](Capabilities::Create create) {
        EXPECT(
            create.create(Engine::factory(), unexpected) ==
            Negotiation::Status::Rejected);
      },

      [&] { EXPECT(False); });
}

VALIDATION_TEST(ComputeTests, cpu_fallback) {
  Count attempts = 0, callbacks = 0;
  const ttx_abstract factories[] = {
    compute_refusing_factory(&attempts), Engine::factory().get_abi()};

  Selection selection{View::Vector<ttx_abstract>(factories)};
  auto callback = [&](Abstract engine) {
    ++callbacks;
    EXPECT(
        engine.supports<Capabilities::Create>() ==
        Negotiation::Status::Satisfied);
  };

  selection.get_abstract().bind<Capabilities::Create>().visit(
      [&](Capabilities::Create create) {
        EXPECT(
            create.create(
                Policies::None::get_none().get_abstract(), callback) ==
            Negotiation::Status::Satisfied);
      },

      [&] { EXPECT(False); });
  EXPECT_EQ(attempts, Count(1));
  EXPECT_EQ(callbacks, Count(1));

  Selection required(
      View::Vector<ttx_abstract>(factories),
      Identity::Uuid(COMPUTE_ACCELERATED_ID_HIGH, COMPUTE_ACCELERATED_ID_LOW));
  required.get_abstract().bind<Capabilities::Create>().visit(
      [&](Capabilities::Create create) {
        EXPECT(
            create.create(
                Policies::None::get_none().get_abstract(), callback) ==
            Negotiation::Status::Rejected);
      },

      [&] { EXPECT(False); });
  EXPECT_EQ(attempts, Count(2));
  EXPECT_EQ(callbacks, Count(1));
}

VALIDATION_TEST(ComputeTests, image_bounds) {
  const U8 input[4] = {1, 2, 3, 4};
  Rgba8::create(1, 1, View::Bytes(input))
      .visit(
          [&](Policies::Retained image) {
            image.get_abstract().bind<Pixels>().visit(
                [&](Pixels pixels) {
                  U8 output[4] = {17, 17, 17, 17};
                  const auto api = pixels.get_abi();
                  EXPECT_EQ(
                      api.operations->read(api.context, output, 3),
                      COMPUTE_IMAGE_INVALID);
                  EXPECT_EQ(
                      api.operations->read(api.context, nullptr, 4),
                      COMPUTE_IMAGE_INVALID);
                  for (auto byte : output) {
                    EXPECT_EQ(byte, U8(17));
                  }
                },

                [&] { EXPECT(False); });
            EXPECT(image.release() == Policies::Retained::Status::Success);
          },

          [&](compute_image_status) { EXPECT(False); });
  Rgba8::create(2, 1, View::Bytes(input))
      .visit(
          [&](Policies::Retained unexpected) {
            EXPECT(False);
            unexpected.release();
          },

          [&](compute_image_status status) {
            EXPECT_EQ(status, COMPUTE_IMAGE_INVALID);
          });
}

VALIDATION_TEST(ComputeTests, selection_preserves_binding_freedom) {
  Count callbacks = 0;
  const ttx_abstract factories[] = {compute_generated_factory()};
  Selection selection{View::Vector<ttx_abstract>(factories)};
  auto callback = [&](Abstract created) {
    ++callbacks;
    EXPECT(created.supports<Abstract>() == Negotiation::Status::Unknown);
    EXPECT(created.supports<Pixels>() == Negotiation::Status::Satisfied);
    EXPECT(!created.bind<Abstract>());
  };

  selection.get_abstract().bind<Capabilities::Create>().visit(
      [&](Capabilities::Create create) {
        EXPECT(
            create.create(
                Policies::None::get_none().get_abstract(), callback) ==
            Negotiation::Status::Satisfied);
      },

      [&] { EXPECT(False); });
  EXPECT_EQ(callbacks, Count(1));
}

// An independent producer must already be sealed. This prevents unsubmitted
// buffers from keeping one another alive through retained input acquisitions.
VALIDATION_TEST(ComputeTests, cross_buffer_submission) {
  const auto engine = create(Engine::factory());
  const auto a = create(engine.get_abstract());
  const auto b = create(engine.get_abstract());
  Storage::Option<Policies::Retained> first;
  auto callback = [&](Abstract input) {
    first = record(a.get_abstract(), input);
  };

  compute_generated_image(&callback, [](void* context, ttx_abstract input) {
    (*Data::cast<decltype(callback)>(context))(Abstract(input));
  });

  first.visit(
      [&](Policies::Retained image) {
        bound<Invert>(b.get_abstract())
            .record(image.get_abstract())
            .visit(
                [&](Policies::Retained unexpected) {
                  EXPECT(False);
                  unexpected.release();
                },

                [&](compute_image_status status) {
                  EXPECT_EQ(status, COMPUTE_IMAGE_INVALID);
                });

        EXPECT_EQ(
            bound<Commands>(a.get_abstract()).submit(),
            COMPUTE_EXECUTION_SUCCESS);

        const auto output = record(b.get_abstract(), image.get_abstract());
        EXPECT(image.release() == Policies::Retained::Status::Success);
        EXPECT(a.release() == Policies::Retained::Status::Success);
        EXPECT_EQ(
            bound<Commands>(b.get_abstract()).submit(),
            COMPUTE_EXECUTION_SUCCESS);
        EXPECT(b.release() == Policies::Retained::Status::Success);

        U8 observed[16]{};
        EXPECT_EQ(
            bound<Pixels>(output.get_abstract())
                .read(Access::Bytes(observed[0], 16)),
            COMPUTE_IMAGE_SUCCESS);
        for (Count i = 0; i < 16; ++i) {
          EXPECT_EQ(observed[i], U8(17 + i * 13));
        }

        EXPECT(output.release() == Policies::Retained::Status::Success);
      },

      [&] { EXPECT(False); });
  EXPECT(engine.release() == Policies::Retained::Status::Success);
}

// The C provider's retention changes both its context layout and operations.
// Reads occur in authored order after both original callbacks have ended.
VALIDATION_TEST(ComputeTests, deferred_inputs_and_failures) {
  for (U8 case_index = 0; case_index < 3; ++case_index) {
    const Bool fail = case_index != 0;
    const auto engine = create(Engine::factory());
    const auto buffer = create(engine.get_abstract());
    Count sequence = 0;
    compute_deferred_state states[2]{};
    Storage::Option<Policies::Retained> images[2];
    for (Count i = 0; i < 2; ++i) {
      states[i].seed = U8(19 + i);
      states[i].sequence = &sequence;
      states[i].fail = case_index == i + 1;

      auto callback = [&](Abstract input) {
        images[i] = record(buffer.get_abstract(), input);
      };

      compute_deferred_image(
          &states[i], &callback, [](void* context, ttx_abstract input) {
            (*Data::cast<decltype(callback)>(context))(Abstract(input));
          });

      EXPECT_EQ(states[i].original_reads, Count(0));
      EXPECT_EQ(states[i].retained_reads, Count(0));
      EXPECT_EQ(states[i].acquisitions, Count(1));
    }

    const auto commands = bound<Commands>(buffer.get_abstract());
    EXPECT_EQ(
        commands.submit(),
        fail ? COMPUTE_EXECUTION_FAILED : COMPUTE_EXECUTION_SUCCESS);
    EXPECT_EQ(
        commands.wait(),
        fail ? COMPUTE_EXECUTION_FAILED : COMPUTE_EXECUTION_SUCCESS);
    EXPECT_EQ(states[0].read_order, Count(1));
    EXPECT_EQ(states[1].read_order, case_index == 1 ? Count(0) : Count(2));
    for (Count i = 0; i < 2; ++i) {
      images[i].visit(
          [&](Policies::Retained image) {
            U8 output[16]{};
            EXPECT_EQ(
                bound<Pixels>(image.get_abstract())
                    .read(Access::Bytes(output[0], 16)),
                fail ? COMPUTE_IMAGE_FAILED : COMPUTE_IMAGE_SUCCESS);
            EXPECT(image.release() == Policies::Retained::Status::Success);
          },

          [&] { EXPECT(False); });
    }

    EXPECT(buffer.release() == Policies::Retained::Status::Success);
    EXPECT(engine.release() == Policies::Retained::Status::Success);
    EXPECT_EQ(states[0].releases, states[0].acquisitions);
    EXPECT_EQ(states[1].releases, states[1].acquisitions);
  }
}

VALIDATION_TEST(ComputeTests, capture_when_retained_binding_is_refused) {
  const auto engine = create(Engine::factory());
  const auto buffer = create(engine.get_abstract());
  compute_deferred_state state{};
  state.seed = 21;
  state.refuse_pixels = 1;

  Storage::Option<Policies::Retained> recorded;
  auto callback = [&](Abstract input) {
    recorded = record(buffer.get_abstract(), input);
  };

  compute_deferred_image(
      &state, &callback, [](void* context, ttx_abstract input) {
        (*Data::cast<decltype(callback)>(context))(Abstract(input));
      });

  EXPECT_EQ(state.original_reads, Count(1));
  EXPECT_EQ(state.retained_reads, Count(0));
  EXPECT_EQ(state.acquisitions, Count(1));
  EXPECT_EQ(state.releases, Count(1));
  EXPECT_EQ(
      bound<Commands>(buffer.get_abstract()).submit(),
      COMPUTE_EXECUTION_SUCCESS);
  recorded.visit(
      [&](Policies::Retained image) {
        U8 output[16]{};
        EXPECT_EQ(
            bound<Pixels>(image.get_abstract())
                .read(Access::Bytes(output[0], 16)),
            COMPUTE_IMAGE_SUCCESS);
        for (Count i = 0; i < 16; ++i) {
          const auto original = U8(21 + i * 13);
          EXPECT_EQ(output[i], i % 4 == 3 ? original : U8(255 - original));
        }

        EXPECT(image.release() == Policies::Retained::Status::Success);
      },

      [&] { EXPECT(False); });
  EXPECT(buffer.release() == Policies::Retained::Status::Success);
  EXPECT(engine.release() == Policies::Retained::Status::Success);
}

VALIDATION_TEST(ComputeTests, empty_and_discarded_recordings) {
  const auto engine = create(Engine::factory());
  const auto empty = create(engine.get_abstract());
  EXPECT_EQ(
      bound<Commands>(empty.get_abstract()).submit(),
      COMPUTE_EXECUTION_SUCCESS);
  EXPECT_EQ(
      bound<Commands>(empty.get_abstract()).wait(), COMPUTE_EXECUTION_SUCCESS);
  EXPECT(empty.release() == Policies::Retained::Status::Success);

  const auto discarded = create(engine.get_abstract());
  compute_deferred_state state{};
  state.seed = 7;

  auto callback = [&](Abstract input) {
    const auto recorded = record(discarded.get_abstract(), input);
    EXPECT(recorded.release() == Policies::Retained::Status::Success);
  };

  compute_deferred_image(
      &state, &callback, [](void* context, ttx_abstract input) {
        (*Data::cast<decltype(callback)>(context))(Abstract(input));
      });

  EXPECT(discarded.release() == Policies::Retained::Status::Success);
  EXPECT_EQ(state.retained_reads, Count(0));
  EXPECT_EQ(state.releases, Count(1));
  EXPECT(engine.release() == Policies::Retained::Status::Success);
}

VALIDATION_TEST(ComputeTests, final_release_reports_input_cleanup) {
  const auto engine = create(Engine::factory());
  const auto buffer = create(engine.get_abstract());
  compute_deferred_state state{};
  state.seed = 3;
  state.fail_release = 1;

  auto callback = [&](Abstract input) {
    const auto image = record(buffer.get_abstract(), input);
    EXPECT(image.release() == Policies::Retained::Status::Success);
  };

  compute_deferred_image(
      &state, &callback, [](void* context, ttx_abstract input) {
        (*Data::cast<decltype(callback)>(context))(Abstract(input));
      });

  EXPECT(buffer.release() == Policies::Retained::Status::CleanupFailed);
  EXPECT_EQ(state.releases, Count(1));
  EXPECT_EQ(state.retained_reads, Count(0));
  EXPECT(engine.release() == Policies::Retained::Status::Success);
}

// Create's callback receives access without a release obligation. Retain
// creates the consumer's acquisition explicitly, and the resulting API remains
// usable even though its Abstract refuses to bind the Retained UUID again.
VALIDATION_TEST(ComputeTests, creation_keeps_acquisition_explicit) {
  auto callback = [&](Abstract engine) {
    EXPECT(!engine.bind<Policies::Retained>());

    auto command_callback = [&](Abstract buffer) {
      EXPECT(!buffer.bind<Policies::Retained>());

      const auto acquired = retained(buffer);
      EXPECT(!acquired.get_abstract().bind<Policies::Retained>());
      EXPECT_EQ(
          bound<Commands>(acquired.get_abstract()).submit(),
          COMPUTE_EXECUTION_SUCCESS);
      EXPECT(acquired.release() == Policies::Retained::Status::Success);
    };

    EXPECT(
        bound<Capabilities::Create>(engine).create(
            Policies::None::get_none().get_abstract(), command_callback) ==
        Negotiation::Status::Satisfied);
  };

  EXPECT(
      bound<Capabilities::Create>(Engine::factory())
          .create(Policies::None::get_none().get_abstract(), callback) ==
      Negotiation::Status::Satisfied);
}
