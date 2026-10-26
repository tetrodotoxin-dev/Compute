// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "toolchain/validation/test.hpp"

#include "perimortem/core/view/null_terminated.hpp"

#include "ttx/graph/capabilities/create.hpp"
#include "ttx/graph/capabilities/retain.hpp"
#include "ttx/graph/policies/none.hpp"

#include "compute/execution/selection.hpp"
#include "compute/image/invert.hpp"
#include "compute/image/rgba8.hpp"
#include "compute/platforms/cpu.hpp"
#include "compute/policies/accelerated.h"
#include "validation/tests/generated.h"

using namespace Perimortem::Core;
using namespace Toolchain::Validation;
using namespace Ttx::Graph;
using namespace Ttx::Semantic;
using namespace Compute::Image;
using namespace Compute::Platforms;
using namespace Compute::Execution;

static Harness ComputeTests = {.name = "Compute::Cpu"};

// This consumer retains the created engine before the callback ends. Its
// operations consume the same Abstract input used by the CUDA acceptance case.
VALIDATION_TEST(ComputeTests, create_invert_release) {
  Storage::Option<Policies::Retained> engine;
  auto callback = [&](Abstract created) {
    created.bind<Capabilities::Retain>().visit(
        [&](Capabilities::Retain retain) {
          retain.retain().visit(
              [&](Policies::Retained result) { engine = result; },
              [&](Negotiation::Status) { EXPECT(False); });
        },

        [&] { EXPECT(False); });
  };

  Cpu::factory().bind<Capabilities::Create>().visit(
      [&](Capabilities::Create create) {
        EXPECT(
            create.create(
                Policies::None::get_none().get_abstract(), callback) ==
            Negotiation::Status::Satisfied);
      },

      [&] { EXPECT(False); });
  ASSERT(engine);

  const U8 input[] = {0, 127, 255, 37, 3, 80, 196, 255};
  Rgba8::create(2, 1, View::Bytes(input))
      .visit(
          [&](Policies::Retained image) {
            engine.visit(
                [&](Policies::Retained selected) {
                  EXPECT(
                      selected.get_abstract().supports(
                          Identity::Uuid(
                              COMPUTE_ACCELERATED_ID_HIGH,
                              COMPUTE_ACCELERATED_ID_LOW)) ==
                      Negotiation::Status::Rejected);
                  selected.get_abstract()
                      .resolve_route("invert"_view)
                      .bind<Invert>()
                      .visit(
                          [&](Invert operation) {
                            operation.apply(image.get_abstract())
                                .visit(
                                    [&](Policies::Retained output) {
                                      EXPECT(
                                          selected.release() ==
                                          Policies::Retained::Status::Success);

                                      U8 pixels[8]{};
                                      output.get_abstract()
                                          .bind<Pixels>()
                                          .visit(
                                              [&](Pixels supplied) {
                                                EXPECT_EQ(
                                                    supplied.read(
                                                        Access::Bytes(
                                                            pixels[0], 8)),
                                                    COMPUTE_IMAGE_SUCCESS);
                                                EXPECT_EQ(
                                                    supplied.dimensions().width,
                                                    U32(2));
                                                EXPECT_EQ(
                                                    supplied.dimensions()
                                                        .height,
                                                    U32(1));
                                              },

                                              [&] { EXPECT(False); });
                                      for (Count i = 0; i < 8; ++i) {
                                        EXPECT_EQ(
                                            pixels[i],
                                            i % 4 == 3 ? input[i]
                                                       : U8(255 - input[i]));
                                      }

                                      EXPECT(
                                          output.release() ==
                                          Policies::Retained::Status::Success);
                                    },

                                    [&](compute_image_status) {
                                      EXPECT(False);
                                      selected.release();
                                    });
                          },

                          [&] {
                            EXPECT(False);
                            selected.release();
                          });
                },

                [&] { EXPECT(False); });
            EXPECT(image.release() == Policies::Retained::Status::Success);
          },

          [&](compute_image_status) { EXPECT(False); });
}

VALIDATION_TEST(ComputeTests, refused_creation) {
  auto unexpected = [&](Abstract unexpected) { EXPECT(False); };
  Cpu::factory().bind<Capabilities::Create>().visit(
      [&](Capabilities::Create create) {
        EXPECT(
            create.create(Cpu::factory(), unexpected) ==
            Negotiation::Status::Rejected);
      },

      [&] { EXPECT(False); });
}

VALIDATION_TEST(ComputeTests, cpu_fallback) {
  Count attempts = 0, callbacks = 0;
  const ttx_abstract factories[] = {
    compute_refusing_factory(&attempts), Cpu::factory().get_abi()};

  Selection selection{View::Vector<ttx_abstract>(factories)};
  auto callback = [&](Abstract engine) {
    ++callbacks;
    EXPECT(engine.supports<Invert>() == Negotiation::Status::Satisfied);
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
