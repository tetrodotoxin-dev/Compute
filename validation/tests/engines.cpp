// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "toolchain/validation/test.hpp"

#include "perimortem/core/diagnostics/log.hpp"
#include "perimortem/core/view/null_terminated.hpp"
#include "perimortem/memory/allocator/arena.hpp"
#include "perimortem/system/process/library.hpp"

#include "ttx/data/transfer.hpp"
#include "ttx/graph/capabilities/create.hpp"
#include "ttx/graph/capabilities/retain.hpp"
#include "ttx/graph/policies/none.hpp"

#include "compute/execution/selection.hpp"
#include "compute/image/invert.hpp"
#include "compute/image/pixels.hpp"
#include "compute/platforms/cpu.hpp"
#include "compute/policies/accelerated.h"
#include "validation/tests/generated.h"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Perimortem::System;
using namespace Toolchain::Validation;
using namespace Ttx::Data;
using namespace Ttx::Graph;
using namespace Ttx::Semantic;
using namespace Compute::Execution;
using namespace Compute::Platforms;
using namespace Compute::Image;

static Harness Engines = {.name = "Compute::Substitution"};

class Provider {
 public:
  // Explicit construction loads the shared library and keeps its executable
  // code available through this fixture's lifetime.
  explicit Provider(View::Bytes path) : library(open(path)) {}
  auto root() -> Abstract {
    Allocator::Arena errors;
    const auto* symbol =
        library.symbol("ttx_entry"_view, errors)
            .visit(
                [](void* value) -> void* { return value; },
                [](View::Bytes) -> void* {
                  Diagnostics::Log::fatal(
                      "Compute provider entry is missing."_view);
                });

    ttx_semantic_query entry{};
    auto destination = Form::Storage::from(entry).get_abi();
    if (ttx_transfer(Data::cast<const ttx_storage>(symbol), &destination) !=
        TTX_DATA_SUCCESS) {
      Diagnostics::Log::fatal("Compute provider entry is incompatible."_view);
    }

    return Negotiation::Query(entry).bind<Abstract>().visit(
        [](Abstract api) -> Abstract { return api; },
        [](Negotiation::Status) -> Abstract {
          Diagnostics::Log::fatal("Compute provider has no Abstract."_view);
        });
  }

 private:
  static auto open(View::Bytes path) -> Process::Library {
    Allocator::Arena errors;
    return Process::Library::open(path, errors)
        .visit(
            [](Process::Library& library) -> Process::Library {
              return Data::take(library);
            },

            [](View::Bytes) -> Process::Library {
              Diagnostics::Log::fatal(
                  "Compute provider library is unavailable."_view);
            });
  }

  Process::Library library;
};

#ifdef TOOLCHAIN_WINDOWS

static constexpr auto cpu_library = "compute.dll"_view;
static constexpr auto cuda_library = "ttx_cuda.dll"_view;

#else

static constexpr auto cpu_library = "libcompute.so"_view;
static constexpr auto cuda_library = "libttx_cuda.so"_view;

#endif

template <typename Contract>
static auto bound(Abstract abstract) -> Contract {
  return abstract.bind<Contract>().visit(
      [](Contract api) -> Contract { return api; },
      []() -> Contract {
        Diagnostics::Log::fatal(
            "Compute fixture contract is unavailable."_view);
      });
}

static auto retain(Abstract abstract) -> Policies::Retained {
  return bound<Capabilities::Retain>(abstract).retain().visit(
      [](Policies::Retained acquired) -> Policies::Retained {
        return acquired;
      },

      [](Negotiation::Status) -> Policies::Retained {
        Diagnostics::Log::fatal("Compute fixture retention failed."_view);
      });
}

static auto create(Abstract factory) -> Policies::Retained {
  Storage::Option<Policies::Retained> acquired;
  auto callback = [&](Abstract engine) { acquired = retain(engine); };
  const auto status = bound<Capabilities::Create>(factory).create(
      Policies::None::get_none().get_abstract(), callback);
  if (status != Negotiation::Status::Satisfied) {
    Diagnostics::Log::fatal("Compute fixture engine creation failed."_view);
  }

  return acquired.visit(
      [](Policies::Retained value) -> Policies::Retained { return value; },
      []() -> Policies::Retained {
        Diagnostics::Log::fatal("Compute factory supplied no engine."_view);
      });
}

static auto invert(Abstract engine, Abstract image) -> Policies::Retained {
  return bound<Invert>(engine.resolve_route("invert"_view))
      .apply(image)
      .visit(
          [](Policies::Retained value) -> Policies::Retained { return value; },
          [](compute_image_status) -> Policies::Retained {
            Diagnostics::Log::fatal("Compute fixture inversion failed."_view);
          });
}

// Every pairing uses the same consumer and image contract. The input provider
// generates pixels in C and disappears after its callback. Both produced images
// remain readable after engine release, including a separately retained first
// output whose original acquisition has already ended.
VALIDATION_TEST(Engines, all_provider_pairs) {
  Provider cpu(cpu_library), cuda(cuda_library);
  const Abstract factories[] = {
    cpu.root(), cuda.root().resolve_route("engine"_view)};

  for (const auto first_factory : factories) {
    for (const auto second_factory : factories) {
      const auto first_engine = create(first_factory);
      const auto second_engine = create(second_factory);
      Storage::Option<Policies::Retained> first;
      auto callback = [&](Abstract input) {
        const auto pixels = bound<Pixels>(input);
        EXPECT(pixels.get_abi().context != input.get_abi().context);
        first = invert(first_engine.get_abstract(), input);
      };

      compute_generated_image(&callback, [](void* context, ttx_abstract input) {
        (*Data::cast<decltype(callback)>(context))(Abstract(input));
      });

      ASSERT(first);
      first.visit(
          [&](Policies::Retained first_image) {
            const auto another_consumer = retain(first_image.get_abstract());
            const auto second_image = invert(
                second_engine.get_abstract(), first_image.get_abstract());
            EXPECT(
                first_image.release() == Policies::Retained::Status::Success);
            EXPECT(
                first_engine.release() == Policies::Retained::Status::Success);
            EXPECT(
                second_engine.release() == Policies::Retained::Status::Success);

            U8 observed[16]{}, inverted[16]{};
            EXPECT_EQ(
                bound<Pixels>(second_image.get_abstract())
                    .read(Access::Bytes(observed[0], 16)),
                COMPUTE_IMAGE_SUCCESS);
            EXPECT_EQ(
                bound<Pixels>(another_consumer.get_abstract())
                    .read(Access::Bytes(inverted[0], 16)),
                COMPUTE_IMAGE_SUCCESS);
            for (Count i = 0; i < 16; ++i) {
              const U8 original = U8(17 + i * 13);
              EXPECT_EQ(observed[i], original);
              EXPECT_EQ(
                  inverted[i], i % 4 == 3 ? original : U8(255 - original));
            }

            EXPECT(
                second_image.release() == Policies::Retained::Status::Success);
            EXPECT(
                another_consumer.release() ==
                Policies::Retained::Status::Success);
          },

          [&] { EXPECT(False); });
    }
  }
}

VALIDATION_TEST(Engines, policy_selection) {
  Provider cpu(cpu_library), cuda(cuda_library);
  const ttx_abstract factories[] = {
    cpu.root().get_abi(), cuda.root().resolve_route("engine"_view).get_abi()};

  const Identity::Uuid accelerated(
      COMPUTE_ACCELERATED_ID_HIGH, COMPUTE_ACCELERATED_ID_LOW);
  Selection selection(View::Vector<ttx_abstract>(factories), accelerated);
  const auto selected = create(selection.get_abstract());
  EXPECT(
      selected.get_abstract().supports(accelerated) ==
      Negotiation::Status::Satisfied);
  EXPECT(selected.release() == Policies::Retained::Status::Success);

  Selection cpu_only(
      View::Vector<ttx_abstract>(factories).slice(0, 1), accelerated);
  auto unexpected = [&](Abstract engine) { EXPECT(False); };
  EXPECT(
      bound<Capabilities::Create>(cpu_only.get_abstract())
          .create(Policies::None::get_none().get_abstract(), unexpected) ==
      Negotiation::Status::Rejected);
}
