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

#include "compute/execution/commands.hpp"
#include "compute/execution/selection.hpp"
#include "compute/image/invert.hpp"
#include "compute/image/pixels.hpp"
#include "compute/policies/accelerated.h"
#include "cuda/platform/device_image.hpp"
#include "validation/tests/deferred.h"
#include "validation/tests/generated.h"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Perimortem::System;
using namespace Toolchain::Validation;
using namespace Ttx::Data;
using namespace Ttx::Graph;
using namespace Ttx::Semantic;
using namespace Compute::Execution;
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

static constexpr auto cpu_library = "ttx_cpu.dll"_view;
static constexpr auto cuda_library = "ttx_cuda.dll"_view;

#else

static constexpr auto cpu_library = "libttx_cpu.so"_view;
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
  const auto buffer = create(engine.resolve_route("commands"_view));
  const auto output =
      bound<Invert>(buffer.get_abstract())
          .record(image)
          .visit(
              [](Policies::Retained value) -> Policies::Retained {
                return value;
              },

              [](compute_image_status) -> Policies::Retained {
                Diagnostics::Log::fatal(
                    "Compute fixture recording failed."_view);
              });

  if (bound<Commands>(buffer.get_abstract()).submit() !=
          COMPUTE_EXECUTION_SUCCESS ||
      buffer.release() != Policies::Retained::Status::Success) {
    Diagnostics::Log::fatal("Compute fixture submission failed."_view);
  }

  return output;
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

// This provider delegates negotiated dependencies and device access while
// refusing all host pixel reads. Its pixel context has a different layout from
// its Abstract. A successful pipeline therefore establishes both substitution
// and resident access through supplied operations.
class ForwardedImage {
 public:
  // Explicit construction makes the additional Retained acquisition visible
  // at the call site. The fixture releases it after its final forwarded use.
  explicit ForwardedImage(Abstract image)
      : acquired(retain(image)),
        pixels(bound<Pixels>(acquired.get_abstract())),
        pixel_state{this} {}
  auto get_abstract() -> Abstract { return Abstract({this, &operations}); }
  auto release() -> ttx_release_status { return release_context(this); }
  Count host_reads = 0;
  Count acquisitions = 1;

 private:
  struct PixelState {
    ForwardedImage* provider;
  };

  static auto graph(void* context) -> ttx_abstract {
    return {context, &operations};
  }

  static auto pixel_graph(void* context) -> ttx_abstract {
    return graph(Data::cast<PixelState>(context)->provider);
  }

  static auto dimensions(void* context) -> compute_image_dimensions {
    return Data::cast<PixelState>(context)->provider->pixels.dimensions();
  }

  static auto read(void* context, U8* output, Count count)
      -> compute_image_status {
    ++Data::cast<PixelState>(context)->provider->host_reads;
    return COMPUTE_IMAGE_FAILED;
  }

  static auto acquire(void* context, ttx_retained* output)
      -> ttx_negotiation_status {
    auto& image = *Data::cast<ForwardedImage>(context);
    ++image.acquisitions;
    *output = {context, &lifetime};
    return TTX_NEGOTIATION_SATISFIED;
  }

  static auto release_context(void* context) -> ttx_release_status {
    auto& image = *Data::cast<ForwardedImage>(context);
    if (--image.acquisitions == 0) {
      return ttx_release_status(image.acquired.release());
    }

    return TTX_RELEASE_SUCCESS;
  }

  static auto supports(void* context, perimortem_uuid id)
      -> ttx_negotiation_status {
    const Identity::Uuid contract(id);
    if (contract == Abstract::contract_id || contract == Pixels::contract_id ||
        contract == Capabilities::Retain::contract_id) {
      return TTX_NEGOTIATION_SATISFIED;
    }

    return ttx_negotiation_status(
        Data::cast<ForwardedImage>(context)->acquired.get_abstract().supports(
            contract));
  }

  static auto bind(void* context, perimortem_uuid id, void* output)
      -> ttx_negotiation_status {
    auto& image = *Data::cast<ForwardedImage>(context);
    const Identity::Uuid contract(id);
    if (contract == Pixels::contract_id) {
      *Data::cast<compute_pixels>(output) = {
        &image.pixel_state, &pixel_operations};
    } else if (contract == Capabilities::Retain::contract_id) {
      *Data::cast<ttx_retain>(output) = {context, &retention};
    } else if (contract == Abstract::contract_id) {
      *Data::cast<ttx_abstract>(output) = graph(context);
    } else {
      const auto supplied = image.acquired.get_abstract().get_abi();
      return supplied.operations->bind(supplied.context, id, output);
    }

    return TTX_NEGOTIATION_SATISFIED;
  }

  static auto route(void* context, perimortem_view_bytes route)
      -> ttx_abstract {
    return ttx_none();
  }

  static auto visit(void* context, ttx_graph_visitor visitor) -> void {}
  static auto state(void* context) -> ttx_abstract_state {
    return TTX_ABSTRACT_PRESENT;
  }

  static const ttx_abstract_ops operations;
  static const ttx_retained_ops lifetime;
  static const ttx_retain_ops retention;
  static const compute_pixels_ops pixel_operations;
  Policies::Retained acquired;
  Pixels pixels;
  PixelState pixel_state;
};

const ttx_abstract_ops ForwardedImage::operations = {
  supports, bind, route, visit, state};

const ttx_retained_ops ForwardedImage::lifetime = {graph, release_context};
const ttx_retain_ops ForwardedImage::retention = {graph, acquire};
const compute_pixels_ops ForwardedImage::pixel_operations = {
  pixel_graph, dimensions, read};

static auto recorded(Abstract buffer, Abstract image) -> Policies::Retained {
  return bound<Invert>(buffer).record(image).visit(
      [](Policies::Retained output) -> Policies::Retained { return output; },
      [](compute_image_status) -> Policies::Retained {
        Diagnostics::Log::fatal("Pipeline recording failed."_view);
      });
}

VALIDATION_TEST(Engines, resident_command_chain) {
  Provider cpu(cpu_library), cuda(cuda_library);
  const Abstract factories[] = {
    cpu.root(), cuda.root().resolve_route("engine"_view)};

  for (auto factory : factories) {
    const auto engine = create(factory);
    const auto buffer =
        create(engine.get_abstract().resolve_route("commands"_view));
    Storage::Option<Policies::Retained> first;
    auto callback = [&](Abstract image) {
      first = recorded(buffer.get_abstract(), image);
    };

    compute_generated_image(&callback, [](void* context, ttx_abstract image) {
      (*Data::cast<decltype(callback)>(context))(Abstract(image));
    });

    first.visit(
        [&](Policies::Retained image) {
          ForwardedImage forwarded(image.get_abstract());
          const auto other = create(engine.get_abstract());
          bound<Invert>(other.get_abstract())
              .record(forwarded.get_abstract())
              .visit(
                  [&](Policies::Retained unexpected) {
                    EXPECT(False);
                    unexpected.release();
                  },

                  [&](compute_image_status status) {
                    EXPECT_EQ(status, COMPUTE_IMAGE_INVALID);
                  });

          EXPECT(other.release() == Policies::Retained::Status::Success);

          auto output =
              recorded(buffer.get_abstract(), forwarded.get_abstract());
          for (Count i = 2; i < 64; ++i) {
            const auto next =
                recorded(buffer.get_abstract(), output.get_abstract());
            EXPECT(output.release() == Policies::Retained::Status::Success);
            output = next;
          }

          EXPECT_EQ(forwarded.release(), TTX_RELEASE_SUCCESS);
          EXPECT_EQ(forwarded.acquisitions, Count(0));
          EXPECT(image.release() == Policies::Retained::Status::Success);
          EXPECT(engine.release() == Policies::Retained::Status::Success);

          U8 unavailable[16]{};
          EXPECT_EQ(
              bound<Pixels>(output.get_abstract())
                  .read(Access::Bytes(unavailable[0], 16)),
              COMPUTE_IMAGE_INVALID);
          EXPECT_EQ(
              bound<Commands>(buffer.get_abstract()).wait(),
              COMPUTE_EXECUTION_INVALID);
          EXPECT_EQ(
              bound<Commands>(buffer.get_abstract()).submit(),
              COMPUTE_EXECUTION_SUCCESS);
          EXPECT(buffer.release() == Policies::Retained::Status::Success);

          U8 observed[16]{};
          EXPECT_EQ(
              bound<Pixels>(output.get_abstract())
                  .read(Access::Bytes(observed[0], 16)),
              COMPUTE_IMAGE_SUCCESS);
          for (Count i = 0; i < 16; ++i) {
            EXPECT_EQ(observed[i], U8(17 + i * 13));
          }

          EXPECT_EQ(forwarded.host_reads, Count(0));
          EXPECT(output.release() == Policies::Retained::Status::Success);
        },

        [&] { EXPECT(False); });
  }
}

VALIDATION_TEST(Engines, cuda_stream_dependency) {
  Provider cuda(cuda_library);
  const auto engine_a = create(cuda.root().resolve_route("engine"_view));
  const auto engine_b = create(cuda.root().resolve_route("engine"_view));
  const auto a = create(engine_a.get_abstract());
  const auto b = create(engine_b.get_abstract());
  Storage::Option<Policies::Retained> first;
  auto callback = [&](Abstract input) {
    first = recorded(a.get_abstract(), input);
  };

  compute_generated_image(&callback, [](void* context, ttx_abstract input) {
    (*Data::cast<decltype(callback)>(context))(Abstract(input));
  });

  EXPECT_EQ(
      bound<Commands>(a.get_abstract()).submit(), COMPUTE_EXECUTION_SUCCESS);
  first.visit(
      [&](Policies::Retained image) {
        ForwardedImage forwarded(image.get_abstract());
        const auto output =
            recorded(b.get_abstract(), forwarded.get_abstract());
        EXPECT_EQ(forwarded.release(), TTX_RELEASE_SUCCESS);
        EXPECT_EQ(forwarded.acquisitions, Count(1));
        EXPECT(image.release() == Policies::Retained::Status::Success);
        EXPECT(a.release() == Policies::Retained::Status::Success);
        EXPECT(engine_a.release() == Policies::Retained::Status::Success);
        EXPECT_EQ(
            bound<Commands>(b.get_abstract()).submit(),
            COMPUTE_EXECUTION_SUCCESS);
        EXPECT(b.release() == Policies::Retained::Status::Success);
        EXPECT(engine_b.release() == Policies::Retained::Status::Success);

        U8 observed[16]{};
        EXPECT_EQ(
            bound<Pixels>(output.get_abstract())
                .read(Access::Bytes(observed[0], 16)),
            COMPUTE_IMAGE_SUCCESS);
        for (Count i = 0; i < 16; ++i) {
          EXPECT_EQ(observed[i], U8(17 + i * 13));
        }

        EXPECT_EQ(forwarded.host_reads, Count(0));
        EXPECT(output.release() == Policies::Retained::Status::Success);
        EXPECT_EQ(forwarded.acquisitions, Count(0));
      },

      [&] { EXPECT(False); });
}

VALIDATION_TEST(Engines, deferred_c_input_order_and_failure) {
  Provider cuda(cuda_library);
  for (U8 failure = 0; failure < 3; ++failure) {
    const auto engine = create(cuda.root().resolve_route("engine"_view));
    const auto buffer = create(engine.get_abstract());
    compute_deferred_state states[2]{};
    Count sequence = 0;
    Storage::Option<Policies::Retained> images[2];
    for (Count i = 0; i < 2; ++i) {
      states[i].seed = 17;
      states[i].sequence = &sequence;
      states[i].fail = failure == i + 1;

      auto callback = [&](Abstract input) {
        images[i] = recorded(buffer.get_abstract(), input);
      };

      compute_deferred_image(
          &states[i], &callback, [](void* context, ttx_abstract input) {
            (*Data::cast<decltype(callback)>(context))(Abstract(input));
          });

      EXPECT_EQ(states[i].original_reads, Count(0));
      EXPECT_EQ(states[i].retained_reads, Count(0));
    }

    const auto commands = bound<Commands>(buffer.get_abstract());
    EXPECT_EQ(
        commands.submit(),
        failure ? COMPUTE_EXECUTION_FAILED : COMPUTE_EXECUTION_SUCCESS);
    EXPECT_EQ(
        commands.wait(),
        failure ? COMPUTE_EXECUTION_FAILED : COMPUTE_EXECUTION_SUCCESS);
    EXPECT_EQ(states[0].read_order, Count(1));
    EXPECT_EQ(states[1].read_order, failure == 1 ? Count(0) : Count(2));
    for (Count i = 0; i < 2; ++i) {
      images[i].visit(
          [&](Policies::Retained image) {
            U8 output[16]{};
            EXPECT_EQ(
                bound<Pixels>(image.get_abstract())
                    .read(Access::Bytes(output[0], 16)),
                failure ? COMPUTE_IMAGE_FAILED : COMPUTE_IMAGE_SUCCESS);
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

VALIDATION_TEST(Engines, cuda_final_release_completes_pending_work) {
  Provider cuda(cuda_library);
  const auto engine = create(cuda.root().resolve_route("engine"_view));
  const auto buffer = create(engine.get_abstract());
  compute_deferred_state input{};
  input.seed = 17;
  input.fail_release = 1;

  Storage::Option<Policies::Retained> image;
  auto callback = [&](Abstract value) {
    image = recorded(buffer.get_abstract(), value);
  };

  compute_deferred_image(
      &input, &callback, [](void* context, ttx_abstract value) {
        (*Data::cast<decltype(callback)>(context))(Abstract(value));
      });

  EXPECT_EQ(
      bound<Commands>(buffer.get_abstract()).submit(),
      COMPUTE_EXECUTION_SUCCESS);
  image.visit(
      [&](Policies::Retained output) {
        EXPECT(output.release() == Policies::Retained::Status::Success);
      },

      [&] { EXPECT(False); });
  EXPECT(engine.release() == Policies::Retained::Status::Success);
  EXPECT(buffer.release() == Policies::Retained::Status::CleanupFailed);
  EXPECT_EQ(input.releases, Count(1));
  EXPECT_EQ(input.retained_reads, Count(1));
}

VALIDATION_TEST(Engines, cuda_creation_keeps_acquisition_explicit) {
  Provider cuda(cuda_library);
  auto callback = [&](Abstract engine) {
    EXPECT(!engine.bind<Policies::Retained>());

    auto command_callback = [&](Abstract buffer) {
      EXPECT(!buffer.bind<Policies::Retained>());

      const auto acquired = retain(buffer);
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
      bound<Capabilities::Create>(cuda.root().resolve_route("engine"_view))
          .create(Policies::None::get_none().get_abstract(), callback) ==
      Negotiation::Status::Satisfied);
}
