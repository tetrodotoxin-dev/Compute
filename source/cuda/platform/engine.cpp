// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cuda/platform/engine.hpp"

#include "toolchain/memory/lifetime.hpp"

#include "perimortem/core/diagnostics/log.hpp"
#include "perimortem/core/view/null_terminated.hpp"
#include "perimortem/memory/allocator/global.hpp"

#include "ttx/graph/capabilities/create.hpp"
#include "ttx/graph/policies/none.h"

#include "compute/image/invert.hpp"
#include "compute/policies/accelerated.h"
#include "cuda/contracts/kernel.hpp"
#include "cuda/platform/image.hpp"
#include "cuda/runtime/program.hpp"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Toolchain::Memory;
using namespace Ttx::Graph;
using namespace Ttx::Data;
using namespace Compute::Image;
using namespace Cuda::Platform;
using namespace Cuda::Runtime;

struct ImageArguments {
  U64 input, output, count;
};

TTX_DATA_RECORD(
    ImageArguments,
    TTX_DATA_MEMBER(ImageArguments, input),
    TTX_DATA_MEMBER(ImageArguments, output),
    TTX_DATA_MEMBER(ImageArguments, count));

static constexpr auto source = R"CUDA(
extern "C" __global__ void invert(const unsigned char* input,
                                  unsigned char* output, unsigned long long count) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) output[i] = i % 4 == 3 ? input[i] : 255 - input[i];
}
)CUDA"_view;

// EngineProvider retains compiled code once. Its output images retain their
// own buffers, so engine release can precede every later pixel read or input
// transfer performed through those images.
class EngineProvider {
 public:
  EngineProvider(
      ttx_retained program,
      cuda_program api,
      ttx_retained kernel,
      cuda_kernel launch)
      : program(program),
        program_api(api),
        kernel(kernel),
        kernel_api(launch) {}

  static auto graph(void* context) -> ttx_abstract {
    return {context, &graph_operations};
  }

  static auto factory_graph(void* context) -> ttx_abstract {
    return {context, &factory_operations};
  }

  static auto supports(void* context, perimortem_uuid id)
      -> ttx_negotiation_status {
    if (!id.high && id.low <= 1) {
      return TTX_NEGOTIATION_REJECTED;
    }

    const Identity::Uuid contract(id);
    return contract == Abstract::contract_id ||
                   contract == Invert::contract_id ||
                   contract == Capabilities::Retain::contract_id ||
                   contract == Policies::Retained::contract_id ||
                   contract == Identity::Uuid(
                                   COMPUTE_ACCELERATED_ID_HIGH,
                                   COMPUTE_ACCELERATED_ID_LOW)
               ? TTX_NEGOTIATION_SATISFIED
               : TTX_NEGOTIATION_UNKNOWN;
  }

  static auto bind(void* context, perimortem_uuid id, void* output)
      -> ttx_negotiation_status {
    const auto status = supports(context, id);
    if (status != TTX_NEGOTIATION_SATISFIED) {
      return status;
    }

    const Identity::Uuid contract(id);
    if (contract == Invert::contract_id) {
      *Data::cast<compute_invert>(output) = {context, &invert_operations};
    } else if (contract == Capabilities::Retain::contract_id) {
      *Data::cast<ttx_retain>(output) = {context, &retain_operations};
    } else if (contract == Policies::Retained::contract_id) {
      *Data::cast<ttx_retained>(output) = {context, &retained_operations};
    } else {
      *Data::cast<ttx_abstract>(output) = graph(context);
    }

    return TTX_NEGOTIATION_SATISFIED;
  }

  static auto route(void* context, perimortem_view_bytes route)
      -> ttx_abstract {
    return View::Bytes(route) == "invert"_view ? graph(context) : ttx_none();
  }

  static auto visit(void* context, ttx_graph_visitor visitor) -> void {
    visitor.callback(visitor.callback_context, "invert"_view, graph(context));
  }

  static auto state(void* context) -> ttx_abstract_state {
    return TTX_ABSTRACT_PRESENT;
  }

  static auto empty_route(void* context, perimortem_view_bytes route)
      -> ttx_abstract {
    return ttx_none();
  }

  static auto empty_visit(void* context, ttx_graph_visitor visitor) -> void {}

  static auto retain(void* context, ttx_retained* output)
      -> ttx_negotiation_status {
    auto& engine = *Data::cast<EngineProvider>(context);
    if (engine.references == Count(-1)) {
      return TTX_NEGOTIATION_REJECTED;
    }

    ++engine.references;
    *output = {context, &retained_operations};
    return TTX_NEGOTIATION_SATISFIED;
  }

  static auto release(void* context) -> ttx_release_status {
    auto* engine = Data::cast<EngineProvider>(context);
    if (--engine->references) {
      return TTX_RELEASE_SUCCESS;
    }

    const auto kernel =
        engine->kernel.operations->release(engine->kernel.context);
    const auto program =
        engine->program.operations->release(engine->program.context);
    engine->~EngineProvider();
    Allocator::Global::release(Data::cast<U8>(engine));
    return kernel == TTX_RELEASE_SUCCESS && program == TTX_RELEASE_SUCCESS
               ? TTX_RELEASE_SUCCESS
               : TTX_RELEASE_CLEANUP_FAILED;
  }

  static auto apply(void* context, ttx_abstract input, ttx_retained* output)
      -> compute_image_status {
    if (!output) {
      return COMPUTE_IMAGE_INVALID;
    }

    auto& engine = *Data::cast<EngineProvider>(context);
    return Abstract(input).bind<Pixels>().visit(
        [&](Pixels pixels) -> compute_image_status {
          const auto size = pixels.dimensions();
          const U64 count = U64(size.width) * size.height;
          if (!count || count > Count(-1) / 4 || count > U64(U32(-1)) * 64) {
            return COMPUTE_IMAGE_INVALID;
          }

          Global::Bytes bytes;
          bytes.forgetful_resize(count * 4);

          const auto read = pixels.read(bytes.get_access());
          if (read != COMPUTE_IMAGE_SUCCESS) {
            return read;
          }

          // Inputs cross the domain boundary as pixels. Only this CUDA provider
          // obtains device addresses from buffers it allocated in its Program.
          ttx_retained source{}, destination{};
          if (engine.program_api.operations->allocate(
                  engine.program_api.context, bytes.get_size(), &source) !=
              CUDA_SUCCESS_STATUS) {
            return COMPUTE_IMAGE_FAILED;
          }

          if (engine.program_api.operations->allocate(
                  engine.program_api.context, bytes.get_size(), &destination) !=
              CUDA_SUCCESS_STATUS) {
            source.operations->release(source.context);
            return COMPUTE_IMAGE_FAILED;
          }

          const auto source_graph =
              Abstract(source.operations->get_abstract(source.context));
          const auto destination_graph = Abstract(
              destination.operations->get_abstract(destination.context));
          compute_image_status status = COMPUTE_IMAGE_FAILED;
          cuda_buffer from{}, to{};
          const auto from_graph = source_graph.get_abi();
          const auto to_graph = destination_graph.get_abi();
          const perimortem_uuid buffer_id{
            TTX_CUDA_BUFFER_HIGH, TTX_CUDA_BUFFER_LOW};

          if (from_graph.operations->bind(
                  from_graph.context, buffer_id, &from) ==
                  TTX_NEGOTIATION_SATISFIED &&
              to_graph.operations->bind(to_graph.context, buffer_id, &to) ==
                  TTX_NEGOTIATION_SATISFIED &&
              from.operations->write(from.context, 0, bytes.get_view()) ==
                  CUDA_SUCCESS_STATUS) {
            ImageArguments frame{
              from.operations->address(from.context),
              to.operations->address(to.context), bytes.get_size()};

            const cuda_launch geometry{
              U32((bytes.get_size() + 255) / 256), 1, 1, 256, 1, 1, 0};

            if (engine.kernel_api.operations->launch(
                    engine.kernel_api.context, geometry,
                    Form::Storage::from(frame).get_abi()) ==
                CUDA_SUCCESS_STATUS) {
              status = COMPUTE_IMAGE_SUCCESS;
            }
          }

          if (source.operations->release(source.context) !=
              TTX_RELEASE_SUCCESS) {
            status = COMPUTE_IMAGE_FAILED;
          }

          if (status != COMPUTE_IMAGE_SUCCESS) {
            destination.operations->release(destination.context);
            return status;
          }

          return Image::create(size, Policies::Retained(destination))
              .visit(
                  [&](Policies::Retained image) -> compute_image_status {
                    *output = image.get_abi();
                    return COMPUTE_IMAGE_SUCCESS;
                  },

                  [](compute_image_status failure) -> compute_image_status {
                    return failure;
                  });
        },

        []() -> compute_image_status { return COMPUTE_IMAGE_INVALID; });
  }

  static auto create(
      void* context,
      ttx_abstract arguments,
      void* callback_context,
      void (*callback)(void*, ttx_abstract)) -> ttx_negotiation_status {
    if (Abstract(arguments).get_state() != Abstract::State::None) {
      return TTX_NEGOTIATION_REJECTED;
    }

    const auto root = Program::compiler().get_abi();
    cuda_compiler compiler{};
    if (root.operations->bind(
            root.context, {TTX_CUDA_COMPILER_HIGH, TTX_CUDA_COMPILER_LOW},
            &compiler) != TTX_NEGOTIATION_SATISFIED) {
      return TTX_NEGOTIATION_REJECTED;
    }

    const cuda_compile_request request{
      {"invert.cu"_view, source}, nullptr, 0, nullptr, 0, 0};

    ttx_retained program{}, kernel{};
    if (compiler.operations->compile(compiler.context, request, {}, &program) !=
        CUDA_SUCCESS_STATUS) {
      return TTX_NEGOTIATION_REJECTED;
    }

    const auto program_graph =
        program.operations->get_abstract(program.context);
    cuda_program api{};
    if (program_graph.operations->bind(
            program_graph.context,
            {TTX_CUDA_PROGRAM_HIGH, TTX_CUDA_PROGRAM_LOW},
            &api) != TTX_NEGOTIATION_SATISFIED) {
      program.operations->release(program.context);
      return TTX_NEGOTIATION_REJECTED;
    }

    const auto& frame = Form::Compiled<
        Form::Native<ImageArguments>::reference>::get_representation();
    const auto& integer =
        Form::Compiled<Form::Native<U64>::reference>::get_representation();
    const cuda_argument fields[] = {
      {__builtin_offsetof(ImageArguments, input), &integer.get_abi()},
      {__builtin_offsetof(ImageArguments, output), &integer.get_abi()},
      {__builtin_offsetof(ImageArguments, count), &integer.get_abi()}};

    if (api.operations->prepare(
            api.context, "invert"_view, &frame.get_abi(), fields, 3, {},
            &kernel) != CUDA_SUCCESS_STATUS) {
      program.operations->release(program.context);
      return TTX_NEGOTIATION_REJECTED;
    }

    const auto kernel_graph = kernel.operations->get_abstract(kernel.context);
    cuda_kernel launch{};
    if (kernel_graph.operations->bind(
            kernel_graph.context, {TTX_CUDA_KERNEL_HIGH, TTX_CUDA_KERNEL_LOW},
            &launch) != TTX_NEGOTIATION_SATISFIED) {
      kernel.operations->release(kernel.context);
      program.operations->release(program.context);
      return TTX_NEGOTIATION_REJECTED;
    }

    static_assert(
        alignof(EngineProvider) <= Allocator::Global::allocation_alignment);
    auto memory = Allocator::Global::allocate(sizeof(EngineProvider));
    auto& engine = Lifetime::construct<EngineProvider>(memory.get_data(), [&] {
      return EngineProvider(program, api, kernel, launch);
    });

    callback(callback_context, graph(&engine));
    if (release(&engine) != TTX_RELEASE_SUCCESS) {
      Diagnostics::Log::error(
          "CUDA engine cleanup failed after its creation callback."_view);
    }

    return TTX_NEGOTIATION_SATISFIED;
  }

  static auto factory_supports(void* context, perimortem_uuid id)
      -> ttx_negotiation_status {
    if (!id.high && id.low <= 1) {
      return TTX_NEGOTIATION_REJECTED;
    }

    const Identity::Uuid contract(id);
    return contract == Abstract::contract_id ||
                   contract == Capabilities::Create::contract_id
               ? TTX_NEGOTIATION_SATISFIED
               : TTX_NEGOTIATION_UNKNOWN;
  }

  static auto factory_bind(void* context, perimortem_uuid id, void* output)
      -> ttx_negotiation_status {
    const auto status = factory_supports(context, id);
    if (status != TTX_NEGOTIATION_SATISFIED) {
      return status;
    }

    if (Identity::Uuid(id) == Capabilities::Create::contract_id) {
      *Data::cast<ttx_create>(output) = {context, &create_operations};
    } else {
      *Data::cast<ttx_abstract>(output) = factory_graph(context);
    }

    return TTX_NEGOTIATION_SATISFIED;
  }

 private:
  Count references = 1;
  ttx_retained program;
  cuda_program program_api;
  ttx_retained kernel;
  cuda_kernel kernel_api;
  static const ttx_abstract_ops graph_operations, factory_operations;
  static const ttx_retained_ops retained_operations;
  static const ttx_retain_ops retain_operations;
  static const compute_invert_ops invert_operations;
  static const ttx_create_ops create_operations;
};

const ttx_abstract_ops EngineProvider::graph_operations = {
  supports, bind, route, visit, state};

const ttx_abstract_ops EngineProvider::factory_operations = {
  factory_supports, factory_bind, empty_route, empty_visit, state};

const ttx_retained_ops EngineProvider::retained_operations = {graph, release};
const ttx_retain_ops EngineProvider::retain_operations = {graph, retain};
const compute_invert_ops EngineProvider::invert_operations = {graph, apply};
const ttx_create_ops EngineProvider::create_operations = {
  factory_graph, create};

auto Engine::factory() -> Abstract {
  static U8 context;
  return Abstract(EngineProvider::factory_graph(&context));
}
