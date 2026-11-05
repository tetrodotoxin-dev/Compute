// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cuda/runtime/engine.hpp"

#include <cuda.h>

#include "toolchain/memory/lifetime.hpp"

#include "perimortem/core/storage/vector.hpp"
#include "perimortem/core/view/null_terminated.hpp"
#include "perimortem/memory/allocator/global.hpp"
#include "perimortem/memory/global/vector.hpp"

#include "compute/image/composite_command.hpp"
#include "compute/image/convolve_command.hpp"
#include "compute/image/invert_command.hpp"
#include "compute/image/posterize_command.hpp"
#include "compute/reflection/module.hpp"
#include "cuda/contracts/buffer_extension.hpp"
#include "cuda/runtime/program.hpp"
#include "ttx/graph/policies/constant.hpp"
#include "ttx/graph/policies/none.h"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Toolchain::Memory;
using namespace Cuda::Runtime;
using namespace Ttx::Graph;
using namespace Compute::Image;
using namespace Compute::Reflection;

struct State {
  CUdevice device;
  CUcontext context;
  Program* program = nullptr;
  Count references = 1;
};

struct DeviceBuffer {
  State* engine;
  CUdeviceptr address;
  U64 extent;
};

static auto buffer_graph(void* context) -> ttx_abstract;

static auto buffer_address(void* context, void* driver_context, U64* output)
    -> ttx_negotiation_status {
  const auto& buffer = *Data::cast<DeviceBuffer>(context);
  if (!driver_context || !output || buffer.engine->context != driver_context) {
    return TTX_NEGOTIATION_REJECTED;
  }

  *output = buffer.address;
  return TTX_NEGOTIATION_SATISFIED;
}

static const cuda_buffer_extension_ops buffer_extension_operations{
  buffer_address};

static auto buffer_supports(void* context, perimortem_uuid id)
    -> ttx_negotiation_status {
  if (!id.high && id.low <= 1) {
    return TTX_NEGOTIATION_REJECTED;
  }

  const Identity::Uuid requested(id);
  return requested == Abstract::contract_id ||
                 requested == Policies::Constant::contract_id ||
                 requested == Cuda::Contracts::BufferExtension::contract_id
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

static auto buffer_bind(void* context, perimortem_uuid id, void* output)
    -> ttx_negotiation_status {
  const auto status = buffer_supports(context, id);
  if (status != TTX_NEGOTIATION_SATISFIED || !output) {
    return status == TTX_NEGOTIATION_SATISFIED ? TTX_NEGOTIATION_REJECTED
                                               : status;
  }

  if (Identity::Uuid(id) == Cuda::Contracts::BufferExtension::contract_id) {
    *Data::cast<cuda_buffer_extension>(output) = {
      context, &buffer_extension_operations};
  } else {
    *Data::cast<ttx_abstract>(output) = buffer_graph(context);
  }

  return TTX_NEGOTIATION_SATISFIED;
}

static auto buffer_route(void* context, perimortem_view_bytes route)
    -> ttx_abstract {
  return ttx_none();
}

static auto buffer_visit(void* context, ttx_graph_visitor visitor) -> void {}

static auto buffer_state(void* context) -> ttx_abstract_state {
  return TTX_ABSTRACT_PRESENT;
}

static const ttx_abstract_ops buffer_abstract_operations{
  buffer_supports, buffer_bind, buffer_route, buffer_visit, buffer_state};

static auto buffer_graph(void* context) -> ttx_abstract {
  return {context, &buffer_abstract_operations};
}

static auto finish(CUresult status) -> compute_execution_status {
  CUcontext previous = nullptr;
  const auto restored = cuCtxPopCurrent(&previous);
  return status == CUDA_SUCCESS && restored == CUDA_SUCCESS
             ? COMPUTE_EXECUTION_SUCCESS
             : COMPUTE_EXECUTION_FAILED;
}

static auto release_engine(void* context) -> compute_execution_status {
  auto* engine = Data::cast<State>(context);
  if (--engine->references) {
    return COMPUTE_EXECUTION_SUCCESS;
  }

  const auto status = cuDevicePrimaryCtxRelease(engine->device);
  auto outcome = status == CUDA_SUCCESS ? COMPUTE_EXECUTION_SUCCESS
                                        : COMPUTE_EXECUTION_FAILED;
  if (engine->program &&
      engine->program->release() != COMPUTE_EXECUTION_SUCCESS) {
    outcome = COMPUTE_EXECUTION_FAILED;
  }

  engine->~State();
  Allocator::Global::release(Data::cast<U8>(engine));
  return outcome;
}

static auto size(void* context) -> U64 {
  return Data::cast<DeviceBuffer>(context)->extent;
}

static auto read(void* context, U64 offset, U8* output, U64 count)
    -> compute_execution_status {
  const auto& buffer = *Data::cast<DeviceBuffer>(context);
  if (offset > buffer.extent || count > buffer.extent - offset ||
      (count && !output)) {
    return COMPUTE_EXECUTION_INVALID;
  }

  if (!count) {
    return COMPUTE_EXECUTION_SUCCESS;
  }

  if (cuCtxPushCurrent(buffer.engine->context) != CUDA_SUCCESS) {
    return COMPUTE_EXECUTION_FAILED;
  }

  return finish(cuMemcpyDtoH(output, buffer.address + offset, count));
}

static auto write(void* context, U64 offset, perimortem_view_bytes input)
    -> compute_execution_status {
  const auto& buffer = *Data::cast<DeviceBuffer>(context);
  if (offset > buffer.extent || input.size > buffer.extent - offset ||
      (input.size && !input.data)) {
    return COMPUTE_EXECUTION_INVALID;
  }

  if (!input.size) {
    return COMPUTE_EXECUTION_SUCCESS;
  }

  if (cuCtxPushCurrent(buffer.engine->context) != CUDA_SUCCESS) {
    return COMPUTE_EXECUTION_FAILED;
  }

  return finish(cuMemcpyHtoD(buffer.address + offset, input.data, input.size));
}

static auto release_buffer(void* context) -> compute_execution_status {
  auto* buffer = Data::cast<DeviceBuffer>(context);
  auto status = COMPUTE_EXECUTION_FAILED;
  if (cuCtxPushCurrent(buffer->engine->context) == CUDA_SUCCESS) {
    status = finish(cuMemFree(buffer->address));
  }

  if (release_engine(buffer->engine) != COMPUTE_EXECUTION_SUCCESS) {
    status = COMPUTE_EXECUTION_FAILED;
  }

  buffer->~DeviceBuffer();
  Allocator::Global::release(Data::cast<U8>(buffer));
  return status;
}

static const compute_buffer_ops buffer_operations{
  buffer_graph, size, read, write, release_buffer};

static auto allocate(void* context, U64 size) -> compute_allocation {
  auto& engine = *Data::cast<State>(context);
  if (!size || size > Count(-1) || engine.references == Count(-1)) {
    return {COMPUTE_EXECUTION_INVALID, {}};
  }

  if (cuCtxPushCurrent(engine.context) != CUDA_SUCCESS) {
    return {COMPUTE_EXECUTION_FAILED, {}};
  }

  CUdeviceptr address = 0;
  const auto allocated = cuMemAlloc(&address, size);
  const U32 synchronized = 1;
  const auto ordered =
      allocated == CUDA_SUCCESS
          ? cuPointerSetAttribute(
                &synchronized, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, address)
          : allocated;
  const auto status = finish(ordered);
  if (status != COMPUTE_EXECUTION_SUCCESS) {
    // Buffer promises that host transfers complete before returning. CUDA can
    // otherwise leave a nominally synchronous transfer pending, allowing the
    // non-blocking command stream to consume a partial write. Allocation fails
    // when that ordering guarantee cannot be established.
    //
    // `cuMemAlloc` can also succeed before attribute or context restoration
    // fails. Free that allocation using its context before returning failure.
    if (allocated == CUDA_SUCCESS &&
        cuCtxPushCurrent(engine.context) == CUDA_SUCCESS) {
      finish(cuMemFree(address));
    }

    return {COMPUTE_EXECUTION_FAILED, {}};
  }

  auto storage = Allocator::Global::allocate(sizeof(DeviceBuffer));
  auto& buffer = Lifetime::construct<DeviceBuffer>(
      storage.get_data(), [&] { return DeviceBuffer{&engine, address, size}; });
  ++engine.references;

  return {COMPUTE_EXECUTION_SUCCESS, {&buffer, &buffer_operations}};
}

class Recording {
 public:
  static auto create(State& engine) -> compute_commands_result;

 private:
  enum class Kind { Invert, Posterize, Convolve, Composite };
  enum class Phase { Recording, Submitted, FailedPending, Complete, Failed };
  struct Operation {
    Kind kind;
    compute_invert_command invert{};
    compute_posterize_command posterize{};
    compute_convolve_command convolve{};
    compute_composite_command composite{};
  };

  struct ResolvedBuffer {
    CUdeviceptr address = 0;
    U64 extent = 0;
  };

  Recording(Program& program, CUstream stream)
      : program(program), stream(stream) {}
  static auto submit(void* context) -> compute_execution_status;
  static auto wait(void* context) -> compute_execution_status;
  static auto release(void* context) -> compute_execution_status;
  static auto supports(void* context, perimortem_uuid command)
      -> ttx_negotiation_status;
  static auto record(
      void* context,
      perimortem_uuid command,
      const void* arguments) -> compute_execution_status;
  auto launch(const Operation& operation) -> compute_execution_status;
  auto function(const char* name, CUfunction& output) -> bool;
  auto buffer(compute_buffer value, ResolvedBuffer& output) -> Bool;
  auto finish() -> compute_execution_status;

  Program& program;
  CUstream stream;
  Phase phase = Phase::Recording;
  Global::Vector<Operation> operations;
};

static constexpr auto commands_source = R"(
extern "C" __global__ void compute_invert(
    const unsigned char* input,
    unsigned char* output,
    unsigned long long count) {
  unsigned long long i =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) output[i] = (i % 4 == 3) ? input[i] : 255 - input[i];
}

extern "C" __global__ void compute_posterize(
    const unsigned char* input,
    unsigned char* output,
    unsigned long long count,
    unsigned levels) {
  unsigned long long i =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    output[i] = (i % 4 == 3)
        ? input[i]
        : ((input[i] * (levels - 1) + 127) / 255) * 255 / (levels - 1);
  }
}

extern "C" __global__ void compute_convolve(
    const unsigned char* input,
    unsigned char* output,
    const float* weights,
    unsigned width,
    unsigned height,
    unsigned kw,
    unsigned kh) {
  unsigned pixel = blockIdx.x * blockDim.x + threadIdx.x;
  if (pixel >= width * height) return;
  unsigned x = pixel % width, y = pixel / width;
  for (unsigned channel = 0; channel != 3; ++channel) {
    float sum = 0;
    for (unsigned ky = 0; ky != kh; ++ky) {
      int sy = int(y) + int(kh / 2) - int(ky);
      if (sy < 0 || sy >= int(height)) continue;
      for (unsigned kx = 0; kx != kw; ++kx) {
        int sx = int(x) + int(kw / 2) - int(kx);
        if (sx >= 0 && sx < int(width)) {
          sum += input[(sy * width + sx) * 4 + channel] *
                 weights[ky * kw + kx];
        }
      }
    }
    output[pixel * 4 + channel] =
        (unsigned char)fminf(255.0f, fmaxf(0.0f, floorf(sum + 0.5f)));
  }
  output[pixel * 4 + 3] = input[pixel * 4 + 3];
}

extern "C" __global__ void compute_composite(
    const unsigned char* back,
    const unsigned char* front,
    unsigned char* output,
    unsigned long long count) {
  unsigned long long pixel =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (pixel >= count) return;
  unsigned long long i = pixel * 4;
  unsigned a = front[i + 3], b = back[i + 3];
  unsigned alpha = a * 255 + b * (255 - a);
  for (unsigned c = 0; c != 3; ++c) {
    unsigned color =
        front[i + c] * a * 255 + back[i + c] * b * (255 - a);
    output[i + c] = alpha ? (color + alpha / 2) / alpha : 0;
  }
  output[i + 3] = (alpha + 127) / 255;
}
)"_view;

auto Recording::create(State& engine) -> compute_commands_result {
  if (!engine.program) {
    Program::create("compute_commands.cu"_view, commands_source, engine.device)
        .visit(
            [&](Program& program) { engine.program = &program; },
            [](compute_execution_status status) {});
  }

  if (!engine.program ||
      cuCtxPushCurrent(engine.program->get_context()) != CUDA_SUCCESS) {
    return {COMPUTE_EXECUTION_FAILED, {}};
  }

  CUstream stream = nullptr;
  const auto status = ::finish(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  if (status != COMPUTE_EXECUTION_SUCCESS) {
    return {status, {}};
  }

  engine.program->retain();

  auto storage = Allocator::Global::allocate(sizeof(Recording));
  auto& recording = Lifetime::construct<Recording>(
      storage.get_data(), [&] { return Recording(*engine.program, stream); });
  static const compute_commands_ops operations{
    submit, wait, release, supports, record};

  return {COMPUTE_EXECUTION_SUCCESS, {&recording, &operations}};
}

auto Recording::supports(void* context, perimortem_uuid command)
    -> ttx_negotiation_status {
  const Identity::Uuid requested(command);
  return requested == InvertCommand::contract_id ||
                 requested == PosterizeCommand::contract_id ||
                 requested == ConvolveCommand::contract_id ||
                 requested == CompositeCommand::contract_id
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

auto Recording::record(
    void* context,
    perimortem_uuid command,
    const void* arguments) -> compute_execution_status {
  auto& recording = *Data::cast<Recording>(context);
  if (recording.phase != Phase::Recording || !arguments ||
      supports(context, command) != TTX_NEGOTIATION_SATISFIED) {
    return COMPUTE_EXECUTION_INVALID;
  }

  Operation operation{};
  const Identity::Uuid requested(command);
  if (requested == InvertCommand::contract_id) {
    operation.kind = Kind::Invert;
    operation.invert = *Data::cast<const compute_invert_command>(arguments);
  } else if (requested == PosterizeCommand::contract_id) {
    operation.kind = Kind::Posterize;
    operation.posterize =
        *Data::cast<const compute_posterize_command>(arguments);
  } else if (requested == ConvolveCommand::contract_id) {
    operation.kind = Kind::Convolve;
    operation.convolve = *Data::cast<const compute_convolve_command>(arguments);
  } else {
    operation.kind = Kind::Composite;
    operation.composite =
        *Data::cast<const compute_composite_command>(arguments);
  }

  recording.operations.insert(operation);
  return COMPUTE_EXECUTION_SUCCESS;
}

auto Recording::buffer(compute_buffer value, ResolvedBuffer& output) -> Bool {
  if (!value.context || !value.operations || !value.operations->get_abstract ||
      !value.operations->size) {
    return False;
  }

  Bool accepted = False;
  Abstract(value.operations->get_abstract(value.context))
      .bind<Cuda::Contracts::BufferExtension>()
      .visit(
          [&](Cuda::Contracts::BufferExtension extension) {
            extension.address(program.get_context())
                .visit(
                    [&](U64 address) {
                      output = {
                        CUdeviceptr(address),
                        value.operations->size(value.context)};

                      accepted = True;
                    },

                    [] {});
          },

          [] {});
  return accepted;
}

auto Recording::function(const char* name, CUfunction& output) -> bool {
  return cuModuleGetFunction(&output, program.get_module(), name) ==
         CUDA_SUCCESS;
}

auto Recording::launch(const Operation& operation) -> compute_execution_status {
  CUfunction kernel = nullptr;
  Storage::Vector<void*, 7> arguments;
  U64 count = 0;
  U32 width = 0, height = 0, kernel_width = 0, kernel_height = 0, levels = 0;
  CUdeviceptr first = 0, second = 0, third = 0;
  ResolvedBuffer first_buffer, second_buffer, third_buffer;
  if (operation.kind == Kind::Invert) {
    const auto& command = operation.invert;
    if (!buffer(command.input, first_buffer) ||
        !buffer(command.output, second_buffer) || !command.count ||
        command.count % 4 || first_buffer.extent < command.count ||
        second_buffer.extent < command.count ||
        !function("compute_invert", kernel)) {
      return COMPUTE_EXECUTION_INVALID;
    }

    first = first_buffer.address;
    second = second_buffer.address;
    count = command.count;
    arguments[0] = &first;
    arguments[1] = &second;
    arguments[2] = &count;
  } else if (operation.kind == Kind::Posterize) {
    const auto& command = operation.posterize;
    if (!buffer(command.input, first_buffer) ||
        !buffer(command.output, second_buffer) || !command.count ||
        command.count % 4 || command.levels < 2 || command.levels > 256 ||
        first_buffer.extent < command.count ||
        second_buffer.extent < command.count ||
        !function("compute_posterize", kernel)) {
      return COMPUTE_EXECUTION_INVALID;
    }

    first = first_buffer.address;
    second = second_buffer.address;
    count = command.count;
    levels = command.levels;
    arguments[0] = &first;
    arguments[1] = &second;
    arguments[2] = &count;
    arguments[3] = &levels;
  } else if (operation.kind == Kind::Convolve) {
    const auto& command = operation.convolve;
    count = U64(command.width) * command.height;

    const U64 weight_size =
        U64(command.kernel_width) * command.kernel_height * sizeof(R32);
    if (!buffer(command.input, first_buffer) ||
        !buffer(command.output, second_buffer) ||
        !buffer(command.weights, third_buffer) || !count || count > U32(-1) ||
        !command.kernel_width || !command.kernel_height ||
        !(command.kernel_width & 1) || !(command.kernel_height & 1) ||
        first_buffer.extent < count * 4 || second_buffer.extent < count * 4 ||
        third_buffer.extent < weight_size ||
        !function("compute_convolve", kernel)) {
      return COMPUTE_EXECUTION_INVALID;
    }

    first = first_buffer.address;
    second = second_buffer.address;
    third = third_buffer.address;
    width = command.width;
    height = command.height;
    kernel_width = command.kernel_width;
    kernel_height = command.kernel_height;
    arguments[0] = &first;
    arguments[1] = &second;
    arguments[2] = &third;
    arguments[3] = &width;
    arguments[4] = &height;
    arguments[5] = &kernel_width;
    arguments[6] = &kernel_height;
  } else {
    const auto& command = operation.composite;
    if (!buffer(command.back, first_buffer) ||
        !buffer(command.front, second_buffer) ||
        !buffer(command.output, third_buffer) || !command.count ||
        command.count > Count(-1) / 4 ||
        first_buffer.extent < command.count * 4 ||
        second_buffer.extent < command.count * 4 ||
        third_buffer.extent < command.count * 4 ||
        !function("compute_composite", kernel)) {
      return COMPUTE_EXECUTION_INVALID;
    }

    first = first_buffer.address;
    second = second_buffer.address;
    third = third_buffer.address;
    count = command.count;
    arguments[0] = &first;
    arguments[1] = &second;
    arguments[2] = &third;
    arguments[3] = &count;
  }

  const U64 work =
      operation.kind == Kind::Convolve ? U64(width) * height : count;
  if (work > U64(U32(-1)) * 256) {
    return COMPUTE_EXECUTION_INVALID;
  }

  return cuLaunchKernel(
             kernel, U32((work + 255) / 256), 1, 1, 256, 1, 1, 0, stream,
             arguments.get_data(), nullptr) == CUDA_SUCCESS
             ? COMPUTE_EXECUTION_SUCCESS
             : COMPUTE_EXECUTION_FAILED;
}

auto Recording::submit(void* context) -> compute_execution_status {
  auto& recording = *Data::cast<Recording>(context);
  if (recording.phase != Phase::Recording ||
      cuCtxPushCurrent(recording.program.get_context()) != CUDA_SUCCESS) {
    return COMPUTE_EXECUTION_INVALID;
  }

  auto outcome = COMPUTE_EXECUTION_SUCCESS;
  for (Count i = 0; i < recording.operations.get_size(); ++i) {
    outcome = recording.launch(recording.operations.get_data()[i]);
    if (outcome != COMPUTE_EXECUTION_SUCCESS) {
      break;
    }
  }

  if (::finish(CUDA_SUCCESS) != COMPUTE_EXECUTION_SUCCESS) {
    outcome = COMPUTE_EXECUTION_FAILED;
  }

  recording.phase = outcome == COMPUTE_EXECUTION_SUCCESS ? Phase::Submitted
                                                         : Phase::FailedPending;
  return outcome;
}

auto Recording::wait(void* context) -> compute_execution_status {
  auto& recording = *Data::cast<Recording>(context);
  if (recording.phase == Phase::Recording) {
    return COMPUTE_EXECUTION_INVALID;
  }

  if (recording.phase == Phase::Complete) {
    return COMPUTE_EXECUTION_SUCCESS;
  }

  if (recording.phase == Phase::Failed) {
    return COMPUTE_EXECUTION_FAILED;
  }

  const auto submitted = recording.phase == Phase::Submitted;
  if (cuCtxPushCurrent(recording.program.get_context()) != CUDA_SUCCESS) {
    recording.phase = Phase::Failed;
    return COMPUTE_EXECUTION_FAILED;
  }

  const auto outcome = ::finish(cuStreamSynchronize(recording.stream));
  recording.phase = outcome == COMPUTE_EXECUTION_SUCCESS && submitted
                        ? Phase::Complete
                        : Phase::Failed;
  return recording.phase == Phase::Complete ? COMPUTE_EXECUTION_SUCCESS
                                            : COMPUTE_EXECUTION_FAILED;
}

auto Recording::finish() -> compute_execution_status {
  auto outcome = COMPUTE_EXECUTION_SUCCESS;
  if ((phase == Phase::Submitted || phase == Phase::FailedPending) &&
      wait(this) != COMPUTE_EXECUTION_SUCCESS) {
    outcome = COMPUTE_EXECUTION_FAILED;
  }

  if (cuCtxPushCurrent(program.get_context()) != CUDA_SUCCESS) {
    outcome = COMPUTE_EXECUTION_FAILED;
  } else if (::finish(cuStreamDestroy(stream)) != COMPUTE_EXECUTION_SUCCESS) {
    outcome = COMPUTE_EXECUTION_FAILED;
  }

  if (program.release() != COMPUTE_EXECUTION_SUCCESS) {
    outcome = COMPUTE_EXECUTION_FAILED;
  }

  return outcome;
}

auto Recording::release(void* context) -> compute_execution_status {
  auto* recording = Data::cast<Recording>(context);
  const auto outcome = recording->finish();
  recording->~Recording();
  Allocator::Global::release(Data::cast<U8>(recording));
  return outcome;
}

static auto create_commands(void* context) -> compute_commands_result {
  return Recording::create(*Data::cast<State>(context));
}

static auto publication() -> Module& {
  static U8 context;
  static Module value("ComputeCuda"_view, {&context, [](void* context) {
                                             return Engine::create();
                                           }});

  return value;
}

static auto supports(void* context, perimortem_uuid id)
    -> ttx_negotiation_status {
  const auto api = publication().get_abstract().get_abi();
  return api.operations->supports(api.context, id);
}

static auto bind(void* context, perimortem_uuid id, void* output)
    -> ttx_negotiation_status {
  const auto api = publication().get_abstract().get_abi();
  return api.operations->bind(api.context, id, output);
}

static auto route(void* context, perimortem_view_bytes path) -> ttx_abstract {
  const auto api = publication().get_abstract().get_abi();
  return api.operations->resolve_route(api.context, path);
}

static auto visit(void* context, ttx_graph_visitor visitor) -> void {
  const auto api = publication().get_abstract().get_abi();
  api.operations->visit_routes(api.context, visitor);
}

static auto state(void* context) -> ttx_abstract_state {
  return TTX_ABSTRACT_PRESENT;
}

auto Engine::create() -> compute_engine_result {
  CUdevice device;
  CUcontext context;
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS) {
    return {COMPUTE_EXECUTION_FAILED, {}};
  }

  auto storage = Allocator::Global::allocate(sizeof(State));
  auto& engine = Lifetime::construct<State>(
      storage.get_data(), [&] { return State{device, context}; });
  static const compute_engine_ops operations{
    allocate, create_commands, release_engine};

  return {COMPUTE_EXECUTION_SUCCESS, {&engine, &operations}};
}

auto Engine::factory() -> Abstract {
  static const ttx_abstract_ops operations{supports, bind, route, visit, state};
  return Abstract({&publication(), &operations});
}
