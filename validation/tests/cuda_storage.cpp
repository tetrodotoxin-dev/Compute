// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "toolchain/validation/test.hpp"

#include "perimortem/core/diagnostics/log.hpp"
#include "perimortem/core/view/null_terminated.hpp"
#include "perimortem/memory/global/bytes.hpp"

#include "compute/execution/buffer.hpp"
#include "compute/execution/commands.hpp"
#include "compute/execution/engine.hpp"
#include "compute/image/composite_command.hpp"
#include "compute/image/invert_command.hpp"
#include "cpu/platform/memory.hpp"
#include "cuda/contracts/buffer_extension.hpp"
#include "cuda/runtime/engine.hpp"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Toolchain::Validation;
using namespace Ttx::Data;
using namespace Compute::Execution;
using namespace Compute::Image;

static Harness CudaStorage{.name = "CUDA::Storage"};

struct ForwardingBuffer {
  compute_buffer source;
};

static auto forwarding_abstract(void* context) -> ttx_abstract {
  const auto source = Data::cast<ForwardingBuffer>(context)->source;
  return source.operations->get_abstract(source.context);
}

static auto forwarding_size(void* context) -> U64 {
  const auto source = Data::cast<ForwardingBuffer>(context)->source;
  return source.operations->size(source.context);
}

static auto forwarding_read(void* context, U64 offset, U8* output, U64 size)
    -> compute_execution_status {
  const auto source = Data::cast<ForwardingBuffer>(context)->source;
  return source.operations->read(source.context, offset, output, size);
}

static auto forwarding_write(
    void* context,
    U64 offset,
    perimortem_view_bytes input) -> compute_execution_status {
  const auto source = Data::cast<ForwardingBuffer>(context)->source;
  return source.operations->write(source.context, offset, input);
}

static auto forwarding_release(void* context) -> compute_execution_status {
  const auto source = Data::cast<ForwardingBuffer>(context)->source;
  return source.operations->release(source.context);
}

static const compute_buffer_ops forwarding_operations{
  forwarding_abstract, forwarding_size, forwarding_read, forwarding_write,
  forwarding_release};

template <typename Type>
static auto accepted(Storage::Result<Type, compute_execution_status> result)
    -> Type {
  return result.visit(
      [](Type value) -> Type { return value; },
      [](compute_execution_status) -> Type {
        Diagnostics::Log::fatal("CUDA storage fixture creation failed."_view);
      });
}

// CUDA Commands recover provider-specific access through Buffer's graph
// location. A typed simulacrum can therefore publish another operation table
// without exposing or reproducing CUDA's private Buffer state.
VALIDATION_TEST(CudaStorage, commands_bind_buffer_extension_from_simulacra) {
  const auto created = Cuda::Runtime::Engine::create();
  ASSERT_EQ(created.status, COMPUTE_EXECUTION_SUCCESS);

  Engine engine(created.engine);
  auto input = accepted(engine.allocate(8));
  auto output = accepted(engine.allocate(8));
  Storage::Vector<U8, 8> source = {{10, 20, 30, 40, 50, 60, 70, 80}};
  Storage::Vector<U8, 8> observed;
  ASSERT_EQ(
      input.write(0, View::Bytes(source[0], source.get_size())),
      COMPUTE_EXECUTION_SUCCESS);

  ForwardingBuffer forwarded_input{input.get_abi()};
  ForwardingBuffer forwarded_output{output.get_abi()};
  const compute_buffer input_api{&forwarded_input, &forwarding_operations};
  const compute_buffer output_api{&forwarded_output, &forwarding_operations};
  EXPECT(input_api.operations != input.get_abi().operations);
  EXPECT(output_api.operations != output.get_abi().operations);

  auto commands = accepted(engine.create_commands());
  const compute_invert_command command{input_api, output_api, 8};
  ASSERT_EQ(
      commands.record(InvertCommand::contract_id, command),
      COMPUTE_EXECUTION_SUCCESS);
  ASSERT_EQ(commands.submit(), COMPUTE_EXECUTION_SUCCESS);
  ASSERT_EQ(commands.wait(), COMPUTE_EXECUTION_SUCCESS);
  ASSERT_EQ(
      output.read(0, Access::Bytes(observed[0], observed.get_size())),
      COMPUTE_EXECUTION_SUCCESS);

  constexpr Storage::Vector<U8, 8> expected = {
    {245, 235, 225, 40, 205, 195, 185, 80}};

  EXPECT_EQ(observed.get_view(), expected.get_view());
  EXPECT_EQ(commands.release(), COMPUTE_EXECUTION_SUCCESS);
  EXPECT_EQ(output.release(), COMPUTE_EXECUTION_SUCCESS);
  EXPECT_EQ(input.release(), COMPUTE_EXECUTION_SUCCESS);
  EXPECT_EQ(engine.release(), COMPUTE_EXECUTION_SUCCESS);
}

VALIDATION_TEST(CudaStorage, cpu_buffer_refuses_cuda_extension) {
  const auto created = Cpu::Platform::Memory::create();
  ASSERT_EQ(created.status, COMPUTE_EXECUTION_SUCCESS);

  Engine engine(created.engine);
  auto buffer = accepted(engine.allocate(8));
  EXPECT(!buffer.get_abstract().bind<Cuda::Contracts::BufferExtension>());
  EXPECT_EQ(buffer.release(), COMPUTE_EXECUTION_SUCCESS);
  EXPECT_EQ(engine.release(), COMPUTE_EXECUTION_SUCCESS);
}

// Buffer writes happen outside the non-blocking command stream. Every returned
// success must make the complete overlay visible before the following command
// consumes it, including repeated writes to the same allocation.
VALIDATION_TEST(CudaStorage, writes_precede_non_blocking_commands) {
  constexpr U32 width = 128, height = 64;
  constexpr Count pixels = Count(width) * height;
  constexpr Count size = pixels * 4;
  constexpr U32 side = height / 3;

  const auto created = Cuda::Runtime::Engine::create();
  ASSERT_EQ(created.status, COMPUTE_EXECUTION_SUCCESS);

  Engine engine(created.engine);
  auto back = accepted(engine.allocate(size));
  auto front = accepted(engine.allocate(size));
  auto output = accepted(engine.allocate(size));
  Global::Bytes source, overlay, observed;
  source.resize(size);
  overlay.resize(size);
  observed.resize(size);
  for (Count pixel = 0; pixel < pixels; ++pixel) {
    const Count offset = pixel * 4;
    source.get_data()[offset] = U8(pixel % 251);
    source.get_data()[offset + 1] = U8((pixel * 3) % 253);
    source.get_data()[offset + 2] = U8((pixel * 7) % 255);
    source.get_data()[offset + 3] = 255;
  }

  ASSERT_EQ(back.write(0, source.get_view()), COMPUTE_EXECUTION_SUCCESS);
  for (U32 iteration = 0; iteration < 100; ++iteration) {
    for (Count i = 0; i < size; ++i) {
      overlay.get_data()[i] = 0;
    }

    const U32 left = (iteration * 17) % (width - side);
    for (U32 y = height / 3; y < height / 3 + side; ++y) {
      for (U32 x = left; x < left + side; ++x) {
        const Count offset = (Count(y) * width + x) * 4;
        overlay.get_data()[offset] = 255;
        overlay.get_data()[offset + 1] = 255;
        overlay.get_data()[offset + 2] = 255;
        overlay.get_data()[offset + 3] = 255;
      }
    }

    ASSERT_EQ(front.write(0, overlay.get_view()), COMPUTE_EXECUTION_SUCCESS);

    auto commands = accepted(engine.create_commands());
    const compute_composite_command command{
      back.get_abi(), front.get_abi(), output.get_abi(), pixels};

    ASSERT_EQ(
        commands.record(CompositeCommand::contract_id, command),
        COMPUTE_EXECUTION_SUCCESS);
    ASSERT_EQ(commands.submit(), COMPUTE_EXECUTION_SUCCESS);
    ASSERT_EQ(commands.wait(), COMPUTE_EXECUTION_SUCCESS);
    ASSERT_EQ(output.read(0, observed.get_access()), COMPUTE_EXECUTION_SUCCESS);
    ASSERT_EQ(commands.release(), COMPUTE_EXECUTION_SUCCESS);

    Bool matches = True;
    for (Count i = 0; i < size; ++i) {
      const U8 expected = overlay.get_data()[i] ? 255 : source.get_data()[i];
      if (observed.get_data()[i] != expected) {
        matches = False;
        break;
      }
    }

    EXPECT(matches);
    if (!matches) {
      break;
    }
  }

  EXPECT_EQ(output.release(), COMPUTE_EXECUTION_SUCCESS);
  EXPECT_EQ(front.release(), COMPUTE_EXECUTION_SUCCESS);
  EXPECT_EQ(back.release(), COMPUTE_EXECUTION_SUCCESS);
  EXPECT_EQ(engine.release(), COMPUTE_EXECUTION_SUCCESS);
}
