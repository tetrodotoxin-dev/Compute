// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cpu/platform/memory.hpp"

#include "toolchain/memory/lifetime.hpp"

#include "perimortem/memory/allocator/global.hpp"
#include "perimortem/memory/global/bytes.hpp"

#include "cpu/execution/recording.hpp"
#include "ttx/graph/policies/none.h"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Toolchain::Memory;

class BufferStorage {
 public:
  BufferStorage(Count size) { bytes.resize(size); }

  static auto get_abstract(void* context) -> ttx_abstract { return ttx_none(); }

  static auto size(void* context) -> U64 {
    return Data::cast<BufferStorage>(context)->bytes.get_size();
  }

  static auto read(void* context, U64 offset, U8* output, U64 size)
      -> compute_execution_status {
    const auto& bytes = Data::cast<BufferStorage>(context)->bytes;
    if (offset > bytes.get_size() || size > bytes.get_size() - offset ||
        (size && !output)) {
      return COMPUTE_EXECUTION_INVALID;
    }

    if (size) {
      Data::copy(output, bytes.get_data() + offset, size);
    }

    return COMPUTE_EXECUTION_SUCCESS;
  }

  static auto write(void* context, U64 offset, perimortem_view_bytes input)
      -> compute_execution_status {
    auto& bytes = Data::cast<BufferStorage>(context)->bytes;
    if (offset > bytes.get_size() || input.size > bytes.get_size() - offset ||
        (input.size && !input.data)) {
      return COMPUTE_EXECUTION_INVALID;
    }

    if (input.size) {
      Data::copy(bytes.get_data() + offset, input.data, input.size);
    }

    return COMPUTE_EXECUTION_SUCCESS;
  }

  static auto release(void* context) -> compute_execution_status {
    auto* allocation = Data::cast<BufferStorage>(context);
    allocation->~BufferStorage();
    Allocator::Global::release(Data::cast<U8>(allocation));
    return COMPUTE_EXECUTION_SUCCESS;
  }

  static const compute_buffer_ops operations;

 private:
  Global::Bytes bytes;
};

const compute_buffer_ops BufferStorage::operations{
  get_abstract, size, read, write, release};

static auto allocate(void* context, U64 size) -> compute_allocation {
  if (!size || size > Count(-1)) {
    return {COMPUTE_EXECUTION_INVALID, {}};
  }

  auto storage = Allocator::Global::allocate(sizeof(BufferStorage));
  auto& allocation = Lifetime::construct<BufferStorage>(
      storage.get_data(), [&] { return BufferStorage(Count(size)); });
  return {COMPUTE_EXECUTION_SUCCESS, {&allocation, &BufferStorage::operations}};
}

static auto release_engine(void* context) -> compute_execution_status {
  return COMPUTE_EXECUTION_SUCCESS;
}

static auto create_commands(void* context) -> compute_commands_result {
  return Cpu::Execution::Recording::create();
}

auto Cpu::Platform::Memory::create() -> compute_engine_result {
  static const compute_engine_ops operations{
    allocate, create_commands, release_engine};

  static U8 context;
  return {COMPUTE_EXECUTION_SUCCESS, {&context, &operations}};
}
