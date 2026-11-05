// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "perimortem/core/identity/uuid.hpp"

#include "compute/execution/buffer.hpp"
#include "compute/execution/commands.hpp"
#include "compute/execution/engine.h"

namespace Compute::Execution {

// Engine invokes the provider's allocation and release operations without
// recovering its state. The acquisition and child-storage guarantees are
// specified by engine.h. Destruction of this interface performs no release.
class Engine {
 public:
  using Api = compute_engine;
  static constexpr auto contract_id = Perimortem::Core::Identity::Uuid(
      COMPUTE_ENGINE_ID_HIGH,
      COMPUTE_ENGINE_ID_LOW);
  explicit constexpr Engine(Api api) : api(api) {}
  constexpr auto get_abi() const -> Api { return api; }
  auto allocate(U64 size) const
      -> Perimortem::Core::Storage::Result<Buffer, compute_execution_status> {
    const auto allocated = api.operations->allocate(api.context, size);
    if (allocated.status != COMPUTE_EXECUTION_SUCCESS) {
      return allocated.status;
    }

    return Buffer(allocated.buffer);
  }

  auto release() const -> compute_execution_status {
    return api.operations->release(api.context);
  }

  auto create_commands() const
      -> Perimortem::Core::Storage::Result<Commands, compute_execution_status> {
    const auto created = api.operations->create_commands(api.context);
    if (created.status != COMPUTE_EXECUTION_SUCCESS) {
      return created.status;
    }

    return Commands(created.commands);
  }

 private:
  Api api;
};

}  // namespace Compute::Execution

TTX_DATA_RECORD(
    compute_engine_ops,
    TTX_DATA_MEMBER(compute_engine_ops, allocate),
    TTX_DATA_MEMBER(compute_engine_ops, create_commands),
    TTX_DATA_MEMBER(compute_engine_ops, release));
TTX_DATA_RECORD(
    compute_engine,
    TTX_DATA_MEMBER(compute_engine, context),
    TTX_DATA_MEMBER(compute_engine, operations));
TTX_DATA_RECORD(
    compute_engine_result,
    TTX_DATA_MEMBER(compute_engine_result, status),
    TTX_DATA_MEMBER(compute_engine_result, engine));
