// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "perimortem/core/identity/uuid.hpp"

#include "compute/execution/engine.hpp"
#include "compute/execution/factory.h"

namespace Compute::Execution {

// Factory invokes the supplied engine constructor. A failed creation supplies
// no Engine acquisition. Successful creations transfer independent release
// obligations under engine.h, including when their records share a context.
// Constructing or copying this interface keeps the API's supplying access
// duration. The consumer keeps the factory state and module accessible through
// each call. Created Engines follow their own acquisition obligations.
class Factory {
 public:
  using Api = compute_factory;
  static constexpr auto contract_id = Perimortem::Core::Identity::Uuid(
      COMPUTE_FACTORY_ID_HIGH,
      COMPUTE_FACTORY_ID_LOW);
  explicit constexpr Factory(Api api) : api(api) {}
  constexpr auto get_abi() const -> Api { return api; }
  auto create_engine() const
      -> Perimortem::Core::Storage::Result<Engine, compute_execution_status> {
    const auto created = api.create_engine(api.context);
    if (created.status != COMPUTE_EXECUTION_SUCCESS) {
      return created.status;
    }

    return Engine(created.engine);
  }

 private:
  Api api;
};

}  // namespace Compute::Execution

TTX_DATA_RECORD(
    compute_factory,
    TTX_DATA_MEMBER(compute_factory, context),
    TTX_DATA_MEMBER(compute_factory, create_engine));
