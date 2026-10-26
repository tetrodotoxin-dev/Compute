// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "perimortem/core/storage/option.hpp"

#include "compute/execution/commands.hpp"
#include "compute/execution/dependency.h"

namespace Compute::Execution {

// Dependency locates an artifact in a particular recording. Locate returns an
// empty Option when the provider refuses that position. A successful position
// remains valid through the producing buffer's lifetime. The caller keeps the
// artifact and requested Commands accessible through the call. This interface
// stores the supplied API record and preserves its existing access duration.
class Dependency {
 public:
  using Api = compute_dependency;
  static constexpr auto contract_id = Perimortem::Core::Identity::Uuid(
      COMPUTE_DEPENDENCY_ID_HIGH,
      COMPUTE_DEPENDENCY_ID_LOW);

  explicit constexpr Dependency(Api api) : api(api) {}
  auto get_abstract() const -> Ttx::Graph::Abstract {
    return Ttx::Graph::Abstract(api.operations->get_abstract(api.context));
  }

  auto locate(Commands commands) const
      -> Perimortem::Core::Storage::Option<Count> {
    Count position = 0;
    if (api.operations->locate(api.context, commands.get_abi(), &position) !=
        TTX_NEGOTIATION_SATISFIED) {
      return {};
    }

    return position;
  }

  auto submitted() const -> Bool {
    return api.operations->submitted(api.context) != 0;
  }

 private:
  Api api;
};

}  // namespace Compute::Execution
