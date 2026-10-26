// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "ttx/graph/abstract.hpp"

#include "compute/execution/commands.h"

namespace Compute::Execution {

// Commands submits one recorded sequence and waits for its completion. Its
// Abstract supplies the typed recording operations. A successful submit can
// return before execution finishes, while wait reports the completed outcome.
// Construction stores an existing API record. The supplying acquisition must
// remain available through every call, including calls through copies of this
// interface. `commands.h` specifies ordering and final release of pending work.
class Commands {
 public:
  using Api = compute_commands;
  static constexpr auto contract_id = Perimortem::Core::Identity::Uuid(
      COMPUTE_COMMANDS_ID_HIGH,
      COMPUTE_COMMANDS_ID_LOW);

  explicit constexpr Commands(Api api) : api(api) {}
  constexpr auto get_abi() const -> Api { return api; }
  auto get_abstract() const -> Ttx::Graph::Abstract {
    return Ttx::Graph::Abstract(api.operations->get_abstract(api.context));
  }

  auto submit() const -> compute_execution_status {
    return api.operations->submit(api.context);
  }

  auto wait() const -> compute_execution_status {
    return api.operations->wait(api.context);
  }

 private:
  Api api;
};

}  // namespace Compute::Execution
