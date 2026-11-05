// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "compute/execution/commands.h"
#include "ttx/graph/abstract.hpp"

namespace Compute::Execution {

// Commands records, submits and waits for one ordered sequence. A successful
// submit can return before execution finishes, while wait reports completion.
// Construction stores an existing API record. The supplying acquisition must
// remain available through every call, including calls through copies of this
// interface. `commands.h` specifies argument identity, ordering and destructive
// release.
class Commands {
 public:
  using Api = compute_commands;
  static constexpr auto contract_id = Perimortem::Core::Identity::Uuid(
      COMPUTE_COMMANDS_ID_HIGH,
      COMPUTE_COMMANDS_ID_LOW);

  explicit constexpr Commands(Api api) : api(api) {}
  constexpr auto get_abi() const -> Api { return api; }
  auto submit() const -> compute_execution_status {
    return api.operations->submit(api.context);
  }

  auto wait() const -> compute_execution_status {
    return api.operations->wait(api.context);
  }

  auto release() const -> compute_execution_status {
    return api.operations->release(api.context);
  }

  auto supports(Perimortem::Core::Identity::Uuid command) const
      -> Ttx::Semantic::Negotiation::Status {
    return Ttx::Semantic::Negotiation::Status(
        api.operations->supports(api.context, command));
  }

  template <typename Arguments>
  auto record(
      Perimortem::Core::Identity::Uuid command,
      const Arguments& arguments) const -> compute_execution_status {
    return api.operations->record(api.context, command, &arguments);
  }

 private:
  Api api;
};

}  // namespace Compute::Execution

TTX_DATA_RECORD(
    compute_commands_ops,
    TTX_DATA_MEMBER(compute_commands_ops, submit),
    TTX_DATA_MEMBER(compute_commands_ops, wait),
    TTX_DATA_MEMBER(compute_commands_ops, release),
    TTX_DATA_MEMBER(compute_commands_ops, supports),
    TTX_DATA_MEMBER(compute_commands_ops, record));
TTX_DATA_RECORD(
    compute_commands,
    TTX_DATA_MEMBER(compute_commands, context),
    TTX_DATA_MEMBER(compute_commands, operations));
TTX_DATA_RECORD(
    compute_commands_result,
    TTX_DATA_MEMBER(compute_commands_result, status),
    TTX_DATA_MEMBER(compute_commands_result, commands));
