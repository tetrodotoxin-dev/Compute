// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "perimortem/core/identity/uuid.hpp"

#include "compute/execution/buffer.hpp"
#include "compute/image/composite_command.h"
#include "ttx/data/form/native.hpp"

namespace Compute::Image {

// CompositeCommand names the UUID and argument record accepted by Commands.
// Direct consumers construct that record without entering the Graph.
class CompositeCommand {
 public:
  using Arguments = compute_composite_command;
  static constexpr auto contract_id = Perimortem::Core::Identity::Uuid(
      COMPUTE_COMPOSITE_COMMAND_ID_HIGH,
      COMPUTE_COMPOSITE_COMMAND_ID_LOW);
};

}  // namespace Compute::Image

TTX_DATA_RECORD(
    compute_composite_command,
    TTX_DATA_MEMBER(compute_composite_command, back),
    TTX_DATA_MEMBER(compute_composite_command, front),
    TTX_DATA_MEMBER(compute_composite_command, output),
    TTX_DATA_MEMBER(compute_composite_command, count));
