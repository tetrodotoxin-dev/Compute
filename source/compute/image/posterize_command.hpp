// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "perimortem/core/identity/uuid.hpp"

#include "compute/execution/buffer.hpp"
#include "compute/image/posterize_command.h"
#include "ttx/data/form/native.hpp"

namespace Compute::Image {

// PosterizeCommand names the UUID and argument record accepted by Commands.
// Direct consumers construct that record without entering the Graph.
class PosterizeCommand {
 public:
  using Arguments = compute_posterize_command;
  static constexpr auto contract_id = Perimortem::Core::Identity::Uuid(
      COMPUTE_POSTERIZE_COMMAND_ID_HIGH,
      COMPUTE_POSTERIZE_COMMAND_ID_LOW);
};

}  // namespace Compute::Image

TTX_DATA_RECORD(
    compute_posterize_command,
    TTX_DATA_MEMBER(compute_posterize_command, input),
    TTX_DATA_MEMBER(compute_posterize_command, output),
    TTX_DATA_MEMBER(compute_posterize_command, count),
    TTX_DATA_MEMBER(compute_posterize_command, levels));
