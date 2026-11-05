// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "perimortem/core/identity/uuid.hpp"

#include "compute/execution/buffer.hpp"
#include "compute/image/convolve_command.h"
#include "ttx/data/form/native.hpp"

namespace Compute::Image {

// ConvolveCommand names the UUID and argument record accepted by Commands.
// Direct consumers construct that record without entering the Graph.
class ConvolveCommand {
 public:
  using Arguments = compute_convolve_command;
  static constexpr auto contract_id = Perimortem::Core::Identity::Uuid(
      COMPUTE_CONVOLVE_COMMAND_ID_HIGH,
      COMPUTE_CONVOLVE_COMMAND_ID_LOW);
};

}  // namespace Compute::Image

TTX_DATA_RECORD(
    compute_convolve_command,
    TTX_DATA_MEMBER(compute_convolve_command, input),
    TTX_DATA_MEMBER(compute_convolve_command, output),
    TTX_DATA_MEMBER(compute_convolve_command, weights),
    TTX_DATA_MEMBER(compute_convolve_command, width),
    TTX_DATA_MEMBER(compute_convolve_command, height),
    TTX_DATA_MEMBER(compute_convolve_command, kernel_width),
    TTX_DATA_MEMBER(compute_convolve_command, kernel_height));
