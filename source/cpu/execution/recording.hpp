// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "perimortem/memory/global/vector.hpp"

#include "compute/execution/commands.h"
#include "compute/image/composite_command.h"
#include "compute/image/convolve_command.h"
#include "compute/image/invert_command.h"
#include "compute/image/posterize_command.h"

namespace Cpu::Execution {

// Recording owns one ordered list of CPU commands. Record copies complete
// command records while the caller keeps every referenced Buffer acquisition
// accessible through release. Submit executes the list once on its calling
// thread. Wait returns the stored submission outcome.
class Recording {
 public:
  static auto create() -> compute_commands_result;

 private:
  enum class Kind { Invert, Posterize, Convolve, Composite };
  enum class Phase { Recording, Complete, Failed };

  struct Operation {
    Kind kind;
    compute_invert_command invert{};
    compute_posterize_command posterize{};
    compute_convolve_command convolve{};
    compute_composite_command composite{};
  };

  static auto submit(void* context) -> compute_execution_status;
  static auto wait(void* context) -> compute_execution_status;
  static auto release(void* context) -> compute_execution_status;
  static auto supports(void* context, perimortem_uuid command)
      -> ttx_negotiation_status;
  static auto record(
      void* context,
      perimortem_uuid command,
      const void* arguments) -> compute_execution_status;
  auto execute(const Operation& operation) -> compute_execution_status;

  Phase phase = Phase::Recording;
  Perimortem::Memory::Global::Vector<Operation> operations;
};

}  // namespace Cpu::Execution
