// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "compute/execution/engine.h"

namespace Cpu::Platform {

// Memory creates independent host engines for Compute's byte-storage contract.
// Each Buffer owns its allocation, so releasing the Engine preserves existing
// Buffers. Provider implementation state stays behind the published functions.
class Memory {
 public:
  static EXPORTED(CPU) auto create() -> compute_engine_result;
};

}  // namespace Cpu::Platform
