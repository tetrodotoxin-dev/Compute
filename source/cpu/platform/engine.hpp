// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "ttx/graph/abstract.hpp"

namespace Cpu::Platform {

// Engine supplies a Create factory for host execution. None requests the
// default engine. Its commands route supplies a Create factory for independent
// command buffers. The consumer retains an engine or buffer during creation
// when later calls require it. CPU submissions execute in authored order on
// the submitting thread. Results follow the Commands and Pixels contracts.
// The host keeps the module loaded through every use and release.
class Engine {
 public:
  static EXPORTED(CPU) auto factory() -> Ttx::Graph::Abstract;
};

}  // namespace Cpu::Platform
