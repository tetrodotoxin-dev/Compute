// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "ttx/graph/abstract.hpp"

namespace Cpu::Platform {

// Engine supplies a Create factory for host execution. None requests the
// default engine. The created Abstract exposes the invert route and its Invert
// contract. The consumer can retain the engine during the creation callback.
// This stateless engine remains backed by the loaded module, while each image
// result holds its own storage. The host keeps the module loaded through use.
// Calls follow the Invert contract's serialization policy for an engine
// creation and its produced images.
class Engine {
 public:
  static EXPORTED(CPU) auto factory() -> Ttx::Graph::Abstract;
};

}  // namespace Cpu::Platform
