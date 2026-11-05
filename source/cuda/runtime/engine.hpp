// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "compute/execution/engine.h"
#include "ttx/graph/abstract.hpp"

namespace Cuda::Runtime {

// Engine publishes CUDA's factory independently of a loaded application
// program. Discovery performs no device initialization. Creation acquires a
// primary context. Buffers retain that context until their own final release.
class Engine {
 public:
  static EXPORTED(CUDA) auto create() -> compute_engine_result;
  static EXPORTED(CUDA) auto factory() -> Ttx::Graph::Abstract;
};

}  // namespace Cuda::Runtime
