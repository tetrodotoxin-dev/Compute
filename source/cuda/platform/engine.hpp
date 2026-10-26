// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "ttx/graph/abstract.hpp"

namespace Cuda::Platform {

// Engine supplies a Create factory for CUDA execution. None requests the
// default device. Creation prepares the engine's compiled code, then exposes
// a commands route whose Create capability supplies independent command
// buffers. Each buffer acquires that code and a dedicated stream. Its results
// keep their producing state available independently of the engine acquisition.
// Creation callback cleanup failures use Perimortem diagnostics. A retained
// engine reports its cleanup outcome through Retained release. The consumer
// serializes calls through each engine and follows Commands for its buffers.
class Engine {
 public:
  static EXPORTED(CUDA) auto factory() -> Ttx::Graph::Abstract;
};

}  // namespace Cuda::Platform
