// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "ttx/graph/abstract.hpp"

namespace Cuda::Platform {

// Engine supplies a Create factory for CUDA image execution. None requests the
// default device. Creation prepares the invert operation before publishing an
// engine Abstract, or refuses the request without invoking the callback.
// The consumer retains the engine inside the callback for later use. Image
// results keep their device storage usable independently of that acquisition.
// If the callback leaves no retained acquisition, the factory releases its
// engine and reports cleanup failure through Perimortem diagnostics. A retained
// engine reports cleanup failure through its consumer's release operation.
// The consumer serializes calls through one engine creation and every image
// it produces, including all bindings, retention and release operations.
class Engine {
 public:
  static EXPORTED(CUDA) auto factory() -> Ttx::Graph::Abstract;
};

}  // namespace Cuda::Platform
