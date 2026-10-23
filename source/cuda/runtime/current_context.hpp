// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include <cuda.h>

#include "toolchain/toolchain.h"

namespace Cuda::Runtime {

// The application may enter this provider with another CUDA context current.
// CurrentContext installs the required context for one scope and restores the
// previous stack entry when that scope ends. A failed push performs no pop,
// allowing callers to report acquisition failure without disturbing host state.
class CurrentContext {
 public:
  EXPORTED(CUDA) explicit CurrentContext(CUcontext context);
  EXPORTED(CUDA) ~CurrentContext();
  CurrentContext(const CurrentContext&) = delete;
  auto operator=(const CurrentContext&) -> CurrentContext& = delete;
  const CUresult status;
};

}  // namespace Cuda::Runtime
