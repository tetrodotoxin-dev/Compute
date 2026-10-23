// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "cuda/contracts/status.h"

namespace Cuda::Contracts {

// ExecutionStatus reports the CUDA operation's outcome. Negotiating an
// interface uses TTX's negotiation status independently of executing its CUDA
// operations.
enum class ExecutionStatus : U8 {
  Success = CUDA_SUCCESS_STATUS,
  Invalid = CUDA_INVALID,
  Bounds = CUDA_BOUNDS,
  Incompatible = CUDA_INCOMPATIBLE,
  Unavailable = CUDA_UNAVAILABLE,
  CompilationFailed = CUDA_COMPILATION_FAILED,
  DriverFailed = CUDA_DRIVER_FAILED,
};

}  // namespace Cuda::Contracts
