// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cuda/runtime/current_context.hpp"

#include "perimortem/core/diagnostics/log.hpp"
#include "perimortem/core/view/null_terminated.hpp"

using namespace Cuda::Runtime;
using namespace Perimortem::Core;

CurrentContext::CurrentContext(CUcontext context)
    : status(cuCtxPushCurrent(context)) {}
CurrentContext::~CurrentContext() {
  if (status != CUDA_SUCCESS) {
    return;
  }

  CUcontext previous;
  if (cuCtxPopCurrent(&previous) != CUDA_SUCCESS) {
    Diagnostics::Log::fatal(
        "CUDA could not restore the caller's current context."_view);
  }
}
