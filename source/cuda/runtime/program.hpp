// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include <cuda.h>

#include "toolchain/memory/lifetime.hpp"

#include "perimortem/core/storage/result.hpp"
#include "perimortem/core/view/bytes.hpp"

#include "compute/execution/status.h"

namespace Cuda::Runtime {

// Program owns one internal CUDA module and the primary context reference used
// by that module. Engine command recordings share it through the internal
// reference count. The final release unloads the module and releases the
// primary context before destroying the host object.
class Program {
 public:
  Program(const Program&) = delete;
  auto operator=(const Program&) -> Program& = delete;
  static auto create(
      Perimortem::Core::View::Bytes name,
      Perimortem::Core::View::Bytes source,
      S32 device)
      -> Perimortem::Core::Storage::Result<Program&, compute_execution_status>;
  auto get_context() const -> CUcontext { return context; }
  auto get_module() const -> CUmodule { return module; }
  auto retain() -> void;
  auto release() -> compute_execution_status;

 private:
  friend Toolchain::Memory::Lifetime;
  Program() = default;
  ~Program() = default;

  Count references = 1;
  CUdevice device = 0;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
};

}  // namespace Cuda::Runtime
