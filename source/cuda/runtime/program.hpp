// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include <cuda.h>

#include "ttx/graph/capabilities/retain.h"

#include "cuda/contracts/compiler.hpp"
#include "cuda/contracts/program.hpp"

namespace Cuda::Runtime {

// Program keeps a CUDA primary context reference and its loaded module alive.
// Native creation returns one reference. Each Kernel and Buffer holds another.
// The caller serializes calls through this Program and every Kernel and Buffer
// created from it, including all bindings, retention and release operations.
// The native module and context accessors support CUDA operations supplied by
// the consuming project. That caller completes its device work before release.
class Program {
 public:
  Program(const Program&) = delete;
  auto operator=(const Program&) -> Program& = delete;
  static EXPORTED(CUDA) auto create(
      cuda_compile_request request,
      cuda_diagnostics diagnostics) -> Perimortem::Core::Storage::
      Result<Program&, Contracts::ExecutionStatus>;
  static EXPORTED(CUDA) auto compile(
      cuda_compile_request request,
      cuda_diagnostics diagnostics,
      ttx_retained* output) -> cuda_status;
  static EXPORTED(CUDA) auto compiler() -> Ttx::Graph::Abstract;
  auto get_context() const -> CUcontext { return context; }
  auto get_module() const -> CUmodule { return module; }
  EXPORTED(CUDA) auto retain() -> void;
  // Consumes one reference. The final release unloads the module and releases
  // its primary context reference. CleanupFailed reports failure of either
  // cleanup operation or restoration of the caller's current context. Both
  // outcomes end this reference's access to the Program.
  EXPORTED(CUDA) auto release() -> Ttx::Graph::Policies::Retained::Status;

 private:
  friend Toolchain::Memory::Lifetime;
  Program() = default;
  ~Program() = default;
  static auto graph(void* context) -> ttx_abstract;
  static auto supports(void* context, perimortem_uuid id)
      -> ttx_negotiation_status;
  static auto bind(void* context, perimortem_uuid id, void* output)
      -> ttx_negotiation_status;
  static auto resolve_route(void* context, perimortem_view_bytes route)
      -> ttx_abstract;
  static auto visit_routes(void* context, ttx_graph_visitor visitor) -> void;
  static auto get_state(void* context) -> ttx_abstract_state;
  static auto acquire(void* context, ttx_retained* output)
      -> ttx_negotiation_status;
  static auto release_context(void* context) -> ttx_release_status;
  static auto prepare(
      void* context,
      perimortem_view_bytes entry,
      const ttx_representation* frame,
      const cuda_argument* arguments,
      Count count,
      cuda_diagnostics diagnostics,
      ttx_retained* output) -> cuda_status;
  static auto allocate(void* context, Count size, ttx_retained* output)
      -> cuda_status;
  static const ttx_abstract_ops graph_operations;
  static const ttx_retained_ops retained_operations;
  static const ttx_retain_ops retain_operations;
  static const cuda_program_ops operations;

  Count references = 1;
  CUdevice device = 0;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
};

}  // namespace Cuda::Runtime
