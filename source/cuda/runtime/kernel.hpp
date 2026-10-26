// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "perimortem/memory/global/bytes.hpp"
#include "perimortem/memory/global/vector.hpp"

#include "cuda/contracts/kernel.hpp"
#include "cuda/runtime/program.hpp"

namespace Cuda::Runtime {

// Kernel retains its Program and an independent copy of the argument form.
// Preparation resolves the CUDA entry and allocates argument pointer scratch.
// Launches reuse that scratch, so the consumer serializes calls through all
// bindings of this Kernel, including bindings acquired by separate retention.
// Final release consumes the Kernel's Program reference and reports its cleanup
// outcome before destroying the Kernel's host storage.
class Kernel {
 public:
  Kernel(const Kernel&) = delete;
  auto operator=(const Kernel&) -> Kernel& = delete;
  static EXPORTED(CUDA) auto prepare(
      Program& program,
      perimortem_view_bytes entry,
      const ttx_representation& frame,
      const cuda_argument* arguments,
      Count count,
      cuda_diagnostics diagnostics,
      ttx_retained* output) -> cuda_status;

 private:
  friend Toolchain::Memory::Lifetime;
  Kernel(
      Program& program,
      CUfunction function,
      const ttx_representation& frame,
      Perimortem::Memory::Global::Vector<Count> offsets);
  ~Kernel() = default;
  auto launch(cuda_launch geometry, ttx_storage arguments) -> cuda_status;
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
  static auto launch_context(
      void* context,
      cuda_launch geometry,
      ttx_storage arguments) -> cuda_status;
  static const ttx_abstract_ops graph_operations;
  static const ttx_retained_ops retained_operations;
  static const ttx_retain_ops retain_operations;
  static const cuda_kernel_ops operations;

  Count references = 1;
  Program& program;
  CUfunction function;
  Perimortem::Memory::Global::Bytes form;
  Perimortem::Memory::Global::Vector<Count> offsets;
  Perimortem::Memory::Global::Vector<void*> arguments;
};

}  // namespace Cuda::Runtime
