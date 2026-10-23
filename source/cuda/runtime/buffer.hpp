// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "cuda/contracts/buffer.hpp"
#include "cuda/runtime/program.hpp"

namespace Cuda::Runtime {

// Buffer holds one device allocation and a Program reference for its context.
// The final release frees the allocation before releasing that Program.
// Each acquired Retained result contributes one reference. Plain API copies
// and bindings share the result that supplied them.
class Buffer {
 public:
  Buffer(const Buffer&) = delete;
  auto operator=(const Buffer&) -> Buffer& = delete;
  static EXPORTED(CUDA) auto allocate(
      Program& program,
      Count size,
      ttx_retained* output) -> cuda_status;

 private:
  friend Toolchain::Memory::Lifetime;
  Buffer(Program& program, CUdeviceptr address, Count size)
      : program(program), address(address), size(size) {
    program.retain();
  }

  ~Buffer();
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
  static auto release_context(void* context) -> void;
  static auto get_address(void* context) -> U64;
  static auto get_size(void* context) -> Count;
  static auto read(void* context, Count offset, U8* output, Count size)
      -> cuda_status;
  static auto write(void* context, Count offset, perimortem_view_bytes input)
      -> cuda_status;
  static const ttx_abstract_ops graph_operations;
  static const ttx_retained_ops retained_operations;
  static const ttx_retain_ops retain_operations;
  static const cuda_buffer_ops operations;

  Count references = 1;
  Program& program;
  CUdeviceptr address;
  Count size;
};

}  // namespace Cuda::Runtime
