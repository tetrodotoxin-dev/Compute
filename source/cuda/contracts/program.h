// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef CUDA_CONTRACTS_PROGRAM_H
#define CUDA_CONTRACTS_PROGRAM_H

#include "ttx/data/form/representation.h"
#include "ttx/graph/policies/retained.h"

#include "cuda/contracts/diagnostics.h"
#include "cuda/contracts/status.h"

#define TTX_CUDA_PROGRAM_HIGH 0xfcdff2fcbe6d4902ULL
#define TTX_CUDA_PROGRAM_LOW 0x82a51e833740ff97ULL

// Each argument supplies a complete canonical form and its offset in the host
// frame. CUDA verifies parameter counts and sizes against the loaded function.
// The source author establishes agreement with the device parameter types.
// Device addresses are eight byte CUDA values, meaningful in that context.
typedef struct cuda_argument {
  Count offset;
  const ttx_representation* representation;
} cuda_argument;

// Program prepares kernels and allocates device buffers. Each successful
// operation supplies an independent Retained result whose Abstract binds Kernel
// or Buffer. Kernels and buffers keep the Program's loaded code and context
// available after the original Program result is released. Calls on related
// resources are serialized by the consumer, including retention and release.
typedef struct cuda_program {
  void* context;
  const struct cuda_program_ops* operations;
} cuda_program;

typedef struct cuda_program_ops {
  // Returns to this binding's graph location using a provider supplied record.
  ttx_abstract (*get_abstract)(void* context);
  // The caller keeps the canonical frame, argument forms and entry bytes
  // readable through preparation. The prepared Kernel copies its frame form.
  cuda_status (*prepare)(
      void* context,
      perimortem_view_bytes entry,
      const ttx_representation* frame,
      const cuda_argument* arguments,
      Count count,
      cuda_diagnostics diagnostics,
      ttx_retained* output);
  cuda_status (*allocate)(void* context, Count size, ttx_retained* output);
} cuda_program_ops;

#endif
