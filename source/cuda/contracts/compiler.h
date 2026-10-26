// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef CUDA_CONTRACTS_COMPILER_H
#define CUDA_CONTRACTS_COMPILER_H

#include "ttx/graph/policies/retained.h"

#include "cuda/contracts/diagnostics.h"
#include "cuda/contracts/status.h"

#define TTX_CUDA_COMPILER_HIGH 0x14421902be5141b4ULL
#define TTX_CUDA_COMPILER_LOW 0xa6b7698f8a109520ULL

// Source supplies a name and text for NVRTC. Names, source, includes and
// options contain UTF 8 bytes without embedded nulls. The caller keeps every
// input range and array readable until compilation returns.
typedef struct cuda_source {
  perimortem_view_bytes name;
  perimortem_view_bytes text;
} cuda_source;

typedef struct cuda_compile_request {
  cuda_source source;
  const cuda_source* headers;
  Count header_count;
  const perimortem_view_bytes* options;
  Count option_count;
  S32 device;
} cuda_compile_request;

// Compiler builds source for the selected device. A successful compile call
// supplies a Retained result whose Abstract binds Program. The consumer
// releases that result after its final use. Compilation failure leaves earlier
// results usable. The host keeps the provider library loaded through every
// result's release.
// The result's release reports cleanup failure through the Retained contract.
typedef struct cuda_compiler {
  void* context;
  const struct cuda_compiler_ops* operations;
} cuda_compiler;

typedef struct cuda_compiler_ops {
  // Returns the graph location that supplied this Compiler binding. The
  // provider publishes the returned Abstract's own context and operations.
  ttx_abstract (*get_abstract)(void* context);
  cuda_status (*compile)(
      void* context,
      cuda_compile_request request,
      cuda_diagnostics diagnostics,
      ttx_retained* output);
} cuda_compiler_ops;

#endif
