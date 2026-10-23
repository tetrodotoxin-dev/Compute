// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef CUDA_CONTRACTS_BUFFER_H
#define CUDA_CONTRACTS_BUFFER_H

#include "ttx/graph/abstract.h"

#include "cuda/contracts/status.h"

#define TTX_CUDA_BUFFER_HIGH 0x8ee853cb736c42efULL
#define TTX_CUDA_BUFFER_LOW 0x982dfe08237b1e65ULL

// Buffer exposes device storage independently of an image or element format.
// Its address belongs to the supplying CUDA context. The consumer keeps the
// Retained result available through every launch using that address.
// Read and write finish their host transfer before returning. The caller
// provides accessible host bytes for the requested range through that call.
typedef struct cuda_buffer {
  void* context;
  const struct cuda_buffer_ops* operations;
} cuda_buffer;

typedef struct cuda_buffer_ops {
  // Returns to this binding's graph location using a provider supplied record.
  ttx_abstract (*get_abstract)(void* context);
  U64 (*address)(void* context);
  Count (*size)(void* context);
  cuda_status (*read)(void* context, Count offset, U8* output, Count size);
  cuda_status (
      *write)(void* context, Count offset, perimortem_view_bytes input);
} cuda_buffer_ops;

#endif
