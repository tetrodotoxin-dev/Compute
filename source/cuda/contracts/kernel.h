// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef CUDA_CONTRACTS_KERNEL_H
#define CUDA_CONTRACTS_KERNEL_H

#include "ttx/data/form/storage.h"
#include "ttx/graph/abstract.h"

#include "cuda/contracts/status.h"

#define TTX_CUDA_KERNEL_HIGH 0x48a47e699a184a2fULL
#define TTX_CUDA_KERNEL_LOW 0x99499b6fd15cbe85ULL

typedef struct cuda_launch {
  U32 grid_x, grid_y, grid_z;
  U32 block_x, block_y, block_z;
  U32 shared_bytes;
} cuda_launch;

// Kernel executes a prepared entry using the caller's launch geometry and
// argument frame. The caller keeps the frame and every referenced device
// allocation alive until launch returns. Device execution completes within
// the call. A failed launch may have modified device buffers.
// The consumer serializes launches because preparation retains argument
// scratch.
typedef struct cuda_kernel {
  void* context;
  const struct cuda_kernel_ops* operations;
} cuda_kernel;

typedef struct cuda_kernel_ops {
  // Returns to this binding's graph location using a provider supplied record.
  ttx_abstract (*get_abstract)(void* context);
  cuda_status (
      *launch)(void* context, cuda_launch geometry, ttx_storage arguments);
} cuda_kernel_ops;

#endif
