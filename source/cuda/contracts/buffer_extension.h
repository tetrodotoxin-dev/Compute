// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef CUDA_CONTRACTS_BUFFER_EXTENSION_H
#define CUDA_CONTRACTS_BUFFER_EXTENSION_H

#include "ttx/graph/abstract.h"

#define CUDA_BUFFER_EXTENSION_ID_HIGH 0xb5edfa1c306b4abfULL
#define CUDA_BUFFER_EXTENSION_ID_LOW 0x89fd3cf95ff02a4cULL

// BufferExtension lets a CUDA Commands provider request device access from a
// Compute Buffer without interpreting its typed record. That provider binds
// this extension from Buffer's Abstract and supplies the opaque CUDA driver
// context that will execute the command. The application transporting the
// Buffer needs no CUDA-specific behavior.
//
// Address returns Satisfied and writes the device address only when the Buffer
// provider permits that context to use its storage. Unknown or Rejected leaves
// output unusable. The operation transfers no acquisition. The caller keeps
// the public Compute Buffer acquisition accessible through every command using
// the address and releases that acquisition through its own Buffer API.
//
// `driver_context` carries a CUDA `CUcontext` handle without requiring CUDA
// headers in this contract. Neither participant dereferences that pointer.
typedef struct cuda_buffer_extension {
  void* context;
  const struct cuda_buffer_extension_ops* operations;
} cuda_buffer_extension;

typedef struct cuda_buffer_extension_ops {
  ttx_negotiation_status (
      *address)(void* context, void* driver_context, U64* output);
} cuda_buffer_extension_ops;

#endif
