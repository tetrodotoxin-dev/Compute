// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_EXECUTION_BUFFER_H
#define COMPUTE_EXECUTION_BUFFER_H

#include "perimortem/core/view/bytes.h"

#include "compute/execution/status.h"
#include "ttx/graph/abstract.h"

// Buffer provides byte storage on the selected engine. Read and write complete
// the requested host transfer before returning. The caller keeps each host
// range accessible through the call. An out of range request leaves both the
// buffer and the caller's destination unchanged.
//
// Allocation supplies one acquisition. Copies of the record share that
// acquisition. Release consumes it, including when cleanup reports failure.
// The consumer keeps the provider module loaded through release. The context
// has meaning only to the operations supplied in this record. Buffer makes no
// promise that its storage is directly addressable by the host.
//
// `get_abstract` returns the graph result associated with this acquisition.
// Provider-specific commands can bind extensions there without interpreting
// the Buffer context or operation table. The returned Abstract transfers no
// acquisition and remains accessible only while this Buffer is accessible.
// A consumer can transport the complete Buffer record without observing that
// graph result.
typedef struct compute_buffer {
  void* context;
  const struct compute_buffer_ops* operations;
} compute_buffer;

typedef struct compute_buffer_ops {
  ttx_abstract (*get_abstract)(void* context);
  U64 (*size)(void* context);
  compute_execution_status (
      *read)(void* context, U64 offset, U8* output, U64 size);
  compute_execution_status (
      *write)(void* context, U64 offset, perimortem_view_bytes input);
  compute_execution_status (*release)(void* context);
} compute_buffer_ops;

// A successful allocation supplies a Buffer acquisition. Other statuses leave
// buffer empty. Reading this result or copying it does not acquire storage.
typedef struct compute_allocation {
  compute_execution_status status;
  compute_buffer buffer;
} compute_allocation;

#endif
