// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef CUDA_CONTRACTS_DIAGNOSTICS_H
#define CUDA_CONTRACTS_DIAGNOSTICS_H

#include "perimortem/core/view/bytes.h"

// Diagnostics delivers compilation messages synchronously. The consumer keeps
// callback_context available during the operation and copies any message bytes
// needed after the callback returns. A null callback discards diagnostics.
typedef struct cuda_diagnostics {
  void* callback_context;
  void (*callback)(void* callback_context, perimortem_view_bytes text);
} cuda_diagnostics;

#endif
