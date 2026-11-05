// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_EXECUTION_ENGINE_H
#define COMPUTE_EXECUTION_ENGINE_H

#include "compute/execution/buffer.h"
#include "compute/execution/commands.h"

#define COMPUTE_ENGINE_ID_HIGH 0x59d8c3ea03834c17ULL
#define COMPUTE_ENGINE_ID_LOW 0x831125ec94f134bcULL

// Engine allocates storage through the selected provider. Each successful
// allocation and command creation has an independent release obligation and
// keeps its required provider state accessible after the Engine acquisition is
// released. Zero size is rejected. Provider failures supply an empty record.
//
// Factory creation supplies one Engine acquisition. Its release consumes that
// acquisition on every outcome. The consumer serializes operations through an
// Engine, Buffers and Commands, and keeps their module loaded through final
// release. These public records are semantic thunks: callers invoke the
// supplied table, whose functions interpret the context. Discovery describes
// the records separately through Compute's Type graph.
typedef struct compute_engine {
  void* context;
  const struct compute_engine_ops* operations;
} compute_engine;

typedef struct compute_engine_ops {
  compute_allocation (*allocate)(void* context, U64 size);
  struct compute_commands_result (*create_commands)(void* context);
  compute_execution_status (*release)(void* context);
} compute_engine_ops;

typedef struct compute_engine_result {
  compute_execution_status status;
  compute_engine engine;
} compute_engine_result;

#endif
