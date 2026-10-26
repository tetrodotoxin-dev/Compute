// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_EXECUTION_DEPENDENCY_H
#define COMPUTE_EXECUTION_DEPENDENCY_H

#include "compute/execution/commands.h"

#define COMPUTE_DEPENDENCY_ID_HIGH 0xb7e5dc940b50410aULL
#define COMPUTE_DEPENDENCY_ID_LOW 0xb89d5a85f5577952ULL

// Dependency identifies an artifact among a command buffer's recorded results.
// Locate returns Satisfied and a position only when the requested buffer owns
// that result. The position identifies the artifact independently of how many
// outputs its producing operation has. Only the requested buffer interprets
// that position, which remains stable for its lifetime. Unknown or Rejected
// leaves output unusable. The caller supplies writable position storage.
//
// The producing operation interprets its own context. The requested Commands
// is a complete published API record, not a native object address to recover.
// A recording provider uses a successful position to reference its earlier
// result directly. This keeps chained results resident and avoids acquiring
// its own buffer through an input artifact. Other inputs require retained
// access or a captured representation under their supplying contracts.
//
// Submitted returns zero while the producing buffer accepts recording and one
// after submission begins, including on execution failure. A consumer records
// dependencies between buffers only after this point, so later recordings
// cannot create a retention cycle with a producer whose recording has ended.
typedef struct compute_dependency {
  void* context;
  const struct compute_dependency_ops* operations;
} compute_dependency;

typedef struct compute_dependency_ops {
  ttx_abstract (*get_abstract)(void* context);
  U8 (*submitted)(void* context);
  ttx_negotiation_status (
      *locate)(void* context, compute_commands commands, Count* position);
} compute_dependency_ops;

#endif
