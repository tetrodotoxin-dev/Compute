// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_EXECUTION_COMMANDS_H
#define COMPUTE_EXECUTION_COMMANDS_H

#include "ttx/graph/abstract.h"

#include "compute/execution/status.h"

#define COMPUTE_COMMANDS_ID_HIGH 0xadb2a591815641deULL
#define COMPUTE_COMMANDS_ID_LOW 0x9820ac1d1223d521ULL

// Commands records one ordered submission through the typed capabilities on
// its Abstract. Recording establishes dependencies and returns retained result
// artifacts. Metadata is available while recording. Payload access requires
// submission. Each recording call must capture arguments or acquire the state
// needed later before returning. A provider may copy an input that cannot be
// retained, or refuse it without appending a command.
//
// Submit ends recording and starts the complete sequence in authored order.
// A successful submission can finish executing later. A failed submission also
// ends recording and may leave earlier commands in flight. Further recording
// and repeated submission return Invalid. Wait before submission returns
// Invalid. After submission, wait completes the buffer's work and reports its
// execution outcome. Repeated waits preserve that outcome.
//
// Separate buffers have no implicit order. A consumer submits a producing
// buffer before recording its results into another buffer. Providers
// establish those dependencies through supplied artifact operations.
//
// Result acquisitions keep their producing state available independently of
// the buffer acquisition. Final release waits for submitted work before
// destroying its resources and reports cleanup through Retained. Releasing an
// acquisition does not cancel submitted work. An unsubmitted buffer discards
// its commands when its final buffer or result acquisition is released.
// The consumer must serialize host calls through one engine creation and its
// buffers and results, including recording, submission, reads, retention and
// release. Submitted device work can overlap across buffers. Retained inputs
// also keep their supplying access and serialization policies.
typedef struct compute_commands {
  void* context;
  const struct compute_commands_ops* operations;
} compute_commands;

typedef struct compute_commands_ops {
  // Returns to the graph location from which this Commands binding was
  // obtained. The provider supplies that Abstract's own context and operations.
  ttx_abstract (*get_abstract)(void* context);
  compute_execution_status (*submit)(void* context);
  compute_execution_status (*wait)(void* context);
} compute_commands_ops;

#endif
