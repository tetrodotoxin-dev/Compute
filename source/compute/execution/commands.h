// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_EXECUTION_COMMANDS_H
#define COMPUTE_EXECUTION_COMMANDS_H

#include "compute/execution/status.h"
#include "ttx/graph/abstract.h"

#define COMPUTE_COMMANDS_ID_HIGH 0xadb2a591815641deULL
#define COMPUTE_COMMANDS_ID_LOW 0x9820ac1d1223d521ULL

// Commands records one ordered submission. A command UUID identifies the
// complete C argument record accepted by `record`. The provider reports
// support independently, then copies that record or establishes any access it
// needs before `record` returns. Each command contract states which referenced
// acquisitions the consumer must keep accessible. A failed record appends no
// command.
//
// Submit ends recording and starts the complete sequence in authored order.
// A successful submission can finish executing later. A failed submission also
// ends recording and may leave earlier commands in flight. Further recording
// and repeated submission return Invalid. Wait before submission returns
// Invalid. After submission, wait completes the buffer's work and reports its
// execution outcome. Repeated waits preserve that outcome.
//
// Release consumes this Commands acquisition on every outcome. It completes
// submitted work before destroying recording resources and reports cleanup
// failure through ExecutionStatus. An unsubmitted acquisition discards its
// recorded commands. Copies of this record share the same acquisition and do
// not add a release obligation. The consumer serializes recording, submission,
// waiting and release through that acquisition. Separate Commands acquisitions
// have no implicit ordering.
typedef struct compute_commands {
  void* context;
  const struct compute_commands_ops* operations;
} compute_commands;

typedef struct compute_commands_ops {
  compute_execution_status (*submit)(void* context);
  compute_execution_status (*wait)(void* context);
  compute_execution_status (*release)(void* context);
  ttx_negotiation_status (*supports)(void* context, perimortem_uuid command);
  compute_execution_status (
      *record)(void* context, perimortem_uuid command, const void* arguments);
} compute_commands_ops;

typedef struct compute_commands_result {
  compute_execution_status status;
  compute_commands commands;
} compute_commands_result;

#endif
