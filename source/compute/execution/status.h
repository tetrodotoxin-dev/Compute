// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_EXECUTION_STATUS_H
#define COMPUTE_EXECUTION_STATUS_H

#include "toolchain/toolchain.h"

// Success means that the requested submission or completion succeeded. Invalid
// identifies a call made in the wrong recording state. Failed reports execution
// failure. Submitted work may already have produced partial results.
typedef U8 compute_execution_status;

#define COMPUTE_EXECUTION_SUCCESS ((compute_execution_status)0)
#define COMPUTE_EXECUTION_INVALID ((compute_execution_status)1)
#define COMPUTE_EXECUTION_FAILED ((compute_execution_status)2)

#endif
