// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_POLICIES_ACCELERATED_H
#define COMPUTE_POLICIES_ACCELERATED_H

#include "ttx/graph/abstract.h"

#define COMPUTE_ACCELERATED_ID_HIGH 0x94e28f4cdf8e496dULL
#define COMPUTE_ACCELERATED_ID_LOW 0x89f444afb0e3569dULL

// Accelerated promises that the engine executes its image transformations on
// a compute device separate from the host CPU. Binding supplies a complete
// Abstract at that engine's graph location. Transfer and host preparation may
// still use the CPU. Execution time depends on the request and device.
typedef ttx_abstract compute_accelerated;

#endif
