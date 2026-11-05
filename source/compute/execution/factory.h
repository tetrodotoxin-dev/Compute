// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_EXECUTION_FACTORY_H
#define COMPUTE_EXECUTION_FACTORY_H

#include "compute/execution/engine.h"

#define COMPUTE_FACTORY_ID_HIGH 0xeb3b05896cf844b6ULL
#define COMPUTE_FACTORY_ID_LOW 0x83ae84ee889d4166ULL

// Factory creates independent Engine acquisitions. Selection happens before
// execution. A failed creation returns no engine and performs no application
// commands. A consumer may then select another factory. Creation supplies the
// complete Engine contract, independently of the factory's graph or native
// state.
typedef struct compute_factory {
  void* context;
  compute_engine_result (*create_engine)(void* context);
} compute_factory;

#endif
