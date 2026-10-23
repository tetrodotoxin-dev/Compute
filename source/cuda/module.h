// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef CUDA_MODULE_H
#define CUDA_MODULE_H

#include "ttx/semantic/negotiation/library.h"

// CUDA publishes its Query as described Storage. Transfer establishes the
// Query representation before the consumer calls it to obtain the compiler
// Abstract. The host keeps this library loaded through all resource releases.
C_LINKAGE EXPORTED(CUDA) const ttx_storage ttx_entry;

#endif
