// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef CUDA_MODULE_H
#define CUDA_MODULE_H

#include "ttx/semantic/negotiation/library.h"

// CUDA publishes its Query as described Storage. Transfer establishes the
// Query representation before the consumer calls it to obtain the compiler
// Abstract. The host keeps this library loaded through all resource releases.
// The explicit C storage declaration keeps the entry defined by the provider.
// C++ receives that declaration through C_LINKAGE.
#ifdef __cplusplus

C_LINKAGE EXPORTED(CUDA) const ttx_storage ttx_entry;

#else

extern EXPORTED(CUDA) const ttx_storage ttx_entry;

#endif

#endif
