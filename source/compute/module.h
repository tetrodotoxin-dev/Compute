// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_MODULE_H
#define COMPUTE_MODULE_H

#include "ttx/semantic/negotiation/library.h"

// Compute exports described Query storage for the CPU engine factory.
// Transfer establishes Query before Semantic negotiation supplies the factory
// Abstract. The host keeps this module loaded through its final acquisition.
#ifdef __cplusplus

C_LINKAGE EXPORTED(COMPUTE) const ttx_storage ttx_entry;

#else

extern EXPORTED(COMPUTE) const ttx_storage ttx_entry;

#endif

#endif
