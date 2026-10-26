// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef CPU_MODULE_H
#define CPU_MODULE_H

#include "ttx/semantic/negotiation/library.h"

// CPU exports described Query storage for its engine factory.
// Transfer establishes Query before Semantic negotiation supplies the factory
// Abstract. The host keeps this module loaded through its final acquisition.
#ifdef __cplusplus

C_LINKAGE EXPORTED(CPU) const ttx_storage ttx_entry;

#else

extern EXPORTED(CPU) const ttx_storage ttx_entry;

#endif

#endif
