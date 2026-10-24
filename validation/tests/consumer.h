// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef CUDA_VALIDATION_CONSUMER_H
#define CUDA_VALIDATION_CONSUMER_H

#include "toolchain/toolchain.h"

#include "ttx/data/form/storage.h"

#include "cuda/contracts/buffer.h"

// Exercises the loaded CUDA provider through C records alone. Zero reports
// that bootstrap, typed binding and independently retained storage succeeded.
C_LINKAGE S32 cuda_validate_c(void);

// Exercises the same C contracts through a linked SDK entry. The caller keeps
// the provider available until this function releases its last result.
C_LINKAGE S32 cuda_validate_entry(const ttx_storage* published);

// The callback receives a temporary C provider whose typed and graph contexts
// have different layouts. Its API remains available until the callback returns.
C_LINKAGE void cuda_substituted_buffer(
    void* callback_context,
    void (*callback)(void* callback_context, cuda_buffer api));

#endif
