// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef CUDA_CONTRACTS_STATUS_H
#define CUDA_CONTRACTS_STATUS_H

#include "toolchain/toolchain.h"

// CUDA operations distinguish invalid inputs, unavailable devices, compilation
// failures and driver failures. A successful creation supplies one Retained
// result. On failure the output record is unusable and no release is required.
// Launch and transfer failures can leave device bytes changed.
typedef U8 cuda_status;

#define CUDA_SUCCESS_STATUS ((cuda_status)0)
#define CUDA_INVALID ((cuda_status)1)
#define CUDA_BOUNDS ((cuda_status)2)
#define CUDA_INCOMPATIBLE ((cuda_status)3)
#define CUDA_UNAVAILABLE ((cuda_status)4)
#define CUDA_COMPILATION_FAILED ((cuda_status)5)
#define CUDA_DRIVER_FAILED ((cuda_status)6)

#endif
