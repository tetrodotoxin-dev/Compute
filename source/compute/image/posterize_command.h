// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_IMAGE_POSTERIZE_COMMAND_H
#define COMPUTE_IMAGE_POSTERIZE_COMMAND_H

#include "compute/execution/buffer.h"

#define COMPUTE_POSTERIZE_COMMAND_ID_HIGH 0xc420c323f27c4af2ULL
#define COMPUTE_POSTERIZE_COMMAND_ID_LOW 0x9ef8742ed78bbc16ULL

// PosterizeCommand reduces each RGB byte to the requested number of evenly
// spaced levels and preserves alpha. Levels must be between two and 256.
// Count covers complete RGBA8 bytes and must be divisible by four. The
// consumer keeps both Buffers accessible until Commands release.
typedef struct compute_posterize_command {
  compute_buffer input;
  compute_buffer output;
  U64 count;
  U32 levels;
} compute_posterize_command;

#endif
