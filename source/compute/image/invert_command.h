// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_IMAGE_INVERT_COMMAND_H
#define COMPUTE_IMAGE_INVERT_COMMAND_H

#include "compute/execution/buffer.h"

#define COMPUTE_INVERT_COMMAND_ID_HIGH 0xa47d9fd315414d63ULL
#define COMPUTE_INVERT_COMMAND_ID_LOW 0xb7713ad642ee1a1aULL

// InvertCommand identifies RGB byte complementation with preserved alpha.
// Count covers complete RGBA8 bytes and must be divisible by four. Input and
// output can name different provider buffers. The consumer keeps both Buffer
// acquisitions accessible until its Commands acquisition is released.
typedef struct compute_invert_command {
  compute_buffer input;
  compute_buffer output;
  U64 count;
} compute_invert_command;

#endif
