// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_IMAGE_COMPOSITE_COMMAND_H
#define COMPUTE_IMAGE_COMPOSITE_COMMAND_H

#include "compute/execution/buffer.h"

#define COMPUTE_COMPOSITE_COMMAND_ID_HIGH 0x3147c72822c24181ULL
#define COMPUTE_COMPOSITE_COMMAND_ID_LOW 0xae228b94444ff014ULL

// CompositeCommand places front over back using straight RGBA8 alpha. Count
// names pixels, so each Buffer must cover count times four bytes. Input and
// output may use different providers. The consumer keeps all three Buffer
// acquisitions accessible until Commands release.
typedef struct compute_composite_command {
  compute_buffer back;
  compute_buffer front;
  compute_buffer output;
  U64 count;
} compute_composite_command;

#endif
