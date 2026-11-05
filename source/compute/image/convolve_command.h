// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_IMAGE_CONVOLVE_COMMAND_H
#define COMPUTE_IMAGE_CONVOLVE_COMMAND_H

#include "compute/execution/buffer.h"

#define COMPUTE_CONVOLVE_COMMAND_ID_HIGH 0xa09cd79ed71f4eb9ULL
#define COMPUTE_CONVOLVE_COMMAND_ID_LOW 0x809183c45b1a408bULL

// ConvolveCommand applies one centered kernel to the RGB channels and treats
// samples outside the packed RGBA8 image as zero. The operation preserves
// alpha. Kernel storage contains native
// R32 values. Image and kernel dimensions must be positive. Kernel dimensions
// must be odd, and their product must fit the weights Buffer. The consumer
// keeps all three Buffers accessible until Commands release.
typedef struct compute_convolve_command {
  compute_buffer input;
  compute_buffer output;
  compute_buffer weights;
  U32 width;
  U32 height;
  U32 kernel_width;
  U32 kernel_height;
} compute_convolve_command;

#endif
