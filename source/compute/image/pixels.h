// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_IMAGE_PIXELS_H
#define COMPUTE_IMAGE_PIXELS_H

#include "ttx/graph/abstract.h"

#define COMPUTE_PIXELS_ID_HIGH 0xc1150ea13c814fd7ULL
#define COMPUTE_PIXELS_ID_LOW 0xb6685c9a79c5a778ULL

// An image operation returns Success after producing its requested result.
// Invalid identifies unsuitable dimensions, input contracts, recording state
// or destination storage. Failed identifies an execution failure. The caller
// uses an output record only after the operation returns Success, under its
// access contract.
typedef U8 compute_image_status;

#define COMPUTE_IMAGE_SUCCESS ((compute_image_status)0)
#define COMPUTE_IMAGE_INVALID ((compute_image_status)1)
#define COMPUTE_IMAGE_FAILED ((compute_image_status)2)

typedef struct compute_image_dimensions {
  U32 width;
  U32 height;
} compute_image_dimensions;

// Pixels supplies a snapshot as tightly packed RGBA8 rows, starting at the
// upper left. Width and height are positive and their product times four fits
// Count. RGB and alpha are independent byte values. The provider may store,
// generate or transfer pixels to supply this representation.
//
// Read copies exactly width * height * 4 bytes into the caller's destination.
// The caller keeps that writable range available until read returns. An
// undersized or null destination is Invalid and remains unchanged. Failed can
// leave destination bytes changed. The provider keeps dimensions and pixels
// stable for this binding's access duration. Retain establishes any additional
// lifetime through the supplied Abstract.
//
// A recorded image exposes dimensions before execution. Read before its
// producing buffer is submitted returns Invalid without changing destination
// bytes. After submission, read waits for the work needed to produce this
// image and reports execution failure as Failed. Metadata queries do not
// submit or wait for work. Reading never implicitly submits another buffer.
typedef struct compute_pixels {
  void* context;
  const struct compute_pixels_ops* operations;
} compute_pixels;

typedef struct compute_pixels_ops {
  // Returns to this binding's graph location through the provider's complete
  // Abstract record, including that record's own context and operations.
  ttx_abstract (*get_abstract)(void* context);
  compute_image_dimensions (*dimensions)(void* context);
  compute_image_status (*read)(void* context, U8* output, Count capacity);
} compute_pixels_ops;

#endif
