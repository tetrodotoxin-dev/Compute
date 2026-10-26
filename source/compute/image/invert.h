// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_IMAGE_INVERT_H
#define COMPUTE_IMAGE_INVERT_H

#include "ttx/graph/policies/retained.h"

#include "compute/image/pixels.h"

#define COMPUTE_INVERT_ID_HIGH 0x73108444b94d4e03ULL
#define COMPUTE_INVERT_ID_LOW 0xbe270de2a555223dULL

// Invert records RGB complementation while preserving alpha and dimensions.
// The input remains unchanged. The binding's Abstract also supplies Commands.
// Record appends to that buffer and returns one Retained image on Success.
// The result supplies Pixels and Dependency before execution. Its dimensions
// are available immediately. Pixel reads require the buffer's submission.
//
// The caller keeps the input accessible through record. An input produced in
// this buffer is an earlier recorded dependency. A different producing buffer
// must already be submitted. Its input is retained through the supplied
// contract or copied while accessible. A retained input is used through its
// newly supplied Abstract and bindings. Recording can therefore capture input
// whose access ends with a callback. A failed record appends nothing and
// transfers no acquisition.
//
// Submit and wait follow Commands, including authored ordering and resource
// lifetimes. Recording after submission returns Invalid. Host calls follow
// the engine serialization policy in Commands and each input's supplying
// access policy.
typedef struct compute_invert {
  void* context;
  const struct compute_invert_ops* operations;
} compute_invert;

typedef struct compute_invert_ops {
  // Returns to this binding's graph location through the provider's complete
  // Abstract record, including that record's own context and operations.
  ttx_abstract (*get_abstract)(void* context);
  compute_image_status (
      *record)(void* context, ttx_abstract image, ttx_retained* output);
} compute_invert_ops;

#endif
