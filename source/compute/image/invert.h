// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_IMAGE_INVERT_H
#define COMPUTE_IMAGE_INVERT_H

#include "ttx/graph/policies/retained.h"

#include "compute/image/pixels.h"

#define COMPUTE_INVERT_ID_HIGH 0x163ec657f7f6492aULL
#define COMPUTE_INVERT_ID_LOW 0xb1b89d3cd7d9ef2bULL

// Invert complements each RGB byte and preserves alpha and dimensions. Apply
// obtains Pixels from the supplied image Abstract and produces an independent
// snapshot. The input remains unchanged. The provider interprets the input
// through that bound contract, allowing another provider to supply its pixels.
//
// The caller keeps the input Abstract accessible until apply returns. On
// success, apply writes one Retained result whose Abstract binds Pixels. The
// consumer releases that acquisition after its final use. A failed call leaves
// output unusable and transfers no release obligation. Execution finishes
// within the call, including any transfer needed to consume another provider's
// image.
//
// The consumer must serialize calls through one engine creation and every
// image produced through it, including all bindings, retention and release.
// Input images also follow their supplying contract's serialization policy.
typedef struct compute_invert {
  void* context;
  const struct compute_invert_ops* operations;
} compute_invert;

typedef struct compute_invert_ops {
  // Returns to this binding's graph location through the provider's complete
  // Abstract record, including that record's own context and operations.
  ttx_abstract (*get_abstract)(void* context);
  compute_image_status (
      *apply)(void* context, ttx_abstract image, ttx_retained* output);
} compute_invert_ops;

#endif
