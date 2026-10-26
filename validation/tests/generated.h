// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_VALIDATION_GENERATED_H
#define COMPUTE_VALIDATION_GENERATED_H

#include "ttx/graph/abstract.h"

// Supplies a generated image for the callback's duration. The C provider has
// no pixel array and exposes different contexts for Pixels and Abstract.
C_LINKAGE void compute_generated_image(
    void* callback_context,
    void (*callback)(void* callback_context, ttx_abstract image));

// Models an available factory whose execution resource cannot be created.
// The caller keeps the attempt counter available through each creation call.
C_LINKAGE ttx_abstract compute_refusing_factory(Count* attempts);

// The generated result supports Pixels while leaving its own Abstract UUID
// undetermined. Selection must preserve that valid result's binding freedom.
C_LINKAGE ttx_abstract compute_generated_factory(void);

#endif
