// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef COMPUTE_TEST_DEFERRED_H
#define COMPUTE_TEST_DEFERRED_H

#include "ttx/graph/abstract.h"

// The test owns this state through every acquired result. The original C
// provider exists only during its callback. Retention substitutes this separate
// layout. Counters make capture, execution order and release observable.
typedef struct compute_deferred_state {
  Count original_reads;
  Count retained_reads;
  Count acquisitions;
  Count releases;
  Count* sequence;
  Count read_order;
  U8 fail;
  U8 fail_release;
  U8 refuse_pixels;
  U8 seed;
} compute_deferred_state;

C_LINKAGE void compute_deferred_image(
    compute_deferred_state* state,
    void* callback_context,
    void (*callback)(void* callback_context, ttx_abstract image));

#endif
