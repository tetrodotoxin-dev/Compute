// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "validation/tests/generated.h"

#include <stddef.h>

#include "ttx/graph/capabilities/create.h"
#include "ttx/graph/policies/none.h"

#include "compute/image/pixels.h"

typedef struct generated_state {
  U64 seed;
  struct generated_state* pixels_context;
} generated_state;

static const ttx_abstract_ops graph_operations;

static ttx_abstract graph(void* context) {
  generated_state* state = *(generated_state**)context;
  return (ttx_abstract){state, &graph_operations};
}

static compute_image_dimensions dimensions(void* context) {
  return (compute_image_dimensions){2, 2};
}

static compute_image_status
    read_pixels(void* context, U8* output, Count capacity) {
  const generated_state* state = *(generated_state**)context;
  if (!output || capacity < 16) {
    return COMPUTE_IMAGE_INVALID;
  }

  for (Count i = 0; i < 16; ++i) {
    output[i] = (U8)(state->seed + i * 13);
  }

  return COMPUTE_IMAGE_SUCCESS;
}

static const compute_pixels_ops pixels_operations = {
  graph, dimensions, read_pixels};

static ttx_negotiation_status supports(void* context, perimortem_uuid id) {
  if (!id.high && id.low <= 1) {
    return TTX_NEGOTIATION_REJECTED;
  }

  return id.high == COMPUTE_PIXELS_ID_HIGH && id.low == COMPUTE_PIXELS_ID_LOW
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

static ttx_negotiation_status
    bind(void* context, perimortem_uuid id, void* output) {
  const ttx_negotiation_status status = supports(context, id);
  if (status != TTX_NEGOTIATION_SATISFIED) {
    return status;
  }

  generated_state* state = context;
  compute_pixels* result = output;
  *result = (compute_pixels){&state->pixels_context, &pixels_operations};
  return TTX_NEGOTIATION_SATISFIED;
}

static ttx_abstract route(void* context, perimortem_view_bytes route) {
  return ttx_none();
}

static void visit(void* context, ttx_graph_visitor visitor) {}
static ttx_abstract_state state(void* context) {
  return TTX_ABSTRACT_PRESENT;
}

static const ttx_abstract_ops graph_operations = {
  supports, bind, route, visit, state};

void compute_generated_image(
    void* callback_context,
    void (*callback)(void* callback_context, ttx_abstract image)) {
  generated_state state = {17, NULL};
  state.pixels_context = &state;
  callback(callback_context, (ttx_abstract){&state, &graph_operations});
}

static const ttx_abstract_ops refusing_operations;

static ttx_abstract refusing_graph(void* context) {
  return (ttx_abstract){context, &refusing_operations};
}

static ttx_negotiation_status refuse(
    void* context,
    ttx_abstract arguments,
    void* callback_context,
    void (*callback)(void* callback_context, ttx_abstract engine)) {
  Count* attempts = context;
  ++*attempts;
  return TTX_NEGOTIATION_REJECTED;
}

static const ttx_create_ops refusing_creation = {refusing_graph, refuse};

static ttx_negotiation_status refusing_supports(
    void* context,
    perimortem_uuid id) {
  if (!id.high && id.low <= 1) {
    return TTX_NEGOTIATION_REJECTED;
  }

  return (id.high == TTX_CREATE_ID_HIGH && id.low == TTX_CREATE_ID_LOW) ||
                 (id.high == TTX_ABSTRACT_ID_HIGH &&
                  id.low == TTX_ABSTRACT_ID_LOW)
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

static ttx_negotiation_status
    refusing_bind(void* context, perimortem_uuid id, void* output) {
  const ttx_negotiation_status status = refusing_supports(context, id);
  if (status != TTX_NEGOTIATION_SATISFIED) {
    return status;
  }

  if (id.high == TTX_CREATE_ID_HIGH && id.low == TTX_CREATE_ID_LOW) {
    ttx_create* api = output;
    *api = (ttx_create){context, &refusing_creation};
  } else {
    ttx_abstract* api = output;
    *api = refusing_graph(context);
  }

  return TTX_NEGOTIATION_SATISFIED;
}

static const ttx_abstract_ops refusing_operations = {
  refusing_supports, refusing_bind, route, visit, state};

ttx_abstract compute_refusing_factory(Count* attempts) {
  return refusing_graph(attempts);
}

static const ttx_abstract_ops generated_factory_operations;

static ttx_abstract generated_factory_graph(void* context) {
  return (ttx_abstract){context, &generated_factory_operations};
}

static ttx_negotiation_status create_generated(
    void* context,
    ttx_abstract arguments,
    void* callback_context,
    void (*callback)(void* callback_context, ttx_abstract image)) {
  compute_generated_image(callback_context, callback);
  return TTX_NEGOTIATION_SATISFIED;
}

static const ttx_create_ops generated_creation = {
  generated_factory_graph, create_generated};

static ttx_negotiation_status
    generated_factory_bind(void* context, perimortem_uuid id, void* output) {
  const ttx_negotiation_status status = refusing_supports(context, id);
  if (status != TTX_NEGOTIATION_SATISFIED) {
    return status;
  }

  if (id.high == TTX_CREATE_ID_HIGH && id.low == TTX_CREATE_ID_LOW) {
    ttx_create* api = output;
    *api = (ttx_create){context, &generated_creation};
  } else {
    ttx_abstract* api = output;
    *api = generated_factory_graph(context);
  }

  return TTX_NEGOTIATION_SATISFIED;
}

static const ttx_abstract_ops generated_factory_operations = {
  refusing_supports, generated_factory_bind, route, visit, state};

ttx_abstract compute_generated_factory(void) {
  static U8 context;
  return generated_factory_graph(&context);
}
