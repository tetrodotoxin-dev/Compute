// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "validation/tests/deferred.h"

#include "ttx/graph/capabilities/retain.h"
#include "ttx/graph/policies/none.h"

#include "compute/execution/dependency.h"
#include "compute/image/pixels.h"

typedef struct source {
  U64 tag;
  compute_deferred_state* retained;
  struct source* pixels;
} source;

static const ttx_abstract_ops source_operations, retained_operations;

static ttx_abstract original_graph(void* context) {
  return (ttx_abstract){context, &source_operations};
}

static ttx_abstract retained_graph(void* context) {
  return (ttx_abstract){context, &retained_operations};
}

static ttx_abstract pixel_graph(void* context) {
  source* value = *(source**)context;
  return original_graph(value);
}

static compute_image_dimensions dimensions(void* context) {
  return (compute_image_dimensions){2, 2};
}

static compute_image_status
    fill(compute_deferred_state* state, U8* output, Count size) {
  if (!output || size < 16) {
    return COMPUTE_IMAGE_INVALID;
  }

  if (state->fail) {
    return COMPUTE_IMAGE_FAILED;
  }

  for (Count i = 0; i < 16; ++i) {
    output[i] = (U8)(state->seed + i * 13);
  }

  return COMPUTE_IMAGE_SUCCESS;
}

static compute_image_status
    original_read(void* context, U8* output, Count size) {
  source* value = *(source**)context;
  ++value->retained->original_reads;
  return fill(value->retained, output, size);
}

static compute_image_status
    retained_read(void* context, U8* output, Count size) {
  compute_deferred_state* state = context;
  ++state->retained_reads;
  if (state->sequence) {
    state->read_order = ++*state->sequence;
  }

  return fill(state, output, size);
}

static ttx_release_status release(void* context) {
  compute_deferred_state* state = context;
  ++state->releases;
  return state->fail_release ? TTX_RELEASE_CLEANUP_FAILED : TTX_RELEASE_SUCCESS;
}

static const ttx_retained_ops lifetime = {retained_graph, release};

static ttx_negotiation_status acquire(void* context, ttx_retained* output) {
  compute_deferred_state* state = context;
  ++state->acquisitions;
  *output = (ttx_retained){state, &lifetime};
  return TTX_NEGOTIATION_SATISFIED;
}

static ttx_negotiation_status acquire_original(
    void* context,
    ttx_retained* output) {
  source* value = context;
  return acquire(value->retained, output);
}

static U8 submitted(void* context) {
  return 1;
}

static ttx_negotiation_status
    locate(void* context, compute_commands commands, Count* position) {
  return TTX_NEGOTIATION_REJECTED;
}

static const ttx_retain_ops original_retention = {
  original_graph, acquire_original};

static const ttx_retain_ops retained_retention = {retained_graph, acquire};
static const compute_pixels_ops original_pixels = {
  pixel_graph, dimensions, original_read};

static const compute_pixels_ops retained_pixels = {
  retained_graph, dimensions, retained_read};

static const compute_dependency_ops original_dependency = {
  original_graph, submitted, locate};

static const compute_dependency_ops retained_dependency = {
  retained_graph, submitted, locate};

static ttx_negotiation_status supports(void* context, perimortem_uuid id) {
  if (!id.high && id.low <= 1) {
    return TTX_NEGOTIATION_REJECTED;
  }

  return (id.high == COMPUTE_PIXELS_ID_HIGH &&
          id.low == COMPUTE_PIXELS_ID_LOW) ||
                 (id.high == COMPUTE_DEPENDENCY_ID_HIGH &&
                  id.low == COMPUTE_DEPENDENCY_ID_LOW) ||
                 (id.high == TTX_RETAIN_ID_HIGH && id.low == TTX_RETAIN_ID_LOW)
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

static ttx_negotiation_status
    original_bind(void* context, perimortem_uuid id, void* output) {
  const ttx_negotiation_status status = supports(context, id);
  if (status != TTX_NEGOTIATION_SATISFIED) {
    return status;
  }

  source* value = context;
  if (id.high == COMPUTE_PIXELS_ID_HIGH && id.low == COMPUTE_PIXELS_ID_LOW) {
    compute_pixels* api = output;
    *api = (compute_pixels){&value->pixels, &original_pixels};
  } else if (
      id.high == COMPUTE_DEPENDENCY_ID_HIGH &&
      id.low == COMPUTE_DEPENDENCY_ID_LOW) {
    compute_dependency* api = output;
    *api = (compute_dependency){context, &original_dependency};
  } else {
    ttx_retain* api = output;
    *api = (ttx_retain){context, &original_retention};
  }

  return TTX_NEGOTIATION_SATISFIED;
}

static ttx_negotiation_status retained_supports(
    void* context,
    perimortem_uuid id) {
  compute_deferred_state* state = context;
  if (state->refuse_pixels && id.high == COMPUTE_PIXELS_ID_HIGH &&
      id.low == COMPUTE_PIXELS_ID_LOW) {
    return TTX_NEGOTIATION_UNKNOWN;
  }

  return supports(context, id);
}

static ttx_negotiation_status
    retained_bind(void* context, perimortem_uuid id, void* output) {
  const ttx_negotiation_status status = retained_supports(context, id);
  if (status != TTX_NEGOTIATION_SATISFIED) {
    return status;
  }

  if (id.high == COMPUTE_PIXELS_ID_HIGH && id.low == COMPUTE_PIXELS_ID_LOW) {
    compute_pixels* api = output;
    *api = (compute_pixels){context, &retained_pixels};
  } else if (
      id.high == COMPUTE_DEPENDENCY_ID_HIGH &&
      id.low == COMPUTE_DEPENDENCY_ID_LOW) {
    compute_dependency* api = output;
    *api = (compute_dependency){context, &retained_dependency};
  } else {
    ttx_retain* api = output;
    *api = (ttx_retain){context, &retained_retention};
  }

  return TTX_NEGOTIATION_SATISFIED;
}

static ttx_abstract route(void* context, perimortem_view_bytes route) {
  return ttx_none();
}

static void visit(void* context, ttx_graph_visitor visitor) {}
static ttx_abstract_state state(void* context) {
  return TTX_ABSTRACT_PRESENT;
}

static const ttx_abstract_ops source_operations = {
  supports, original_bind, route, visit, state};

static const ttx_abstract_ops retained_operations = {
  retained_supports, retained_bind, route, visit, state};

void compute_deferred_image(
    compute_deferred_state* state,
    void* callback_context,
    void (*callback)(void* callback_context, ttx_abstract image)) {
  source original = {91, state, 0};
  original.pixels = &original;
  callback(callback_context, original_graph(&original));
}
