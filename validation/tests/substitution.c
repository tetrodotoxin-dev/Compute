// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "ttx/graph/policies/none.h"

#include "validation/tests/consumer.h"

// The Buffer callbacks interpret a pointer slot inside graph_state. Graph
// callbacks interpret the whole graph_state record, whose first field counts
// reads. Interchanging these contexts would interpret that count as a pointer.
typedef struct graph_state {
  U64 reads;
  struct graph_state* first;
  struct graph_state* second;
  U8 selected;
} graph_state;

static ttx_abstract graph(void* context);
static const cuda_buffer_ops buffer_operations;
static const ttx_abstract_ops graph_operations;

static ttx_abstract graph(void* context) {
  graph_state* state = *(graph_state**)context;
  return (ttx_abstract){state, &graph_operations};
}

static U64 address(void* context) {
  graph_state* state = *(graph_state**)context;
  ++state->reads;
  return state->reads;
}

static Count size(void* context) {
  return 17;
}

static cuda_status
    read_bytes(void* context, Count offset, U8* output, Count count) {
  return CUDA_UNAVAILABLE;
}

static cuda_status
    write_bytes(void* context, Count offset, perimortem_view_bytes input) {
  return CUDA_UNAVAILABLE;
}

static ttx_negotiation_status supports(
    void* context,
    perimortem_uuid contract) {
  if (!contract.high && contract.low <= 1) {
    return TTX_NEGOTIATION_REJECTED;
  }

  return contract.high == TTX_CUDA_BUFFER_HIGH &&
                 contract.low == TTX_CUDA_BUFFER_LOW
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

static ttx_negotiation_status
    bind(void* context, perimortem_uuid contract, void* output) {
  const ttx_negotiation_status status = supports(context, contract);
  if (status != TTX_NEGOTIATION_SATISFIED) {
    return status;
  }

  graph_state* state = context;
  state->selected = !state->selected;

  cuda_buffer* api = output;
  *api = (cuda_buffer){
    state->selected ? &state->first : &state->second, &buffer_operations};

  return TTX_NEGOTIATION_SATISFIED;
}

static ttx_abstract resolve_route(void* context, perimortem_view_bytes route) {
  return ttx_none();
}

static void visit_routes(void* context, ttx_graph_visitor visitor) {}
static ttx_abstract_state get_state(void* context) {
  return TTX_ABSTRACT_PRESENT;
}

static const cuda_buffer_ops buffer_operations = {
  graph, address, size, read_bytes, write_bytes};

static const ttx_abstract_ops graph_operations = {
  supports, bind, resolve_route, visit_routes, get_state};

void cuda_substituted_buffer(
    void* callback_context,
    void (*callback)(void*, cuda_buffer)) {
  graph_state state = {0};
  state.first = &state;
  state.second = &state;
  callback(callback_context, (cuda_buffer){&state.first, &buffer_operations});
}
