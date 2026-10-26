// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cpu/module.h"

#include "ttx/data/transfer.h"
#include "ttx/graph/abstract.h"
#include "ttx/semantic/negotiation/query.h"

extern HIDDEN const ttx_representation cpu_query_representation;
HIDDEN ttx_abstract cpu_factory(void);

static ttx_negotiation_status supports(void* context, perimortem_uuid id) {
  return id.high == TTX_ABSTRACT_ID_HIGH && id.low == TTX_ABSTRACT_ID_LOW
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

static ttx_negotiation_status
    bind(void* context, perimortem_uuid id, ttx_storage requested) {
  if (supports(context, id) != TTX_NEGOTIATION_SATISFIED) {
    return TTX_NEGOTIATION_UNKNOWN;
  }

  ttx_abstract api = cpu_factory();
  const ttx_storage provided = {
    ttx_abstract_representation(), (U8*)&api, sizeof(api)};

  return ttx_transfer(&provided, &requested) == TTX_DATA_SUCCESS
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_REJECTED;
}

static U8 context;
static ttx_semantic_query query = {&context, bind, supports};

EXPORTED(CPU) const ttx_storage ttx_entry = {
  &cpu_query_representation, (U8*)&query, sizeof(query)};
