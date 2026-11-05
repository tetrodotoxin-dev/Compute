// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cpu/module.h"

#include "compute/execution/factory.h"
#include "ttx/data/transfer.h"
#include "ttx/graph/abstract.h"
#include "ttx/semantic/negotiation/query.h"

extern HIDDEN const ttx_representation cpu_query_representation;
extern HIDDEN const ttx_representation cpu_factory_representation;
HIDDEN ttx_abstract cpu_factory(void);
HIDDEN compute_engine_result cpu_create_engine(void* context);

static ttx_negotiation_status supports(void* context, perimortem_uuid id) {
  return (id.high == TTX_ABSTRACT_ID_HIGH && id.low == TTX_ABSTRACT_ID_LOW) ||
                 (id.high == COMPUTE_FACTORY_ID_HIGH &&
                  id.low == COMPUTE_FACTORY_ID_LOW)
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

static ttx_negotiation_status
    bind(void* context, perimortem_uuid id, ttx_storage requested) {
  if (supports(context, id) != TTX_NEGOTIATION_SATISFIED) {
    return TTX_NEGOTIATION_UNKNOWN;
  }

  if (id.high == COMPUTE_FACTORY_ID_HIGH && id.low == COMPUTE_FACTORY_ID_LOW) {
    compute_factory factory = {context, cpu_create_engine};
    const ttx_storage provided = {
      &cpu_factory_representation, (U8*)&factory, sizeof(factory)};

    return ttx_transfer(&provided, &requested) == TTX_DATA_SUCCESS
               ? TTX_NEGOTIATION_SATISFIED
               : TTX_NEGOTIATION_REJECTED;
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
