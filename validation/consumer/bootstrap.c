// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "compute/execution/factory.h"

#ifdef COMPUTE_CPU_PROVIDER
#include "cpu/module.h"
#else
#include "cuda/module.h"
#endif

#include "ttx/data/transfer.h"
#include "ttx/semantic/negotiation/query.h"

S32 main(void) {
  ttx_semantic_query query = {0};
  ttx_storage requested = {
    ttx_semantic_query_representation(), (U8*)&query, sizeof(query)};

  if (ttx_transfer(&ttx_entry, &requested) != TTX_DATA_SUCCESS) {
    return 1;
  }

  const perimortem_uuid factory = {
    COMPUTE_FACTORY_ID_HIGH, COMPUTE_FACTORY_ID_LOW};

  return query.supports(query.context, factory) == TTX_NEGOTIATION_SATISFIED
             ? 0
             : 2;
}
