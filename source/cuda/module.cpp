// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cuda/module.h"

#include "ttx/data/transfer.hpp"
#include "ttx/semantic/negotiation/library.h"
#include "ttx/semantic/negotiation/query.hpp"

#include "cuda/runtime/program.hpp"

using namespace Perimortem::Core;
using namespace Ttx::Data;
using namespace Ttx::Graph;
using namespace Cuda::Runtime;

static auto supports(void* context, perimortem_uuid contract)
    -> ttx_negotiation_status {
  return Identity::Uuid(contract) == Abstract::contract_id
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

static auto bind(void* context, perimortem_uuid contract, ttx_storage requested)
    -> ttx_negotiation_status {
  if (supports(context, contract) != TTX_NEGOTIATION_SATISFIED) {
    return TTX_NEGOTIATION_UNKNOWN;
  }

  auto api = Program::compiler().get_abi();
  return Transfer::copy(Form::Storage::from(api), Form::Storage(requested)) ==
                 Status::Success
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_REJECTED;
}

static U8 context;
static ttx_semantic_query query{&context, bind, supports};

// Data Transfer establishes the exported Query representation before the
// consumer calls its functions. The exported record, its descriptors and the
// compiler graph live with the module.
C_LINKAGE EXPORTED(CUDA) const ttx_storage ttx_entry{
  &Form::Compiled<
       Form::Native<ttx_semantic_query>::reference>::get_representation()
       .get_abi(),
  Data::cast<U8>(&query), sizeof(query)};
