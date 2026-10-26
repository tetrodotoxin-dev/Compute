// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cuda/module.h"

#include "ttx/semantic/negotiation/query.hpp"

#include "cuda/runtime/program.hpp"

using namespace Ttx::Data;
using namespace Cuda::Runtime;

// C++ compiles the Query description. The C translation unit publishes the
// exported Storage record using static address initializers. Keeping that
// record free of runtime writes lets several loaded providers use ttx_entry
// independently even when the platform permits symbol interposition.
C_LINKAGE HIDDEN constinit const ttx_representation cuda_query_representation =
    Form::Compiled<
        Form::Native<ttx_semantic_query>::reference>::get_representation()
        .get_abi();

C_LINKAGE HIDDEN auto cuda_factory() -> ttx_abstract {
  return Program::compiler().get_abi();
}
