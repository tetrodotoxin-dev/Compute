// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "compute/module.h"

#include "ttx/semantic/negotiation/query.hpp"

#include "compute/platforms/cpu.hpp"

using namespace Ttx::Data;
using namespace Compute::Platforms;

// C++ compiles the Query description. The C translation unit publishes the
// exported Storage record using static address initializers. Keeping that
// record free of runtime writes lets several loaded providers use ttx_entry
// independently even when the platform permits symbol interposition.
C_LINKAGE HIDDEN constinit const ttx_representation
    compute_query_representation =
        Form::Compiled<
            Form::Native<ttx_semantic_query>::reference>::get_representation()
            .get_abi();

C_LINKAGE HIDDEN auto compute_factory() -> ttx_abstract {
  return Cpu::factory().get_abi();
}
