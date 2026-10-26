// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cpu/module.h"

#include "ttx/semantic/negotiation/query.hpp"

#include "cpu/platform/engine.hpp"

using namespace Ttx::Data;
using namespace Cpu::Platform;

// C++ compiles the Query description. The C translation unit publishes the
// exported Storage record using static address initializers. Keeping that
// record free of runtime writes lets several loaded providers use ttx_entry
// independently even when the platform permits symbol interposition.
C_LINKAGE HIDDEN constinit const ttx_representation cpu_query_representation =
    Form::Compiled<
        Form::Native<ttx_semantic_query>::reference>::get_representation()
        .get_abi();

C_LINKAGE HIDDEN auto cpu_factory() -> ttx_abstract {
  return Engine::factory().get_abi();
}
