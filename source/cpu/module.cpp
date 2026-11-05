// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cpu/module.h"

#include "perimortem/core/view/null_terminated.hpp"

#include "compute/reflection/module.hpp"
#include "cpu/platform/memory.hpp"
#include "ttx/semantic/negotiation/query.hpp"

using namespace Ttx::Data;
using namespace Cpu::Platform;
using namespace Compute::Reflection;

// C++ compiles the Query description. The C translation unit publishes the
// exported Storage record using static address initializers. Keeping that
// record free of runtime writes lets several loaded providers use ttx_entry
// independently even when the platform permits symbol interposition.
C_LINKAGE HIDDEN constinit const ttx_representation cpu_query_representation =
    Form::Compiled<
        Form::Native<ttx_semantic_query>::reference>::get_representation()
        .get_abi();

C_LINKAGE HIDDEN constinit const ttx_representation cpu_factory_representation =
    Form::Compiled<
        Form::Native<compute_factory>::reference>::get_representation()
        .get_abi();

C_LINKAGE HIDDEN auto cpu_create_engine(void* context)
    -> compute_engine_result {
  return Cpu::Platform::Memory::create();
}

C_LINKAGE HIDDEN auto cpu_factory() -> ttx_abstract {
  using namespace Perimortem::Core;
  static U8 context;
  static Module factory("ComputeCpu"_view, {&context, cpu_create_engine});

  return factory.get_abstract().get_abi();
}
