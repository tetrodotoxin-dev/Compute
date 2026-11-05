// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "toolchain/validation/benchmark.hpp"

#include "coherence/type_system/type.hpp"
#include "compute/execution/values.hpp"

using namespace Toolchain::Validation;
using namespace Coherence::TypeSystem;
using namespace Compute::Execution;

static Harness Reflection = {.name = "Compute::Reflection"};

VALIDATION_BENCHMARK(Reflection, type_binding) {
  auto binding = Values::engine().bind<Type::Contract>();
  Benchmark::prevent_optimization(binding);
}
