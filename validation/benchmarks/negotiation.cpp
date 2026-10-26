// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "toolchain/validation/benchmark.hpp"

#include "ttx/graph/capabilities/create.hpp"

#include "cpu/platform/engine.hpp"

using namespace Toolchain::Validation;
using namespace Ttx::Graph;
using namespace Cpu::Platform;

static Harness Negotiation = {.name = "Compute::Negotiation"};

VALIDATION_BENCHMARK(Negotiation, factory_binding) {
  auto binding = Engine::factory().bind<Capabilities::Create>();
  Benchmark::prevent_optimization(binding);
}
