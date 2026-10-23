// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "toolchain/validation/benchmark.hpp"

#include "perimortem/core/diagnostics/log.hpp"
#include "perimortem/core/view/null_terminated.hpp"

#include "ttx/data/transfer.hpp"

#include "cuda/contracts/compiler.hpp"
#include "cuda/module.h"

using namespace Toolchain::Validation;
using namespace Perimortem::Core;
using namespace Ttx::Data;
using namespace Ttx::Graph;
using namespace Ttx::Semantic;
using namespace Cuda::Contracts;

static Storage::Option<Abstract> compiler;

static auto setup() -> void {
  ttx_semantic_query entry{};
  auto requested = Form::Storage::from(entry).get_abi();
  if (ttx_transfer(&ttx_entry, &requested) != TTX_DATA_SUCCESS) {
    Diagnostics::Log::fatal("CUDA bootstrap representation was rejected."_view);
  }

  Negotiation::Query(entry).bind<Abstract>().visit(
      [](Abstract abstract) { compiler = abstract; },
      [](Negotiation::Status status) {
        Diagnostics::Log::fatal("CUDA compiler graph was unavailable."_view);
      });
}

static Harness CudaNegotiation = {.name = "CUDA::Negotiation", .init = setup};

VALIDATION_BENCHMARK(CudaNegotiation, compiler_binding) {
  compiler.visit(
      [](Abstract abstract) {
        auto result = abstract.bind<Compiler>();
        Benchmark::prevent_optimization(result);
      },

      [] {});
}
