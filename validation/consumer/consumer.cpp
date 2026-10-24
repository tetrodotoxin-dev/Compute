// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "ttx/data/transfer.hpp"
#include "ttx/semantic/negotiation/query.hpp"

#include "cuda/contracts/compiler.hpp"
#include "cuda/module.h"

using namespace Cuda::Contracts;
using namespace Perimortem::Memory;
using namespace Ttx::Data;
using namespace Ttx::Graph;
using namespace Ttx::Semantic;

S32 main() {
  ttx_semantic_query entry{};
  auto requested = Form::Storage::from(entry).get_abi();
  if (ttx_transfer(&ttx_entry, &requested) != TTX_DATA_SUCCESS) {
    return 1;
  }

  return Negotiation::Query(entry).bind<Abstract>().visit(
      [](Abstract abstract) -> S32 {
        return abstract.bind<Compiler>().visit(
            [](Compiler compiler) -> S32 {
              static const U8 name[] = "cpp_consumer.cu";
              static const U8 source[] =
                  "extern \"C\" __global__ void empty() {}";
              const cuda_compile_request request{
                {{name, sizeof(name) - 1}, {source, sizeof(source) - 1}},
                nullptr,
                0,
                nullptr,
                0,
                0};

              Global::Bytes diagnostics;
              return compiler.compile(request, diagnostics)
                  .visit(
                      [](Policies::Retained result) -> S32 {
                        result.release();
                        return 0;
                      },

                      [](ExecutionStatus status) -> S32 { return 3; });
            },

            []() -> S32 { return 4; });
      },

      [](auto status) -> S32 { return 5; });
}
