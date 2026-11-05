// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "perimortem/core/access/bytes.hpp"
#include "perimortem/core/view/bytes.hpp"

#include "compute/execution/factory.hpp"
#ifdef COMPUTE_CPU_PROVIDER
#include "cpu/module.h"
#else
#include "cuda/module.h"
#endif
#include "ttx/data/form/storage.hpp"
#include "ttx/data/transfer.hpp"
#include "ttx/semantic/negotiation/query.hpp"

using namespace Perimortem::Core;
using namespace Compute::Execution;
using namespace Ttx::Data;
using namespace Ttx::Semantic;

S32 main() {
  ttx_semantic_query api{};
  if (Transfer::copy(Form::Storage(ttx_entry), Form::Storage::from(api)) !=
      Status::Success) {
    return 1;
  }

  auto bound = Negotiation::Query(api).bind<Factory>();

#ifdef COMPUTE_RUN_ENGINE

  return bound.visit(
      [](Factory factory) {
        return factory.create_engine().visit(
            [](Engine engine) {
              return engine.allocate(4).visit(
                  [&](Buffer buffer) {
                    const U8 source[] = {1, 2, 3, 4};
                    U8 output[4] = {};
                    auto result = buffer.write(0, View::Bytes(source));
                    if (result == COMPUTE_EXECUTION_SUCCESS) {
                      result = buffer.read(0, Access::Bytes(output));
                    }

                    if (buffer.release() != COMPUTE_EXECUTION_SUCCESS ||
                        engine.release() != COMPUTE_EXECUTION_SUCCESS) {
                      return 5;
                    }

                    return result == COMPUTE_EXECUTION_SUCCESS &&
                                   View::Bytes(source) == View::Bytes(output)
                               ? 0
                               : 4;
                  },

                  [&](compute_execution_status) {
                    engine.release();
                    return 3;
                  });
            },

            [](compute_execution_status) { return 3; });
      },

      [](Negotiation::Status) { return 2; });

#else

  return bound.visit(
      [](Factory) { return 0; }, [](Negotiation::Status) { return 2; });

#endif
}
