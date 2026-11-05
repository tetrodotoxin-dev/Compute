// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include <cuda.h>

#include "toolchain/validation/test.hpp"

#include "compute/execution/engine.hpp"
#include "cuda/runtime/engine.hpp"

using namespace Compute::Execution;
using namespace Toolchain::Validation;

static Harness Native = {.name = "CUDA::Native"};

VALIDATION_TEST(Native, context_restoration) {
  CUcontext before = nullptr, after = nullptr;
  ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
  ASSERT_EQ(cuCtxGetCurrent(&before), CUDA_SUCCESS);

  const auto created = Cuda::Runtime::Engine::create();
  ASSERT_EQ(created.status, COMPUTE_EXECUTION_SUCCESS);

  Engine engine(created.engine);
  engine.create_commands().visit(
      [&](Commands commands) {
        EXPECT_EQ(cuCtxGetCurrent(&after), CUDA_SUCCESS);
        EXPECT(after == before);
        EXPECT_EQ(commands.release(), COMPUTE_EXECUTION_SUCCESS);
      },

      [&](compute_execution_status status) { EXPECT(False); });

  EXPECT_EQ(cuCtxGetCurrent(&after), CUDA_SUCCESS);
  EXPECT(after == before);
  EXPECT_EQ(engine.release(), COMPUTE_EXECUTION_SUCCESS);
}
