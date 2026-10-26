// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "toolchain/validation/test.hpp"

#include "perimortem/core/view/null_terminated.hpp"

#include "cuda/runtime/current_context.hpp"
#include "cuda/runtime/program.hpp"

using namespace Cuda::Runtime;
using namespace Toolchain::Validation;
using namespace Ttx::Graph;

static Harness Native = {.name = "CUDA::Native"};

VALIDATION_TEST(Native, context_restoration) {
  CUcontext before = nullptr, after = nullptr;
  ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
  ASSERT_EQ(cuCtxGetCurrent(&before), CUDA_SUCCESS);

  const cuda_compile_request request{
    {"empty.cu"_view, "extern \"C\" __global__ void empty() {}"_view},
    nullptr,
    0,
    nullptr,
    0,
    0};

  Program::create(request, {})
      .visit(
          [&](Program& program) {
            EXPECT(program.get_module() != nullptr);
            {
              CurrentContext current(program.get_context());
              EXPECT_EQ(current.status, CUDA_SUCCESS);
              EXPECT_EQ(cuCtxGetCurrent(&after), CUDA_SUCCESS);
              EXPECT(after == program.get_context());
            }

            EXPECT_EQ(cuCtxGetCurrent(&after), CUDA_SUCCESS);
            EXPECT(after == before);
            EXPECT(program.release() == Policies::Retained::Status::Success);
          },

          [&](auto status) { EXPECT(False); });
}
