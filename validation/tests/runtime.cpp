// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "toolchain/validation/test.hpp"

#include "perimortem/core/diagnostics/log.hpp"
#include "perimortem/core/storage/vector.hpp"
#include "perimortem/core/view/null_terminated.hpp"
#include "perimortem/memory/allocator/bibliotheca.hpp"
#include "perimortem/system/process/library.hpp"

#include "ttx/data/transfer.hpp"
#include "ttx/graph/capabilities/retain.hpp"
#include "ttx/graph/policies/none.hpp"
#include "ttx/graph/policies/unknown.hpp"

#include "cuda/contracts/buffer.hpp"
#include "cuda/contracts/compiler.hpp"
#include "cuda/contracts/kernel.hpp"
#include "cuda/contracts/program.hpp"
#include "cuda/module.h"
#include "validation/tests/consumer.h"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Perimortem::System;
using namespace Toolchain::Validation;
using namespace Cuda::Contracts;
using namespace Ttx::Graph;
using namespace Ttx::Data;
using namespace Ttx::Semantic;

static Harness CudaTests = {.name = "CUDA"};

// Each test creation result is released at scope exit, including after an
// assertion returns early. Explicit release consumes that result first.
class Held {
 public:
  // Explicit construction transfers the result's release obligation into
  // this test scope. An implicit conversion would hide who must release it.
  explicit Held(Policies::Retained retained) : retained(retained.get_abi()) {}
  Held(const Held&) = delete;
  auto operator=(const Held&) -> Held& = delete;
  ~Held() { release(); }
  auto get_abstract() const -> Abstract {
    return Abstract(retained.operations->get_abstract(retained.context));
  }

  auto release() -> Policies::Retained::Status {
    if (retained.operations) {
      const auto completed = retained;
      retained = {};
      return Policies::Retained::Status(
          completed.operations->release(completed.context));
    }

    return Policies::Retained::Status::Success;
  }

 private:
  ttx_retained retained;
};

template <typename Type, typename Error>
static auto accepted(Storage::Result<Type, Error> value) -> Type {
  return value.visit(
      [](Type& result) -> Type {
        if constexpr (__is_lvalue_reference(Type)) {
          return result;
        } else {
          return Data::take(result);
        }
      },

      [](Error error) -> Type {
        Diagnostics::Log::fatal("CUDA fixture preparation failed."_view);
      });
}

template <typename Type>
static auto bound(Abstract abstract) -> Type {
  return abstract.bind<Type>().visit(
      [](Type result) -> Type { return result; },
      []() -> Type {
        Diagnostics::Log::fatal("CUDA fixture binding failed."_view);
      });
}

static auto compiler() -> Compiler {
  Allocator::Arena errors;

#ifdef TOOLCHAIN_WINDOWS

  constexpr auto path = "ttx_cuda.dll"_view;

#else

  constexpr auto path = "libttx_cuda.so"_view;

#endif

  static auto library = accepted(Process::Library::open(path, errors));
  const auto* symbol = accepted(library.symbol("ttx_entry"_view, errors));
  const auto& published = *Data::cast<const ttx_storage>(symbol);
  ttx_semantic_query entry{};
  auto destination = Form::Storage::from(entry).get_abi();
  if (ttx_transfer(&published, &destination) != TTX_DATA_SUCCESS) {
    Diagnostics::Log::fatal("CUDA bootstrap representation was rejected."_view);
  }

  return bound<Compiler>(accepted(Negotiation::Query(entry).bind<Abstract>()));
}

static constexpr auto source_name = "project.cu"_view;
static constexpr auto header_name = "parameters.cuh"_view;
static constexpr auto header_text =
    "struct Parameters { float scale; unsigned bias; };"_view;
static constexpr auto source_text = R"CUDA(
#include "parameters.cuh"
extern "C" __global__ void transform(float* out, const float* in, unsigned count,
                                     Parameters parameters, double adjustment) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = in[i] * parameters.scale + parameters.bias + adjustment;
}
)CUDA"_view;
static const cuda_source header{header_name, header_text};
static const cuda_compile_request request{
  {source_name, source_text}, &header, 1, nullptr, 0, 0};

struct Parameters {
  R32 scale;
  U32 bias;
};

struct Arguments {
  U64 output;
  U64 input;
  U32 count;
  Parameters parameters;
  R64 adjustment;
};

TTX_DATA_RECORD(
    Parameters,
    TTX_DATA_MEMBER(Parameters, scale),
    TTX_DATA_MEMBER(Parameters, bias));
TTX_DATA_RECORD(
    Arguments,
    TTX_DATA_MEMBER(Arguments, output),
    TTX_DATA_MEMBER(Arguments, input),
    TTX_DATA_MEMBER(Arguments, count),
    TTX_DATA_MEMBER(Arguments, parameters),
    TTX_DATA_MEMBER(Arguments, adjustment));

template <typename Type>
static auto form() -> const Form::Representation& {
  return Form::Compiled<Form::Native<Type>::reference>::get_representation();
}

static auto arguments() -> Storage::Vector<cuda_argument, 5> {
  return {
    {cuda_argument{
       __builtin_offsetof(Arguments, output), &form<U64>().get_abi()},

     {__builtin_offsetof(Arguments, input), &form<U64>().get_abi()},
     {__builtin_offsetof(Arguments, count), &form<U32>().get_abi()},
     {__builtin_offsetof(Arguments, parameters), &form<Parameters>().get_abi()},
     {__builtin_offsetof(Arguments, adjustment), &form<R64>().get_abi()}}};
}

static auto outcome(Storage::Result<Policies::Retained, ExecutionStatus> value)
    -> ExecutionStatus {
  return value.visit(
      [](Policies::Retained retained) {
        retained.release();
        return ExecutionStatus::Success;
      },

      [](ExecutionStatus status) { return status; });
}

VALIDATION_TEST(CudaTests, c_consumer) {
  EXPECT_EQ(cuda_validate_c(), S32(0));
}

VALIDATION_TEST(CudaTests, bootstrap) {
  Allocator::Arena errors;

#ifdef TOOLCHAIN_WINDOWS

  constexpr auto path = "ttx_cuda.dll"_view;

#else

  constexpr auto path = "libttx_cuda.so"_view;

#endif

  auto library = accepted(Process::Library::open(path, errors));
  auto* symbol = accepted(library.symbol("ttx_entry"_view, errors));
  const auto& published = *Data::cast<const ttx_storage>(symbol);
  ttx_semantic_query entry{};
  auto requested = Form::Storage::from(entry).get_abi();
  ASSERT_EQ(ttx_transfer(&published, &requested), TTX_DATA_SUCCESS);

  const Negotiation::Query query(entry);
  auto abstract = accepted(query.bind<Abstract>());
  auto api = bound<Compiler>(abstract);
  EXPECT(api.get_abstract() == abstract);
  EXPECT(abstract.get_state() == Abstract::State::Present);
  EXPECT(abstract.supports<Policies::None>() == Negotiation::Status::Rejected);
  EXPECT(
      abstract.supports<Policies::Unknown>() == Negotiation::Status::Rejected);
  EXPECT(!abstract.bind<Policies::None>());
  EXPECT(!abstract.bind<Policies::Unknown>());
  EXPECT(
      abstract.resolve_route("missing"_view).get_state() ==
      Abstract::State::None);

  U32 wrong = 0;
  requested = Form::Storage::from(wrong).get_abi();
  EXPECT_EQ(ttx_transfer(&published, &requested), TTX_DATA_INCOMPATIBLE);
  EXPECT_EQ(wrong, U32(0));
}

VALIDATION_TEST(CudaTests, compilation_failures) {
  Global::Bytes errors;
  auto invalid = request;
  invalid.source.text = "not valid CUDA source"_view;
  EXPECT(
      outcome(compiler().compile(invalid, errors)) ==
      ExecutionStatus::CompilationFailed);
  EXPECT(!errors.is_empty());
  invalid = request;
  invalid.options = nullptr;
  invalid.option_count = 1;
  EXPECT(
      outcome(compiler().compile(invalid, errors)) == ExecutionStatus::Invalid);
  invalid = request;
  invalid.source.name = "embedded\0null"_view;
  EXPECT(
      outcome(compiler().compile(invalid, errors)) == ExecutionStatus::Invalid);
}

VALIDATION_TEST(CudaTests, retained_launch) {
  Global::Bytes errors;
  Held program_lifetime(accepted(compiler().compile(request, errors)));
  const auto program = bound<Program>(program_lifetime.get_abstract());
  EXPECT(program.get_abstract() == program_lifetime.get_abstract());

  Held input_lifetime(accepted(program.allocate(4 * sizeof(R32))));
  Held output_lifetime(accepted(program.allocate(4 * sizeof(R32))));
  const auto input = bound<Buffer>(input_lifetime.get_abstract());
  const auto output = bound<Buffer>(output_lifetime.get_abstract());
  EXPECT(input.get_abstract() == input_lifetime.get_abstract());

  Storage::Vector<R32, 4> values{{1, 2, 3, 4}};
  const View::Bytes bytes(
      Data::cast<const U8>(values.get_data())[0], sizeof(values));
  EXPECT(input.write(0, bytes) == ExecutionStatus::Success);
  EXPECT(input.write(sizeof(values), bytes) == ExecutionStatus::Bounds);

  const auto buffer_api = input.get_abi();
  EXPECT_EQ(
      buffer_api.operations->read(buffer_api.context, 0, nullptr, 1),
      CUDA_INVALID);
  EXPECT_EQ(
      buffer_api.operations->read(
          buffer_api.context, input.get_size(), nullptr, 0),
      CUDA_SUCCESS_STATUS);

  auto parameters = arguments();
  Held kernel_lifetime(accepted(program.prepare(
      "transform"_view, form<Arguments>(), parameters.get_view(), errors)));
  const auto kernel = bound<Kernel>(kernel_lifetime.get_abstract());
  EXPECT(kernel.get_abstract() == kernel_lifetime.get_abstract());

  auto wrong = parameters;
  wrong[2].representation = &form<U64>().get_abi();
  EXPECT(
      outcome(program.prepare(
          "transform"_view, form<Arguments>(), wrong.get_view(), errors)) ==
      ExecutionStatus::Incompatible);
  EXPECT(
      outcome(program.prepare(
          "absent"_view, form<Arguments>(), parameters.get_view(), errors)) ==
      ExecutionStatus::Unavailable);
  EXPECT(
      outcome(program.prepare(
          "transform"_view, form<Arguments>(), parameters.slice(0, 4),
          errors)) == ExecutionStatus::Incompatible);

  auto invalid = request;
  invalid.source.text = "not valid CUDA source"_view;
  EXPECT(
      outcome(compiler().compile(invalid, errors)) ==
      ExecutionStatus::CompilationFailed);

  // The second acquisition survives release of the first. Bind again through
  // that acquisition, keeping the original result's API unused after release.
  Held second_input(
      accepted(bound<Capabilities::Retain>(input.get_abstract()).retain()));
  EXPECT(input_lifetime.release() == Policies::Retained::Status::Success);
  EXPECT(program_lifetime.release() == Policies::Retained::Status::Success);

  const auto retained_input = bound<Buffer>(second_input.get_abstract());
  Arguments frame{
    output.get_address(), retained_input.get_address(), 4, {2.0f, 3}, 0.5};

  const auto storage = Form::Storage::from(frame);
  const cuda_launch geometry{1, 1, 1, 32, 1, 1, 0};
  ASSERT(kernel.launch(geometry, storage) == ExecutionStatus::Success);

  Storage::Vector<R32, 4> observed;
  EXPECT(
      output.read(
          0, Access::Bytes(
                 Data::cast<U8>(observed.get_data())[0], sizeof(observed))) ==
      ExecutionStatus::Success);
  for (Count i = 0; i < 4; ++i) {
    EXPECT_EQ(observed[i], values[i] * 2 + 3.5f);
  }

  U32 unrelated = 0;
  EXPECT(
      kernel.launch(geometry, Form::Storage::from(unrelated)) ==
      ExecutionStatus::Incompatible);

  auto short_frame = storage.get_abi();
  short_frame.size = 1;
  EXPECT(
      kernel.launch(geometry, Form::Storage(short_frame)) ==
      ExecutionStatus::Bounds);

  auto missing_form = storage.get_abi();
  missing_form.representation = nullptr;
  EXPECT(
      kernel.launch(geometry, Form::Storage(missing_form)) ==
      ExecutionStatus::Invalid);

  const Count allocations = Allocator::Bibliotheca::check_out_requests();
  for (Count i = 0; i < 1000; ++i) {
    EXPECT(kernel.launch(geometry, storage) == ExecutionStatus::Success);
  }

  EXPECT_EQ(Allocator::Bibliotheca::check_out_requests(), allocations);
  EXPECT(kernel_lifetime.release() == Policies::Retained::Status::Success);
  EXPECT(second_input.release() == Policies::Retained::Status::Success);
  EXPECT(output_lifetime.release() == Policies::Retained::Status::Success);
}

VALIDATION_TEST(CudaTests, substituted_contexts) {
  cuda_substituted_buffer(
      &result, [](void* callback_context, cuda_buffer supplied) {
        auto& result = *Data::cast<Test::TestResult>(callback_context);
        const Buffer first(supplied);
        const auto abstract = first.get_abstract();
        EXPECT(abstract.get_abi().context != supplied.context);
        EXPECT_EQ(first.get_address(), U64(1));

        const auto second = bound<Buffer>(abstract);
        const auto third = bound<Buffer>(second.get_abstract());
        EXPECT(second.get_abi().context != third.get_abi().context);
        EXPECT(third.get_abstract() == abstract);
        EXPECT_EQ(second.get_size(), Count(17));
        EXPECT_EQ(third.get_address(), U64(2));
        EXPECT_EQ(first.get_address(), U64(3));
      });
}
