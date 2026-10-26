// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cuda/platform/engine.hpp"

#include "toolchain/memory/lifetime.hpp"

#include "perimortem/core/diagnostics/log.hpp"
#include "perimortem/core/view/null_terminated.hpp"
#include "perimortem/memory/allocator/global.hpp"

#include "ttx/graph/capabilities/create.hpp"
#include "ttx/graph/policies/none.h"

#include "compute/policies/accelerated.h"
#include "cuda/execution/submission.hpp"
#include "cuda/runtime/program.hpp"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Toolchain::Memory;
using namespace Ttx::Graph;
using namespace Cuda::Platform;
using namespace Cuda::Execution;
using namespace Cuda::Runtime;

static constexpr auto source = R"CUDA(
extern "C" __global__ void invert(const unsigned char* input,
                                  unsigned char* output, unsigned long long count) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) output[i] = i % 4 == 3 ? input[i] : 255 - input[i];
}
)CUDA"_view;

// The engine compiles its own program once. Each command buffer acquires a
// separate Program reference and stream. Releasing the engine leaves those
// submissions and their images usable under their own retained contracts.
class EngineProvider {
 public:
  EngineProvider(Program& program, CUfunction function)
      : program(program), function(function) {}

  static auto graph(void* context) -> ttx_abstract {
    return {context, &graph_operations};
  }

  static auto factory_graph(void* context) -> ttx_abstract {
    return {context, &factory_operations};
  }

  static auto supports(void* context, perimortem_uuid id)
      -> ttx_negotiation_status {
    if (!id.high && id.low <= 1) {
      return TTX_NEGOTIATION_REJECTED;
    }

    const Identity::Uuid contract(id);
    return contract == Abstract::contract_id ||
                   contract == Capabilities::Create::contract_id ||
                   contract == Capabilities::Retain::contract_id ||
                   contract == Identity::Uuid(
                                   COMPUTE_ACCELERATED_ID_HIGH,
                                   COMPUTE_ACCELERATED_ID_LOW)
               ? TTX_NEGOTIATION_SATISFIED
               : TTX_NEGOTIATION_UNKNOWN;
  }

  static auto bind(void* context, perimortem_uuid id, void* output)
      -> ttx_negotiation_status {
    const auto status = supports(context, id);
    if (status != TTX_NEGOTIATION_SATISFIED) {
      return status;
    }

    const Identity::Uuid contract(id);
    if (contract == Capabilities::Create::contract_id) {
      *Data::cast<ttx_create>(output) = {context, &command_creation};
    } else if (contract == Capabilities::Retain::contract_id) {
      *Data::cast<ttx_retain>(output) = {context, &retain_operations};
    } else {
      *Data::cast<ttx_abstract>(output) = graph(context);
    }

    return TTX_NEGOTIATION_SATISFIED;
  }

  static auto route(void* context, perimortem_view_bytes route)
      -> ttx_abstract {
    return View::Bytes(route) == "commands"_view ? graph(context) : ttx_none();
  }

  static auto visit(void* context, ttx_graph_visitor visitor) -> void {
    visitor.callback(visitor.callback_context, "commands"_view, graph(context));
  }

  static auto state(void* context) -> ttx_abstract_state {
    return TTX_ABSTRACT_PRESENT;
  }

  static auto empty_route(void* context, perimortem_view_bytes route)
      -> ttx_abstract {
    return ttx_none();
  }

  static auto empty_visit(void* context, ttx_graph_visitor visitor) -> void {}

  static auto retain(void* context, ttx_retained* output)
      -> ttx_negotiation_status {
    auto& engine = *Data::cast<EngineProvider>(context);
    if (engine.references == Count(-1)) {
      return TTX_NEGOTIATION_REJECTED;
    }

    ++engine.references;
    *output = {context, &retained_operations};
    return TTX_NEGOTIATION_SATISFIED;
  }

  static auto release(void* context) -> ttx_release_status {
    auto* engine = Data::cast<EngineProvider>(context);
    if (--engine->references) {
      return TTX_RELEASE_SUCCESS;
    }

    const auto status = engine->program.release();
    engine->~EngineProvider();
    Allocator::Global::release(Data::cast<U8>(engine));
    return ttx_release_status(status);
  }

  static auto create_commands(
      void* context,
      ttx_abstract arguments,
      void* callback_context,
      void (*callback)(void*, ttx_abstract)) -> ttx_negotiation_status {
    if (Abstract(arguments).get_state() != Abstract::State::None) {
      return TTX_NEGOTIATION_REJECTED;
    }

    auto& engine = *Data::cast<EngineProvider>(context);
    return Submission::create(engine.program, engine.function)
        .visit(
            [&](Policies::Retained commands) -> ttx_negotiation_status {
              callback(callback_context, commands.get_abstract().get_abi());
              if (commands.release() != Policies::Retained::Status::Success) {
                Diagnostics::Log::error(
                    "CUDA commands cleanup failed after creation."_view);
              }

              return TTX_NEGOTIATION_SATISFIED;
            },

            [](compute_execution_status) -> ttx_negotiation_status {
              return TTX_NEGOTIATION_REJECTED;
            });
  }

  static auto create(
      void* context,
      ttx_abstract arguments,
      void* callback_context,
      void (*callback)(void*, ttx_abstract)) -> ttx_negotiation_status {
    if (Abstract(arguments).get_state() != Abstract::State::None) {
      return TTX_NEGOTIATION_REJECTED;
    }

    const cuda_compile_request request{
      {"invert.cu"_view, source}, nullptr, 0, nullptr, 0, 0};

    return Program::create(request, {})
        .visit(
            [&](Program& program) -> ttx_negotiation_status {
              CUfunction function = nullptr;
              auto status = cuCtxPushCurrent(program.get_context());
              if (status == CUDA_SUCCESS) {
                status = cuModuleGetFunction(
                    &function, program.get_module(), "invert");

                CUcontext previous = nullptr;
                const auto restored = cuCtxPopCurrent(&previous);
                if (restored != CUDA_SUCCESS) {
                  status = restored;
                }
              }

              if (status != CUDA_SUCCESS) {
                program.release();
                return TTX_NEGOTIATION_REJECTED;
              }

              static_assert(
                  alignof(EngineProvider) <=
                  Allocator::Global::allocation_alignment);
              auto memory = Allocator::Global::allocate(sizeof(EngineProvider));
              auto& engine = Lifetime::construct<EngineProvider>(
                  memory.get_data(),
                  [&] { return EngineProvider(program, function); });
              callback(callback_context, graph(&engine));
              if (release(&engine) != TTX_RELEASE_SUCCESS) {
                Diagnostics::Log::error(
                    "CUDA engine cleanup failed after its creation callback."_view);
              }

              return TTX_NEGOTIATION_SATISFIED;
            },

            [](Cuda::Contracts::ExecutionStatus) -> ttx_negotiation_status {
              return TTX_NEGOTIATION_REJECTED;
            });
  }

  static auto factory_supports(void* context, perimortem_uuid id)
      -> ttx_negotiation_status {
    if (!id.high && id.low <= 1) {
      return TTX_NEGOTIATION_REJECTED;
    }

    const Identity::Uuid contract(id);
    return contract == Abstract::contract_id ||
                   contract == Capabilities::Create::contract_id
               ? TTX_NEGOTIATION_SATISFIED
               : TTX_NEGOTIATION_UNKNOWN;
  }

  static auto factory_bind(void* context, perimortem_uuid id, void* output)
      -> ttx_negotiation_status {
    const auto status = factory_supports(context, id);
    if (status != TTX_NEGOTIATION_SATISFIED) {
      return status;
    }

    if (Identity::Uuid(id) == Capabilities::Create::contract_id) {
      *Data::cast<ttx_create>(output) = {context, &create_operations};
    } else {
      *Data::cast<ttx_abstract>(output) = factory_graph(context);
    }

    return TTX_NEGOTIATION_SATISFIED;
  }

 private:
  Count references = 1;
  Program& program;
  CUfunction function;
  static const ttx_abstract_ops graph_operations, factory_operations;
  static const ttx_retained_ops retained_operations;
  static const ttx_retain_ops retain_operations;
  static const ttx_create_ops command_creation, create_operations;
};

const ttx_abstract_ops EngineProvider::graph_operations = {
  supports, bind, route, visit, state};

const ttx_abstract_ops EngineProvider::factory_operations = {
  factory_supports, factory_bind, empty_route, empty_visit, state};

const ttx_retained_ops EngineProvider::retained_operations = {graph, release};
const ttx_retain_ops EngineProvider::retain_operations = {graph, retain};
const ttx_create_ops EngineProvider::command_creation = {
  graph, create_commands};

const ttx_create_ops EngineProvider::create_operations = {
  factory_graph, create};

auto Engine::factory() -> Abstract {
  static U8 context;
  return Abstract(EngineProvider::factory_graph(&context));
}
