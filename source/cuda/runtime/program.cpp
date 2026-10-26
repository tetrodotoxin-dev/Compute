// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cuda/runtime/program.hpp"

#include <nvrtc.h>
#include <stdio.h>

#include "toolchain/memory/lifetime.hpp"

#include "perimortem/core/diagnostics/log.hpp"
#include "perimortem/core/view/null_terminated.hpp"
#include "perimortem/memory/allocator/global.hpp"
#include "perimortem/memory/global/vector.hpp"

#include "ttx/graph/policies/none.h"
#include "ttx/graph/policies/unknown.h"

#include "cuda/platform/engine.hpp"
#include "cuda/runtime/buffer.hpp"
#include "cuda/runtime/current_context.hpp"
#include "cuda/runtime/kernel.hpp"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Toolchain::Memory;
using namespace Cuda::Runtime;
using namespace Cuda::Platform;
using namespace Ttx::Graph;

static auto valid_text(perimortem_view_bytes bytes) -> Bool {
  if ((bytes.size && !bytes.data) || bytes.size == Count(-1)) {
    return False;
  }

  for (Count i = 0; i < bytes.size; ++i) {
    if (bytes.data[i] == 0) {
      return False;
    }
  }

  return True;
}

static auto text(perimortem_view_bytes value) -> Global::Bytes {
  Global::Bytes result;
  result.forgetful_resize(value.size + 1);
  if (value.size) {
    Data::copy(result.get_access().get_data(), value.data, value.size);
  }

  result.get_access().get_data()[value.size] = 0;
  return result;
}

static auto report(cuda_diagnostics sink, View::Bytes message) -> void {
  if (sink.callback) {
    sink.callback(sink.callback_context, message);
  }
}

auto Program::create(cuda_compile_request request, cuda_diagnostics diagnostics)
    -> Storage::Result<Program&, Contracts::ExecutionStatus> {
  if (!valid_text(request.source.name) || !valid_text(request.source.text) ||
      (request.header_count && !request.headers) ||
      (request.option_count && !request.options) ||
      request.header_count > 0x7fffffff || request.option_count >= 0x7fffffff ||
      request.device < 0) {
    return Contracts::ExecutionStatus::Invalid;
  }

  for (Count i = 0; i < request.header_count; ++i) {
    if (!valid_text(request.headers[i].name) ||
        !valid_text(request.headers[i].text)) {
      return Contracts::ExecutionStatus::Invalid;
    }
  }

  for (Count i = 0; i < request.option_count; ++i) {
    if (!valid_text(request.options[i])) {
      return Contracts::ExecutionStatus::Invalid;
    }
  }

  CUresult driver = cuInit(0);
  if (driver != CUDA_SUCCESS) {
    report(diagnostics, "CUDA initialization failed."_view);
    return Contracts::ExecutionStatus::Unavailable;
  }

  static_assert(alignof(Program) <= Allocator::Global::allocation_alignment);
  auto memory = Allocator::Global::allocate(sizeof(Program));
  auto* result = &Lifetime::construct<Program>(
      memory.get_data(), [] { return Program(); });
  driver = cuDeviceGet(&result->device, request.device);
  if (driver == CUDA_SUCCESS) {
    driver = cuDevicePrimaryCtxRetain(&result->context, result->device);
  }

  if (driver != CUDA_SUCCESS) {
    result->release();
    report(diagnostics, "CUDA device or context is unavailable."_view);
    return Contracts::ExecutionStatus::Unavailable;
  }

  // NVRTC needs terminated strings and parallel include arrays. All backing
  // Bytes are populated before publishing their addresses for compilation.
  auto source = text(request.source.text);
  auto name = text(request.source.name);
  Global::Vector<Global::Bytes> names, headers, options;
  for (Count i = 0; i < request.header_count; ++i) {
    names.insert(text(request.headers[i].name));
    headers.insert(text(request.headers[i].text));
  }

  for (Count i = 0; i < request.option_count; ++i) {
    options.insert(text(request.options[i]));
  }

  Global::Vector<const char*> named, included, selected;
  for (Count i = 0; i < request.header_count; ++i) {
    named.insert(Data::cast<const char>(names.get_data()[i].get_data()));
    included.insert(Data::cast<const char>(headers.get_data()[i].get_data()));
  }

  Bool targeted = False;
  for (Count i = 0; i < request.option_count; ++i) {
    const auto option = View::Bytes(request.options[i]);
    targeted = targeted || option.slice(0, 5) == "-arch"_view ||
               option.slice(0, 18) == "--gpu-architecture"_view;
    selected.insert(Data::cast<const char>(options.get_data()[i].get_data()));
  }

  // An explicit target belongs to the source author. Otherwise NVRTC receives
  // the selected device's compute capability, independent of the build host.
  char architecture[64];
  if (!targeted) {
    S32 major = 0, minor = 0;
    driver = cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, result->device);
    if (driver == CUDA_SUCCESS) {
      driver = cuDeviceGetAttribute(
          &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, result->device);
    }

    const S32 length = snprintf(
        architecture, sizeof(architecture), "--gpu-architecture=compute_%d%d",
        major, minor);
    if (driver != CUDA_SUCCESS || length < 0 ||
        Count(length) >= sizeof(architecture)) {
      result->release();
      report(diagnostics, "CUDA target architecture is unavailable."_view);
      return Contracts::ExecutionStatus::Unavailable;
    }

    selected.insert(architecture);
  }

  nvrtcProgram compiler = nullptr;
  auto status = nvrtcCreateProgram(
      &compiler, Data::cast<const char>(source.get_data()),
      Data::cast<const char>(name.get_data()), S32(included.get_size()),
      included.get_data(), named.get_data());
  if (status != NVRTC_SUCCESS) {
    result->release();
    report(diagnostics, "NVRTC could not create the compiler program."_view);
    return Contracts::ExecutionStatus::CompilationFailed;
  }

  status = nvrtcCompileProgram(
      compiler, S32(selected.get_size()), selected.get_data());
  if (status != NVRTC_SUCCESS) {
    CppSize size = 0;
    Global::Bytes message;
    if (nvrtcGetProgramLogSize(compiler, &size) == NVRTC_SUCCESS && size) {
      message.forgetful_resize(size);
      if (nvrtcGetProgramLog(
              compiler, Data::cast<char>(message.get_access().get_data())) ==
          NVRTC_SUCCESS) {
        report(diagnostics, message.get_view().slice(0, size - 1));
      }
    }

    nvrtcDestroyProgram(&compiler);
    result->release();
    return Contracts::ExecutionStatus::CompilationFailed;
  }

  CppSize size = 0;
  Global::Bytes ptx;
  status = nvrtcGetPTXSize(compiler, &size);
  if (status == NVRTC_SUCCESS) {
    ptx.forgetful_resize(size);
    status =
        nvrtcGetPTX(compiler, Data::cast<char>(ptx.get_access().get_data()));
  }

  nvrtcDestroyProgram(&compiler);
  if (status != NVRTC_SUCCESS) {
    result->release();
    report(diagnostics, "NVRTC could not produce PTX."_view);
    return Contracts::ExecutionStatus::CompilationFailed;
  }

  // Finish the context scope before releasing a failed Program. Its destructor
  // can then install the same context for resource cleanup independently.
  {
    CurrentContext current(result->context);
    driver = current.status;
    if (driver == CUDA_SUCCESS) {
      driver = cuModuleLoadData(&result->module, ptx.get_data());
    }
  }

  if (driver != CUDA_SUCCESS) {
    result->release();
    report(diagnostics, "CUDA module load failed."_view);
    return Contracts::ExecutionStatus::DriverFailed;
  }

  return *result;
}

auto Program::compile(
    cuda_compile_request request,
    cuda_diagnostics diagnostics,
    ttx_retained* output) -> cuda_status {
  if (!output) {
    return CUDA_INVALID;
  }

  return create(request, diagnostics)
      .visit(
          [&](Program& result) -> cuda_status {
            *output = {&result, &retained_operations};
            return CUDA_SUCCESS_STATUS;
          },

          [](Contracts::ExecutionStatus status) -> cuda_status {
            return cuda_status(status);
          });
}

// CompilerProvider is available for the loaded module's duration. Compilation
// creates a new graph location and a separately released CUDA resource result.
class CompilerProvider {
 public:
  static auto graph(void* context) -> ttx_abstract {
    return {context, &graph_operations};
  }

  static auto supports(void* context, perimortem_uuid id)
      -> ttx_negotiation_status {
    const Identity::Uuid contract(id);
    if (contract == Identity::Uuid(TTX_NONE_ID_HIGH, TTX_NONE_ID_LOW) ||
        contract == Identity::Uuid(TTX_UNKNOWN_ID_HIGH, TTX_UNKNOWN_ID_LOW)) {
      return TTX_NEGOTIATION_REJECTED;
    }

    return contract == Abstract::contract_id ||
                   contract ==
                       Identity::Uuid(
                           TTX_CUDA_COMPILER_HIGH, TTX_CUDA_COMPILER_LOW)
               ? TTX_NEGOTIATION_SATISFIED
               : TTX_NEGOTIATION_UNKNOWN;
  }

  static auto bind(void* context, perimortem_uuid id, void* output)
      -> ttx_negotiation_status {
    const auto status = supports(context, id);
    if (status != TTX_NEGOTIATION_SATISFIED) {
      return status;
    }

    if (Identity::Uuid(id) == Abstract::contract_id) {
      *Data::cast<ttx_abstract>(output) = graph(context);
    } else {
      *Data::cast<cuda_compiler>(output) = {context, &operations};
    }

    return TTX_NEGOTIATION_SATISFIED;
  }

  static auto resolve_route(void* context, perimortem_view_bytes route)
      -> ttx_abstract {
    if (View::Bytes(route) == "engine"_view) {
      return Engine::factory().get_abi();
    }

    return ttx_none();
  }

  static auto visit_routes(void* context, ttx_graph_visitor visitor) -> void {
    visitor.callback(
        visitor.callback_context, "engine"_view, Engine::factory().get_abi());
  }

  static auto get_state(void* context) -> ttx_abstract_state {
    return TTX_ABSTRACT_PRESENT;
  }

  static auto compile(
      void* context,
      cuda_compile_request request,
      cuda_diagnostics diagnostics,
      ttx_retained* output) -> cuda_status {
    return Program::compile(request, diagnostics, output);
  }

  static constexpr ttx_abstract_ops graph_operations = {
    supports, bind, resolve_route, visit_routes, get_state};

  static constexpr cuda_compiler_ops operations = {graph, compile};
};

auto Program::compiler() -> Abstract {
  static CompilerProvider compiler;
  return Abstract(CompilerProvider::graph(&compiler));
}

auto Program::retain() -> void {
  if (references == Count(-1)) {
    Diagnostics::Log::fatal("CUDA Program reference count overflow."_view);
  }

  ++references;
}

auto Program::release() -> Policies::Retained::Status {
  if (--references) {
    return Policies::Retained::Status::Success;
  }

  auto outcome = Policies::Retained::Status::Success;
  if (module) {
    if (cuCtxPushCurrent(context) == CUDA_SUCCESS) {
      if (cuModuleUnload(module) != CUDA_SUCCESS) {
        outcome = Policies::Retained::Status::CleanupFailed;
      }

      CUcontext previous;
      if (cuCtxPopCurrent(&previous) != CUDA_SUCCESS) {
        outcome = Policies::Retained::Status::CleanupFailed;
      }
    } else {
      outcome = Policies::Retained::Status::CleanupFailed;
    }
  }

  if (context && cuDevicePrimaryCtxRelease(device) != CUDA_SUCCESS) {
    outcome = Policies::Retained::Status::CleanupFailed;
  }

  this->~Program();
  Allocator::Global::release(Data::cast<U8>(this));
  return outcome;
}

auto Program::prepare(
    void* context,
    perimortem_view_bytes entry,
    const ttx_representation* frame,
    const cuda_argument* arguments,
    Count count,
    cuda_diagnostics diagnostics,
    ttx_retained* output) -> cuda_status {
  if (!frame) {
    return CUDA_INVALID;
  }

  return Kernel::prepare(
      *Data::cast<Program>(context), entry, *frame, arguments, count,
      diagnostics, output);
}

auto Program::allocate(void* context, Count size, ttx_retained* output)
    -> cuda_status {
  return Buffer::allocate(*Data::cast<Program>(context), size, output);
}

const ttx_abstract_ops Program::graph_operations = {
  supports, bind, resolve_route, visit_routes, get_state};

const ttx_retained_ops Program::retained_operations = {graph, release_context};

const ttx_retain_ops Program::retain_operations = {graph, acquire};
const cuda_program_ops Program::operations = {graph, prepare, allocate};

auto Program::graph(void* context) -> ttx_abstract {
  return {context, &graph_operations};
}

auto Program::supports(void* context, perimortem_uuid id)
    -> ttx_negotiation_status {
  const Identity::Uuid contract(id);
  if (contract == Identity::Uuid(TTX_NONE_ID_HIGH, TTX_NONE_ID_LOW) ||
      contract == Identity::Uuid(TTX_UNKNOWN_ID_HIGH, TTX_UNKNOWN_ID_LOW)) {
    return TTX_NEGOTIATION_REJECTED;
  }

  return contract == Abstract::contract_id ||
                 contract ==
                     Identity::Uuid(TTX_RETAIN_ID_HIGH, TTX_RETAIN_ID_LOW) ||
                 contract == Identity::Uuid(
                                 TTX_RETAINED_ID_HIGH, TTX_RETAINED_ID_LOW) ||
                 contract ==
                     Identity::Uuid(TTX_CUDA_PROGRAM_HIGH, TTX_CUDA_PROGRAM_LOW)
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

auto Program::bind(void* context, perimortem_uuid id, void* output)
    -> ttx_negotiation_status {
  const auto status = supports(context, id);
  if (status != TTX_NEGOTIATION_SATISFIED) {
    return status;
  }

  const Identity::Uuid contract(id);
  if (contract == Abstract::contract_id) {
    *Data::cast<ttx_abstract>(output) = graph(context);
  } else if (
      contract == Identity::Uuid(TTX_RETAIN_ID_HIGH, TTX_RETAIN_ID_LOW)) {
    *Data::cast<ttx_retain>(output) = {context, &retain_operations};
  } else if (
      contract == Identity::Uuid(TTX_RETAINED_ID_HIGH, TTX_RETAINED_ID_LOW)) {
    *Data::cast<ttx_retained>(output) = {context, &retained_operations};
  } else {
    *Data::cast<cuda_program>(output) = {context, &operations};
  }

  return TTX_NEGOTIATION_SATISFIED;
}

auto Program::resolve_route(void* context, perimortem_view_bytes route)
    -> ttx_abstract {
  return ttx_none();
}

auto Program::visit_routes(void* context, ttx_graph_visitor visitor) -> void {}

auto Program::get_state(void* context) -> ttx_abstract_state {
  return TTX_ABSTRACT_PRESENT;
}

// Binding preserves the supplying release obligation. Only acquisition adds
// a reference and grants an independently released duration of use.
auto Program::acquire(void* context, ttx_retained* output)
    -> ttx_negotiation_status {
  auto& value = *Data::cast<Program>(context);
  if (value.references == Count(-1)) {
    return TTX_NEGOTIATION_REJECTED;
  }

  ++value.references;
  *output = {context, &retained_operations};
  return TTX_NEGOTIATION_SATISFIED;
}

auto Program::release_context(void* context) -> ttx_release_status {
  return ttx_release_status(Data::cast<Program>(context)->release());
}
