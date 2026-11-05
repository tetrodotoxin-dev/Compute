// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cuda/runtime/program.hpp"

#include <nvrtc.h>
#include <stdio.h>

#include "toolchain/memory/lifetime.hpp"

#include "perimortem/core/diagnostics/log.hpp"
#include "perimortem/core/storage/vector.hpp"
#include "perimortem/core/view/null_terminated.hpp"
#include "perimortem/memory/allocator/global.hpp"
#include "perimortem/memory/global/bytes.hpp"

#include "cuda/runtime/current_context.hpp"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Toolchain::Memory;
using namespace Cuda::Runtime;

static auto valid_text(View::Bytes bytes) -> Bool {
  if (bytes.get_size() == Count(-1) ||
      (bytes.get_size() && !bytes.get_data())) {
    return False;
  }

  for (Count i = 0; i < bytes.get_size(); ++i) {
    if (bytes[i] == 0) {
      return False;
    }
  }

  return True;
}

static auto text(View::Bytes value) -> Global::Bytes {
  Global::Bytes result;
  result.forgetful_resize(value.get_size() + 1);
  if (!value.is_empty()) {
    Data::copy(
        result.get_access().get_data(), value.get_data(), value.get_size());
  }

  result.get_access().get_data()[value.get_size()] = 0;
  return result;
}

auto Program::create(View::Bytes name, View::Bytes source, S32 device_index)
    -> Storage::Result<Program&, compute_execution_status> {
  if (!valid_text(name) || !valid_text(source) || device_index < 0) {
    return COMPUTE_EXECUTION_INVALID;
  }

  CUresult driver = cuInit(0);
  if (driver != CUDA_SUCCESS) {
    return COMPUTE_EXECUTION_FAILED;
  }

  static_assert(alignof(Program) <= Allocator::Global::allocation_alignment);
  auto memory = Allocator::Global::allocate(sizeof(Program));
  auto* result = &Lifetime::construct<Program>(
      memory.get_data(), [] { return Program(); });
  driver = cuDeviceGet(&result->device, device_index);
  if (driver == CUDA_SUCCESS) {
    driver = cuDevicePrimaryCtxRetain(&result->context, result->device);
  }

  if (driver != CUDA_SUCCESS) {
    result->release();
    return COMPUTE_EXECUTION_FAILED;
  }

  S32 major = 0, minor = 0;
  driver = cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, result->device);
  if (driver == CUDA_SUCCESS) {
    driver = cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, result->device);
  }

  Storage::Vector<char, 64> architecture;
  const S32 length = snprintf(
      architecture.get_data(), architecture.get_size(),
      "--gpu-architecture=compute_%d%d", major, minor);
  if (driver != CUDA_SUCCESS || length < 0 ||
      Count(length) >= architecture.get_size()) {
    result->release();
    return COMPUTE_EXECUTION_FAILED;
  }

  auto source_text = text(source);
  auto source_name = text(name);
  nvrtcProgram compiler = nullptr;
  auto status = nvrtcCreateProgram(
      &compiler, Data::cast<const char>(source_text.get_data()),
      Data::cast<const char>(source_name.get_data()), 0, nullptr, nullptr);
  if (status == NVRTC_SUCCESS) {
    const char* options[] = {architecture.get_data()};
    status = nvrtcCompileProgram(compiler, 1, options);
  }

  CppSize size = 0;
  Global::Bytes ptx;
  if (status == NVRTC_SUCCESS) {
    status = nvrtcGetPTXSize(compiler, &size);
  }

  if (status == NVRTC_SUCCESS) {
    ptx.forgetful_resize(size);
    status =
        nvrtcGetPTX(compiler, Data::cast<char>(ptx.get_access().get_data()));
  }

  if (compiler) {
    nvrtcDestroyProgram(&compiler);
  }

  if (status != NVRTC_SUCCESS) {
    result->release();
    return COMPUTE_EXECUTION_FAILED;
  }

  // Finish the context scope before releasing a failed Program. Its final
  // release can then install the same context for cleanup independently.
  {
    CurrentContext current(result->context);
    driver = current.status;
    if (driver == CUDA_SUCCESS) {
      driver = cuModuleLoadData(&result->module, ptx.get_data());
    }
  }

  if (driver != CUDA_SUCCESS) {
    result->release();
    return COMPUTE_EXECUTION_FAILED;
  }

  return *result;
}

auto Program::retain() -> void {
  if (references == Count(-1)) {
    Diagnostics::Log::fatal("CUDA Program reference count overflow."_view);
  }

  ++references;
}

auto Program::release() -> compute_execution_status {
  if (--references) {
    return COMPUTE_EXECUTION_SUCCESS;
  }

  auto outcome = COMPUTE_EXECUTION_SUCCESS;
  if (module) {
    if (cuCtxPushCurrent(context) == CUDA_SUCCESS) {
      if (cuModuleUnload(module) != CUDA_SUCCESS) {
        outcome = COMPUTE_EXECUTION_FAILED;
      }

      CUcontext previous;
      if (cuCtxPopCurrent(&previous) != CUDA_SUCCESS) {
        outcome = COMPUTE_EXECUTION_FAILED;
      }
    } else {
      outcome = COMPUTE_EXECUTION_FAILED;
    }
  }

  if (context && cuDevicePrimaryCtxRelease(device) != CUDA_SUCCESS) {
    outcome = COMPUTE_EXECUTION_FAILED;
  }

  this->~Program();
  Allocator::Global::release(Data::cast<U8>(this));
  return outcome;
}
