// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "validation/tests/consumer.h"

#ifdef TOOLCHAIN_WINDOWS
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <string.h>

#include "ttx/data/transfer.h"
#include "ttx/graph/capabilities/retain.h"
#include "ttx/semantic/negotiation/query.h"

#include "cuda/contracts/buffer.h"
#include "cuda/contracts/compiler.h"
#include "cuda/contracts/program.h"

S32 cuda_validate_c(void) {

#ifdef TOOLCHAIN_WINDOWS

  HMODULE library = LoadLibraryA("ttx_cuda.dll");
  const ttx_storage* published =
      library ? (const ttx_storage*)GetProcAddress(library, "ttx_entry") : NULL;

#else

  void* library = dlopen("libttx_cuda.so", RTLD_NOW | RTLD_LOCAL);
  const ttx_storage* published = library ? dlsym(library, "ttx_entry") : NULL;

#endif

  if (!library) {
    return 1;
  }

  S32 result = 2;
  U8 program_live = 0, buffer_live = 0, additional_live = 0;
  ttx_retained program_lifetime = {0}, buffer_lifetime = {0}, additional = {0};
  ttx_semantic_query query = {0};
  ttx_storage requested = {
    ttx_semantic_query_representation(), (U8*)&query, sizeof(query)};

  if (!published || ttx_transfer(published, &requested) != TTX_DATA_SUCCESS) {
    goto cleanup;
  }

  result = 3;

  ttx_abstract root = {0};
  requested =
      (ttx_storage){ttx_abstract_representation(), (U8*)&root, sizeof(root)};
  if (query.bind(
          query.context,
          (perimortem_uuid){TTX_ABSTRACT_ID_HIGH, TTX_ABSTRACT_ID_LOW},
          requested) != TTX_NEGOTIATION_SATISFIED) {
    goto cleanup;
  }

  result = 4;

  cuda_compiler compiler = {0};
  if (root.operations->bind(
          root.context,
          (perimortem_uuid){TTX_CUDA_COMPILER_HIGH, TTX_CUDA_COMPILER_LOW},
          &compiler) != TTX_NEGOTIATION_SATISFIED) {
    goto cleanup;
  }

  result = 5;

  static const U8 name[] = "consumer.cu";
  static const U8 source[] = "extern \"C\" __global__ void empty() {}";
  const cuda_compile_request request = {
    {{name, sizeof(name) - 1}, {source, sizeof(source) - 1}},
    NULL,
    0,
    NULL,
    0,
    0};

  if (compiler.operations->compile(
          compiler.context, request, (cuda_diagnostics){0},
          &program_lifetime) != CUDA_SUCCESS_STATUS) {
    goto cleanup;
  }

  program_live = 1;
  result = 6;

  ttx_abstract abstract =
      program_lifetime.operations->get_abstract(program_lifetime.context);
  cuda_program program = {0};
  if (abstract.operations->bind(
          abstract.context,
          (perimortem_uuid){TTX_CUDA_PROGRAM_HIGH, TTX_CUDA_PROGRAM_LOW},
          &program) != TTX_NEGOTIATION_SATISFIED) {
    goto cleanup;
  }

  result = 7;
  if (program.operations->allocate(program.context, 4, &buffer_lifetime) !=
      CUDA_SUCCESS_STATUS) {
    goto cleanup;
  }

  buffer_live = 1;
  result = 8;
  abstract = buffer_lifetime.operations->get_abstract(buffer_lifetime.context);

  ttx_retain retain = {0};
  if (abstract.operations->bind(
          abstract.context,
          (perimortem_uuid){TTX_RETAIN_ID_HIGH, TTX_RETAIN_ID_LOW},
          &retain) != TTX_NEGOTIATION_SATISFIED ||
      retain.operations->retain(retain.context, &additional) !=
          TTX_NEGOTIATION_SATISFIED) {
    goto cleanup;
  }

  additional_live = 1;
  buffer_lifetime.operations->release(buffer_lifetime.context);
  buffer_live = 0;
  program_lifetime.operations->release(program_lifetime.context);
  program_live = 0;

  result = 9;
  abstract = additional.operations->get_abstract(additional.context);

  cuda_buffer buffer = {0};
  if (abstract.operations->bind(
          abstract.context,
          (perimortem_uuid){TTX_CUDA_BUFFER_HIGH, TTX_CUDA_BUFFER_LOW},
          &buffer) != TTX_NEGOTIATION_SATISFIED) {
    goto cleanup;
  }

  result = 10;

  const U8 expected[] = {1, 2, 3, 4};
  U8 observed[4] = {0};
  if (buffer.operations->write(
          buffer.context, 0,
          (perimortem_view_bytes){expected, sizeof(expected)}) !=
          CUDA_SUCCESS_STATUS ||
      buffer.operations->read(buffer.context, 0, observed, sizeof(observed)) !=
          CUDA_SUCCESS_STATUS ||
      memcmp(expected, observed, sizeof(expected)) != 0) {
    goto cleanup;
  }

  result = 0;
cleanup:
  if (additional_live) {
    additional.operations->release(additional.context);
  }

  if (buffer_live) {
    buffer_lifetime.operations->release(buffer_lifetime.context);
  }

  if (program_live) {
    program_lifetime.operations->release(program_lifetime.context);
  }

#ifdef TOOLCHAIN_WINDOWS

  FreeLibrary(library);

#else

  dlclose(library);

#endif

  return result;
}
