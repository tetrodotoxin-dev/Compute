// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "validation/tests/consumer.h"

#include "cuda/module.h"

S32 main(void) {
  return cuda_validate_entry(&ttx_entry);
}
