// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#ifndef CUDA_PLATFORM_DEVICE_IMAGE_H
#define CUDA_PLATFORM_DEVICE_IMAGE_H

#include <cuda.h>

#include "compute/image/pixels.h"

#define CUDA_DEVICE_IMAGE_ID_HIGH 0x38b391199a9a4747ULL
#define CUDA_DEVICE_IMAGE_ID_LOW 0x9f446fc368716222ULL

// DeviceImage supplies RGBA8 storage for work on a compatible CUDA context
// and stream. Address establishes the producing dependency in that stream
// before returning the device address. The bytes have the dimensions and
// snapshot policy supplied by Pixels on this artifact's Abstract.
//
// The caller supplies a live nonnull stream belonging to the requested context
// and retains the image until that stream's reads complete. The caller uses
// the address only for reads in that ordered work. The provider returns Success
// after establishing both the address and dependency. A call returning Invalid
// leaves output unchanged when contexts differ or the producer has not been
// submitted. Failed reports execution or CUDA dependency failure. A rejected
// DeviceImage binding permits the caller to request Pixels for a host transfer.
typedef struct cuda_device_image {
  void* context;
  const struct cuda_device_image_ops* operations;
} cuda_device_image;

typedef struct cuda_device_image_ops {
  ttx_abstract (*get_abstract)(void* context);
  // Identifies the CUDA context containing this snapshot. The caller compares
  // this native handle to its destination context before requesting access.
  CUcontext (*get_context)(void* context);
  compute_image_status (*address)(
      void* context,
      CUcontext destination,
      CUstream stream,
      U64* output);
} cuda_device_image_ops;

#endif
