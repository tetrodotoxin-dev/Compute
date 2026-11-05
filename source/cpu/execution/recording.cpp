// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cpu/execution/recording.hpp"

#include "toolchain/memory/lifetime.hpp"

#include "perimortem/core/identity/uuid.hpp"
#include "perimortem/memory/allocator/global.hpp"
#include "perimortem/memory/global/bytes.hpp"

#include "compute/image/composite_command.hpp"
#include "compute/image/convolve_command.hpp"
#include "compute/image/invert_command.hpp"
#include "compute/image/posterize_command.hpp"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Toolchain::Memory;
using namespace Compute::Image;
using namespace Cpu::Execution;

static auto read(compute_buffer buffer, U64 size, Global::Bytes& output)
    -> compute_execution_status {
  if (!buffer.context || !buffer.operations ||
      buffer.operations->size(buffer.context) < size || size > Count(-1)) {
    return COMPUTE_EXECUTION_INVALID;
  }

  output.forgetful_resize(size);
  return buffer.operations->read(
      buffer.context, 0, output.get_data(), output.get_size());
}

static auto write(compute_buffer buffer, View::Bytes input)
    -> compute_execution_status {
  if (!buffer.context || !buffer.operations ||
      buffer.operations->size(buffer.context) < input.get_size()) {
    return COMPUTE_EXECUTION_INVALID;
  }

  return buffer.operations->write(buffer.context, 0, input);
}

auto Recording::create() -> compute_commands_result {
  auto storage = Allocator::Global::allocate(sizeof(Recording));
  auto& recording = Lifetime::construct<Recording>(storage.get_data());
  static const compute_commands_ops operations{
    submit, wait, release, supports, record};

  return {COMPUTE_EXECUTION_SUCCESS, {&recording, &operations}};
}

auto Recording::supports(void* context, perimortem_uuid command)
    -> ttx_negotiation_status {
  const Identity::Uuid requested(command);
  return requested == InvertCommand::contract_id ||
                 requested == PosterizeCommand::contract_id ||
                 requested == ConvolveCommand::contract_id ||
                 requested == CompositeCommand::contract_id
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

auto Recording::record(
    void* context,
    perimortem_uuid command,
    const void* arguments) -> compute_execution_status {
  auto& recording = *Data::cast<Recording>(context);
  if (recording.phase != Phase::Recording || !arguments ||
      supports(context, command) != TTX_NEGOTIATION_SATISFIED) {
    return COMPUTE_EXECUTION_INVALID;
  }

  Operation operation{};
  const Identity::Uuid requested(command);
  if (requested == InvertCommand::contract_id) {
    operation.kind = Kind::Invert;
    operation.invert = *Data::cast<const compute_invert_command>(arguments);
  } else if (requested == PosterizeCommand::contract_id) {
    operation.kind = Kind::Posterize;
    operation.posterize =
        *Data::cast<const compute_posterize_command>(arguments);
  } else if (requested == ConvolveCommand::contract_id) {
    operation.kind = Kind::Convolve;
    operation.convolve = *Data::cast<const compute_convolve_command>(arguments);
  } else {
    operation.kind = Kind::Composite;
    operation.composite =
        *Data::cast<const compute_composite_command>(arguments);
  }

  recording.operations.insert(operation);
  return COMPUTE_EXECUTION_SUCCESS;
}

auto Recording::execute(const Operation& operation)
    -> compute_execution_status {
  if (operation.kind == Kind::Invert) {
    const auto& command = operation.invert;
    if (!command.count || command.count % 4) {
      return COMPUTE_EXECUTION_INVALID;
    }

    Global::Bytes pixels;
    auto status = read(command.input, command.count, pixels);
    if (status != COMPUTE_EXECUTION_SUCCESS) {
      return status;
    }

    for (Count i = 0; i < pixels.get_size(); i += 4) {
      pixels.get_data()[i] ^= 255;
      pixels.get_data()[i + 1] ^= 255;
      pixels.get_data()[i + 2] ^= 255;
    }

    return write(command.output, pixels.get_view());
  }

  if (operation.kind == Kind::Posterize) {
    const auto& command = operation.posterize;
    if (!command.count || command.count % 4 || command.levels < 2 ||
        command.levels > 256) {
      return COMPUTE_EXECUTION_INVALID;
    }

    Global::Bytes pixels;
    auto status = read(command.input, command.count, pixels);
    if (status != COMPUTE_EXECUTION_SUCCESS) {
      return status;
    }

    for (Count i = 0; i < pixels.get_size(); i += 4) {
      for (Count channel = 0; channel < 3; ++channel) {
        const U32 value = pixels.get_data()[i + channel];
        pixels.get_data()[i + channel] =
            ((value * (command.levels - 1) + 127) / 255) * 255 /
            (command.levels - 1);
      }
    }

    return write(command.output, pixels.get_view());
  }

  if (operation.kind == Kind::Composite) {
    const auto& command = operation.composite;
    if (!command.count || command.count > Count(-1) / 4) {
      return COMPUTE_EXECUTION_INVALID;
    }

    const Count size = command.count * 4;
    Global::Bytes back, front;
    auto status = read(command.back, size, back);
    if (status != COMPUTE_EXECUTION_SUCCESS) {
      return status;
    }

    status = read(command.front, size, front);
    if (status != COMPUTE_EXECUTION_SUCCESS) {
      return status;
    }

    for (Count i = 0; i < size; i += 4) {
      const U32 foreground = front.get_data()[i + 3];
      const U32 background = back.get_data()[i + 3];
      const U32 alpha = foreground * 255 + background * (255 - foreground);
      for (Count channel = 0; channel < 3; ++channel) {
        const U32 color =
            front.get_data()[i + channel] * foreground * 255 +
            back.get_data()[i + channel] * background * (255 - foreground);
        back.get_data()[i + channel] = alpha ? (color + alpha / 2) / alpha : 0;
      }

      back.get_data()[i + 3] = (alpha + 127) / 255;
    }

    return write(command.output, back.get_view());
  }

  const auto& command = operation.convolve;
  const U64 pixels = U64(command.width) * command.height;
  const U64 kernel = U64(command.kernel_width) * command.kernel_height;
  if (!pixels || pixels > Count(-1) / 4 || !kernel ||
      kernel > Count(-1) / sizeof(R32) || !(command.kernel_width & 1) ||
      !(command.kernel_height & 1)) {
    return COMPUTE_EXECUTION_INVALID;
  }

  Global::Bytes input, weights, output;
  auto status = read(command.input, pixels * 4, input);
  if (status != COMPUTE_EXECUTION_SUCCESS) {
    return status;
  }

  status = read(command.weights, kernel * sizeof(R32), weights);
  if (status != COMPUTE_EXECUTION_SUCCESS) {
    return status;
  }

  output.resize(pixels * 4);

  const auto* coefficients = Data::cast<const R32>(weights.get_data());
  for (U32 y = 0; y < command.height; ++y) {
    for (U32 x = 0; x < command.width; ++x) {
      const Count destination = (Count(y) * command.width + x) * 4;
      for (Count channel = 0; channel < 3; ++channel) {
        R32 sum = 0;
        for (U32 ky = 0; ky < command.kernel_height; ++ky) {
          const S64 source_y = S64(y) + command.kernel_height / 2 - S64(ky);
          if (source_y < 0 || source_y >= command.height) {
            continue;
          }

          for (U32 kx = 0; kx < command.kernel_width; ++kx) {
            const S64 source_x = S64(x) + command.kernel_width / 2 - S64(kx);
            if (source_x < 0 || source_x >= command.width) {
              continue;
            }

            const Count source =
                (Count(source_y) * command.width + Count(source_x)) * 4;
            sum += input.get_data()[source + channel] *
                   coefficients[Count(ky) * command.kernel_width + kx];
          }
        }

        const S64 rounded = S64(__builtin_floorf(sum + 0.5F));
        output.get_data()[destination + channel] = rounded < 0 ? 0
                                                   : rounded > 255
                                                       ? 255
                                                       : U8(rounded);
      }

      output.get_data()[destination + 3] = input.get_data()[destination + 3];
    }
  }

  return write(command.output, output.get_view());
}

auto Recording::submit(void* context) -> compute_execution_status {
  auto& recording = *Data::cast<Recording>(context);
  if (recording.phase != Phase::Recording) {
    return COMPUTE_EXECUTION_INVALID;
  }

  for (Count i = 0; i < recording.operations.get_size(); ++i) {
    const auto status = recording.execute(recording.operations.get_data()[i]);
    if (status != COMPUTE_EXECUTION_SUCCESS) {
      recording.phase = Phase::Failed;
      return status;
    }
  }

  recording.phase = Phase::Complete;
  return COMPUTE_EXECUTION_SUCCESS;
}

auto Recording::wait(void* context) -> compute_execution_status {
  const auto phase = Data::cast<Recording>(context)->phase;
  if (phase == Phase::Recording) {
    return COMPUTE_EXECUTION_INVALID;
  }

  return phase == Phase::Complete ? COMPUTE_EXECUTION_SUCCESS
                                  : COMPUTE_EXECUTION_FAILED;
}

auto Recording::release(void* context) -> compute_execution_status {
  auto* recording = Data::cast<Recording>(context);
  recording->~Recording();
  Allocator::Global::release(Data::cast<U8>(recording));
  return COMPUTE_EXECUTION_SUCCESS;
}
