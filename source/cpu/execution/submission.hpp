// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "toolchain/memory/lifetime.hpp"

#include "perimortem/memory/global/bytes.hpp"
#include "perimortem/memory/global/vector.hpp"

#include "ttx/graph/capabilities/retain.hpp"

#include "compute/execution/commands.h"
#include "compute/execution/dependency.h"
#include "compute/image/invert.h"

namespace Cpu::Execution {

// Submission owns one host submission and the storage for its result artifacts.
// Submission executes the authored sequence on the calling thread. Result
// acquisitions retain the complete recording, so their API contexts remain
// stable even when later commands grow the command list.
class Submission {
 public:
  static EXPORTED(CPU) auto create() -> Ttx::Graph::Policies::Retained;

 private:
  friend Toolchain::Memory::Lifetime;
  // Each image has a stable address for its published operations. A local
  // input names an earlier position, while a foreign input carries a retained
  // API or bytes captured before its original access ended.
  struct Image {
    Image(Submission& commands, Count position, compute_image_dimensions size)
        : commands(commands), position(position), size(size) {}
    Submission& commands;
    Count position;
    compute_image_dimensions size;
    Perimortem::Core::Storage::Option<Count> previous;
    Perimortem::Core::Storage::Option<ttx_retained> input;
    Perimortem::Core::Storage::Option<compute_pixels> pixels;
    Perimortem::Memory::Global::Bytes data;
  };

  Submission() = default;
  enum class Phase { Recording, Complete, Failed };
  static auto graph(void* context) -> ttx_abstract;
  static auto supports(void* context, perimortem_uuid id)
      -> ttx_negotiation_status;
  static auto bind(void* context, perimortem_uuid id, void* output)
      -> ttx_negotiation_status;
  static auto route(void* context, perimortem_view_bytes route) -> ttx_abstract;
  static auto visit(void* context, ttx_graph_visitor visitor) -> void;
  static auto state(void* context) -> ttx_abstract_state;
  static auto retain(void* context, ttx_retained* output)
      -> ttx_negotiation_status;
  static auto release(void* context) -> ttx_release_status;
  static auto submit(void* context) -> compute_execution_status;
  static auto wait(void* context) -> compute_execution_status;
  static auto record(void* context, ttx_abstract input, ttx_retained* output)
      -> compute_image_status;
  static auto image_graph(void* context) -> ttx_abstract;
  static auto image_supports(void* context, perimortem_uuid id)
      -> ttx_negotiation_status;
  static auto image_bind(void* context, perimortem_uuid id, void* output)
      -> ttx_negotiation_status;
  static auto image_retain(void* context, ttx_retained* output)
      -> ttx_negotiation_status;
  static auto image_release(void* context) -> ttx_release_status;
  static auto dimensions(void* context) -> compute_image_dimensions;
  static auto read(void* context, U8* output, Count capacity)
      -> compute_image_status;
  static auto submitted(void* context) -> U8;
  static auto locate(void* context, compute_commands commands, Count* position)
      -> ttx_negotiation_status;
  static const ttx_abstract_ops graph_operations, image_operations;
  static const ttx_retain_ops retain_operations, image_retention;
  static const ttx_retained_ops retained_operations, image_retained;
  static const compute_commands_ops command_operations;
  static const compute_invert_ops invert_operations;
  static const compute_pixels_ops pixel_operations;
  static const compute_dependency_ops dependency_operations;
  Count references = 1;
  Phase phase = Phase::Recording;
  Perimortem::Memory::Global::Vector<Image*> images;
};

}  // namespace Cpu::Execution
