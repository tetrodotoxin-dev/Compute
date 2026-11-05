// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "perimortem/core/access/bytes.hpp"
#include "perimortem/core/view/bytes.hpp"

#include "compute/execution/buffer.h"
#include "ttx/data/form/compiled.hpp"
#include "ttx/graph/abstract.hpp"

namespace Compute::Execution {

// Buffer forwards the supplied byte storage contract. The caller performs
// explicit release under buffer.h. Copying this interface keeps the same
// acquisition and its existing access duration.
class Buffer {
 public:
  explicit constexpr Buffer(compute_buffer api) : api(api) {}
  constexpr auto get_abi() const -> compute_buffer { return api; }
  auto get_abstract() const -> Ttx::Graph::Abstract {
    return Ttx::Graph::Abstract(api.operations->get_abstract(api.context));
  }

  auto size() const -> U64 { return api.operations->size(api.context); }
  auto read(U64 offset, Perimortem::Core::Access::Bytes output) const
      -> compute_execution_status {
    return api.operations->read(
        api.context, offset, output.get_data(), output.get_size());
  }

  auto write(U64 offset, Perimortem::Core::View::Bytes input) const
      -> compute_execution_status {
    return api.operations->write(api.context, offset, input);
  }

  auto release() const -> compute_execution_status {
    return api.operations->release(api.context);
  }

 private:
  compute_buffer api;
};

}  // namespace Compute::Execution

TTX_DATA_RECORD(
    compute_buffer_ops,
    TTX_DATA_MEMBER(compute_buffer_ops, get_abstract),
    TTX_DATA_MEMBER(compute_buffer_ops, size),
    TTX_DATA_MEMBER(compute_buffer_ops, read),
    TTX_DATA_MEMBER(compute_buffer_ops, write),
    TTX_DATA_MEMBER(compute_buffer_ops, release));
TTX_DATA_RECORD(
    compute_buffer,
    TTX_DATA_MEMBER(compute_buffer, context),
    TTX_DATA_MEMBER(compute_buffer, operations));
TTX_DATA_RECORD(
    compute_allocation,
    TTX_DATA_MEMBER(compute_allocation, status),
    TTX_DATA_MEMBER(compute_allocation, buffer));
