// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "perimortem/core/view/vector.hpp"

#include "ttx/graph/capabilities/create.hpp"

namespace Compute::Execution {

// Selection publishes a Create factory over ordered candidate factories. Each
// candidate receives the same argument graph. The first created engine that
// satisfies the optional required contract reaches the consumer's callback.
// An empty requirement accepts the supplied Abstract directly. Each further
// binding remains an independent request to that result's provider.
//
// Selection performs factory selection before consumer execution. Refusing one
// candidate permits the next to be tried. The requirement applies to every
// candidate, so fallback preserves requested policies such as Accelerated.
// The consumer retains a selected engine inside its callback for later use.
//
// The caller keeps Selection, its factory array and each factory's supplying
// state and code available through creation. The selected engine's retention
// contract independently governs its subsequent access duration.
class Selection {
 public:
  Selection(
      Perimortem::Core::View::Vector<ttx_abstract> factories,
      Perimortem::Core::Storage::Option<Perimortem::Core::Identity::Uuid>
          required = {})
      : factories(factories), required(required) {}
  EXPORTED(COMPUTE) auto get_abstract() -> Ttx::Graph::Abstract;

 private:
  static auto graph(void* context) -> ttx_abstract;
  static auto supports(void* context, perimortem_uuid id)
      -> ttx_negotiation_status;
  static auto bind(void* context, perimortem_uuid id, void* output)
      -> ttx_negotiation_status;
  static auto route(void* context, perimortem_view_bytes route) -> ttx_abstract;
  static auto visit(void* context, ttx_graph_visitor visitor) -> void;
  static auto state(void* context) -> ttx_abstract_state;
  static auto create(
      void* context,
      ttx_abstract arguments,
      void* callback_context,
      void (*callback)(void*, ttx_abstract)) -> ttx_negotiation_status;
  static const ttx_abstract_ops operations;
  static const ttx_create_ops creation;
  Perimortem::Core::View::Vector<ttx_abstract> factories;
  Perimortem::Core::Storage::Option<Perimortem::Core::Identity::Uuid> required;
};

}  // namespace Compute::Execution
