// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "compute/execution/selection.hpp"

#include "ttx/graph/policies/none.h"

using namespace Perimortem::Core;
using namespace Ttx::Graph;
using namespace Ttx::Semantic;
using namespace Compute::Execution;

auto Selection::get_abstract() -> Abstract {
  return Abstract(this, operations);
}

auto Selection::graph(void* context) -> ttx_abstract {
  return {context, &operations};
}

auto Selection::supports(void* context, perimortem_uuid id)
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

auto Selection::bind(void* context, perimortem_uuid id, void* output)
    -> ttx_negotiation_status {
  const auto status = supports(context, id);
  if (status != TTX_NEGOTIATION_SATISFIED) {
    return status;
  }

  if (Identity::Uuid(id) == Capabilities::Create::contract_id) {
    *Data::cast<ttx_create>(output) = {context, &creation};
  } else {
    *Data::cast<ttx_abstract>(output) = graph(context);
  }

  return TTX_NEGOTIATION_SATISFIED;
}

auto Selection::route(void* context, perimortem_view_bytes route)
    -> ttx_abstract {
  return ttx_none();
}

auto Selection::visit(void* context, ttx_graph_visitor visitor) -> void {}
auto Selection::state(void* context) -> ttx_abstract_state {
  return TTX_ABSTRACT_PRESENT;
}

auto Selection::create(
    void* context,
    ttx_abstract arguments,
    void* callback_context,
    void (*callback)(void*, ttx_abstract)) -> ttx_negotiation_status {
  const auto& selection = *Data::cast<Selection>(context);
  Bool undetermined = False;
  for (Count i = 0; i < selection.factories.get_size(); ++i) {
    Bool selected = False;
    const auto factory = Abstract(selection.factories[i]);
    auto candidate_callback = [&](Abstract engine) {
      const auto policy = selection.required.visit(
          [&](Identity::Uuid required) { return engine.supports(required); },
          [] { return Negotiation::Status::Satisfied; });
      undetermined = undetermined || policy == Negotiation::Status::Unknown;
      if (policy == Negotiation::Status::Satisfied) {
        selected = True;
        callback(callback_context, engine.get_abi());
      }
    };

    factory.bind<Capabilities::Create>().visit(
        [&](Capabilities::Create candidate) {
          const auto status =
              candidate.create(Abstract(arguments), candidate_callback);
          undetermined = undetermined || status == Negotiation::Status::Unknown;
        },

        [&] { undetermined = True; });
    if (selected) {
      return TTX_NEGOTIATION_SATISFIED;
    }
  }

  return undetermined ? TTX_NEGOTIATION_UNKNOWN : TTX_NEGOTIATION_REJECTED;
}

const ttx_abstract_ops Selection::operations = {
  supports, bind, route, visit, state};

const ttx_create_ops Selection::creation = {graph, create};
