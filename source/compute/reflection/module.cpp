// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "compute/reflection/module.hpp"

#include "perimortem/core/view/null_terminated.hpp"

#include "coherence/type_system/callable.hpp"
#include "coherence/type_system/section.hpp"
#include "coherence/type_system/sections.hpp"
#include "coherence/type_system/selections/intersection.hpp"
#include "coherence/type_system/type.hpp"
#include "compute/execution/engine.hpp"
#include "compute/execution/values.hpp"
#include "ttx/graph/capabilities/describe.hpp"
#include "ttx/graph/capabilities/selection.hpp"
#include "ttx/graph/policies/constant.hpp"

using namespace Perimortem::Core;
using namespace Ttx::Data;
using namespace Ttx::Graph;
using namespace Compute::Reflection;
using namespace Compute::Execution;

static auto factory_graph(void* context) -> ttx_abstract;
static auto call_graph(void* context) -> ttx_abstract;

static auto name(void* context, perimortem_view_bytes* output) -> U8 {
  *output = Data::cast<Module>(context)->get_name();
  return 1;
}

static auto call_name(void* context, perimortem_view_bytes* output) -> U8 {
  *output = "create_engine"_view;
  return 1;
}

static auto factory_description(void* context, perimortem_view_bytes* output)
    -> U8 {
  *output =
      "A factory Type that creates an acquired Compute execution engine."_view;
  return 1;
}

static auto call_description(void* context, perimortem_view_bytes* output)
    -> U8 {
  *output = "Creates one acquired Compute execution engine."_view;
  return 1;
}

static auto visit_empty(
    void* context,
    void* callback_context,
    void (*callback)(void*, ttx_abstract)) -> void {}
static auto state(void* context) -> ttx_abstract_state {
  return TTX_ABSTRACT_PRESENT;
}

static auto result(void* context) -> ttx_abstract {
  return Values::engine().get_abi();
}

static auto invoke(void* context, const void* input, void* output)
    -> ttx_data_status {
  const auto created = Data::cast<Module>(context)->create_engine();
  if (created.status != COMPUTE_EXECUTION_SUCCESS) {
    return TTX_DATA_DENIED;
  }

  *Data::cast<compute_engine>(output) = created.engine;
  return TTX_DATA_SUCCESS;
}

static auto operation(void* context) -> coherence_operation {
  return {context, invoke};
}

static auto visit_callables(
    void* context,
    void* callback_context,
    void (*callback)(void*, ttx_abstract)) -> void {
  callback(callback_context, call_graph(context));
}

static auto selection_matches(Abstract request) -> Storage::Option<Bool> {
  const Identity::Uuid sections[] = {
    Coherence::TypeSystem::Sections::callables().get_identity(),
    Coherence::TypeSystem::Sections::statics().get_identity(),
  };

  auto section = request.bind<Coherence::TypeSystem::Section::Contract>();
  if (section) {
    return section.visit(
        [&](Coherence::TypeSystem::Section::Contract value) {
          const auto identity = value.get_identity();
          return Storage::Option<Bool>(
              identity == sections[0] || identity == sections[1]);
        },

        []() -> Storage::Option<Bool> { return {}; });
  }

  Storage::Option<Bool> result;
  request.bind<Coherence::TypeSystem::Selections::Intersection::Contract>()
      .visit(
          [&](Coherence::TypeSystem::Selections::Intersection::Contract all) {
            Bool matched = True;
            Bool valid = True;
            auto visit = [&](Abstract nested) {
              auto nested_section =
                  nested.bind<Coherence::TypeSystem::Section::Contract>();
              nested_section.visit(
                  [&](Coherence::TypeSystem::Section::Contract value) {
                    const auto identity = value.get_identity();
                    matched = matched && (identity == sections[0] ||
                                          identity == sections[1]);
                  },

                  [&] { valid = False; });
            };

            all.visit_sections(visit);
            if (valid) {
              result = matched;
            }
          },

          [] {});
  return result;
}

static auto select(
    void* context,
    ttx_abstract request,
    ttx_graph_visitor visitor) -> ttx_negotiation_status {
  Storage::Option<Bool> accepted = selection_matches(Abstract(request));
  if (!accepted) {
    return TTX_NEGOTIATION_UNKNOWN;
  }

  Bool matched = False;
  accepted.visit([&](Bool value) { matched = value; }, [] {});
  if (matched) {
    visitor.callback(
        visitor.callback_context, "create_engine"_view, call_graph(context));
  }

  return TTX_NEGOTIATION_SATISFIED;
}

static auto selection_supports(void* context, perimortem_uuid policy)
    -> ttx_negotiation_status {
  (void)context;
  (void)policy;
  return TTX_NEGOTIATION_UNKNOWN;
}

static auto supports(void* context, perimortem_uuid id)
    -> ttx_negotiation_status {
  if (!id.high && id.low <= 1) {
    return TTX_NEGOTIATION_REJECTED;
  }

  const Identity::Uuid contract(id);
  return contract == Abstract::contract_id ||
                 contract ==
                     Coherence::TypeSystem::Type::Contract::contract_id ||
                 contract == Capabilities::Describe::contract_id ||
                 contract == Capabilities::Selection::contract_id ||
                 contract == Policies::Constant::contract_id ||
                 contract == Factory::contract_id
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

static auto bind(void* context, perimortem_uuid id, void* output)
    -> ttx_negotiation_status {
  const auto status = supports(context, id);
  if (status != TTX_NEGOTIATION_SATISFIED) {
    return status;
  }

  static const ttx_describe_ops descriptions{name, factory_description};

  static const coherence_type_ops types{
    factory_graph, visit_empty, visit_callables};

  static const ttx_selection_ops selections{selection_supports, select};
  const Identity::Uuid contract(id);
  if (contract == Capabilities::Describe::contract_id) {
    *Data::cast<ttx_describe>(output) = {context, &descriptions};
  } else if (contract == Coherence::TypeSystem::Type::Contract::contract_id) {
    *Data::cast<coherence_type>(output) = {context, &types};
  } else if (contract == Capabilities::Selection::contract_id) {
    *Data::cast<ttx_selection>(output) = {context, &selections};
  } else if (contract == Policies::Constant::contract_id) {
    *Data::cast<ttx_abstract>(output) = factory_graph(context);
  } else if (contract == Factory::contract_id) {
    *Data::cast<compute_factory>(output) =
        Data::cast<Module>(context)->get_abi();
  } else {
    *Data::cast<ttx_abstract>(output) = factory_graph(context);
  }

  return status;
}

static auto call_supports(void* context, perimortem_uuid id)
    -> ttx_negotiation_status {
  if (!id.high && id.low <= 1) {
    return TTX_NEGOTIATION_REJECTED;
  }

  const Identity::Uuid contract(id);
  return contract == Abstract::contract_id ||
                 contract == Capabilities::Describe::contract_id ||
                 contract ==
                     Coherence::TypeSystem::Callable::Contract::contract_id
             ? TTX_NEGOTIATION_SATISFIED
             : TTX_NEGOTIATION_UNKNOWN;
}

static auto call_bind(void* context, perimortem_uuid id, void* output)
    -> ttx_negotiation_status {
  static const ttx_describe_ops descriptions{call_name, call_description};

  static const coherence_callable_ops calls{
    call_graph, visit_empty, result, operation};

  const Identity::Uuid contract(id);
  if (contract == Capabilities::Describe::contract_id) {
    *Data::cast<ttx_describe>(output) = {context, &descriptions};
  } else if (
      contract == Coherence::TypeSystem::Callable::Contract::contract_id) {
    *Data::cast<coherence_callable>(output) = {context, &calls};
  } else if (contract == Abstract::contract_id) {
    *Data::cast<ttx_abstract>(output) = call_graph(context);
  } else if (!id.high && id.low <= 1) {
    return TTX_NEGOTIATION_REJECTED;
  } else {
    return TTX_NEGOTIATION_UNKNOWN;
  }

  return TTX_NEGOTIATION_SATISFIED;
}

static auto route(void* context, perimortem_view_bytes path) -> ttx_abstract {
  return View::Bytes(path) == "create_engine"_view ? call_graph(context)
                                                   : ttx_none();
}

static auto visit(void* context, ttx_graph_visitor visitor) -> void {
  visitor.callback(
      visitor.callback_context, "create_engine"_view, call_graph(context));
}

static auto empty_route(void* context, perimortem_view_bytes path)
    -> ttx_abstract {
  return ttx_none();
}

static auto empty_visit(void* context, ttx_graph_visitor visitor) -> void {}

static auto factory_graph(void* context) -> ttx_abstract {
  static const ttx_abstract_ops operations{supports, bind, route, visit, state};
  return {context, &operations};
}

static auto call_graph(void* context) -> ttx_abstract {
  static const ttx_abstract_ops operations{
    call_supports, call_bind, empty_route, empty_visit, state};

  return {context, &operations};
}

auto Module::get_abstract() -> Abstract {
  return Abstract(factory_graph(this));
}
