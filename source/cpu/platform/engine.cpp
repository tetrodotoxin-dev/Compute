// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "cpu/platform/engine.hpp"

#include "perimortem/core/diagnostics/log.hpp"
#include "perimortem/core/view/null_terminated.hpp"

#include "ttx/graph/capabilities/create.hpp"
#include "ttx/graph/policies/none.h"

#include "compute/policies/accelerated.h"
#include "cpu/execution/submission.hpp"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Ttx::Graph;
using namespace Cpu::Platform;
using namespace Cpu::Execution;

static U8 engine_context, factory_context;
class CpuProvider {
 public:
  static auto engine(void* context) -> ttx_abstract {
    return {context, &engine_operations};
  }

  static auto factory_graph(void* context) -> ttx_abstract {
    return {context, &factory_operations};
  }

  static auto release(void* context) -> ttx_release_status {
    return TTX_RELEASE_SUCCESS;
  }

  static auto retain(void* context, ttx_retained* output)
      -> ttx_negotiation_status {
    *output = {context, &retained_operations};
    return TTX_NEGOTIATION_SATISFIED;
  }

  static auto create_commands(
      void* context,
      ttx_abstract arguments,
      void* callback_context,
      void (*callback)(void*, ttx_abstract)) -> ttx_negotiation_status {
    if (Abstract(arguments).get_state() != Abstract::State::None) {
      return TTX_NEGOTIATION_REJECTED;
    }

    const auto commands = Submission::create();
    callback(callback_context, commands.get_abstract().get_abi());
    if (commands.release() != Policies::Retained::Status::Success) {
      Diagnostics::Log::error(
          "CPU commands cleanup failed after creation."_view);
    }

    return TTX_NEGOTIATION_SATISFIED;
  }

  static auto supports(void* context, perimortem_uuid id)
      -> ttx_negotiation_status {
    const Identity::Uuid contract(id);
    if ((!id.high && id.low <= 1) ||
        contract ==
            Identity::Uuid(
                COMPUTE_ACCELERATED_ID_HIGH, COMPUTE_ACCELERATED_ID_LOW)) {
      return TTX_NEGOTIATION_REJECTED;
    }

    return contract == Abstract::contract_id ||
                   contract == Capabilities::Create::contract_id ||
                   contract == Capabilities::Retain::contract_id
               ? TTX_NEGOTIATION_SATISFIED
               : TTX_NEGOTIATION_UNKNOWN;
  }

  static auto bind(void* context, perimortem_uuid id, void* output)
      -> ttx_negotiation_status {
    const auto status = supports(context, id);
    if (status != TTX_NEGOTIATION_SATISFIED) {
      return status;
    }

    const Identity::Uuid contract(id);
    if (contract == Capabilities::Create::contract_id) {
      *Data::cast<ttx_create>(output) = {context, &command_creation};
    } else if (contract == Capabilities::Retain::contract_id) {
      *Data::cast<ttx_retain>(output) = {context, &retain_operations};
    } else {
      *Data::cast<ttx_abstract>(output) = engine(context);
    }

    return TTX_NEGOTIATION_SATISFIED;
  }

  static auto route(void* context, perimortem_view_bytes route)
      -> ttx_abstract {
    return View::Bytes(route) == "commands"_view ? engine(context) : ttx_none();
  }

  static auto visit(void* context, ttx_graph_visitor visitor) -> void {
    visitor.callback(
        visitor.callback_context, "commands"_view, engine(context));
  }

  static auto state(void* context) -> ttx_abstract_state {
    return TTX_ABSTRACT_PRESENT;
  }

  static auto empty_route(void* context, perimortem_view_bytes route)
      -> ttx_abstract {
    return ttx_none();
  }

  static auto empty_visit(void* context, ttx_graph_visitor visitor) -> void {}

  static auto create(
      void* context,
      ttx_abstract arguments,
      void* callback_context,
      void (*callback)(void*, ttx_abstract)) -> ttx_negotiation_status {
    if (Abstract(arguments).get_state() != Abstract::State::None) {
      return TTX_NEGOTIATION_REJECTED;
    }

    callback(callback_context, engine(&engine_context));
    return TTX_NEGOTIATION_SATISFIED;
  }

  static auto factory_supports(void* context, perimortem_uuid id)
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

  static auto factory_bind(void* context, perimortem_uuid id, void* output)
      -> ttx_negotiation_status {
    const auto status = factory_supports(context, id);
    if (status != TTX_NEGOTIATION_SATISFIED) {
      return status;
    }

    if (Identity::Uuid(id) == Capabilities::Create::contract_id) {
      *Data::cast<ttx_create>(output) = {context, &create_operations};
    } else {
      *Data::cast<ttx_abstract>(output) = factory_graph(context);
    }

    return TTX_NEGOTIATION_SATISFIED;
  }

 private:
  static const ttx_retained_ops retained_operations;
  static const ttx_retain_ops retain_operations;
  static const ttx_create_ops command_creation;
  static const ttx_create_ops create_operations;
  static const ttx_abstract_ops engine_operations;
  static const ttx_abstract_ops factory_operations;
};

const ttx_retained_ops CpuProvider::retained_operations = {engine, release};
const ttx_retain_ops CpuProvider::retain_operations = {engine, retain};
const ttx_create_ops CpuProvider::command_creation = {engine, create_commands};
const ttx_create_ops CpuProvider::create_operations = {factory_graph, create};
const ttx_abstract_ops CpuProvider::engine_operations = {
  supports, bind, route, visit, state};

const ttx_abstract_ops CpuProvider::factory_operations = {
  factory_supports, factory_bind, empty_route, empty_visit, state};

auto Engine::factory() -> Abstract {
  return Abstract(CpuProvider::factory_graph(&factory_context));
}
