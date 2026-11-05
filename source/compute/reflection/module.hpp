// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "compute/execution/factory.hpp"
#include "ttx/graph/abstract.hpp"

namespace Compute::Reflection {

// Module publishes a named factory Type and its create_engine callable. The
// callable's result refers to Compute's Engine Type. The factory itself
// remains a separate Type. Direct consumers can bind the same factory UUID
// without enumerating its reflection graph.
//
// The publisher keeps this object, its name bytes and its supplied operation
// accessible and unchanged until all consumers finish. Each successful call
// transfers an Engine acquisition under engine.h. Discovery itself creates no
// engine.
class Module {
 public:
  using Api = compute_factory;
  constexpr Module(Perimortem::Core::View::Bytes name, Api api)
      : name(name), api(api) {}
  EXPORTED(COMPUTE) auto get_abstract() -> Ttx::Graph::Abstract;
  auto create_engine() const -> compute_engine_result {
    return api.create_engine(api.context);
  }

  auto get_name() const -> Perimortem::Core::View::Bytes { return name; }
  auto get_abi() const -> Api { return api; }

 private:
  Perimortem::Core::View::Bytes name;
  Api api;
};

}  // namespace Compute::Reflection
