// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#pragma once

#include "ttx/graph/abstract.hpp"

namespace Compute::Execution {

// Values supplies Compute's stable Coherence Type identities to direct users
// and graph discovery. `get` routes each published name to the same Abstract
// returned by its typed getter. Each Type enumerates member Callables whose
// authored arguments exclude the receiver. Coherence compiles that Type as the
// leading invocation input before the explicit arguments.
//
// Descriptions and terminal Schemas remain unchanged and accessible while the
// Compute module is loaded. Compilation derives canonical forms from those
// definitions without storing layouts in the Type graph. Calling an operation
// dispatches through the receiver record supplied by the application,
// independently of the description's graph context.
//
// Values and their acquisitions follow their individual contracts.
// Reflected Buffer reads return a View into staging owned by this module. That
// View remains readable until the next reflected Buffer read on the calling
// thread. A reflection host copies it before allowing another call. Direct
// Buffer reads continue to use the caller's Access storage.
class Values {
 public:
  static EXPORTED(COMPUTE) auto get() -> Ttx::Graph::Abstract;
  static EXPORTED(COMPUTE) auto engine() -> Ttx::Graph::Abstract;
  static EXPORTED(COMPUTE) auto buffer() -> Ttx::Graph::Abstract;
  static EXPORTED(COMPUTE) auto commands() -> Ttx::Graph::Abstract;
  static EXPORTED(COMPUTE) auto execution_status() -> Ttx::Graph::Abstract;
};

}  // namespace Compute::Execution
