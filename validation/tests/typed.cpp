// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "toolchain/validation/test.hpp"

#include "perimortem/core/storage/vector.hpp"
#include "perimortem/core/view/null_terminated.hpp"
#include "perimortem/memory/global/bytes.hpp"

#include "coherence/compilation/compiler.hpp"
#include "coherence/type_system/callable.hpp"
#include "coherence/type_system/sections.hpp"
#include "coherence/type_system/selections/intersection.hpp"
#include "coherence/type_system/type.hpp"
#include "compute/execution/buffer.hpp"
#include "compute/execution/engine.hpp"
#include "compute/execution/factory.hpp"
#include "compute/execution/values.hpp"
#include "compute/image/invert_command.hpp"
#include "compute/reflection/module.hpp"
#include "cpu/platform/memory.hpp"
#include "ttx/graph/capabilities/describe.hpp"
#include "ttx/graph/capabilities/selection.hpp"
#include "ttx/graph/policies/constant.hpp"
#include "ttx/graph/policies/none.hpp"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Toolchain::Validation;
using namespace Ttx::Graph;
using namespace Ttx::Data;
using namespace Ttx::Semantic;
using namespace Compute::Reflection;
using namespace Compute::Execution;
using namespace Compute::Image;
using namespace Coherence::TypeSystem;
using namespace Coherence::Compilation;

static Harness TypedTests{.name = "Compute::Typed"};

static auto reflected_form(Abstract abstract)
    -> Storage::Option<Form::Representation> {
  return Compiler::compile(abstract).visit(
      [](Form::Representation& form) -> Storage::Option<Form::Representation> {
        return Data::take(form);
      },

      [](Status) -> Storage::Option<Form::Representation> { return {}; });
}

static auto method(Abstract type, View::Bytes requested) -> Abstract {
  auto result = Ttx::Graph::Policies::None::get_none().get_abstract();
  const Storage::Vector<Abstract, 2> sections = {{
    Sections::callables().get_abstract(),
    Sections::members().get_abstract(),
  }};

  auto query = Selections::Intersection::create(sections.get_view());
  type.bind<Ttx::Graph::Capabilities::Selection>().visit(
      [&](Ttx::Graph::Capabilities::Selection selection) {
        auto visit = [&](View::Bytes route, Abstract callable) {
          if (route == requested) {
            result = callable;
          }
        };

        selection.select(query.get_abstract(), Abstract::Visitor(visit));
      },

      [] {});
  return result;
}

VALIDATION_TEST(TypedTests, direct_cpp_handles) {
  const Factory factory(
      {nullptr, [](void* context) { return Cpu::Platform::Memory::create(); }});
  factory.create_engine().visit(
      [&](Engine engine) {
        engine.allocate(64).visit(
            [&](Buffer buffer) {
              EXPECT_EQ(buffer.size(), U64(64));
              EXPECT_EQ(buffer.release(), COMPUTE_EXECUTION_SUCCESS);
            },

            [&](compute_execution_status status) { EXPECT(False); });
        EXPECT_EQ(engine.release(), COMPUTE_EXECUTION_SUCCESS);
      },

      [&](compute_execution_status status) { EXPECT(False); });
}

VALIDATION_TEST(TypedTests, ordered_commands) {
  const auto created = Cpu::Platform::Memory::create();
  ASSERT_EQ(created.status, COMPUTE_EXECUTION_SUCCESS);

  Engine engine(created.engine);
  engine.allocate(8).visit(
      [&](Buffer input) {
        engine.allocate(8).visit(
            [&](Buffer output) {
              engine.create_commands().visit(
                  [&](Commands commands) {
                    Storage::Vector<U8, 8> source = {
                      {10, 20, 30, 40, 50, 60, 70, 80}};

                    Storage::Vector<U8, 8> observed;
                    EXPECT_EQ(
                        input.write(
                            0, View::Bytes(source[0], source.get_size())),
                        COMPUTE_EXECUTION_SUCCESS);

                    const compute_invert_command command{
                      input.get_abi(), output.get_abi(), 8};

                    EXPECT_EQ(
                        commands.record(InvertCommand::contract_id, command),
                        COMPUTE_EXECUTION_SUCCESS);
                    EXPECT_EQ(commands.submit(), COMPUTE_EXECUTION_SUCCESS);
                    EXPECT_EQ(commands.wait(), COMPUTE_EXECUTION_SUCCESS);
                    EXPECT_EQ(
                        output.read(
                            0, Access::Bytes(observed[0], observed.get_size())),
                        COMPUTE_EXECUTION_SUCCESS);

                    constexpr Storage::Vector<U8, 8> expected = {
                      {245, 235, 225, 40, 205, 195, 185, 80}};

                    EXPECT_EQ(
                        View::Bytes(observed[0], observed.get_size()),
                        View::Bytes(expected[0], expected.get_size()));
                    EXPECT_EQ(commands.release(), COMPUTE_EXECUTION_SUCCESS);
                  },

                  [&](compute_execution_status status) { EXPECT(False); });
              EXPECT_EQ(output.release(), COMPUTE_EXECUTION_SUCCESS);
            },

            [&](compute_execution_status status) { EXPECT(False); });
        EXPECT_EQ(input.release(), COMPUTE_EXECUTION_SUCCESS);
      },

      [&](compute_execution_status status) { EXPECT(False); });
  EXPECT_EQ(engine.release(), COMPUTE_EXECUTION_SUCCESS);
}

VALIDATION_TEST(TypedTests, command_lifecycle_rejects_invalid_transitions) {
  const auto created = Cpu::Platform::Memory::create();
  ASSERT_EQ(created.status, COMPUTE_EXECUTION_SUCCESS);

  Engine engine(created.engine);
  engine.create_commands().visit(
      [&](Commands commands) {
        EXPECT_EQ(commands.wait(), COMPUTE_EXECUTION_INVALID);

        constexpr auto unsupported =
            Identity::Uuid(0x1919b2c111f442d2ULL, 0x95809d2935a96845ULL);
        EXPECT(commands.supports(unsupported) == Negotiation::Status::Unknown);

        const U8 arguments = 0;
        EXPECT_EQ(
            commands.record(unsupported, arguments), COMPUTE_EXECUTION_INVALID);

        EXPECT_EQ(commands.submit(), COMPUTE_EXECUTION_SUCCESS);
        EXPECT_EQ(commands.submit(), COMPUTE_EXECUTION_INVALID);
        EXPECT_EQ(
            commands.record(InvertCommand::contract_id, arguments),
            COMPUTE_EXECUTION_INVALID);
        EXPECT_EQ(commands.wait(), COMPUTE_EXECUTION_SUCCESS);
        EXPECT_EQ(commands.wait(), COMPUTE_EXECUTION_SUCCESS);
        EXPECT_EQ(commands.release(), COMPUTE_EXECUTION_SUCCESS);
      },

      [&](compute_execution_status status) { EXPECT(False); });
  EXPECT_EQ(engine.release(), COMPUTE_EXECUTION_SUCCESS);
}

VALIDATION_TEST(TypedTests, factory_type_is_not_engine_type) {
  U64 calls = 0;
  Module module(
      "FixtureFactory"_view,
      {&calls, [](void* context) -> compute_engine_result {
         ++*Data::cast<U64>(context);
         return {COMPUTE_EXECUTION_FAILED, {}};
       }});

  const auto abstract = module.get_abstract();
  abstract.bind<Ttx::Graph::Capabilities::Describe>().visit(
      [&](Ttx::Graph::Capabilities::Describe describe) {
        describe.get_description().visit(
            [&](View::Bytes value) {
              EXPECT_EQ(
                  value,
                  "A factory Type that creates an acquired Compute execution engine."_view);
            },

            [&] { EXPECT(False); });
      },

      [&] { EXPECT(False); });

  auto factory = abstract.bind<Factory>();
  ASSERT(factory);
  EXPECT(!abstract.bind<Engine>());

  auto type = abstract.bind<Coherence::TypeSystem::Type::Contract>();
  ASSERT(type);
  EXPECT_NOT(Type::Abstract::convert(abstract));
  abstract.bind<Ttx::Graph::Policies::Constant>().visit(
      [&](Ttx::Graph::Policies::Constant constant) {
        EXPECT_EQ(
            constant.get_abstract().get_abi().context,
            abstract.get_abi().context);
      },

      [&] { EXPECT(False); });
  type.visit(
      [&](Coherence::TypeSystem::Type::Contract value) {
        EXPECT_EQ(
            value.get_abstract().get_abi().context, abstract.get_abi().context);

        Count fields = 0;
        auto field = [&](Abstract) { ++fields; };
        value.visit_fields(field);
        EXPECT_EQ(fields, Count(0));

        Count callables = 0;
        auto callable = [&](Abstract) { ++callables; };
        value.visit_callables(callable);
        EXPECT_EQ(callables, Count(1));
      },

      [&] { EXPECT(False); });

  const auto call = abstract.resolve_route("create_engine"_view);
  const Storage::Vector<Abstract, 2> static_sections = {{
    Sections::callables().get_abstract(),
    Sections::statics().get_abstract(),
  }};

  auto static_query =
      Selections::Intersection::create(static_sections.get_view());
  Count selected_calls = 0;
  auto selected_call = [&](View::Bytes route, Abstract selected) {
    EXPECT_EQ(route, "create_engine"_view);
    EXPECT_EQ(selected, call);
    ++selected_calls;
  };

  abstract.bind<Ttx::Graph::Capabilities::Selection>().visit(
      [&](Ttx::Graph::Capabilities::Selection selection) {
        EXPECT(
            selection.select(
                static_query.get_abstract(),
                Abstract::Visitor(selected_call)) ==
            Negotiation::Status::Satisfied);
      },

      [&] { EXPECT(False); });
  EXPECT_EQ(selected_calls, Count(1));

  const Storage::Vector<Abstract, 2> member_sections = {{
    Sections::callables().get_abstract(),
    Sections::members().get_abstract(),
  }};

  auto member_query =
      Selections::Intersection::create(member_sections.get_view());
  abstract.bind<Ttx::Graph::Capabilities::Selection>().visit(
      [&](Ttx::Graph::Capabilities::Selection selection) {
        EXPECT(
            selection.select(
                member_query.get_abstract(),
                Abstract::Visitor(selected_call)) ==
            Negotiation::Status::Satisfied);
      },

      [&] { EXPECT(False); });
  EXPECT_EQ(selected_calls, Count(1));
  call.bind<Ttx::Graph::Capabilities::Describe>().visit(
      [&](Ttx::Graph::Capabilities::Describe describe) {
        describe.get_description().visit(
            [&](View::Bytes value) {
              EXPECT_EQ(
                  value, "Creates one acquired Compute execution engine."_view);
            },

            [&] { EXPECT(False); });
      },

      [&] { EXPECT(False); });

  call.bind<Coherence::TypeSystem::Callable::Contract>().visit(
      [&](Coherence::TypeSystem::Callable::Contract callable) {
        EXPECT_EQ(callable.get_result(), Values::engine());
      },

      [&] { EXPECT(False); });

  Compiler::compile_callable(call).visit(
      [&](Executable& executable) {
        EXPECT_EQ(executable.get_inputs().get_extent(), Count(0));
        reflected_form(Values::engine())
            .visit(
                [&](const Form::Representation& engine) {
                  EXPECT(executable.get_output().compatible(engine));
                },

                [&] { EXPECT(False); });

        compute_engine output;
        EXPECT(executable.invoke(nullptr, &output) == Status::Denied);
      },

      [&](Status) { EXPECT(False); });
  EXPECT_EQ(calls, U64(1));

  factory.visit(
      [&](Factory factory) {
        factory.create_engine().visit(
            [&](Engine unexpected) {
              EXPECT(False);
              unexpected.release();
            },

            [&](compute_execution_status status) {
              EXPECT_EQ(status, COMPUTE_EXECUTION_FAILED);
            });
      },

      [] {});
  EXPECT_EQ(calls, U64(2));
}

VALIDATION_TEST(TypedTests, reflection_descriptions_and_canonical_types) {
  auto require_description = [&](Abstract abstract) {
    abstract.bind<Ttx::Graph::Capabilities::Describe>().visit(
        [&](Ttx::Graph::Capabilities::Describe describe) {
          describe.get_description().visit(
              [&](View::Bytes value) { EXPECT(!value.is_empty()); },
              [&] { EXPECT(False); });
        },

        [&] { EXPECT(False); });
  };

  const Storage::Vector<Abstract, 3> values = {
    {Values::engine(), Values::buffer(), Values::commands()}};

  Count methods = 0;
  const Storage::Vector<Abstract, 2> member_sections = {{
    Sections::callables().get_abstract(),
    Sections::members().get_abstract(),
  }};

  auto member_query =
      Selections::Intersection::create(member_sections.get_view());
  for (Count value_index = 0; value_index < values.get_size(); ++value_index) {
    const auto value = values[value_index];
    require_description(value);

    auto projected = Type::Abstract::convert(value);
    ASSERT(projected);
    projected.visit(
        [&](const Type::Abstract& receiver) {
          value.bind<Ttx::Graph::Capabilities::Selection>().visit(
              [&](Ttx::Graph::Capabilities::Selection selection) {
                auto visit = [&](View::Bytes, Abstract callable) {
                  ++methods;
                  require_description(callable);
                  callable.bind<Coherence::TypeSystem::Callable::Contract>()
                      .visit(
                          [&](Coherence::TypeSystem::Callable::Contract
                                  function) {
                            function.visit_arguments(require_description);
                            require_description(function.get_result());

                            Compiler::compile_member(receiver, callable)
                                .visit(
                                    [&](Executable& executable) {
                                      EXPECT(executable.accepts(
                                          executable.get_inputs(),
                                          executable.get_output()));
                                    },

                                    [&](Status) { EXPECT(False); });
                          },

                          [&] { EXPECT(False); });
                };

                EXPECT(
                    selection.select(
                        member_query.get_abstract(),
                        Abstract::Visitor(visit)) ==
                    Negotiation::Status::Satisfied);
              },

              [&] { EXPECT(False); });
        },

        [&] { EXPECT(False); });
  }

  EXPECT_EQ(methods, Count(14));

  const auto catalog = Values::get();
  EXPECT_EQ(catalog.resolve_route("ComputeEngine"_view), Values::engine());
  EXPECT_EQ(catalog.resolve_route("ComputeBuffer"_view), Values::buffer());
  EXPECT_EQ(catalog.resolve_route("ComputeCommands"_view), Values::commands());
  EXPECT_EQ(
      catalog.resolve_route("ComputeExecutionStatus"_view),
      Values::execution_status());

  auto status_form = reflected_form(Values::execution_status());
  ASSERT(status_form);
  status_form.visit(
      [&](const Form::Representation& status) {
        EXPECT(status.compatible(
            Form::Compiled<Form::Native<compute_execution_status>::reference>::
                get_representation()));
      },

      [] {});

  const auto allocate = method(Values::engine(), "allocate"_view);
  allocate.bind<Coherence::TypeSystem::Callable::Contract>().visit(
      [&](Coherence::TypeSystem::Callable::Contract callable) {
        Count argument = 0;
        auto visit = [&](Abstract declaration) {
          reflected_form(declaration.resolve_route("type"_view))
              .visit(
                  [&](const Form::Representation& form) {
                    EXPECT(form.compatible(
                        Form::Compiled<Form::Native<U64>::reference>::
                            get_representation()));
                  },

                  [&] { EXPECT(False); });
          ++argument;
        };

        callable.visit_arguments(visit);
        EXPECT_EQ(argument, Count(1));
        EXPECT_EQ(callable.get_result(), Values::buffer());
      },

      [&] { EXPECT(False); });

  method(Values::buffer(), "read"_view)
      .bind<Coherence::TypeSystem::Callable::Contract>()
      .visit(
          [&](Coherence::TypeSystem::Callable::Contract callable) {
            reflected_form(callable.get_result())
                .visit(
                    [&](const Form::Representation& form) {
                      EXPECT(form.compatible(
                          Form::Compiled<Form::Native<perimortem_view_bytes>::
                                             reference>::get_representation()));
                    },

                    [&] { EXPECT(False); });
          },

          [&] { EXPECT(False); });

  method(Values::commands(), "invert"_view)
      .bind<Coherence::TypeSystem::Callable::Contract>()
      .visit(
          [&](Coherence::TypeSystem::Callable::Contract callable) {
            Count argument = 0;
            auto visit = [&](Abstract declaration) {
              if (argument == 0 || argument == 1) {
                EXPECT_EQ(
                    declaration.resolve_route("type"_view), Values::buffer());
              }

              ++argument;
            };

            callable.visit_arguments(visit);
            EXPECT_EQ(argument, Count(3));
            EXPECT_EQ(callable.get_result(), Values::execution_status());
          },

          [&] { EXPECT(False); });
}

VALIDATION_TEST(TypedTests, independent_buffer_access) {
  const auto first = Cpu::Platform::Memory::create();
  const auto second = Cpu::Platform::Memory::create();
  ASSERT_EQ(first.status, COMPUTE_EXECUTION_SUCCESS);
  ASSERT_EQ(second.status, COMPUTE_EXECUTION_SUCCESS);

  Engine engine(first.engine);
  const auto allocated =
      first.engine.operations->allocate(first.engine.context, 16);
  ASSERT_EQ(allocated.status, COMPUTE_EXECUTION_SUCCESS);

  Buffer buffer(allocated.buffer);
  engine.allocate(0).visit(
      [&](Buffer unexpected) {
        EXPECT(False);
        unexpected.release();
      },

      [&](compute_execution_status status) {
        EXPECT_EQ(status, COMPUTE_EXECUTION_INVALID);
      });

  EXPECT_EQ(engine.release(), COMPUTE_EXECUTION_SUCCESS);
  EXPECT_EQ(buffer.size(), U64(16));

  Storage::Vector<U8, 4> input = {{11, 22, 33, 44}};
  Storage::Vector<U8, 4> output;
  const auto input_bytes = View::Bytes(input[0], input.get_size());
  const auto output_bytes = Access::Bytes(output[0], output.get_size());
  EXPECT_EQ(buffer.write(12, input_bytes), COMPUTE_EXECUTION_SUCCESS);
  EXPECT_EQ(buffer.read(12, output_bytes), COMPUTE_EXECUTION_SUCCESS);
  EXPECT_EQ(input_bytes, output_bytes.get_view());
  EXPECT_EQ(buffer.write(13, input_bytes), COMPUTE_EXECUTION_INVALID);
  EXPECT_EQ(buffer.read(13, output_bytes), COMPUTE_EXECUTION_INVALID);
  EXPECT_EQ(input_bytes, output_bytes.get_view());
  EXPECT_EQ(buffer.release(), COMPUTE_EXECUTION_SUCCESS);
  EXPECT_EQ(Engine(second.engine).release(), COMPUTE_EXECUTION_SUCCESS);
}

// This provider has a different state layout and no native CPU allocation.
// The shared description must use its public Buffer record and supplied table.
VALIDATION_TEST(TypedTests, description_dispatches_foreign_context) {
  struct Foreign {
    Storage::Vector<U8, 29> prefix;
    U64 extent;
  } foreign{{}, 513};

  static const compute_buffer_ops operations{
    [](void* context) -> ttx_abstract { return ttx_none(); },
    [](void* context) -> U64 { return Data::cast<Foreign>(context)->extent; },
    [](void* context, U64 offset, U8* output, U64 size)
        -> compute_execution_status { return COMPUTE_EXECUTION_FAILED; },
    [](void* context, U64 offset, perimortem_view_bytes input)
        -> compute_execution_status { return COMPUTE_EXECUTION_FAILED; },
    [](void* context) -> compute_execution_status {
      return COMPUTE_EXECUTION_SUCCESS;
    }};

  compute_buffer buffer{&foreign, &operations};
  const auto callable = method(Values::buffer(), "size"_view);
  auto receiver = Type::Abstract::convert(Values::buffer());
  ASSERT(receiver);
  receiver.visit(
      [&](const Type::Abstract& type) {
        Compiler::compile_member(type, callable)
            .visit(
                [&](Executable& executable) {
                  const auto& inputs =
                      Form::Compiled<Form::Native<compute_buffer>::reference>::
                          get_representation();
                  const auto& outputs = Form::Compiled<
                      Form::Native<U64>::reference>::get_representation();
                  ASSERT(executable.accepts(inputs, outputs));

                  U64 size = 0;
                  EXPECT(executable.invoke(&buffer, &size) == Status::Success);
                  EXPECT_EQ(size, U64(513));
                  foreign.extent = 79;
                  EXPECT(executable.invoke(&buffer, &size) == Status::Success);
                  EXPECT_EQ(size, U64(79));
                },

                [&](Status) { EXPECT(False); });
      },

      [&] { EXPECT(False); });
}
