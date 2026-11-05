// # Tetrodotoxin
// Copyright (c) 2023-present Matt Kaes and contributors

#include "compute/execution/values.hpp"

#include "perimortem/core/storage/vector.hpp"
#include "perimortem/core/view/null_terminated.hpp"
#include "perimortem/memory/global/bytes.hpp"

#include "coherence/type_system/callable.hpp"
#include "coherence/type_system/descriptor.hpp"
#include "coherence/type_system/field.hpp"
#include "coherence/type_system/primitive.hpp"
#include "coherence/type_system/sections.hpp"
#include "coherence/type_system/type.hpp"
#include "compute/execution/engine.hpp"
#include "compute/image/composite_command.hpp"
#include "compute/image/convolve_command.hpp"
#include "compute/image/invert_command.hpp"
#include "compute/image/posterize_command.hpp"
#include "ttx/data/form/native.hpp"

using namespace Perimortem::Core;
using namespace Perimortem::Memory;
using namespace Ttx::Data;
using namespace Ttx::Graph;
using namespace Compute::Execution;
using namespace Compute::Image;
using namespace Coherence::TypeSystem;

using AbstractType = Type::Abstract;

struct AllocateArguments {
  compute_engine engine;
  U64 size;
};

struct InvertArguments {
  compute_commands commands;
  compute_buffer input;
  compute_buffer output;
  U64 count;
};

struct PosterizeArguments {
  compute_commands commands;
  compute_buffer input;
  compute_buffer output;
  U64 count;
  U64 levels;
};

struct ConvolveArguments {
  compute_commands commands;
  compute_buffer input;
  compute_buffer output;
  compute_buffer weights;
  U64 width;
  U64 height;
  U64 kernel_width;
  U64 kernel_height;
};

struct CompositeArguments {
  compute_commands commands;
  compute_buffer back;
  compute_buffer front;
  compute_buffer output;
  U64 count;
};

struct BufferWriteArguments {
  compute_buffer buffer;
  U64 offset;
  perimortem_view_bytes bytes;
};

static auto allocate(void* context, const void* input, void* output)
    -> ttx_data_status {
  const auto& arguments = *Data::cast<const AllocateArguments>(input);
  const auto result = arguments.engine.operations->allocate(
      arguments.engine.context, arguments.size);
  if (result.status != COMPUTE_EXECUTION_SUCCESS) {
    return TTX_DATA_DENIED;
  }

  *Data::cast<compute_buffer>(output) = result.buffer;
  return TTX_DATA_SUCCESS;
}

static auto engine_release(void* context, const void* input, void* output)
    -> ttx_data_status {
  const auto& engine = *Data::cast<const compute_engine>(input);
  *Data::cast<U8>(output) = engine.operations->release(engine.context);
  return TTX_DATA_SUCCESS;
}

static auto create_commands(void* context, const void* input, void* output)
    -> ttx_data_status {
  const auto& engine = *Data::cast<const compute_engine>(input);
  const auto result = engine.operations->create_commands(engine.context);
  if (result.status != COMPUTE_EXECUTION_SUCCESS) {
    return TTX_DATA_DENIED;
  }

  *Data::cast<compute_commands>(output) = result.commands;
  return TTX_DATA_SUCCESS;
}

static auto buffer_size(void* context, const void* input, void* output)
    -> ttx_data_status {
  const auto& buffer = *Data::cast<const compute_buffer>(input);
  *Data::cast<U64>(output) = buffer.operations->size(buffer.context);
  return TTX_DATA_SUCCESS;
}

static auto buffer_read(void* context, const void* input, void* output)
    -> ttx_data_status {
  const auto& buffer = *Data::cast<const compute_buffer>(input);
  static thread_local Global::Bytes bytes;
  bytes.forgetful_resize(buffer.operations->size(buffer.context));
  if (buffer.operations->read(
          buffer.context, 0, bytes.get_data(), bytes.get_size()) !=
      COMPUTE_EXECUTION_SUCCESS) {
    return TTX_DATA_DENIED;
  }

  *Data::cast<perimortem_view_bytes>(output) = bytes.get_view();
  return TTX_DATA_SUCCESS;
}

static auto buffer_write(void* context, const void* input, void* output)
    -> ttx_data_status {
  const auto& arguments = *Data::cast<const BufferWriteArguments>(input);
  *Data::cast<U8>(output) = arguments.buffer.operations->write(
      arguments.buffer.context, arguments.offset, arguments.bytes);
  return TTX_DATA_SUCCESS;
}

static auto buffer_release(void* context, const void* input, void* output)
    -> ttx_data_status {
  const auto& buffer = *Data::cast<const compute_buffer>(input);
  *Data::cast<U8>(output) = buffer.operations->release(buffer.context);
  return TTX_DATA_SUCCESS;
}

static auto command_submit(void* context, const void* input, void* output)
    -> ttx_data_status {
  const auto& commands = *Data::cast<const compute_commands>(input);
  *Data::cast<U8>(output) = commands.operations->submit(commands.context);
  return TTX_DATA_SUCCESS;
}

static auto command_wait(void* context, const void* input, void* output)
    -> ttx_data_status {
  const auto& commands = *Data::cast<const compute_commands>(input);
  *Data::cast<U8>(output) = commands.operations->wait(commands.context);
  return TTX_DATA_SUCCESS;
}

static auto command_release(void* context, const void* input, void* output)
    -> ttx_data_status {
  const auto& commands = *Data::cast<const compute_commands>(input);
  *Data::cast<U8>(output) = commands.operations->release(commands.context);
  return TTX_DATA_SUCCESS;
}

static auto invert(void* context, const void* input, void* output)
    -> ttx_data_status {
  const auto& arguments = *Data::cast<const InvertArguments>(input);
  const compute_invert_command command{
    arguments.input, arguments.output, arguments.count};

  *Data::cast<U8>(output) = arguments.commands.operations->record(
      arguments.commands.context, InvertCommand::contract_id, &command);
  return TTX_DATA_SUCCESS;
}

static auto posterize(void* context, const void* input, void* output)
    -> ttx_data_status {
  const auto& arguments = *Data::cast<const PosterizeArguments>(input);
  if (arguments.levels > U32(-1)) {
    *Data::cast<U8>(output) = COMPUTE_EXECUTION_INVALID;
    return TTX_DATA_SUCCESS;
  }

  const compute_posterize_command command{
    arguments.input, arguments.output, arguments.count, U32(arguments.levels)};

  *Data::cast<U8>(output) = arguments.commands.operations->record(
      arguments.commands.context, PosterizeCommand::contract_id, &command);
  return TTX_DATA_SUCCESS;
}

static auto convolve(void* context, const void* input, void* output)
    -> ttx_data_status {
  const auto& arguments = *Data::cast<const ConvolveArguments>(input);
  if (arguments.width > U32(-1) || arguments.height > U32(-1) ||
      arguments.kernel_width > U32(-1) || arguments.kernel_height > U32(-1)) {
    *Data::cast<U8>(output) = COMPUTE_EXECUTION_INVALID;
    return TTX_DATA_SUCCESS;
  }

  const compute_convolve_command command{
    arguments.input,
    arguments.output,
    arguments.weights,
    U32(arguments.width),
    U32(arguments.height),
    U32(arguments.kernel_width),
    U32(arguments.kernel_height)};

  *Data::cast<U8>(output) = arguments.commands.operations->record(
      arguments.commands.context, ConvolveCommand::contract_id, &command);
  return TTX_DATA_SUCCESS;
}

static auto composite(void* context, const void* input, void* output)
    -> ttx_data_status {
  const auto& arguments = *Data::cast<const CompositeArguments>(input);
  const compute_composite_command command{
    arguments.back, arguments.front, arguments.output, arguments.count};

  *Data::cast<U8>(output) = arguments.commands.operations->record(
      arguments.commands.context, CompositeCommand::contract_id, &command);
  return TTX_DATA_SUCCESS;
}

// Occurrence keeps an exact route separate from its concrete value. The
// current Compute declarations use the same bytes for route and described name,
// while the two values remain independently authored facts.
template <typename Concrete>
class Occurrence {
 public:
  static auto create(View::Bytes route, Concrete value) -> Occurrence {
    return Occurrence(route, value);
  }

  auto get_route() const -> View::Bytes { return route; }
  auto get_value() -> Concrete& { return value; }
  auto get_value() const -> const Concrete& { return value; }

 private:
  Occurrence(View::Bytes route, Concrete value) : route(route), value(value) {}

  View::Bytes route;
  Concrete value;
};

// Definitions owns the immutable values behind Compute's public Type graph.
// Published stores occurrences and their graph Abstracts together so every View
// points into final storage before any containing Type is exposed.
template <typename Concrete, Count count>
class Published {
 public:
  static auto create(const Storage::Vector<Occurrence<Concrete>, count>& values)
      -> Published {
    return Published(values);
  }

  auto get_route(Count index) const -> View::Bytes {
    return occurrences[index].get_route();
  }

  auto get_value(Count index) -> Concrete& {
    return occurrences[index].get_value();
  }

  auto get_references() const -> View::Vector<typename Concrete::Reference> {
    return references.get_view();
  }

 private:
  Published(const Storage::Vector<Occurrence<Concrete>, count>& source)
      : occurrences(source) {
    for (Count index = 0; index < count; ++index) {
      auto projected = Concrete::Abstract::convert(
          occurrences[index].get_value().get_abstract());
      projected.visit(
          [&](const typename Concrete::Abstract& abstract) {
            references[index] = typename Concrete::Reference(abstract);
          },

          [] { __builtin_trap(); });
    }
  }

  Storage::Vector<Occurrence<Concrete>, count> occurrences;
  Storage::Vector<typename Concrete::Reference, count> references;
};

template <typename Concrete>
static auto project(Concrete& value) -> typename Concrete::Abstract {
  auto projected = Concrete::Abstract::convert(value.get_abstract());
  return projected.visit(
      [](const typename Concrete::Abstract& abstract) { return abstract; },
      []() -> typename Concrete::Abstract { __builtin_trap(); });
}

template <Count... indexes>
class Indices {};

template <Count count, Count... indexes>
class MakeIndices : public MakeIndices<count - 1, count - 1, indexes...> {};

template <Count... indexes>
class MakeIndices<0, indexes...> {
 public:
  using Type = Indices<indexes...>;
};

template <
    Count field_count,
    Count callable_count,
    Count... field_indexes,
    Count... callable_indexes>
static auto descriptors(
    Published<Field, field_count>& fields,
    Published<Callable, callable_count>& callables,
    View::Vector<Abstract> field_sections,
    View::Vector<Abstract> callable_sections,
    Indices<field_indexes...>,
    Indices<callable_indexes...>)
    -> Storage::Vector<Descriptor, field_count + callable_count> {
  return {{
    Descriptor::create(
        fields.get_route(field_indexes),
        fields.get_value(field_indexes).get_abstract(), field_sections)...,
    Descriptor::create(
        callables.get_route(callable_indexes),
        callables.get_value(callable_indexes).get_abstract(),
        callable_sections)...,
  }};
}

template <Count field_count, Count callable_count>
static auto descriptors(
    Published<Field, field_count>& fields,
    Published<Callable, callable_count>& callables,
    View::Vector<Abstract> field_sections,
    View::Vector<Abstract> callable_sections)
    -> Storage::Vector<Descriptor, field_count + callable_count> {
  return descriptors(
      fields, callables, field_sections, callable_sections,
      typename MakeIndices<field_count>::Type(),
      typename MakeIndices<callable_count>::Type());
}

static auto project_type(Primitive& value) -> AbstractType {
  auto projected = Type::Abstract::convert(value.get_abstract());
  return projected.visit(
      [](const AbstractType& abstract) { return abstract; },
      []() -> AbstractType { __builtin_trap(); });
}

static auto require_type(Storage::Option<Type> candidate) -> Type {
  return candidate.visit(
      [](const Type& type) { return type; },
      []() -> Type { __builtin_trap(); });
}

static constexpr auto describe(View::Bytes name, View::Bytes documentation)
    -> Description {
  return Description::create(name, documentation);
}

static constexpr auto field(
    View::Bytes name,
    View::Bytes documentation,
    AbstractType type) -> Occurrence<Field> {
  return Occurrence<Field>::create(
      name, Field::create(describe(name, documentation), type.get_abstract()));
}

static constexpr auto callable(
    View::Bytes name,
    View::Bytes documentation,
    View::Vector<Callable::Argument> arguments,
    AbstractType result,
    Operation::Adapter adapter) -> Occurrence<Callable> {
  return Occurrence<Callable>::create(
      name, Callable::create(
                describe(name, documentation), arguments, result.get_abstract(),
                Operation::create(nullptr, adapter)));
}

struct Definitions {
  Storage::Vector<Abstract, 2> field_sections = {{
    Sections::fields().get_abstract(),
    Sections::members().get_abstract(),
  }};

  Storage::Vector<Abstract, 2> callable_sections = {{
    Sections::callables().get_abstract(),
    Sections::members().get_abstract(),
  }};

  Storage::Vector<Abstract, 1> type_sections = {{
    Sections::types().get_abstract(),
  }};

  Primitive pointer = Primitive::create(
      describe("pointer"_view, "An opaque provider-owned address."_view),
      Form::Native<void*>::reference);

  Primitive u64 = Primitive::create(
      describe("u64"_view, "An unsigned 64-bit integer."_view),
      Form::Native<U64>::reference);

  Primitive view_bytes = Primitive::create(
      describe(
          "view_bytes"_view,
          "A read-only byte range with a supplier-defined access window."_view),
      Form::Native<perimortem_view_bytes>::reference);

  Primitive execution_status = Primitive::create(
      describe(
          "ComputeExecutionStatus"_view,
          "The outcome of a Compute execution or cleanup operation."_view),
      Form::Native<compute_execution_status>::reference);

  Primitive buffer_operations_pointer = Primitive::create(
      describe(
          "ComputeBufferOperationsPointer"_view,
          "A pointer to the complete Compute Buffer operation table."_view),
      Form::Native<const compute_buffer_ops*>::reference);

  Primitive command_operations_pointer = Primitive::create(
      describe(
          "ComputeCommandOperationsPointer"_view,
          "A pointer to the complete Compute command operation table."_view),
      Form::Native<const compute_commands_ops*>::reference);

  Primitive engine_operations_pointer = Primitive::create(
      describe(
          "ComputeEngineOperationsPointer"_view,
          "A pointer to the complete Compute Engine operation table."_view),
      Form::Native<const compute_engine_ops*>::reference);

  AbstractType pointer_type = project_type(pointer);
  AbstractType u64_type = project_type(u64);
  AbstractType view_bytes_type = project_type(view_bytes);
  AbstractType status_type = project_type(execution_status);
  AbstractType buffer_operations_type = project_type(buffer_operations_pointer);
  AbstractType command_operations_type =
      project_type(command_operations_pointer);
  AbstractType engine_operations_type = project_type(engine_operations_pointer);

  Published<Field, 2> buffer_storage = Published<Field, 2>::create({{
    field(
        "context"_view,
        "The opaque context supplied with this Buffer."_view,
        pointer_type),
    field(
        "operations"_view,
        "The operations published with this Buffer context."_view,
        buffer_operations_type),
  }});

  Published<Field, 2> write_arguments = Published<Field, 2>::create({{
    field(
        "offset"_view,
        "The byte offset where writing begins."_view,
        u64_type),
    field(
        "bytes"_view,
        "The host bytes copied into the Buffer."_view,
        view_bytes_type),
  }});

  Published<Callable, 4> buffer_callables = Published<Callable, 4>::create({{
    callable(
        "size"_view,
        "Returns the Buffer capacity in bytes."_view,
        View::Vector<Callable::Argument>(),
        u64_type,
        buffer_size),
    callable(
        "read"_view,
        "Copies the Buffer into temporary host bytes valid until this thread performs another reflected read."_view,
        View::Vector<Callable::Argument>(),
        view_bytes_type,
        buffer_read),
    callable(
        "write"_view,
        "Copies host bytes into the Buffer at the requested offset."_view,
        write_arguments.get_references(),
        status_type,
        buffer_write),
    callable(
        "release"_view,
        "Releases this Buffer acquisition and reports cleanup status."_view,
        View::Vector<Callable::Argument>(),
        status_type,
        buffer_release),
  }});

  Storage::Vector<Descriptor, 6> buffer_descriptors = descriptors(
      buffer_storage,
      buffer_callables,
      field_sections,
      callable_sections);

  Type buffer = require_type(
      Type::create(
          describe(
              "ComputeBuffer"_view,
              "An acquired byte Buffer on a Compute engine. Host reads and writes complete before returning."_view),
          buffer_descriptors.get_view()));
  AbstractType buffer_type = project(buffer);

  Published<Field, 2> commands_storage = Published<Field, 2>::create({{
    field(
        "context"_view,
        "The opaque context supplied with this command recording."_view,
        pointer_type),
    field(
        "operations"_view,
        "The operations published with this command-recording context."_view,
        command_operations_type),
  }});

  Published<Field, 3> invert_arguments = Published<Field, 3>::create({{
    field("input"_view, "The source RGBA8 Buffer."_view, buffer_type),
    field("output"_view, "The destination RGBA8 Buffer."_view, buffer_type),
    field("count"_view, "The number of bytes to transform."_view, u64_type),
  }});

  Published<Field, 4> posterize_arguments = Published<Field, 4>::create({{
    field("input"_view, "The source RGBA8 Buffer."_view, buffer_type),
    field("output"_view, "The destination RGBA8 Buffer."_view, buffer_type),
    field("count"_view, "The number of bytes to transform."_view, u64_type),
    field(
        "levels"_view,
        "The number of color levels from 2 through 256."_view,
        u64_type),
  }});

  Published<Field, 7> convolve_arguments = Published<Field, 7>::create({{
    field("input"_view, "The source RGBA8 Buffer."_view, buffer_type),
    field("output"_view, "The destination RGBA8 Buffer."_view, buffer_type),
    field(
        "weights"_view,
        "The row-major floating-point kernel weights."_view,
        buffer_type),
    field("width"_view, "The image width in pixels."_view, u64_type),
    field("height"_view, "The image height in pixels."_view, u64_type),
    field(
        "kernel_width"_view,
        "The odd kernel width in elements."_view,
        u64_type),
    field(
        "kernel_height"_view,
        "The odd kernel height in elements."_view,
        u64_type),
  }});

  Published<Field, 4> composite_arguments = Published<Field, 4>::create({{
    field("back"_view, "The background RGBA8 Buffer."_view, buffer_type),
    field("front"_view, "The foreground RGBA8 Buffer."_view, buffer_type),
    field("output"_view, "The destination RGBA8 Buffer."_view, buffer_type),
    field("count"_view, "The number of pixels to composite."_view, u64_type),
  }});

  Published<Callable, 7> command_callables = Published<Callable, 7>::create({{
    callable(
        "submit"_view,
        "Submits the recorded commands in their authored order."_view,
        View::Vector<Callable::Argument>(),
        status_type,
        command_submit),
    callable(
        "wait"_view,
        "Waits for submitted work and reports its outcome."_view,
        View::Vector<Callable::Argument>(),
        status_type,
        command_wait),
    callable(
        "release"_view,
        "Releases this command recording and reports cleanup status."_view,
        View::Vector<Callable::Argument>(),
        status_type,
        command_release),
    callable(
        "invert"_view,
        "Records an RGBA8 color inversion from input to output."_view,
        invert_arguments.get_references(),
        status_type,
        invert),
    callable(
        "posterize"_view,
        "Records RGBA8 color quantization from input to output."_view,
        posterize_arguments.get_references(),
        status_type,
        posterize),
    callable(
        "convolve"_view,
        "Records a centered RGBA8 convolution using floating-point weights."_view,
        convolve_arguments.get_references(),
        status_type,
        convolve),
    callable(
        "composite"_view,
        "Records front-over-back RGBA8 alpha composition."_view,
        composite_arguments.get_references(),
        status_type,
        composite),
  }});

  Storage::Vector<Descriptor, 9> command_descriptors = descriptors(
      commands_storage,
      command_callables,
      field_sections,
      callable_sections);

  Type commands = require_type(
      Type::create(
          describe(
              "ComputeCommands"_view,
              "An ordered one-shot command recording. Submit seals it, wait reports completion and release consumes it."_view),
          command_descriptors.get_view()));
  AbstractType commands_type = project(commands);

  Published<Field, 2> engine_storage = Published<Field, 2>::create({{
    field(
        "context"_view,
        "The opaque context supplied with this Engine."_view,
        pointer_type),
    field(
        "operations"_view,
        "The operations published with this Engine context."_view,
        engine_operations_type),
  }});

  Published<Field, 1> allocation_arguments = Published<Field, 1>::create({{
    field(
        "size"_view,
        "The requested Buffer capacity in bytes."_view,
        u64_type),
  }});

  Published<Callable, 3> engine_callables = Published<Callable, 3>::create({{
    callable(
        "allocate"_view,
        "Allocates a Buffer containing the requested number of bytes."_view,
        allocation_arguments.get_references(),
        buffer_type,
        allocate),
    callable(
        "create_commands"_view,
        "Creates an empty ordered command recording for this Engine."_view,
        View::Vector<Callable::Argument>(),
        commands_type,
        create_commands),
    callable(
        "release"_view,
        "Releases this Engine acquisition and reports cleanup status."_view,
        View::Vector<Callable::Argument>(),
        status_type,
        engine_release),
  }});

  Storage::Vector<Descriptor, 5> engine_descriptors = descriptors(
      engine_storage,
      engine_callables,
      field_sections,
      callable_sections);

  Type engine = require_type(
      Type::create(
          describe(
              "ComputeEngine"_view,
              "An acquired Compute engine that allocates Buffers and creates ordered command recordings."_view),
          engine_descriptors.get_view()));
  AbstractType engine_type = project(engine);

  Storage::Vector<Descriptor, 4> types = {{
    Descriptor::create(
        "ComputeEngine"_view,
        engine_type.get_abstract(),
        type_sections.get_view()),
    Descriptor::create(
        "ComputeBuffer"_view,
        buffer_type.get_abstract(),
        type_sections.get_view()),
    Descriptor::create(
        "ComputeCommands"_view,
        commands_type.get_abstract(),
        type_sections.get_view()),
    Descriptor::create(
        "ComputeExecutionStatus"_view,
        status_type.get_abstract(),
        type_sections.get_view()),
  }};

  Type root = require_type(
      Type::create(
          describe(
              "ComputeTypes"_view,
              "The Type identities supplied by the Compute SDK."_view),
          types.get_view()));
};

static auto definitions() -> Definitions& {
  static Definitions value;
  return value;
}

auto Values::get() -> Abstract {
  return definitions().root.get_abstract();
}

auto Values::engine() -> Abstract {
  return definitions().engine.get_abstract();
}

auto Values::buffer() -> Abstract {
  return definitions().buffer.get_abstract();
}

auto Values::commands() -> Abstract {
  return definitions().commands.get_abstract();
}

auto Values::execution_status() -> Abstract {
  return definitions().status_type.get_abstract();
}
