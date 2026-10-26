# Using the Compute SDKs

The Compute SDK bundle contains the shared TTX compute contracts and separate
CPU and CUDA implementation packages. Each package supplies public headers and
static and shared libraries. The C contracts use complete TTX API records;
C++ interfaces call the supplied operations with their published context.

## Shared compute domain

The Compute package supplies the image contracts and factory selection through
`compute/` headers. Consumers and engine implementations use the same C records
and C++ interfaces. The package's runtime implements Selection, which composes
existing factories through TTX Create.

Engine modules publish their own entry points. The CPU and CUDA modules each
export `ttx_entry`; look up that symbol on the chosen module and establish its
Query through Data Transfer before requesting an Abstract.

An engine's `commands` route supplies a Create factory for command buffers.
Retain the buffer inside the creation callback. Its Abstract binds Commands
for submission and completion, and typed operations such as Invert for
recording. A successful record returns a Retained image whose Pixels contract
exposes its dimensions immediately. Payload reads before submission return
Invalid. After submission, reads wait for the work producing that image.

Commands are recorded in authored order and submitted once. A buffer can use
its own earlier result directly. Recording an input from another buffer requires
that producer to be submitted first. Inputs are retained through their supplied
contracts or copied while their original access is valid. Retention can supply
a different Abstract and different typed contexts. Recording requests Pixels
again on that returned Abstract before keeping a deferred read.

Submit reports whether execution was started. Wait reports completion or
execution failure without copying data. Submissions can fail after earlier
commands have started, and recording remains closed after either outcome.
The consumer serializes host calls through one engine creation, all of its
buffers and their results. Submitted device work can overlap. Retained inputs
continue to follow their supplying access and serialization policies.

Each result acquisition keeps its producing buffer available independently of
the original engine or buffer acquisition. Final release completes any pending
work before freeing its resources and reports cleanup through TTX Retained.
Discarding an unsubmitted buffer releases captured inputs without executing it.

## CPU module

The CPU package supplies its engine and host image storage through `cpu/`
headers. Its entry supplies the CPU Create factory. None requests the default
engine, whose `commands` route creates independent buffers. Retain the engine
inside the creation callback for later use.

Native C++ callers can use `Cpu::Platform::Engine::factory()` and create host
images through `Cpu::Image::Rgba8::create()`. The resulting Abstracts provide
the same Compute contracts used across a module boundary.

## CUDA module

The optional CUDA package adds a compiler Abstract whose `engine` route selects
the CUDA Create factory. Its image operations use the same Compute contracts,
with device storage private to the provider. Each buffer has a dedicated CUDA
stream, with completion events for its output images. CUDA allocates storage
before launching its sequence, and preserves local dependencies as device
addresses supplied by its own recording.

The optional DeviceImage contract supplies read access in a requested CUDA
context and stream. Compatible buffers order that access through CUDA events.
Other inputs use Pixels and captured host storage. The consumer keeps each
provider module loaded through every operation and release using its code.

CUDA's additional public C records live under `cuda/contracts`. The headers
use TTX 0.4.1 and Perimortem 0.4.0. The build uses Toolchain 0.4.2.

Load `ttx_entry` as the Storage export described by TTX's native library
contract. Transfer its Query into local storage, negotiate an Abstract, and
bind the Compiler. Compilation, preparation and allocation return Retained
results. Each result supplies an Abstract through its `get_abstract` operation.
Bind Program, Kernel or Buffer through that Abstract and release the result
after its final use. Additional retained access requires the Retain capability.
Keep the provider library loaded through every release.

Native C++ consumers can use `cuda/runtime` with NVIDIA's CUDA headers and a
compatible C++ toolchain. Program exposes its loaded module and context for
operations implemented by the consuming project. Those consumers complete
their device work before releasing the Program. cuFFT remains an optional
NVIDIA dependency selected through `//build:fft` in a source checkout.

## Packages and linkage

The root `:sdk` target builds the complete three-package bundle.
`:compute_sdk` and `:cpu_sdk` package their respective modules for Linux, Windows
and Wasm. `:cuda_sdk` packages CUDA for Linux and Windows. Each package contains
static and shared variants. Each module's ordinary source target uses shared
TTX and Perimortem, while its static variant selects the static dependencies.

All three packages belong to the [Compute repository](https://github.com/tetrodotoxin-dev/Compute).
Toolchain SDK imports use `project = "Compute"`, with `library = "compute"`,
`library = "cpu"` or `library = "cuda"` selecting the package. Archive names use
the corresponding prefix: `compute-<version>-...zip`, `cpu-<version>-...zip` or
`cuda-<version>-...zip`. The candidate import declarations are in
[validation/consumer/MODULE.tpl](validation/consumer/MODULE.tpl).

CUDA packaging uses Toolchain 0.4.2's runtime-only DLL and system interface
support. The driver link interfaces accompany the static CUDA SDK; the installed
NVIDIA driver supplies their runtime implementation. NVRTC and its builtins
are bundled runtime dependencies.

`python3 validation/consumer/check.py` builds the bundle and consumes its local
archives from a separate Bazel workspace. C and C++ consumers exercise CUDA
compilation and retained storage, and a composed consumer uses CPU and CUDA
image providers. The CPU test suite also runs against the packaged Compute and
CPU libraries with no CUDA dependency. Linux consumers execute in both linkage
modes, including a relocated CUDA consumer. Windows consumers are compiled and
linked.

The refreshed Compute 0.1.0 bundle replaces the earlier eager execution API.
Commands and Dependency describe recorded work, and Invert and Pixels use new
UUIDs for recording and deferred result access. Consumers of the earlier
bundle must update their archive checksums and migrate to command buffers.
NVIDIA components in the CUDA package retain their supplied license terms and notices
in each archive's LICENSE file. Compute's own source remains MIT licensed.
