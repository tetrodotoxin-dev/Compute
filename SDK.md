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

Engines expose image inversion through their `invert` route. Bind Invert there
and supply an image Abstract that binds Pixels.
Successful execution returns a Retained image whose Abstract binds Pixels.
Each retained result carries one release obligation. Its resources remain
available independently of the original engine acquisition, and another
consumer can request its own acquisition through Retain.

The current image operations finish before returning. Their declarations
specify serialization, input access and result validity. CUDA currently reads
input pixels through host memory, including inputs supplied by another CUDA
engine; its output image keeps device storage behind the Pixels contract.

## CPU module

The CPU package supplies its engine and host image storage through `cpu/`
headers. Its entry supplies the CPU Create factory. None requests the default
engine, whose Abstract exposes the `invert` route. Retain the engine inside the
creation callback for later use.

Native C++ callers can use `Cpu::Platform::Engine::factory()` and create host
images through `Cpu::Image::Rgba8::create()`. The resulting Abstracts provide
the same Compute contracts used across a module boundary.

## CUDA module

The optional CUDA package adds a compiler Abstract whose `engine` route selects
the CUDA Create factory. Its image operations use the same Compute contracts,
with device storage private to the provider. Keep every provider module loaded
through all operations and releases that use its code.

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

The [Compute 0.1.0 release](https://github.com/tetrodotoxin-dev/Compute/releases/tag/v0.1.0)
contains all three packages, their checksums and the source snapshot. NVIDIA
components in the CUDA package retain their supplied license terms and notices
in each archive's LICENSE file. Compute's own source remains MIT licensed.
