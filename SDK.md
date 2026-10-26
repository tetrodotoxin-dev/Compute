# Using the Compute SDKs

Compute packages shared domain contracts and independently implemented CPU and
CUDA providers. Each package supplies public headers and static and shared
libraries. The C contracts use complete TTX API records; C++ interfaces call
the supplied operations with their published context.

## Shared compute domain

The Compute package supplies the image contracts, factory selection, host image
storage and CPU provider through `compute/` headers. Its exported `ttx_entry`
supplies the CPU Create factory through the Storage, Transfer, Query and
Abstract bootstrap. Engine creation uses None for the default configuration.
The returned Abstract exposes the `invert` route. Retain the engine inside the
creation callback for later use.

Bind Invert at that route and supply an image Abstract that binds Pixels.
Successful execution returns a Retained image whose Abstract binds Pixels.
Each retained result carries one release obligation. Its resources remain
available independently of the original engine acquisition, and another
consumer can request its own acquisition through Retain.

The current image operations finish before returning. Their declarations
specify serialization, input access and result validity. CUDA currently reads
input pixels through host memory, including inputs supplied by another CUDA
engine; its output image keeps device storage behind the Pixels contract.

## CUDA platform

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

The root `:sdk` target declares Compute packages for Linux, Windows and Wasm.
`:cuda_sdk` declares static and shared CUDA packages for Linux and Windows.
The library's ordinary source target uses shared TTX and Perimortem, while its
static variant selects the static dependencies.

Both packages belong to the [Compute repository](https://github.com/tetrodotoxin-dev/Compute).
Toolchain SDK imports use `project = "Compute"` for both, with
`library = "compute"` or `library = "cuda"` selecting the package. Package
names remain distinct: `compute-<version>-...zip` and `cuda-<version>-...zip`.
The candidate import declarations are in
[validation/consumer/MODULE.tpl](validation/consumer/MODULE.tpl).

CUDA packaging uses Toolchain 0.4.2's runtime-only DLL and system interface
support. The driver link interfaces accompany the static CUDA SDK; the installed
NVIDIA driver supplies their runtime implementation. NVRTC and its builtins
are bundled runtime dependencies.

`python3 validation/consumer/check.py` builds both SDKs and consumes their local
archives from a separate Bazel workspace. C and C++ consumers exercise CUDA
compilation and retained storage, and a composed consumer uses CPU and CUDA
image providers. Linux consumers execute in both linkage modes, including a
relocated CUDA consumer. Windows consumers are compiled and linked.

The Compute and CUDA 0.4.0 SDKs are not published. Existing CUDA releases
describe the earlier interfaces. NVIDIA runtime distribution and notices must
be checked with the completed package before publication.
