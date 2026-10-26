# Using the Compute SDKs

The Compute package supplies the shared domain and CPU platform through
`compute/` headers. Its exported `ttx_entry` supplies the CPU Create factory
through the standard Storage, Transfer, Query and Abstract bootstrap. Engine
creation uses None for the default configuration. The returned Abstract exposes
the `invert` route. Retain the engine inside the callback for later use.

The optional CUDA package adds a compiler Abstract whose `engine` route selects
the CUDA Create factory. Its image operations use the same Compute contracts,
with device storage private to the provider. Keep both provider modules loaded
through all operations and releases that use their code.

CUDA's public C records live under `cuda/contracts`. C++ interfaces forward
those records through the context and operations supplied by the provider.
The headers use TTX 0.4.1 and Perimortem 0.4.0. The build uses Toolchain 0.4.2.

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

## Candidate packaging

The root `:sdk` target declares Compute packages for Linux, Windows and Wasm.
`:cuda_sdk` declares static and shared CUDA packages for Linux and Windows.
The library's ordinary source target uses shared TTX and Perimortem, while its
static variant selects the static dependencies.

CUDA packaging uses Toolchain 0.4.2's runtime-only DLL and system interface
support. The driver link interfaces accompany the static CUDA SDK; the installed
NVIDIA driver supplies their runtime implementation. NVRTC and its builtins
are bundled runtime dependencies.

`python3 validation/consumer/check.py` builds the CUDA SDK and validates independent
C and C++ consumers in both linkage modes. Linux consumers compile CUDA source
and exercise retained storage, including from a relocated package. Windows
consumers are compiled and linked. Additional Bazel arguments can be supplied
for explicitly approved development experiments.

The Compute and CUDA 0.4.0 SDKs are not published. NVIDIA runtime distribution and notices
must be checked with the completed package before publication.
