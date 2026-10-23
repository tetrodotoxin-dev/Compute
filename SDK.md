# Using the CUDA SDK

CUDA's public C records live under `cuda/contracts`. C++ interfaces forward
those records through the context and operations supplied by the provider.
The headers use TTX 0.4.0 and Perimortem 0.4.0. The build uses Toolchain 0.4.1.

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

The root `:sdk` target declares static and shared Linux and Windows packages.
The library's ordinary source target uses shared TTX and Perimortem, while its
static variant selects the static dependencies.

Full SDK packaging currently stops on the Windows NVRTC builtins dependency:
NVIDIA supplies that DLL for runtime loading without an import library, while
Toolchain 0.4.1 requires an import library for every bundled Windows DLL.
The CUDA 0.4.0 SDK is not published. NVIDIA runtime distribution and notices
must be checked with the completed package before publication.
