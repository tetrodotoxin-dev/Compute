# CUDA

This library exposes CUDA compilation, kernels and device storage through TTX
contracts. Consumers supply source bytes, named includes and compiler options.
NVRTC is used to compile provided source for the selected device on demand, with
the resulting TTX Program owning the loaded code and a reference to its CUDA context.

A Program prepares Kernels and allocates Buffers. Each operation supplies a
Retained result and keeps the Program available, so releasing the original result leaves
existing kernels and allocations usable. Kernel arguments are described with
TTX Data representations so preparation can check their sizes and placement against
the loaded function and the caller's frame. The source author remains responsible
for matching the argument types.

The provider keeps CUDA resource management behind the interface. Tetrodotoxin
and Godot consumers contain examples of defining their own operations and policies on
top of the CUDA layer. An image operation for example chooses its pixel format, kernel
and launch geometry in the consuming project.

## TTX compatibility

The module exports `ttx_entry` as described Storage for TTX `0.4.0`.
Data Transfer establishes its Query representation before the consumer asks
for the compiler Abstract and binds the Compiler contract.
The public contracts use standard TTX C records and function pointers, with C++ interfaces
over those records. Native callers can also use the runtime directly which saves the effort
of negotiating the system's native ABI and calling convention.

Calls on related resources are serialized by the consumer, including retention
and release. The caller keeps the module loaded until every Retained result has
been released. Launches and host transfers are guaranteed
to finish before returning.

## Building and testing

The build targets Linux and Windows x86_64 with x86-64-v3 and RDRAND. Install
Python 3 and the Bazel version in [.bazelversion](.bazelversion). Bazel downloads
the pinned LLVM tools, target SDK, TTX and Perimortem SDKs, and NVIDIA's CUDA
13.3 headers and libraries. The machine running GPU work supplies its NVIDIA
driver.

From the repository root:

```sh
bazel build //source/... --config=release
bazel run :tests --config=debug
bazel run :tests --config=release
bazel run :benchmarks --config=release
bazel run :format -- --check
bazel run :vscode
```

The Linux plugin is written to `.bin/bin/source/libttx_cuda.so`. Add
`--config=windows` to cross-build `ttx_cuda.dll` and its import library from Linux.
The pinned NVRTC and builtins libraries are carried through the build's runtime
dependencies. The tests require a CUDA capable GPU and exercise both C
and C++ consumers, compilation diagnostics, argument agreement, buffer transfers
and retained resource lifetimes.

Consumers use `//:contracts` for the TTX contracts and `//:runtime` for the
native CUDA interfaces. The optional
`//build:fft` target acquires cuFFT for callers that need it.
