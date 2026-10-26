# Compute

Compute provides a [TTX](https://github.com/tetrodotoxin-dev/TTX) SDK for compute
infrastructure used by Tetrodotoxin projects. Its primary purpose is to provide
the base for Tetrodotoxin's image acceleration and serve as a practical template
for wrapping infrastructure with TTX.

The SDK bundle includes two execution modules:

- A simplified CUDA engine.
- A CPU engine.

Both implement the same Compute contracts. Applications use those contracts to
request work, while each engine manages its own execution and storage. Other
systems can implement the same contracts or extend the domain with their own.

## The Compute SDK

The SDK defines capabilities and policies for negotiating compute operations
through TTX. Its current image contracts supply RGBA8 snapshots through
[Pixels](source/compute/image/pixels.h) and color inversion through
[Invert](source/compute/image/invert.h). An image can supply its pixels from
host storage, device storage or a generated result. The operation uses the
published interface to obtain those pixels.

Policies express requirements independently of an engine's implementation.
[Accelerated](source/compute/policies/accelerated.h) promises execution on a
compute device separate from the host CPU. A consumer can require that policy
when selecting an engine. Each candidate must satisfy the requirement,
including any fallback.

Engine factories use TTX's Create capability. A consumer can use a factory
directly or compose several with
[Selection](source/compute/execution/selection.hpp), which provides the same
Create interface. The consumer retains an engine for use beyond its creation
callback. Produced images retain the resources they need independently of that
engine acquisition.

## Modules

The shared SDK and its implementations have separate source and build ownership:

| Module | Provides |
| --- | --- |
| [Compute](source/compute) | TTX compute contracts, C++ interfaces and factory selection |
| [CPU](source/cpu) | The host execution engine and host image storage |
| [CUDA](source/cuda) | The CUDA engine, source compilation, prepared kernels and device buffers |

CPU and CUDA each publish a module entry through TTX's Storage, Transfer and
Query bootstrap. Consumers obtain an Abstract and bind the capabilities they
need. The host keeps a module loaded through the final use and release of its
interfaces.

The CPU module runs independently of NVIDIA libraries. CUDA also exposes its
platform contracts and native C++ runtime for consumers that need to compile
CUDA source or manage device work directly. The shared Compute contracts remain
independent of both engine implementations.

The bundle contains a separate package for each module, with public headers and
static and shared libraries. Compute and CPU target Linux, Windows and Wasm;
CUDA targets Linux and Windows. See [Using the Compute SDKs](SDK.md) for entry
points, package selection and independent consumption.

Download the [Compute 0.1.0 SDK bundle](https://github.com/tetrodotoxin-dev/Compute/releases/tag/v0.1.0)
for the shared contracts and both engine implementations.

## Building and testing

Python 3 and Bazel are required. The Bazel version is specified in
[.bazelversion](.bazelversion), and [MODULE.bazel](MODULE.bazel) declares the
Toolchain dependency that supplies the compilers and shared build rules.
Dependencies are acquired from their declared releases.

Run the CPU tests independently of CUDA:

```sh
bazel run :cpu_tests --config=release
```

The complete source build and test suite include CUDA. Bazel acquires NVIDIA's
headers and compiler libraries; GPU execution requires the host NVIDIA driver
and a compatible GPU. Run these commands from the repository root:

```sh
bazel run :format -- --check
bazel build //source/...
bazel run :tests --config=release
bazel run :benchmarks --config=release
bazel build :sdk --config=release
bazel run :snapshot
```

Tests and benchmarks are standalone executables. `:sdk` builds the complete
bundle. Use `:compute_sdk`, `:cpu_sdk` or `:cuda_sdk` to package one module. A
snapshot archives the clean Git commit that reproduces the source and
validation state.

For VS Code tasks, debugging and header navigation, run `bazel run :vscode`.

Compute is available under the [MIT license](LICENSE). NVIDIA dependencies
retain their own licenses.
