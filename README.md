# Compute

Compute supplies execution domains and their platform implementations for the
Tetrodotoxin family of projects. It builds on [TTX](https://github.com/tetrodotoxin-dev/TTX)
so applications can request an operation through a shared contract while each
provider controls how the work executes and where its data lives.

An application often needs to combine work performed by several systems: a
host implementation, an accelerator, or a provider supplied by another project.
Compute gives those systems common capabilities and policies to negotiate.
Applications compose the operations they need, and providers can substitute
different implementations while preserving the requested behavior.

## Compute domains

A domain defines what an operation means and the guarantees required to use
its inputs and results. Its contracts are independent of the platform that
implements them. New operations and policies can extend a domain through TTX's
existing graph and binding model.

The current image domain supplies RGBA8 snapshots through
[Pixels](source/compute/image/pixels.h) and color inversion through
[Invert](source/compute/image/invert.h). An image provider can store its pixels,
generate them on demand, or transfer them from a device. Both CPU and CUDA
engines consume that same interface and produce independently retained results.

Policies express additional requirements separately from operations.
[Accelerated](source/compute/policies/accelerated.h) promises execution on a
compute device separate from the host CPU. A consumer can require that promise
without identifying a particular vendor or assuming a performance advantage.

## Providers and composition

Engine factories are Abstracts that supply TTX's Create capability. Creation
receives an argument graph and supplies an engine through a callback. A consumer
can use the engine during that callback or retain it for later operations.
Results retain the resources they need independently of the engine acquisition
that produced them.

A consumer can choose a factory directly or compose several factories with
[Selection](source/compute/execution/selection.hpp). Selection tries candidates
before execution and applies the required policy to every candidate, including
fallbacks. It publishes the same Create capability as an individual factory.

The CPU and CUDA providers are separate libraries. Loading a provider establishes
its TTX interface; creating an engine establishes the resources for a request.
The host keeps each provider library loaded through the final use and release
of its interfaces. Providers can expose additional routes and contracts through
their graphs while consumers use the capabilities they understand.

## SDKs

The repository builds two SDK packages for independently developed consumers:

| Package | Provides | Targets |
| --- | --- | --- |
| Compute | Shared image contracts, selection, host image storage and the CPU provider | Linux, Windows, Wasm |
| CUDA | The CUDA image provider, source compilation, prepared kernels and device buffers | Linux, Windows |

Both packages contain C contract headers, C++ interfaces, and static and shared
libraries. The Compute runtime depends on TTX and
[Perimortem](https://github.com/tetrodotoxin-dev/Perimortem). CUDA uses the shared
Compute contracts and supplies its own runtime implementation. The CPU package
can run independently of NVIDIA libraries.

Consumers that need CUDA-specific behavior can use its compilation and device
contracts directly, or the native C++ runtime with NVIDIA's headers. Portable
operations such as Invert have an implementation for each supported provider;
CUDA source compilation remains a CUDA capability.

See [Using the Compute SDKs](SDK.md) for the module entry points, lifetime
requirements and package layout. The Compute 0.4 SDKs are under development;
existing releases preserve the earlier CUDA-only interface.

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
bazel build :cuda_sdk --config=release
bazel run :snapshot
```

Tests and benchmarks are standalone executables. `:sdk` packages Compute, and
`:cuda_sdk` packages the optional CUDA platform. A snapshot archives the clean
Git commit that reproduces the source and validation state.

For VS Code tasks, debugging and header navigation, run `bazel run :vscode`.

Compute is available under the [MIT license](LICENSE). NVIDIA dependencies
retain their own licenses.
