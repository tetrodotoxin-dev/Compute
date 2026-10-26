# Compute

Compute provides a [TTX](https://github.com/tetrodotoxin-dev/TTX) SDK for compute
infrastructure used by Tetrodotoxin projects. Its purpose is to support
Tetrodotoxin's image acceleration but it also provides a practical template for
wrapping existing infrastructure with TTX.

Applications request operations through shared TTX contracts. Providers can
substitute different implementations while preserving the requested behavior.
Results expose access through their own contracts so applications can use data
independently of where it is stored or produced.

The [shared SDK](source/compute) is independent of its included
[CPU](source/cpu) and [CUDA](source/cuda) engines. Each engine is a separate
module implementing the same contracts. Other systems can provide their own
implementations or extend the domain through TTX. See the [SDK guide](SDK.md)
for usage and the [releases](https://github.com/tetrodotoxin-dev/Compute/releases)
for packaged libraries.

## Building and testing

Python 3 and the Bazel version in [.bazelversion](.bazelversion) are required.
Bazel acquires the build dependencies declared in [MODULE.bazel](MODULE.bazel).

Run these commands from the repository root:

```sh
bazel run :format -- --check
bazel build //source/...
bazel run :tests
bazel run :benchmarks --config=release
bazel build :sdk --config=release
bazel run :snapshot
```

The full test suite requires a compatible NVIDIA GPU and driver. Run the CPU
suite independently with `bazel run :cpu_tests`.

Tests and benchmarks are standalone executables. The SDK bundles each module
for independent consumption. A snapshot archives the clean Git commit.

Generate the VS Code configuration with `bazel run :vscode`.

Compute is available under the [MIT license](LICENSE). NVIDIA dependencies
retain their own licenses.
