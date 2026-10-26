# Compute

Compute connects independently implemented execution engines through TTX.
Consumers navigate an Abstract graph and bind the domain contracts they
understand. Providers control their code, storage and execution platform.
The same operation can therefore be supplied by a host CPU, CUDA or another
implementation that fulfills its complete behavioral contract.

Engine creation and selection happen before execution. Required policies apply
to every candidate, including a fallback provider. Retained results keep their
required resources available independently of the factory or engine acquisition
that produced them.

The shared image domain currently provides RGBA8 snapshots and inversion.
CPU and CUDA implementations consume image Abstracts through the same contract.
The CUDA platform also exposes source compilation, prepared kernels and device
buffers for consumers that explicitly need those operations.

## Building and testing

Install Python 3 and the Bazel version in [.bazelversion](.bazelversion).
Bazel acquires the declared Toolchain, Perimortem and TTX SDKs. CUDA targets
also acquire NVIDIA headers and compiler libraries; executing them requires
the host NVIDIA driver and a compatible GPU.

```sh
bazel run :cpu_tests --config=release
bazel run :tests --config=release
bazel run :benchmarks --config=release
bazel run :format -- --check
bazel build :sdk --config=release
bazel build :cuda_sdk --config=release
```

The Compute library and CPU tests have no NVIDIA runtime dependency. The full
test suite also executes CUDA. `:sdk` packages the Compute library for Linux,
Windows and Wasm; `:cuda_sdk` packages the optional CUDA platform for Linux and
Windows. Both packages contain static and shared variants.
