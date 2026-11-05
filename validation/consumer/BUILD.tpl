# Copyright (c) 2023-present Matt Kaes and contributors

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

cc_binary(
    name = "cpu_c_consumer",
    srcs = ["validation/consumer/bootstrap.c"],
    defines = ["COMPUTE_CPU_PROVIDER=1"],
    deps = [
        "@compute",
        "@cpu",
    ],
)

cc_binary(
    name = "cpu_cpp_consumer",
    srcs = ["validation/consumer/typed.cpp"],
    defines = [
        "COMPUTE_CPU_PROVIDER=1",
        "COMPUTE_RUN_ENGINE=1",
    ],
    deps = [
        "@compute",
        "@cpu",
    ],
)

cc_binary(
    name = "cuda_c_consumer",
    srcs = ["validation/consumer/bootstrap.c"],
    deps = [
        "@compute",
        "@cuda",
    ],
)

cc_binary(
    name = "cuda_cpp_consumer",
    srcs = ["validation/consumer/typed.cpp"],
    deps = [
        "@compute",
        "@cuda",
    ],
)
