# Copyright (c) 2023-present Matt Kaes and contributors

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@tetro_toolchain//:defs.bzl", "LINUX")

cc_binary(
    name = "c_consumer",
    srcs = [
        "validation/consumer/consumer.c",
        "validation/tests/consumer.c",
        "validation/tests/consumer.h",
    ],
    linkopts = select({
        LINUX: ["-ldl"],
        "//conditions:default": [],
    }),
    deps = ["@cuda"],
)

cc_binary(
    name = "cpp_consumer",
    srcs = ["validation/consumer/consumer.cpp"],
    deps = ["@cuda"],
)

cc_binary(
    name = "cpu_consumer",
    srcs = [
        "validation/tests/compute.cpp",
        "validation/tests/deferred.c",
        "validation/tests/deferred.h",
        "validation/tests/generated.c",
        "validation/tests/generated.h",
    ],
    deps = [
        "@compute",
        "@cpu",
        "@tetro_toolchain//source/toolchain/validation:test",
    ],
)

cc_binary(
    name = "compute_consumer",
    srcs = ["validation/consumer/compute.cpp"],
    deps = [
        "@compute",
        "@cpu",
        "@cuda",
    ],
)
