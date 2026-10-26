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
    name = "compute_consumer",
    srcs = ["validation/consumer/compute.cpp"],
    deps = [
        "@compute",
        "@cuda",
    ],
)
