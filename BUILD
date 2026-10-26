# Copyright (c) 2023-present Matt Kaes and contributors

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@tetro_toolchain//:defs.bzl", "format", "package", "snapshot", "validation", "vscode")

package(
    linkage = "shared",
    module = "compute",
    shared = "//source:compute_shared",
    static = "//source:compute_static",
)

package(
    name = "cuda_sdk",
    components = {
        "contracts": [],
        "runtime": ["contracts"],
    },
    linkage = "shared",
    module = "cuda",
    platforms = [
        "linux-x86_64-v3",
        "windows-x86_64-msvc",
    ],
    shared = "//source:cuda_shared",
    static = "//source:cuda_static",
)

format()

snapshot()

validation(
    test_data = ["//source:cuda_shared"],
    test_deps = [
        ":cuda",
        "//build:driver",
        "@perimortem//:system",
    ],
    deps = [":compute"],
)

vscode()

cc_binary(
    name = "cpu_tests",
    testonly = True,
    srcs = [
        "validation/tests/compute.cpp",
        "validation/tests/generated.c",
        "validation/tests/generated.h",
    ],
    deps = [
        ":compute",
        "@tetro_toolchain//source/toolchain/validation:test",
    ],
)
