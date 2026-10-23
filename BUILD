# Copyright (c) 2023-present Matt Kaes and contributors

load("@tetro_toolchain//:defs.bzl", "format", "package", "snapshot", "validation", "vscode")

package(
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
        "//build:driver",
        "@perimortem//:system",
    ],
)

vscode()
