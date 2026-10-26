# Copyright (c) 2023-present Matt Kaes and contributors

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@tetro_toolchain//:defs.bzl", "WINDOWS", "library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "compute_headers",
    hdrs = glob([
        "compute/execution/commands.h",
        "compute/execution/commands.hpp",
        "compute/execution/dependency.h",
        "compute/execution/dependency.hpp",
        "compute/execution/status.h",
        "compute/image/*.h",
        "compute/image/*.hpp",
        "compute/policies/*.h",
    ]),
    includes = ["."],
)

library(
    name = "compute_static",
    srcs = glob(["compute/**/*.cpp"]),
    hdrs = glob([
        "compute/**/*.h",
        "compute/**/*.hpp",
    ]),
    includes = ["."],
    module = "compute",
    deps = [
        "@perimortem_static//:memory",
        "@ttx_static//:graph",
    ],
)

library(
    name = "compute_shared",
    srcs = glob(["compute/**/*.cpp"]),
    hdrs = glob([
        "compute/**/*.h",
        "compute/**/*.hpp",
    ]),
    includes = ["."],
    linkage = "shared",
    module = "compute",
    shared_lib_name = select({
        WINDOWS: "compute.dll",
        "//conditions:default": "libcompute.so",
    }),
    deps = [
        "@perimortem//:memory",
        "@ttx//:graph",
    ],
)

library(
    name = "cpu_static",
    srcs = glob([
        "cpu/**/*.cpp",
        "cpu/**/*.c",
    ]),
    hdrs = glob([
        "cpu/**/*.h",
        "cpu/**/*.hpp",
    ]),
    includes = ["."],
    module = "cpu",
    deps = [
        ":compute_headers",
        "@perimortem_static//:memory",
        "@ttx_static//:graph",
    ],
)

library(
    name = "cpu_shared",
    srcs = glob([
        "cpu/**/*.cpp",
        "cpu/**/*.c",
    ]),
    hdrs = glob([
        "cpu/**/*.h",
        "cpu/**/*.hpp",
    ]),
    includes = ["."],
    linkage = "shared",
    module = "cpu",
    shared_lib_name = select({
        WINDOWS: "ttx_cpu.dll",
        "//conditions:default": "libttx_cpu.so",
    }),
    deps = [
        ":compute_headers",
        "@perimortem//:memory",
        "@ttx//:graph",
    ],
)

_COMPONENTS = {
    "contracts": [],
    "runtime": ["contracts"],
}

_HEADERS = glob([
    "cuda/**/*.h",
    "cuda/**/*.hpp",
])

_SOURCES = glob([
    "cuda/**/*.cpp",
    "cuda/**/*.c",
])

library(
    name = "cuda_static",
    srcs = _SOURCES,
    hdrs = _HEADERS,
    component_deps = {
        "contracts": ["@ttx//:graph"],
        "runtime": ["//build:driver_headers"],
    },
    components = _COMPONENTS,
    implementation_deps = ["//build:compute"],
    includes = ["."],
    module = "cuda",
    deps = [
        ":compute_headers",
        "//build:driver_headers",
        "@perimortem_static//:memory",
        "@ttx_static//:graph",
    ],
)

library(
    name = "cuda_shared",
    srcs = _SOURCES,
    hdrs = _HEADERS,
    implementation_deps = ["//build:compute"],
    includes = ["."],
    linkage = "shared",
    module = "cuda",
    shared_lib_name = select({
        WINDOWS: "ttx_cuda.dll",
        "//conditions:default": "libttx_cuda.so",
    }),
    deps = [
        ":compute_headers",
        "//build:driver_headers",
        "@perimortem//:memory",
        "@ttx//:graph",
    ],
)
