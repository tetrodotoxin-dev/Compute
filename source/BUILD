# Copyright (c) 2023-present Matt Kaes and contributors

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@tetro_toolchain//:defs.bzl", "WINDOWS", "library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "compute_headers",
    hdrs = glob([
        "compute/image/*.h",
        "compute/policies/*.h",
    ]) + [
        "compute/image/invert.hpp",
        "compute/image/pixels.hpp",
    ],
    includes = ["."],
)

library(
    name = "compute_static",
    srcs = glob([
        "compute/**/*.cpp",
        "compute/**/*.c",
    ]),
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
    srcs = glob([
        "compute/**/*.cpp",
        "compute/**/*.c",
    ]),
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
