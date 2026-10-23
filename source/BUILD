# Copyright (c) 2023-present Matt Kaes and contributors

load("@tetro_toolchain//:defs.bzl", "WINDOWS", "library")

package(default_visibility = ["//visibility:public"])

_COMPONENTS = {
    "contracts": [],
    "runtime": ["contracts"],
}

_HEADERS = glob([
    "cuda/**/*.h",
    "cuda/**/*.hpp",
])

_SOURCES = glob(["cuda/**/*.cpp"])

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
        "//build:driver_headers",
        "@perimortem//:memory",
        "@ttx//:graph",
    ],
)
