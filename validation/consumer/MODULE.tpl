# Copyright (c) 2023-present Matt Kaes and contributors

sdks.release(
    name = "cuda",
    archives = {archives},
    linkage = "{linkage}",
    project = "CUDA",
    version = "{version}",
    deps = [
        "@perimortem{suffix}//:core",
        "@perimortem{suffix}//:memory",
        "@ttx{suffix}//:graph",
    ],
)
use_repo(sdks, "cuda")
