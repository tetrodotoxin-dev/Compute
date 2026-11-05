# Copyright (c) 2023-present Matt Kaes and contributors

sdks.release(
    name = "compute",
    archives = {compute_archives},
    library = "compute",
    linkage = "{linkage}",
    project = "Compute",
    version = "{version}",
    deps = [
        {coherence_dependency}
        "{perimortem}",
        "{ttx}",
    ],
)
use_repo(sdks, "compute")

sdks.release(
    name = "cpu",
    archives = {cpu_archives},
    library = "cpu",
    linkage = "{linkage}",
    project = "Compute",
    version = "{version}",
    deps = ["@compute//:compute"],
)
use_repo(sdks, "cpu")

sdks.release(
    name = "cuda",
    archives = {archives},
    library = "cuda",
    linkage = "{linkage}",
    project = "Compute",
    version = "{version}",
    deps = ["@compute//:compute"],
)
use_repo(sdks, "cuda")
