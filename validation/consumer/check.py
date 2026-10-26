# Copyright (c) 2023-present Matt Kaes and contributors

"""Build Compute and CUDA SDKs and consume them from a separate workspace.

Run `python3 validation/consumer/check.py` from a source checkout. Additional
Bazel flags are forwarded to each build, allowing an explicitly approved local
Toolchain override during development. Perimortem and TTX keep their source
module's published pins. Linux consumers execute the C lifetime and GPU checks;
Windows consumers establish compilation and linkage.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


def check():
    repository = Path(__file__).resolve().parents[2]
    fixtures = repository / "validation/consumer"
    flags = ["--config=release", *sys.argv[1:]]
    subprocess.run(["bazel", "build", ":sdk", ":cuda_sdk", *flags], cwd=repository, check=True)
    directory = Path(subprocess.check_output(
        ["bazel", "info", "bazel-bin", *flags], cwd=repository, text=True).strip())
    source_module = (repository / "MODULE.bazel").read_text()
    version = re.search(r'version\s*=\s*"([^"]+)"', source_module).group(1)
    prefix = "cuda-" + version + "-"
    packages = {}
    for package in ["cuda", "compute"]:
        package_prefix = package + "-" + version + "-"
        pins = {}
        for line in (directory / (package_prefix + "sha256.txt")).read_text().splitlines():
            digest, name = line.split()
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError("SDK checksum mismatch: " + name)
            pins[name.removeprefix(package_prefix).removesuffix(".zip")] = digest
        packages[package] = pins

    # The consumer uses binary SDKs. NVIDIA source acquisition stays with the
    # producer; its link interfaces and runtime dependencies come from the SDK.
    dependencies = source_module.split('cuda = use_extension("//build:sdk.bzl"', 1)[0]
    dependencies = re.sub(r"\Amodule\([\s\S]*?\)",
                          'module(name="cuda_consumer")', dependencies, count=1)
    with tempfile.TemporaryDirectory(prefix="cuda-sdk-consumer-") as temporary:
        root = Path(temporary)
        workspace = root / "consumer"
        (workspace / "validation/tests").mkdir(parents=True)
        (workspace / "validation/consumer").mkdir()
        for name in [".bazelrc", ".bazelversion"]:
            shutil.copyfile(repository / name, workspace / name)
        shutil.copyfile(fixtures / "BUILD.tpl", workspace / "BUILD")
        for name in ["consumer.c", "consumer.h"]:
            shutil.copyfile(repository / "validation/tests" / name,
                            workspace / "validation/tests" / name)
        for name in ["consumer.c", "consumer.cpp", "compute.cpp"]:
            shutil.copyfile(fixtures / name, workspace / "validation/consumer" / name)
        command = ["bazel", "--batch", "--output_base=" + str(root / "bazel")]
        targets = [":c_consumer", ":cpp_consumer", ":compute_consumer"]
        for linkage in ["static", "shared"]:
            module = dependencies + (fixtures / "MODULE.tpl").read_text()
            for key, value in {"archives": repr(packages["cuda"]), "compute_archives": repr(packages["compute"]), "version": version,
                               "linkage": linkage, "suffix": "_static" if linkage == "static" else ""}.items():
                module = module.replace("{" + key + "}", value)
            (workspace / "MODULE.bazel").write_text(module)
            options = [*flags, "--distdir=" + str(directory)]
            subprocess.run([*command, "build", *targets, *options], cwd=workspace, check=True)
            output = Path(subprocess.check_output(
                [*command, "info", "bazel-bin", *options], cwd=workspace, text=True).strip())
            for target in targets:
                binary = output / target.removeprefix(":")
                subprocess.run([str(binary)], cwd=workspace, check=True)
                dynamic = subprocess.check_output(["readelf", "-d", str(binary)], text=True)
                required = re.findall(r"\(NEEDED\)[^\n]*\[([^\]]+)\]", dynamic)
                if linkage == "static" and any(name in required for name in
                                               ["libcompute.so", "libttx_cuda.so", "libttx.so", "libperimortem.so"]):
                    raise ValueError("Static consumer requires a first-party shared library")
                if linkage == "shared" and "libttx_cuda.so" not in required:
                    raise ValueError("Shared consumer must use the CUDA shared library")
                print(f"{linkage} {binary.name}: {binary.stat().st_size:,} bytes; "
                      f"dependencies: {', '.join(required)}", flush=True)
            if linkage == "shared":
                shutil.copyfile(output / "c_consumer", root / "shared-consumer")
            subprocess.run([*command, "build", *targets, *options,
                            "--platforms=@tetro_toolchain//:windows_x64"], cwd=workspace, check=True)
            print(linkage + " SDK consumers built for Windows.", flush=True)

        # Runtime closure is also checked after extraction, outside Bazel's
        # library paths. The CUDA driver remains supplied by the host.
        relocated = root / "relocated"
        with zipfile.ZipFile(directory / (prefix + "linux-x86_64-v3-shared.zip")) as archive:
            archive.extractall(relocated)
        metadata = json.loads((relocated / "sdk.json").read_text())
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("LD_", "RUNFILES_"))}
        binary = relocated / "lib/consumer"
        shutil.copyfile(root / "shared-consumer", binary)
        binary.chmod(0o755)
        environment["LD_LIBRARY_PATH"] = str(binary.parent)
        environment["LD_DEBUG"] = "libs"
        result = subprocess.run([str(binary)], cwd=relocated, env=environment,
                                capture_output=True, text=True, check=True)
        loaded = {Path(path.strip()).resolve()
                  for path in re.findall(r"calling init: ([^\n]+)", result.stderr)}
        for library in metadata["runtime"]:
            path = (relocated / library["library"]).resolve()
            if path not in loaded:
                raise ValueError("Relocated consumer did not load its packaged " + path.name)
        print("Relocated CUDA compilation and retained storage passed using packaged runtimes.", flush=True)


if __name__ == "__main__":
    check()
