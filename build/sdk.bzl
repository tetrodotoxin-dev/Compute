# # Tetrodotoxin
# Copyright (c) 2023-present Matt Kaes and contributors

"""Acquire the CUDA 13.3 components used by the selected target."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Versions and Linux/Windows checksums come from redistrib_13.3.0.json.
# Driver headers and link libraries come from cudart. The actual driver is
# supplied by the machine running the program.
_PACKAGES = {
    "cuda_crt": ("13.3.33", "4755d36d24c6ef7697a2d3e1dbb23c4562c9c0d97d48390d4cbd8ab32dec5b5f", "752c528281a06a0ddf89237d760ffd6acde1b9cd59efc35803c2591127ef55f0"),
    "cuda_cudart": ("13.3.29", "1e59c4888267d27ba1a9bd0f3669a6439db1334a96e754cd9013c7c73e18dc9d", "1feb7dd266813ffe8dbc24e115183a5ac35a4795c8d34aca0df85ab616b64d9c"),
    "cuda_nvrtc": ("13.3.33", "9e8f78278215babd1236b137252424ca7912c185bd093201f5d97f7dd763b74a", "8519f678588610bf380ccaac130729aa1a624c407183e7ad9c319c19ecc63d2f"),
    "libcufft": ("12.3.0.29", "b2404952a5d630fbbc13e12d36975ef87a9c05c71c321c2cb5b3820789e47462", "83df908ae67e2b3a86201de8463562ab49dd9ee8b3b5efc3fdc2e681b14b5dd9"),
}

_BUILD = """load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
package(default_visibility = ["//visibility:public"])
exports_files(["LICENSE"])
cc_library(name = "headers", hdrs = glob(["include/**/*.h"]), includes = ["include"], deps = %s)
"""

# Bazel supplies the extension context. Package selection uses the pinned
# target inventory independently of that context.
# buildifier: disable=unused-variable
def _dependencies(ctx):
    """Declare independent NVIDIA repositories for each native target.

    Args:
        ctx: Module extension context. Package versions come from _PACKAGES.
    """
    for component, (version, linux_sha, windows_sha) in _PACKAGES.items():
        for platform, checksum in [("linux", linux_sha), ("windows", windows_sha)]:
            windows = platform == "windows"
            headers = []
            if component == "cuda_cudart":
                headers = ["@cuda_crt_" + platform + "//:headers"]
            elif component == "libcufft":
                headers = ["@cuda_cudart_" + platform + "//:headers"]
            build = _BUILD % repr(headers)
            library = None
            runtime = None
            if component == "cuda_cudart":
                library = 'static_library = "lib/x64/cuda.lib"' if windows else 'interface_library = "lib/stubs/libcuda.so", system_provided = True'
            elif component == "cuda_nvrtc":
                library = 'shared_library = "' + ("bin/x64/nvrtc64_130_0.dll" if windows else "lib/libnvrtc.so.13") + '"'
                if windows:
                    library += ', interface_library = "lib/x64/nvrtc.lib"'
                runtime = "bin/x64/nvrtc-builtins64_133.dll" if windows else "lib/libnvrtc-builtins.so.13.3"
            elif component == "libcufft":
                library = 'shared_library = "' + ("bin/x64/cufft64_12.dll" if windows else "lib/libcufft.so.12") + '"'
                if windows:
                    library += ', interface_library = "lib/x64/cufft.lib"'
            if library:
                deps = [":headers"]
                if runtime:
                    # NVRTC loads builtins during compilation. A shared library
                    # dependency carries that version through linking and DLL staging.
                    build += 'cc_import(name = "builtins", shared_library = ' + repr(runtime) + ")\n"
                    deps.append(":builtins")
                build += 'cc_import(name = "library", ' + library + ", deps = " + repr(deps) + ")\n"
            target = platform + "-x86_64"
            directory = component + "-" + target + "-" + version + "-archive"
            http_archive(
                name = component + "_" + platform,
                urls = ["https://developer.download.nvidia.com/compute/cuda/redist/" + component + "/" + target + "/" + directory + (".zip" if windows else ".tar.xz")],
                sha256 = checksum,
                strip_prefix = directory,
                build_file_content = build,
            )

# Separate repositories keep Windows archives and optional cuFFT lazy. Merely
# importing this extension does not download components a target never uses.
dependencies = module_extension(implementation = _dependencies)
