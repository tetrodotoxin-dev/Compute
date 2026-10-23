#!/usr/bin/env bash
# Copyright (c) 2023-present Matt Kaes and contributors
set -euo pipefail
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
exec bazel run :format -- "$@"
