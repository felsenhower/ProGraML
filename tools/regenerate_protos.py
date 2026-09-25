#!/usr/bin/env python3
# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regenerate the Python protobuf bindings with a modern protoc.

The C++ side of ProGraML is built with Bazel against an old, pinned protobuf
version for compatibility with the rest of the native toolchain. Since
protobuf's wire format is stable across versions, and the two sides only ever
exchange serialized bytes (never protobuf library objects), the Python
`*_pb2.py` files can safely be generated independently by a modern `protoc`.
This lets the `protobuf` pip dependency be a current release without touching
the Bazel/C++ build at all.

Usage: tools/regenerate_protos.py [dest_root]

`dest_root` is the directory that the "programl/..." proto paths are rooted
at. Defaults to the repository root (useful for local development); the
release build instead points this at the Bazel py_package runfiles tree so
that the freshly generated files overwrite the ones Bazel produced.
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PROTO_FILES = [
    "programl/proto/program_graph.proto",
    "programl/proto/util.proto",
    "programl/third_party/tensorflow/features.proto",
    "programl/third_party/tensorflow/xla.proto",
]


def _protoc_command() -> list:
    try:
        import grpc_tools.protoc  # noqa: F401

        return [sys.executable, "-m", "grpc_tools.protoc"]
    except ImportError:
        protoc = shutil.which("protoc")
        if not protoc:
            raise SystemExit(
                "No protoc found. Install grpcio-tools (`pip install "
                "grpcio-tools`) or a protoc binary before running this script."
            )
        return [protoc]


def main(dest_root: Path) -> None:
    command_prefix = _protoc_command()
    for proto_file in PROTO_FILES:
        subprocess.run(
            command_prefix
            + [
                f"--proto_path={REPO_ROOT}",
                f"--python_out={dest_root}",
                proto_file,
            ],
            cwd=REPO_ROOT,
            check=True,
        )


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT
    main(dest)
