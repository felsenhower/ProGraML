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
"""Vendor the build machine's C++ standard library headers into the wheel.

clang2graph/llvm2graph are built against a fixed, old LLVM/Clang release for
portability. If asked to parse C++ using the *host* system's include search
path (as `programl.util.py.cc_system_includes.get_system_includes()` does
when no vendored headers are present), a much newer host libstdc++ may use
syntax this old Clang frontend cannot parse. To make graph construction work
regardless of the end user's installed compiler, this script captures a
known-compatible header set once, from the same environment the native
binaries are built in, so it can be bundled into the wheel.

Usage: tools/vendor_stdlib_headers.py [compiler] [dest]
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

DEFAULT_DEST = Path(
    "bazel-bin/py_package.runfiles/programl/third_party/libstdcxx_headers"
)


def _discover_include_dirs(compiler: str) -> List[Path]:
    with tempfile.TemporaryDirectory() as d:
        process = subprocess.run(
            [compiler, "-xc++", "-v", "-c", "-", "-o", str(Path(d) / "a.out")],
            input="",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    if process.returncode:
        raise SystemExit(
            f"Failed to invoke {compiler} to discover include paths:\n"
            f"{process.stderr}"
        )

    dirs = []
    in_search_list = False
    for line in process.stderr.split("\n"):
        if in_search_list and line.startswith("End of search list"):
            break
        elif in_search_list:
            dirs.append(Path(line.strip()))
        elif line.startswith("#include <...> search starts here:"):
            in_search_list = True
    if not dirs:
        raise SystemExit(f"Failed to parse include paths from {compiler} output")
    return dirs


def main(compiler: str, dest: Path) -> None:
    if dest.is_dir():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # Directory names must sort in include-search order: get_system_includes()
    # reads them back with a plain directory listing.
    for i, src_dir in enumerate(_discover_include_dirs(compiler)):
        shutil.copytree(src_dir, dest / f"{i:02d}", symlinks=True)


if __name__ == "__main__":
    compiler = sys.argv[1] if len(sys.argv) > 1 else "g++"
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DEST
    main(compiler, dest)
