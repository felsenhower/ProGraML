#!/usr/bin/env python3
#
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

import distutils.util
import os

import setuptools


# When building a bdist_wheel we need to set the appropriate tags: this package
# includes compiled binaries, and does not include compiled python extensions.
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False

        def get_tag(self):
            python, abi, plat = _bdist_wheel.get_tag(self)
            python, abi = "py3", "none"
            return python, abi, plat


except ImportError:
    bdist_wheel = None

_PACKAGE_ROOT = "bazel-bin/py_package.runfiles/programl"

_PROGRAML_PKG_DIR = os.path.join(_PACKAGE_ROOT, "programl")

_libstdcxx_headers_dir = os.path.join(_PROGRAML_PKG_DIR, "third_party/libstdcxx_headers")
_libstdcxx_headers = [
    os.path.relpath(os.path.join(root, name), _PROGRAML_PKG_DIR)
    for root, _, names in os.walk(_libstdcxx_headers_dir)
    for name in names
]

setuptools.setup(
    packages=[
        "programl.ir.futhark",
        "programl.ir.llvm",
        "programl.proto",
        "programl.third_party.inst2vec",
        "programl.third_party.tensorflow",
        "programl.util.py",
        "programl",
    ],
    package_dir={
        "": _PACKAGE_ROOT,
    },
    package_data={
        "programl": [
            "bin/*",
            "ir/llvm/internal/*.pickle",
            *_libstdcxx_headers,
        ],
    },
    include_package_data=True,
    cmdclass={"bdist_wheel": bdist_wheel},
    platforms=[distutils.util.get_platform()],
    zip_safe=False,
)
