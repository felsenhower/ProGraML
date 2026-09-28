# This package implements support for Futhark.
#
# Copyright 2019-2026 the ProGraML authors.
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

py_library(
    name = "futhark",
    srcs = [
        "__init__.py",
        "futhark.py",
        "ir_parser.py",
        "program_graph_builder.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//programl:exceptions",
        "//programl/proto",
    ],
)
