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
"""Construct Program Graphs from Futhark source code.

Unlike the LLVM/clang/XLA frontends, this does not link against the
compiler's internals. Futhark does not expose its internal representation as
a library API or a machine-readable serialization, so this shells out to the
``futhark`` command line tool (an unmodified, ordinary dependency) and parses
the pretty-printed IR that it writes to stdout.

Pipeline stage: ``futhark dev --standard`` prints the "Core IR" after
Futhark's full simplification/fusion/inlining/defunctionalisation pipeline
has run, but *before* the pipeline forks into a backend-specific
representation (``--gpu``, ``--mc``, ``--seq``, and their ``-mem`` variants,
which lower SOACs into explicit kernels/loops). This is the last point at
which the program is backend-agnostic while still being expressed in terms of
high-level parallel operations (``map``, ``reduce``, ``redomap``, ``scan``,
...) rather than loops -- which is the representation we want the graph to
capture (see :mod:`programl.ir.futhark.program_graph_builder`).
"""
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from programl.exceptions import GraphCreationError
from programl.ir.futhark import ir_parser
from programl.ir.futhark.program_graph_builder import build_program_graph
from programl.proto import ProgramGraph

FUTHARK_BINARY = "futhark"


def program_graph_from_futhark_ir(ir_text: str, module_name: str = "futhark") -> ProgramGraph:
    """Parse the text produced by ``futhark dev`` into a :code:`ProgramGraph`.

    :param ir_text: The stdout of a ``futhark dev`` invocation.

    :param module_name: The name to give the resulting graph's module.

    :raises GraphCreationError: If the text cannot be parsed.
    """
    try:
        program = ir_parser.parse(ir_text)
    except ir_parser.FutharkIRParseError as e:
        raise GraphCreationError(f"Failed to parse Futhark IR: {e}") from e
    return build_program_graph(program, module_name=module_name)


def from_futhark(
    src: str,
    entry_points: Optional[List[str]] = None,
    stage: str = "--standard",
    module_name: str = "futhark",
    futhark_binary: str = FUTHARK_BINARY,
    timeout: int = 300,
) -> ProgramGraph:
    """Construct a Program Graph from a string of Futhark code.

    This requires the ``futhark`` compiler to be installed and on ``PATH``.
    It is invoked as an ordinary compiler dependency (via
    :code:`futhark dev`) -- this project does not modify or link against
    Futhark itself.

    For example:

        >>> programl.from_futhark(\"\"\"
        ... def fact (n: i32): i32 = reduce (*) 1 (1...n)
        ...
        ... def main (n: i32): i32 = fact n
        ... \"\"\")

    :param src: A string of Futhark source code.

    :param entry_points: Additional entry points to compile, beyond those
        already declared with the ``entry`` keyword in :code:`src` (and
        :code:`main`, which is treated as an entry point by convention even
        without the keyword).

    :param stage: The ``futhark dev`` pipeline stage flag to intercept.
        Defaults to :code:`--standard`, the fully-optimized backend-agnostic
        Core IR. See ``futhark dev --help`` for the full list of stages.

    :param module_name: The name to give the resulting graph's module.

    :param futhark_binary: The name (or path) of the Futhark compiler
        executable.

    :param timeout: The maximum number of seconds to wait for the ``futhark``
        invocation before raising an error.

    :return: A :code:`programl.ProgramGraph` instance.

    :raises GraphCreationError: If compilation or graph construction fails.

    :raises TimeoutError: If the specified timeout is reached.
    """
    with tempfile.TemporaryDirectory(prefix="programl_futhark_") as tmpdir:
        src_path = Path(tmpdir) / "input.fut"
        src_path.write_text(src, encoding="utf-8")

        args = [futhark_binary, "dev", stage]
        for entry_point in entry_points or []:
            args.append(f"--entry-points={entry_point}")
        args.append(str(src_path))

        try:
            process = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except FileNotFoundError as e:
            raise GraphCreationError(
                f"Could not find the `{futhark_binary}` executable. "
                "programl.from_futhark() requires the Futhark compiler to "
                "be installed: https://futhark.readthedocs.io/"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(str(e)) from e

        if process.returncode:
            raise GraphCreationError(process.stderr.decode("utf-8", errors="replace"))

        ir_text = process.stdout.decode("utf-8")

    return program_graph_from_futhark_ir(ir_text, module_name=module_name)
