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
import shutil

import pytest

import programl as pg
from programl.proto import Edge, Node
from tests.test_main import main

# The Futhark compiler is an external tool, not built by this project, so
# skip these tests rather than fail outright when it isn't installed.
pytestmark = pytest.mark.skipif(
    shutil.which("futhark") is None, reason="futhark compiler not found on PATH"
)

FACT = """
def fact (n: i32): i32 = reduce (*) 1 (1...n)

def main (n: i32): i32 = fact n
"""


def test_from_invalid_futhark():
    with pytest.raises(pg.GraphCreationError):
        pg.from_futhark("this is not valid futhark syntax !!!")


def test_fact_input():
    graph = pg.from_futhark(FACT)
    assert isinstance(graph, pg.ProgramGraph)


def test_fact_has_entry_function():
    graph = pg.from_futhark(FACT)
    assert "entry_main" in {f.name for f in graph.function}


def test_fact_contains_redomap_instruction():
    """`reduce` lowers to a `redomap` SOAC at the --standard IR stage."""
    graph = pg.from_futhark(FACT)
    instruction_texts = {
        n.text for n in graph.node if n.type == Node.INSTRUCTION
    }
    assert "redomap" in instruction_texts


def test_fact_redomap_has_call_edges_to_lambdas():
    """The redomap SOAC's map/reduce operators become their own functions."""
    graph = pg.from_futhark(FACT)
    redomap_index = next(
        i for i, n in enumerate(graph.node) if n.type == Node.INSTRUCTION and n.text == "redomap"
    )
    call_targets = {
        e.target for e in graph.edge if e.flow == Edge.CALL and e.source == redomap_index
    }
    assert len(call_targets) == 2


def test_fact_graph_round_trips_through_serialization():
    graph = pg.from_futhark(FACT)
    also_graph = pg.ProgramGraph()
    also_graph.ParseFromString(graph.SerializeToString())
    assert also_graph == graph


def test_if_else_produces_diamond_control_flow():
    graph = pg.from_futhark(
        """
        def clamp (n: i32): i32 = if n > 10 then 10 else n
        def main (n: i32): i32 = clamp n
        """
    )
    instruction_texts = {n.text for n in graph.node if n.type == Node.INSTRUCTION}
    assert {"if", "then", "else", "endif"} <= instruction_texts


def test_for_loop_produces_back_edge():
    graph = pg.from_futhark(
        """
        def sumto (n: i32): i32 = loop acc = 0 for i < n do acc + i
        def main (n: i32): i32 = sumto n
        """
    )
    loop_index = next(
        i for i, n in enumerate(graph.node) if n.type == Node.INSTRUCTION and n.text == "loop"
    )
    # The loop instruction is both a successor (entering the body) and a
    # predecessor (the back edge from the last body instruction).
    incoming_control = [e for e in graph.edge if e.flow == Edge.CONTROL and e.target == loop_index]
    assert len(incoming_control) == 2


def test_derived_array_size_and_reshape():
    """Regression test for two Core IR constructs used by `tabulate_2d`.

    `--standard` inlining can introduce statements that bind an
    existentially-derived size to a name like `d<{(+) n 1}>_10250` (see
    `ir_parser.tokenize`), and `flatten`/`unflatten`-style code lowers to a
    `reshape`/`rearrange` call whose non-array arguments are bare `(...)`/
    `[...]` shape metadata (see `ir_parser.RawGroup`).
    """
    graph = pg.from_futhark(
        """
        module float_type = f32
        type t = float_type.t

        let pi: t = 3.14159265358979323846

        let matrix_size (interlines: i64): i64 = interlines * 8 + 8

        let get_matrix (interlines: i64): [][]t =
          let n = matrix_size interlines
          in tabulate_2d (n + 1) (n + 1) (\\_ _ -> 0.0)

        let jacobi_iteration [m] (h: t) (matrix: [m][m]t): [m][m]t =
          let n = m - 1
          let pih = pi * h
          let fpisin = 0.25 * 2.0 * pi * pi * h * h
          in tabulate_2d m m
               (\\i j ->
                  if i == 0 || i == n || j == 0 || j == n
                  then matrix[i][j]
                  else let star = 0.25 * (matrix[i - 1][j] + matrix[i][j - 1] +
                                           matrix[i][j + 1] + matrix[i + 1][j])
                       in star + fpisin * float_type.sin (pih * float_type.i64 i) * float_type.sin (pih * float_type.i64 j))

        let get_residuum [m] (old: [m][m]t) (new: [m][m]t): t =
          map2 (\\row_old row_new -> map2 (\\a b -> float_type.abs (a - b)) row_old row_new) old new
          |> flatten
          |> float_type.maximum

        let main (interlines: i64): t =
          let n = matrix_size interlines
          let h = 1.0 / float_type.i64 n
          let m0 = get_matrix interlines
          let m1 = jacobi_iteration h m0
          in get_residuum m0 m1
        """
    )
    assert isinstance(graph, pg.ProgramGraph)
    assert "entry_main" in {f.name for f in graph.function}


if __name__ == "__main__":
    main()
