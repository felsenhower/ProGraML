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


if __name__ == "__main__":
    main()
