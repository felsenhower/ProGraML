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
"""The graph transform ops are used to modify or convert Program Graphs to
another representation.
"""
from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, Iterable, Optional, Union

import networkx as nx
from networkx.readwrite import json_graph as nx_json

from programl.exceptions import GraphTransformError
from programl.proto import ProgramGraph
from programl.util.py.executor import ExecutorLike, execute
from programl.util.py.runfiles_path import runfiles_path

GRAPH2DOT = str(runfiles_path("programl/bin/graph2dot"))
GRAPH2JSON = str(runfiles_path("programl/bin/graph2json"))

JsonDict = Dict[str, Any]


def _run_graph_transform_binary(
    binary: str,
    graph: ProgramGraph,
    timeout: int = 300,
) -> Iterable[bytes]:
    process = subprocess.Popen(
        [binary, "--stdin_fmt=pb"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        stdout, stderr = process.communicate(graph.SerializeToString(), timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(str(e)) from e

    if process.returncode:
        try:
            raise GraphTransformError(stderr.decode("utf-8"))
        except UnicodeDecodeError as e:
            raise GraphTransformError("Unknown error in graph transformation") from e

    return stdout


def to_json(
    graphs: Union[ProgramGraph, Iterable[ProgramGraph]],
    timeout: int = 300,
    executor: Optional[ExecutorLike] = None,
    chunksize: Optional[int] = None,
) -> Union[JsonDict, Iterable[JsonDict]]:
    """Convert one or more Program Graphs to JSON node-link data.

    :param graphs: A Program Graph, or a sequence of Program Graphs.

    :param timeout: The maximum number of seconds to wait for an individual
        graph conversion before raising an error. If multiple inputs are
        provided, this timeout is per-input.

    :param executor: An executor object, with method :code:`submit(callable,
        *args, **kwargs)` and returning a Future-like object with methods
        :code:`done() -> bool` and :code:`result() -> float`. The executor role
        is to dispatch the execution of the jobs locally/on a cluster/with
        multithreading depending on the implementation. Eg:
        :code:`concurrent.futures.ThreadPoolExecutor`. Defaults to single
        threaded execution. This is only used when multiple inputs are given.

    :param chunksize: The number of inputs to read and process at a time. A
        larger chunksize improves parallelism but increases memory consumption
        as more inputs must be stored in memory. This is only used when multiple
        inputs are given.

    :return: If a single input is provided, return a single JSON dictionary.
        Else returns an iterable sequence of JSON dictionaries.

    :raises GraphTransformError: If graph conversion fails.

    :raises TimeoutError: If the specified timeout is reached.
    """

    def _run_one(graph: ProgramGraph):
        try:
            return json.loads(
                _run_graph_transform_binary(
                    GRAPH2JSON,
                    graph,
                    timeout,
                )
            )
        except json.JSONDecodeError as e:
            raise GraphTransformError(str(e)) from e

    if isinstance(graphs, ProgramGraph):
        return _run_one(graphs)
    return execute(_run_one, graphs, executor, chunksize)


def to_networkx(
    graphs: Union[ProgramGraph, Iterable[ProgramGraph]],
    timeout: int = 300,
    executor: Optional[ExecutorLike] = None,
    chunksize: Optional[int] = None,
) -> Union[nx.MultiDiGraph, Iterable[nx.MultiDiGraph]]:
    """Convert one or more Program Graphs to `NetworkX MultiDiGraphs
    <https://networkx.org/documentation/stable/reference/classes/multidigraph.html>`_.

    :param graphs: A Program Graph, or a sequence of Program Graphs.

    :param timeout: The maximum number of seconds to wait for an individual
        graph conversion before raising an error. If multiple inputs are
        provided, this timeout is per-input.

    :param executor: An executor object, with method :code:`submit(callable,
        *args, **kwargs)` and returning a Future-like object with methods
        :code:`done() -> bool` and :code:`result() -> float`. The executor role
        is to dispatch the execution of the jobs locally/on a cluster/with
        multithreading depending on the implementation. Eg:
        :code:`concurrent.futures.ThreadPoolExecutor`. Defaults to single
        threaded execution. This is only used when multiple inputs are given.

    :param chunksize: The number of inputs to read and process at a time. A
        larger chunksize improves parallelism but increases memory consumption
        as more inputs must be stored in memory. This is only used when multiple
        inputs are given.

    :return: If a single input is provided, return a single :code:`nx.MultiDiGraph`.
        Else returns an iterable sequence of :code:`nx.MultiDiGraph` instances.

    :raises GraphTransformError: If graph conversion fails.

    :raises TimeoutError: If the specified timeout is reached.
    """

    def _run_one(json_data):
        # graph2json always emits the pre-networkx-3.x "links" edge key; pass
        # it explicitly since node_link_graph's own default has since changed.
        return nx_json.node_link_graph(json_data, multigraph=True, directed=True, edges="links")

    if isinstance(graphs, ProgramGraph):
        return _run_one(to_json(graphs, timeout=timeout))
    return execute(
        _run_one,
        to_json(graphs, timeout=timeout, executor=executor, chunksize=chunksize),
        executor,
        chunksize,
    )


def to_dgl(
    graphs: Union[ProgramGraph, Iterable[ProgramGraph]],
    timeout: int = 300,
    executor: Optional[ExecutorLike] = None,
    chunksize: Optional[int] = None,
) -> Union[dgl.DGLGraph, Iterable[dgl.DGLGraph]]:
    """Convert one or more Program Graphs to `DGLGraphs
    <https://docs.dgl.ai/en/latest/api/python/dgl.DGLGraph.html#dgl.DGLGraph>`_.

    :param graphs: A Program Graph, or a sequence of Program Graphs.

    :param timeout: The maximum number of seconds to wait for an individual
        graph conversion before raising an error. If multiple inputs are
        provided, this timeout is per-input.

    :param executor: An executor object, with method :code:`submit(callable,
        *args, **kwargs)` and returning a Future-like object with methods
        :code:`done() -> bool` and :code:`result() -> float`. The executor role
        is to dispatch the execution of the jobs locally/on a cluster/with
        multithreading depending on the implementation. Eg:
        :code:`concurrent.futures.ThreadPoolExecutor`. Defaults to single
        threaded execution. This is only used when multiple inputs are given.

    :param chunksize: The number of inputs to read and process at a time. A
        larger chunksize improves parallelism but increases memory consumption
        as more inputs must be stored in memory. This is only used when multiple
        inputs are given.

    :return: If a single input is provided, return a single
        :code:`dgl.DGLGraph`. Else returns an iterable sequence of
        :code:`dgl.DGLGraph` instances.

    :raises GraphTransformError: If graph conversion fails.

    :raises TimeoutError: If the specified timeout is reached.
    """
    try:
        import dgl
    except ImportError as e:
        raise ImportError(
            "to_dgl() requires the optional 'dgl' dependency. Install it "
            "with `pip install programl[dgl]`."
        ) from e

    def _run_one(nx_graph):
        return dgl.from_networkx(nx_graph)

    if isinstance(graphs, ProgramGraph):
        return _run_one(to_networkx(graphs))
    return execute(
        _run_one,
        to_networkx(graphs, timeout=timeout, executor=executor, chunksize=chunksize),
        executor,
        chunksize,
    )


def to_dot(
    graphs: Union[ProgramGraph, Iterable[ProgramGraph]],
    timeout: int = 300,
    executor: Optional[ExecutorLike] = None,
    chunksize: Optional[int] = None,
) -> Union[str, Iterable[str]]:
    """Convert one or more Program Graphs to DOT Graph Description Language.

    This produces a DOT source string representing the input graph. This can
    then be rendered using the graphviz command line tools, or parsed using
    `pydot <https://pypi.org/project/pydot/>`_.

    :param graphs: A Program Graph, or a sequence of Program Graphs.

    :param timeout: The maximum number of seconds to wait for an individual
        graph conversion before raising an error. If multiple inputs are
        provided, this timeout is per-input.

    :param executor: An executor object, with method :code:`submit(callable,
        *args, **kwargs)` and returning a Future-like object with methods
        :code:`done() -> bool` and :code:`result() -> float`. The executor role
        is to dispatch the execution of the jobs locally/on a cluster/with
        multithreading depending on the implementation. Eg:
        :code:`concurrent.futures.ThreadPoolExecutor`. Defaults to single
        threaded execution. This is only used when multiple inputs are given.

    :param chunksize: The number of inputs to read and process at a time. A
        larger chunksize improves parallelism but increases memory consumption
        as more inputs must be stored in memory. This is only used when multiple
        inputs are given.

    :return: A graphviz dot string when a single input is provided, else an
        iterable sequence of graphviz dot strings.

    :raises GraphTransformError: If graph conversion fails.

    :raises TimeoutError: If the specified timeout is reached.
    """

    def _run_one(graph: ProgramGraph) -> str:
        return _run_graph_transform_binary(GRAPH2DOT, graph, timeout).decode("utf-8")

    if isinstance(graphs, ProgramGraph):
        return _run_one(graphs)
    return execute(_run_one, graphs, executor, chunksize)


def to_pyg(
    graphs: Union[ProgramGraph, Iterable[ProgramGraph]],
    timeout: int = 300,
    vocabulary: Optional[Dict[str, int]] = None,
    node_stat_feature_key: str = "stat_feat",
    node_stat_feature_width: Optional[int] = None,
    executor: Optional[ExecutorLike] = None,
    chunksize: Optional[int] = None,
) -> Union[HeteroData, Iterable[HeteroData]]:
    """Convert one or more Program Graphs to Pytorch-Geometrics's HeteroData.
    This graphs can be used as input for any deep learning model built with
    Pytorch-Geometric:

    https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html

    :param graphs: A Program Graph, or a sequence of Program Graphs.

    :param timeout: The maximum number of seconds to wait for an individual
        graph conversion before raising an error. If multiple inputs are
        provided, this timeout is per-input.

    :param vocabulary: A dictionary containing ProGraML's vocabulary, where the
        keys are the text attribute of the nodes and the values their respective
        indexes.

    :param node_stat_feature_key: The node :code:`features` key under which a
        per-node float feature vector may be stored. Only used when
        :code:`node_stat_feature_width` is set.

    :param node_stat_feature_width: If set, every node's
        :code:`hetero_graph['nodes'].stat_feat` row is populated with the
        :code:`node_stat_feature_key` feature's :code:`float_list.value` (if
        present and of this length), else zeros; presence is recorded in
        :code:`hetero_graph['nodes'].stat_feat_mask`. Defaults to :code:`None`,
        which disables this (no :code:`stat_feat`/:code:`stat_feat_mask`
        attributes are added).

    :param executor: An executor object, with method :code:`submit(callable,
        *args, **kwargs)` and returning a Future-like object with methods
        :code:`done() -> bool` and :code:`result() -> float`. The executor role
        is to dispatch the execution of the jobs locally/on a cluster/with
        multithreading depending on the implementation. Eg:
        :code:`concurrent.futures.ThreadPoolExecutor`. Defaults to single
        threaded execution. This is only used when multiple inputs are given.

    :param chunksize: The number of inputs to read and process at a time. A
        larger chunksize improves parallelism but increases memory consumption
        as more inputs must be stored in memory. This is only used when multiple
        inputs are given.

    :return: A HeteroData graph when a single input is provided, else an
        iterable sequence of HeteroData graphs.

    :raises GraphTransformError: If a node's :code:`node_stat_feature_key`
        feature is present but doesn't have :code:`node_stat_feature_width`
        values.
    """
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as e:
        raise ImportError(
            "to_pyg() requires the optional 'torch' and 'torch_geometric' "
            "dependencies. Install them with `pip install programl[pyg]`."
        ) from e

    def _run_one(graph: ProgramGraph) -> HeteroData:
        # 4 lists, one per edge type
        # (control, data, call and type edges)
        adjacencies = [[], [], [], []]
        edge_positions = [[], [], [], []]

        # Create the adjacency lists and the positions
        for edge in graph.edge:
            adjacencies[edge.flow].append([edge.source, edge.target])
            edge_positions[edge.flow].append(edge.position)

        # Store the text attributes
        node_text_list = []
        node_full_text_list = []

        # Store the text and full text attributes
        for node in graph.node:
            node_text = node_full_text = node.text

            if (
                node.features
                and node.features.feature["full_text"].bytes_list.value
            ):
                node_full_text = node.features.feature["full_text"].bytes_list.value[0]

            node_text_list.append(node_text)
            node_full_text_list.append(node_full_text)

        stat_feat_rows = None
        stat_feat_mask = None
        if node_stat_feature_width is not None:
            stat_feat_rows = []
            stat_feat_mask = []
            for node in graph.node:
                if node_stat_feature_key in node.features.feature:
                    values = list(
                        node.features.feature[node_stat_feature_key].float_list.value
                    )
                    if len(values) != node_stat_feature_width:
                        raise GraphTransformError(
                            f"Node feature {node_stat_feature_key!r} has "
                            f"{len(values)} values, expected "
                            f"{node_stat_feature_width}"
                        )
                    stat_feat_rows.append(values)
                    stat_feat_mask.append(1.0)
                else:
                    stat_feat_rows.append([0.0] * node_stat_feature_width)
                    stat_feat_mask.append(0.0)


        vocab_ids = None
        if vocabulary is not None:
            vocab_ids = [
                vocabulary.get(node.text, len(vocabulary.keys()))
                for node in graph.node
            ]

        # Pass from list to tensor
        adjacencies = [torch.tensor(adj_flow_type) for adj_flow_type in adjacencies]
        edge_positions = [torch.tensor(edge_pos_flow_type) for edge_pos_flow_type in edge_positions]

        if vocabulary is not None:
            vocab_ids = torch.tensor(vocab_ids)

        # Create the graph structure
        hetero_graph = HeteroData()

        # Vocabulary index of each node
        hetero_graph['nodes']['text'] = node_text_list
        hetero_graph['nodes']['full_text'] = node_full_text_list
        hetero_graph['nodes'].x = vocab_ids

        if node_stat_feature_width is not None:
            hetero_graph['nodes'].stat_feat = torch.tensor(stat_feat_rows, dtype=torch.float32)
            hetero_graph['nodes'].stat_feat_mask = torch.tensor(stat_feat_mask, dtype=torch.float32)

        # Add the adjacency lists
        hetero_graph['nodes', 'control', 'nodes'].edge_index = (
            adjacencies[0].t().contiguous() if adjacencies[0].nelement() > 0 else torch.tensor([[], []])
        )
        hetero_graph['nodes', 'data', 'nodes'].edge_index = (
            adjacencies[1].t().contiguous() if adjacencies[1].nelement() > 0 else torch.tensor([[], []])
        )
        hetero_graph['nodes', 'call', 'nodes'].edge_index = (
            adjacencies[2].t().contiguous() if adjacencies[2].nelement() > 0 else torch.tensor([[], []])
        )
        hetero_graph['nodes', 'type', 'nodes'].edge_index = (
            adjacencies[3].t().contiguous() if adjacencies[3].nelement() > 0 else torch.tensor([[], []])
        )

        # Add the edge positions
        hetero_graph['nodes', 'control', 'nodes'].edge_attr = edge_positions[0]
        hetero_graph['nodes', 'data', 'nodes'].edge_attr = edge_positions[1]
        hetero_graph['nodes', 'call', 'nodes'].edge_attr = edge_positions[2]
        hetero_graph['nodes', 'type', 'nodes'].edge_attr = edge_positions[3]

        return hetero_graph

    if isinstance(graphs, ProgramGraph):
        return _run_one(graphs)

    return execute(_run_one, graphs, executor, chunksize)
