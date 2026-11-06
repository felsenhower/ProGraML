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
"""Builds a :class:`programl.proto.ProgramGraph` from parsed Futhark Core IR.

This follows the same node/edge conventions as the C++ LLVM frontend
(:code:`programl/ir/llvm/internal/program_graph_builder.cc`):

* Node ``text`` is a short, generic label (an opcode name, or ``"var"`` /
  ``"val"`` for variables/constants); the exact source text is stashed as a
  ``full_text`` feature instead, to keep the node vocabulary small.
* Types are separate ``TYPE`` nodes, connected via ``TYPE`` edges, and are
  de-duplicated by their textual signature.
* ``DATA`` edges connect ``VARIABLE``/``CONSTANT`` nodes to the
  ``INSTRUCTION`` nodes that produce or consume them; ``position`` gives the
  operand/result index.
* ``CONTROL`` edges chain the instructions of a function in program order.
* ``CALL`` edges connect a call site to the callee's entry instruction, and
  the callee's exit ("return") instruction back to the call site. Every
  function additionally gets a ``CALL`` edge from/to the graph root, mirroring
  the LLVM frontend's convention for "may be invoked from outside the module".

Futhark-specific mapping decisions:

* Each SOAC (``map``, ``reduce``, ``redomap``, ``scan``, ...) becomes a single
  ``INSTRUCTION`` node, with a ``CALL`` edge to each of its lambda operands
  (each lambda is built as its own anonymous ``Function``), and ``DATA``
  edges for its non-lambda operands (array/scalar arguments, neutral
  elements). This reuses the existing call-edge convention instead of
  inventing SOAC-specific edge types.
* ``if``/``else`` is modelled as a classic diamond CFG: an ``if``
  instruction (with a ``DATA`` edge from its condition) has two outgoing
  ``CONTROL`` edges (position 0/1) to ``then``/``else`` instructions, whose
  respective branches converge on a synthetic ``endif`` instruction. Each
  variable bound by the enclosing `let` gets two incoming ``DATA`` edges
  (position 0/1, one per branch) -- a phi node.
* ``loop`` is modelled as a CFG back edge: a ``loop`` instruction has a
  ``CONTROL`` edge (position 0) to the first instruction of its body, and the
  body's last instruction has a ``CONTROL`` edge back to the ``loop``
  instruction; the code that follows the loop instead takes the ``loop``
  instruction's position-1 successor. Each loop-carried variable is a phi
  node with two incoming ``DATA`` edges: the initial value (position 0) and
  the value produced by one iteration of the body (position 1, the back
  edge).
* Certificates (``#{...}``) are currently discarded -- see
  :mod:`programl.ir.futhark.ir_parser` for the rationale.
"""
import itertools
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from programl.ir.futhark.ir_parser import (
    Apply,
    Body,
    Expr,
    If,
    Lambda,
    Literal,
    Loop,
    NilFn,
    PrimCall,
    Program,
    RawGroup,
    Tuple as IrTuple,
    VarRef,
)
from programl.proto import Edge, Node, ProgramGraph


class FutharkGraphBuilderError(ValueError):
    """Raised when parsed Futhark IR cannot be turned into a program graph."""


class ProgramGraphBuilder:
    """A minimal, index-based analogue of the C++ ``ProgramGraphBuilder``."""

    def __init__(self):
        self._graph = ProgramGraph()
        root = self._graph.node.add()
        root.type = Node.INSTRUCTION
        root.text = "[external]"

    @property
    def proto(self) -> ProgramGraph:
        return self._graph

    def get_root_node(self) -> int:
        return 0

    def add_module(self, name: str) -> int:
        module = self._graph.module.add()
        module.name = name
        return len(self._graph.module) - 1

    def add_function(self, name: str, module: int) -> int:
        function = self._graph.function.add()
        function.name = name
        function.module = module
        return len(self._graph.function) - 1

    def _add_node(self, node_type, text: str, function: Optional[int]) -> int:
        node = self._graph.node.add()
        node.type = node_type
        node.text = text
        if function is not None:
            node.function = function
        return len(self._graph.node) - 1

    def add_instruction(self, text: str, function: int) -> int:
        return self._add_node(Node.INSTRUCTION, text, function)

    def add_variable(self, text: str, function: int) -> int:
        return self._add_node(Node.VARIABLE, text, function)

    def add_constant(self, text: str) -> int:
        return self._add_node(Node.CONSTANT, text, None)

    def add_type(self, text: str) -> int:
        return self._add_node(Node.TYPE, text, None)

    def _add_edge(self, flow, position: int, source: int, target: int) -> int:
        edge = self._graph.edge.add()
        edge.flow = flow
        edge.position = position
        edge.source = source
        edge.target = target
        return len(self._graph.edge) - 1

    def add_control_edge(self, position: int, source: int, target: int) -> int:
        return self._add_edge(Edge.CONTROL, position, source, target)

    def add_data_edge(self, position: int, source: int, target: int) -> int:
        return self._add_edge(Edge.DATA, position, source, target)

    def add_call_edge(self, source: int, target: int) -> int:
        return self._add_edge(Edge.CALL, 0, source, target)

    def add_type_edge(self, position: int, source: int, target: int) -> int:
        return self._add_edge(Edge.TYPE, position, source, target)

    def set_text_feature(self, node_index: int, key: str, value: str) -> None:
        feature = self._graph.node[node_index].features.feature[key]
        feature.bytes_list.value.append(value.encode("utf-8"))

    def set_int_feature(self, node_index: int, key: str, value: int) -> None:
        feature = self._graph.node[node_index].features.feature[key]
        feature.int64_list.value.append(value)


@dataclass
class _FunctionInfo:
    function: int
    entry: int
    exits: List[int] = field(default_factory=list)
    param_vars: Dict[str, int] = field(default_factory=dict)


@dataclass
class _Ctx:
    builder: ProgramGraphBuilder
    module: int
    functions: Dict[str, _FunctionInfo]
    all_functions: List[_FunctionInfo]
    type_cache: Dict[str, int]
    lambda_counter: "itertools.count"


def build_program_graph(program: Program, module_name: str = "futhark") -> ProgramGraph:
    """Build a :class:`ProgramGraph` from a parsed Futhark Core IR program."""
    builder = ProgramGraphBuilder()
    module = builder.add_module(module_name)
    ctx = _Ctx(
        builder=builder,
        module=module,
        functions={},
        all_functions=[],
        type_cache={},
        lambda_counter=itertools.count(1),
    )

    # Pass 1: create a Function + entry instruction + parameter variables for
    # every top-level definition, so that `apply` call targets can always be
    # resolved regardless of definition order.
    for fun in program.funs:
        info = _create_function_skeleton(
            ctx, fun.name, fun.params, fun.entry_param_names if fun.is_entry else None
        )
        ctx.functions[fun.name] = info
        ctx.all_functions.append(info)

    # Pass 2: build the bodies.
    for fun in program.funs:
        info = ctx.functions[fun.name]
        env = dict(info.param_vars)
        exit_node = _build_function_body(ctx, info.function, info.entry, env, fun.body)
        info.exits.append(exit_node)

    # Every function may be invoked from outside the program.
    root = builder.get_root_node()
    for info in ctx.all_functions:
        builder.add_call_edge(root, info.entry)
        for exit_node in info.exits:
            builder.add_call_edge(exit_node, root)

    return builder.proto


def _create_function_skeleton(
    ctx: _Ctx,
    name: str,
    params: List[Tuple[str, str]],
    entry_param_names: Optional[List[str]] = None,
) -> _FunctionInfo:
    function = ctx.builder.add_function(name, ctx.module)
    entry = ctx.builder.add_instruction("entry", function)
    surface_index_by_name = (
        {n: i for i, n in enumerate(entry_param_names)} if entry_param_names else {}
    )
    param_vars = {}
    for i, (pname, ptype) in enumerate(params):
        var = ctx.builder.add_variable("var", function)
        ctx.builder.set_text_feature(var, "full_text", pname)
        type_node = _get_or_create_type(ctx, ptype)
        ctx.builder.add_type_edge(0, type_node, var)
        ctx.builder.add_data_edge(i, entry, var)
        param_vars[pname] = var

        # Lowered Core IR names are the original name plus a fresh numeric
        # suffix (e.g. `a` -> `a_7706`); strip it before matching against
        # the real surface parameter names. Implicit params the compiler
        # inserts (e.g. array sizes) won't match anything and are left
        # untagged. This doesn't attempt to match destructured/tuple
        # surface parameters (e.g. `(p: (i32, i32))`, lowered to `0_N`,
        # `1_N`), which also end up untagged.
        m = re.match(r"^(.*)_\d+$", pname)
        base_name = m.group(1) if m else pname
        if base_name in surface_index_by_name:
            ctx.builder.set_int_feature(var, "entry_param_index", surface_index_by_name[base_name])
    return _FunctionInfo(function=function, entry=entry, param_vars=param_vars)


def _build_function_body(
    ctx: _Ctx, function: int, entry: int, env: Dict[str, int], body: Body
) -> int:
    last, next_pos = entry, 0
    for stmt in body.stmts:
        last, next_pos = _build_stmt(ctx, function, env, last, next_pos, stmt)
    exit_node = ctx.builder.add_instruction("return", function)
    ctx.builder.add_control_edge(next_pos, last, exit_node)
    for i, result_expr in enumerate(body.result):
        value_node = _resolve_value(ctx, function, env, result_expr)
        ctx.builder.add_data_edge(i, value_node, exit_node)
    return exit_node


def _build_stmt(
    ctx: _Ctx, function: int, env: Dict[str, int], last: int, control_position: int, stmt
) -> Tuple[int, int]:
    """Build one statement, returning (new `last`, control position for the
    edge into whatever comes next).

    The position is almost always 0 (a single predecessor -> single
    successor edge), except right after an `if`/`loop`, whose instruction
    node has two outgoing control edges (branch/exit) and so needs its
    successor-numbering preserved -- mirroring the LLVM frontend's
    `successorNumber` convention for conditional branches.
    """
    expr = stmt.expr
    if isinstance(expr, If):
        return _build_if_stmt(ctx, function, env, last, control_position, stmt, expr)
    if isinstance(expr, Loop):
        return _build_loop_stmt(ctx, function, env, last, control_position, stmt, expr)

    text = _instruction_label(expr)
    instr = ctx.builder.add_instruction(text, function)
    if stmt.location:
        ctx.builder.set_text_feature(instr, "location", stmt.location)
    ctx.builder.add_control_edge(control_position, last, instr)

    if isinstance(expr, Apply):
        values, lambdas = _flatten_operands(expr.args)
        _wire_data_operands(ctx, function, env, instr, values)
        for lam in lambdas:
            _wire_lambda_call(ctx, instr, _build_lambda(ctx, lam))
        ctx.builder.set_text_feature(instr, "full_text", expr.fn)
        callee = ctx.functions.get(expr.fn)
        if callee is not None:
            _wire_lambda_call(ctx, instr, callee)
        # Otherwise this calls a compiler intrinsic (e.g. `sin32`, `sqrt32`)
        # rather than a function defined in the program, so there's no body
        # to add a CALL edge to.
    elif isinstance(expr, PrimCall):
        values, lambdas = _flatten_operands(expr.args)
        _wire_data_operands(ctx, function, env, instr, values)
        for lam in lambdas:
            _wire_lambda_call(ctx, instr, _build_lambda(ctx, lam))
        if expr.meta:
            for key in ("from", "to"):
                if key in expr.meta:
                    ctx.builder.set_text_feature(instr, f"convert_{key}_type", expr.meta[key])
    else:
        raise FutharkGraphBuilderError(
            f"Unsupported statement expression: {type(expr).__name__}"
        )

    _bind_pattern(ctx, function, env, stmt.pattern, lambda i: instr)

    return instr, 0


def _bind_pattern(ctx, function, env, pattern, source_fn) -> List[int]:
    """Create a VARIABLE node for each name in a `let` pattern.

    `source_fn(i)` returns the node that DATA edge position `i` should come
    from; the default (a single instruction producing all of its results) is
    used by plain instructions, while `if`/`loop` wire their phi'd
    loop/branch variable directly instead.
    """
    var_nodes = []
    for i, (pname, ptype) in enumerate(pattern):
        var = ctx.builder.add_variable("var", function)
        ctx.builder.set_text_feature(var, "full_text", pname)
        type_node = _get_or_create_type(ctx, ptype)
        ctx.builder.add_type_edge(0, type_node, var)
        ctx.builder.add_data_edge(0, source_fn(i), var)
        env[pname] = var
        var_nodes.append(var)
    return var_nodes


def _build_if_stmt(
    ctx: _Ctx, function: int, env: Dict[str, int], last: int, control_position: int, stmt, if_expr: If
) -> Tuple[int, int]:
    instr = ctx.builder.add_instruction("if", function)
    ctx.builder.add_control_edge(control_position, last, instr)
    ctx.builder.add_data_edge(0, _resolve_value(ctx, function, env, if_expr.cond), instr)

    then_last, then_next_pos, then_results = _build_branch(
        ctx, function, dict(env), instr, 0, "then", if_expr.then_body
    )
    else_last, else_next_pos, else_results = _build_branch(
        ctx, function, dict(env), instr, 1, "else", if_expr.else_body
    )

    endif = ctx.builder.add_instruction("endif", function)
    ctx.builder.add_control_edge(then_next_pos, then_last, endif)
    ctx.builder.add_control_edge(else_next_pos, else_last, endif)

    for i, (pname, ptype) in enumerate(stmt.pattern):
        var = ctx.builder.add_variable("var", function)
        ctx.builder.set_text_feature(var, "full_text", pname)
        type_node = _get_or_create_type(ctx, ptype)
        ctx.builder.add_type_edge(0, type_node, var)
        ctx.builder.add_data_edge(0, then_results[i], var)
        ctx.builder.add_data_edge(1, else_results[i], var)
        env[pname] = var

    return endif, 0


def _build_branch(
    ctx: _Ctx, function: int, env: Dict[str, int], pred: int, position: int, label: str, body: Body
) -> Tuple[int, int, List[int]]:
    entry = ctx.builder.add_instruction(label, function)
    ctx.builder.add_control_edge(position, pred, entry)
    last, next_pos = entry, 0
    for stmt in body.stmts:
        last, next_pos = _build_stmt(ctx, function, env, last, next_pos, stmt)
    results = [_resolve_value(ctx, function, env, e) for e in body.result]
    return last, next_pos, results


def _build_loop_stmt(
    ctx: _Ctx, function: int, env: Dict[str, int], last: int, control_position: int, stmt, loop: Loop
) -> Tuple[int, int]:
    instr = ctx.builder.add_instruction("loop", function)
    ctx.builder.add_control_edge(control_position, last, instr)

    # Loop-carried variables are phi-like: their value is either the initial
    # value (entering the loop, position 0) or the previous iteration's
    # result (the back edge, position 1, added once the body is built).
    body_env = dict(env)
    loop_vars = []
    for i, (pname, ptype) in enumerate(loop.pattern):
        var = ctx.builder.add_variable("var", function)
        ctx.builder.set_text_feature(var, "full_text", pname)
        type_node = _get_or_create_type(ctx, ptype)
        ctx.builder.add_type_edge(0, type_node, var)
        ctx.builder.add_data_edge(0, _resolve_value(ctx, function, env, loop.init[i]), var)
        body_env[pname] = var
        loop_vars.append(var)

    if loop.loop_var is not None:
        var_name, var_type = loop.loop_var
        induction_var = ctx.builder.add_variable("var", function)
        ctx.builder.set_text_feature(induction_var, "full_text", var_name)
        type_node = _get_or_create_type(ctx, var_type)
        ctx.builder.add_type_edge(0, type_node, induction_var)
        ctx.builder.add_data_edge(0, instr, induction_var)
        body_env[var_name] = induction_var
        ctx.builder.add_data_edge(
            0, _resolve_value(ctx, function, env, loop.bound), instr
        )

    body_last, body_next_pos = instr, 0
    for body_stmt in loop.body.stmts:
        body_last, body_next_pos = _build_stmt(ctx, function, body_env, body_last, body_next_pos, body_stmt)
    body_results = [_resolve_value(ctx, function, body_env, e) for e in loop.body.result]

    ctx.builder.add_control_edge(body_next_pos, body_last, instr)
    for var, result in zip(loop_vars, body_results):
        ctx.builder.add_data_edge(1, result, var)

    for i, (pname, ptype) in enumerate(stmt.pattern):
        var = ctx.builder.add_variable("var", function)
        ctx.builder.set_text_feature(var, "full_text", pname)
        type_node = _get_or_create_type(ctx, ptype)
        ctx.builder.add_type_edge(0, type_node, var)
        ctx.builder.add_data_edge(0, loop_vars[i], var)
        env[pname] = var

    return instr, 1


def _wire_data_operands(
    ctx: _Ctx, function: int, env: Dict[str, int], instr: int, values: List[Expr]
) -> None:
    for i, value_expr in enumerate(values):
        value_node = _resolve_value(ctx, function, env, value_expr)
        ctx.builder.add_data_edge(i, value_node, instr)


def _wire_lambda_call(ctx: _Ctx, call_site: int, callee: _FunctionInfo) -> None:
    ctx.builder.add_call_edge(call_site, callee.entry)
    for exit_node in callee.exits:
        ctx.builder.add_call_edge(exit_node, call_site)


def _build_lambda(ctx: _Ctx, lam: Lambda) -> _FunctionInfo:
    name = f"lambda_{next(ctx.lambda_counter)}"
    info = _create_function_skeleton(ctx, name, lam.params)
    env = dict(info.param_vars)
    exit_node = _build_function_body(ctx, info.function, info.entry, env, lam.body)
    info.exits.append(exit_node)
    ctx.all_functions.append(info)
    return info



def _flatten_operands(exprs: List[Expr]) -> Tuple[List[Expr], List[Lambda]]:
    """Split SOAC/call arguments into data operands and lambda operands.

    Recurses into tuples (used by the IR for e.g. an operator + its neutral
    elements, `{op_lambda, {ne}}`) but not into a lambda's own body. `nilFn`
    (an absent optional SOAC operator) contributes nothing.
    """
    values: List[Expr] = []
    lambdas: List[Lambda] = []

    def walk(e: Expr) -> None:
        if isinstance(e, (NilFn, RawGroup)):
            return
        if isinstance(e, IrTuple):
            for el in e.elems:
                walk(el)
            return
        if isinstance(e, Lambda):
            lambdas.append(e)
            return
        values.append(e)

    for e in exprs:
        walk(e)
    return values, lambdas


def _resolve_value(ctx: _Ctx, function: int, env: Dict[str, int], expr: Expr) -> int:
    if isinstance(expr, VarRef):
        node = env.get(expr.name)
        if node is None:
            # Defensive fallback: the supported IR subset should never
            # reference a variable that isn't in scope (defunctionalisation
            # closes over all free lambda variables), but don't crash if it
            # does -- create a placeholder rather than produce an invalid
            # graph.
            node = ctx.builder.add_variable("var", function)
            ctx.builder.set_text_feature(node, "full_text", expr.name)
            env[expr.name] = node
        return node
    if isinstance(expr, Literal):
        node = ctx.builder.add_constant("val")
        ctx.builder.set_text_feature(node, "full_text", expr.text)
        ty = getattr(expr, "type_annotation", None) or _infer_literal_type(expr)
        if ty:
            type_node = _get_or_create_type(ctx, ty)
            ctx.builder.add_type_edge(0, type_node, node)
        return node
    raise FutharkGraphBuilderError(
        f"Cannot resolve a data value for expression of type {type(expr).__name__}"
    )


def _instruction_label(expr: Expr) -> str:
    if isinstance(expr, Apply):
        return "apply"
    if isinstance(expr, PrimCall):
        return expr.op
    raise FutharkGraphBuilderError(f"Unsupported expression kind: {type(expr).__name__}")


_LITERAL_SUFFIX_RE = re.compile(
    r"^-?[0-9]+(\.[0-9]+)?(?:[eE][+-]?[0-9]+)?([a-zA-Z][a-zA-Z0-9]*)?$"
)


def _infer_literal_type(literal: Literal) -> Optional[str]:
    if literal.is_string:
        return "string"
    if literal.text in ("true", "false"):
        return "bool"
    m = _LITERAL_SUFFIX_RE.match(literal.text)
    if m and m.group(2):
        return m.group(2)
    return None


def _get_or_create_type(ctx: _Ctx, type_text: str) -> int:
    node = ctx.type_cache.get(type_text)
    if node is None:
        node = ctx.builder.add_type(type_text)
        ctx.type_cache[type_text] = node
    return node
