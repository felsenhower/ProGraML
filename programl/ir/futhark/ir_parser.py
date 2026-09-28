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
"""A parser for the textual Futhark "Core IR" as printed by ``futhark dev``.

Futhark does not expose a machine-readable (e.g. JSON or protocol buffer)
serialization of its internal representation, so this module works from the
pretty-printed text that ``futhark dev`` writes to stdout. This is
necessarily a partial grammar: it covers the subset of the Core IR produced
by the ``--standard`` pipeline stage (see :mod:`programl.ir.futhark.futhark`
for why that stage was chosen), namely straight-line code, function
application, ``if``/``else`` and ``for``/``while`` ``loop``, and SOACs
(``map``/``reduce``/``redomap``/``scan``/...).

Known simplifications (acceptable for a first prototype):

* Certificates (``#{...}``, Futhark's mechanism for tying a value to the
  bounds/safety check that justifies it) are parsed so that they don't break
  parsing, but the dependency they represent is currently discarded rather
  than turned into a graph edge.
* User-defined types (the ``types { ... }`` preamble) are skipped rather than
  modelled.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


class FutharkIRParseError(ValueError):
    """Raised when the Futhark Core IR text cannot be parsed."""


# --------------------------------------------------------------------------
# AST.
# --------------------------------------------------------------------------
#
# Expr is a Union of the node kinds below. Any Expr instance may carry two
# optional dynamic attributes, attached after construction (not part of the
# dataclass fields so that every expression kind can share the same
# incidental extras without repeating the fields everywhere):
#
#   .certs: List[str]           -- names of certificates from a `#{...}`
#                                   prefix (may be absent/empty).
#   .type_annotation: str       -- a trailing `: TYPE` suffix, when present
#                                   (used inside tuples, e.g. assert messages).


@dataclass
class Literal:
    text: str  # Raw literal text, e.g. "1i32", "true", or a string body.
    is_string: bool = False  # True if this came from a "..." string token.


@dataclass
class VarRef:
    name: str


@dataclass
class NilFn:
    """The ``nilFn`` sentinel: an absent optional SOAC operator."""


@dataclass
class RawGroup:
    """A bare ``(...)``/``[...]`` argument that carries no runtime data
    dependency.

    Used for things like a ``rearrange`` permutation tuple (``(1, 0)``), a
    ``reshape`` shape-change spec (``(0::2=>[n]; [n])``), or a ``replicate``
    shape argument (``[n]``): all are constant shape/index metadata, not
    values computed at runtime. Parsed (as a balanced token span) so that it
    doesn't break parsing, then discarded -- the same simplification already
    applied to certificates (see the module docstring).
    """


@dataclass
class Tuple:
    elems: "List[Expr]"


@dataclass
class Lambda:
    params: List[Tuple[str, str]]
    rettypes: List[str]
    body: "Body"


@dataclass
class PrimCall:
    """A generic ``op(args...)`` call.

    This is used both for basic arithmetic/comparison/etc. operations
    (``add32``, ``slt32``, ...) and for SOACs (``map``, ``reduce``,
    ``redomap``, ``scan``, ...) -- the grammar is identical, and the two are
    only distinguished when building the graph (see
    :mod:`programl.ir.futhark.program_graph_builder`).
    """

    op: str
    args: "List[Expr]"
    meta: Optional[dict] = None  # Used for conversions: {"from": t, "to": t}.


@dataclass
class Apply:
    fn: str
    args: "List[Expr]"
    rettypes: Optional[List[str]] = None


@dataclass
class If:
    cond: "Expr"
    then_body: "Body"
    else_body: "Body"
    rettypes: Optional[List[str]] = None


@dataclass
class Loop:
    """A ``loop {pattern} = {init} for/while ... do {body}`` expression.

    Exactly one of (``loop_var``, ``bound``) or ``cond_var`` is set,
    depending on whether this is a ``for`` or a ``while`` loop. For a
    ``while`` loop, the condition is itself one of the loop-carried
    variables named in ``pattern`` (Futhark's Core IR re-evaluates it at the
    end of each iteration alongside the other loop-carried values), so
    ``cond_var`` is one of ``pattern``'s names, not a separate value.
    """

    pattern: List[Tuple[str, str]]
    init: "List[Expr]"
    body: "Body"
    loop_var: Optional[Tuple[str, str]] = None
    bound: Optional["Expr"] = None
    cond_var: Optional[str] = None


Expr = Union[Literal, VarRef, NilFn, RawGroup, Tuple, Lambda, PrimCall, Apply, If, Loop]


@dataclass
class Stmt:
    pattern: List[Tuple[str, str]]  # [(name, type), ...]
    expr: Expr
    location: Optional[str] = None


@dataclass
class Body:
    stmts: List[Stmt]
    result: List[Expr]


@dataclass
class FunDef:
    name: str
    params: List[Tuple[str, str]]
    rettypes: List[str]
    body: Body
    is_entry: bool = False


@dataclass
class Program:
    funs: List[FunDef] = field(default_factory=list)


# --------------------------------------------------------------------------
# Lexer.
# --------------------------------------------------------------------------

_PUNCT_CHARS = "{}()[]:,=#\\"
_OPEN_TO_CLOSE = {"(": ")", "{": "}", "[": "]"}


@dataclass
class Token:
    kind: str  # "string" | "punct" | "arrow" | "word"
    text: str


def _unescape(s: str) -> str:
    return s.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")


def tokenize(text: str) -> List[Token]:
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" and j + 1 < n else 1
            tokens.append(Token("string", _unescape(text[i + 1 : j])))
            i = j + 1
            continue
        if text[i : i + 2] == "->":
            tokens.append(Token("arrow", "->"))
            i += 2
            continue
        if c in _PUNCT_CHARS:
            tokens.append(Token("punct", c))
            i += 1
            continue
        # A "word": a run of otherwise-unbroken characters, with one twist --
        # Futhark's debug printer names existentially-bound sizes after the
        # expression that defines them, e.g. `d<{(+) n 1}>_10250` for "the
        # dimension equal to n+1". The `{...}` (and any nested `(...)`/`[...]`
        # it contains) would otherwise be lexed as separate punctuation and
        # break the name apart, so when a bare `<` is immediately followed by
        # an opening bracket, consume through to the matching `>` (tracking
        # nested brackets, so this also works when the expression itself
        # contains a further `d<...>` name) as part of the same word.
        start = i
        while i < n:
            c = text[i]
            if c.isspace() or c in _PUNCT_CHARS or c == '"':
                break
            i += 1
            if c == "<" and i < n and text[i] in _OPEN_TO_CLOSE:
                stack = []
                while i < n:
                    cc = text[i]
                    if not stack and cc == ">":
                        i += 1
                        break
                    if cc in _OPEN_TO_CLOSE:
                        stack.append(_OPEN_TO_CLOSE[cc])
                        i += 1
                    elif stack and cc == stack[-1]:
                        stack.pop()
                        i += 1
                    else:
                        i += 1
        tokens.append(Token("word", text[start:i]))
    return tokens


_LITERAL_WORD_RE = re.compile(r"^-?[0-9]")
_TYPE_WORDS = {
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
    "f16",
    "f32",
    "f64",
    "bool",
    "unit",
}
# Words that terminate an expression rather than starting a new one. Used to
# disambiguate a bare variable reference from the start of a unary prefix
# call such as `neg_bool x` -- e.g. without "do" here, parsing the `n` in
# `for i:i32 < n do {...}` would wrongly treat `n` as a unary call on `do`.
_EXPR_TERMINATORS = {"in", "to", "let", "then", "else", "do", "for", "while"}


def _is_literal_word(text: str) -> bool:
    return text in ("true", "false") or bool(_LITERAL_WORD_RE.match(text))


def _is_type_word(text: str) -> bool:
    return text in _TYPE_WORDS


# --------------------------------------------------------------------------
# Parser.
# --------------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: List[Token]):
        self._tokens = tokens
        self._i = 0

    def _peek(self) -> Optional[Token]:
        return self._tokens[self._i] if self._i < len(self._tokens) else None

    def _peek_text(self) -> Optional[str]:
        t = self._peek()
        return t.text if t is not None else None

    def _advance(self) -> Token:
        if self._i >= len(self._tokens):
            raise FutharkIRParseError("Unexpected end of input")
        t = self._tokens[self._i]
        self._i += 1
        return t

    def _expect(self, text: str) -> Token:
        t = self._advance()
        if t.text != text:
            raise FutharkIRParseError(f"Expected {text!r} but got {t.text!r}")
        return t

    def _expect_word(self, text: Optional[str] = None) -> str:
        t = self._advance()
        if t.kind != "word":
            raise FutharkIRParseError(f"Expected an identifier but got {t.text!r}")
        if text is not None and t.text != text:
            raise FutharkIRParseError(f"Expected {text!r} but got {t.text!r}")
        return t.text

    def at_eof(self) -> bool:
        return self._i >= len(self._tokens)

    # -- Top level. --------------------------------------------------------

    def parse_program(self) -> Program:
        self._skip_named_braced_block("name_source")
        self._skip_named_braced_block("types")
        funs = []
        while not self.at_eof():
            if self._peek_text() == "fun":
                funs.append(self._parse_fun())
            elif self._peek_text() == "entry":
                funs.append(self._parse_entry())
            else:
                raise FutharkIRParseError(
                    f"Unexpected token {self._peek_text()!r} at top level"
                )
        return Program(funs=funs)

    def _skip_named_braced_block(self, name: str) -> None:
        if self._peek_text() == name:
            self._advance()
            self._expect("{")
            self._skip_balanced_rest()

    def _skip_balanced_rest(self) -> None:
        # Assumes a single opening bracket has already been consumed.
        depth = 1
        openers = {"(", "{", "["}
        closers = {")", "}", "]"}
        while depth > 0:
            t = self._advance()
            if t.text in openers:
                depth += 1
            elif t.text in closers:
                depth -= 1

    def _parse_fun(self) -> FunDef:
        self._expect_word("fun")
        name = self._expect_word()
        self._expect("(")
        params = self._parse_typed_name_list(")")
        self._expect(")")
        self._expect(":")
        rettypes = self._parse_type_list_braced()
        self._expect("=")
        body = self._parse_braced_body()
        return FunDef(name=name, params=params, rettypes=rettypes, body=body)

    def _parse_entry(self) -> FunDef:
        # The `entry(...)` header repeats information (under the exported
        # name) that is given again in full below (under the internal
        # `entry_NAME` symbol), so we only need to skip over it.
        self._expect_word("entry")
        self._expect("(")
        self._skip_balanced_rest()
        name = self._expect_word()
        self._expect("(")
        params = self._parse_typed_name_list(")")
        self._expect(")")
        self._expect(":")
        rettypes = self._parse_type_list_braced()
        self._expect("=")
        body = self._parse_braced_body()
        return FunDef(name=name, params=params, rettypes=rettypes, body=body, is_entry=True)

    # -- Types. --------------------------------------------------------

    def _parse_type(self) -> str:
        parts = []
        while self._peek_text() == "[":
            self._advance()
            if self._peek_text() != "]":
                parts.append("[" + self._advance().text + "]")
            else:
                parts.append("[]")
            self._expect("]")
        parts.append(self._expect_word())
        # A trailing `#(...)` is an aliasing annotation (e.g. `f32#([2], [0])`)
        # attached to the type by the debug printer; discard it.
        if self._peek_text() == "#":
            self._advance()
            self._expect("(")
            self._skip_balanced_rest()
        return "".join(parts)

    def _parse_type_list_braced(self) -> List[str]:
        self._expect("{")
        types = []
        if self._peek_text() != "}":
            types.append(self._parse_type())
            while self._peek_text() == ",":
                self._advance()
                types.append(self._parse_type())
        self._expect("}")
        return types

    def _parse_typed_name_list(self, closer: str) -> List[Tuple[str, str]]:
        items = []
        if self._peek_text() == closer:
            return items
        while True:
            name = self._expect_word()
            self._expect(":")
            ty = self._parse_type()
            items.append((name, ty))
            if self._peek_text() == ",":
                self._advance()
                continue
            break
        return items

    # -- Bodies and statements. --------------------------------------------

    def _parse_braced_body(self) -> Body:
        """Parse a `{ ... }`-wrapped function/entry body.

        The contents are either a full `stmt* in tuple` core, or -- when
        there are no statements at all -- just a bare result tuple with no
        `in` keyword (e.g. a trivial function body prints as `{ {x} }`: the
        function's own wrapping braces, plus the result tuple's own braces).
        """
        self._expect("{")
        if self._stmt_starts_here():
            body = self._parse_body_core()
        else:
            result = self._parse_tuple()
            body = Body(stmts=[], result=result.elems)
        self._expect("}")
        return body

    def _parse_branch_body(self) -> Body:
        """Parse a `{ ... }`-wrapped `if` then/else branch.

        Unlike a function body (see `_parse_braced_body`), a bare (no
        statements) branch does *not* get an extra nested tuple: `futhark
        dev` prints `then {10i32} else {n}` with a single brace pair per
        branch, not `then { {10i32} }`.
        """
        self._expect("{")
        if self._stmt_starts_here():
            body = self._parse_body_core()
        else:
            body = Body(stmts=[], result=self._parse_expr_list_until("}"))
        self._expect("}")
        return body

    def _parse_body_core(self) -> Body:
        stmts = []
        while self._stmt_starts_here():
            stmts.append(self._parse_stmt())
        self._expect_word("in")
        result = self._parse_tuple()
        return Body(stmts=stmts, result=result.elems)

    def _stmt_starts_here(self) -> bool:
        t = self._peek()
        if t is None:
            return False
        return t.kind == "string" or t.text == "let"

    def _parse_stmt(self) -> Stmt:
        location = None
        if self._peek() is not None and self._peek().kind == "string":
            location = self._advance().text
        self._expect_word("let")
        self._expect("{")
        pattern = self._parse_typed_name_list("}")
        self._expect("}")
        self._expect("=")
        expr = self._parse_expr()
        return Stmt(pattern=pattern, expr=expr, location=location)

    # -- Expressions. --------------------------------------------------------

    def _parse_certs_opt(self) -> List[str]:
        if self._peek_text() == "#":
            self._advance()
            self._expect("{")
            names = self._parse_ident_list("}")
            self._expect("}")
            return names
        return []

    def _parse_ident_list(self, closer: str) -> List[str]:
        names = []
        if self._peek_text() == closer:
            return names
        names.append(self._expect_word())
        while self._peek_text() == ",":
            self._advance()
            names.append(self._expect_word())
        return names

    def _parse_expr(self) -> Expr:
        certs = self._parse_certs_opt()
        e = self._parse_operand()
        if certs:
            e.certs = certs
        return e

    def _maybe_type_annotation(self, node) -> None:
        if self._peek_text() == ":":
            self._advance()
            node.type_annotation = self._parse_type()

    def _parse_expr_list_until(self, closer: str) -> List[Expr]:
        args = []
        if self._peek_text() == closer:
            return args
        args.append(self._parse_expr())
        while self._peek_text() == ",":
            self._advance()
            args.append(self._parse_expr())
        return args

    def _parse_tuple(self) -> Tuple:
        self._expect("{")
        elems = self._parse_expr_list_until("}")
        self._expect("}")
        return Tuple(elems=elems)

    def _parse_apply(self) -> Apply:
        self._expect_word("apply")
        name = self._expect_word()
        self._expect("(")
        args = self._parse_expr_list_until(")")
        self._expect(")")
        rettypes = None
        if self._peek_text() == ":":
            self._advance()
            rettypes = self._parse_type_list_braced()
        return Apply(fn=name, args=args, rettypes=rettypes)

    def _parse_lambda(self) -> Lambda:
        self._expect("\\")
        self._expect("{")
        params = self._parse_typed_name_list("}")
        self._expect("}")
        self._expect(":")
        rettypes = self._parse_type_list_braced()
        self._expect("->")
        body = self._parse_lambda_body()
        return Lambda(params=params, rettypes=rettypes, body=body)

    def _parse_lambda_body(self) -> Body:
        if self._peek_text() == "{":
            result = self._parse_tuple()
            return Body(stmts=[], result=result.elems)
        return self._parse_body_core()

    def _parse_if(self) -> If:
        self._expect_word("if")
        cond = self._parse_expr()
        self._expect_word("then")
        then_body = self._parse_branch_body()
        self._expect_word("else")
        else_body = self._parse_branch_body()
        rettypes = None
        if self._peek_text() == ":":
            self._advance()
            rettypes = self._parse_type_list_braced()
        return If(cond=cond, then_body=then_body, else_body=else_body, rettypes=rettypes)

    def _parse_loop(self) -> Loop:
        self._expect_word("loop")
        self._expect("{")
        pattern = self._parse_typed_name_list("}")
        self._expect("}")
        self._expect("=")
        self._expect("{")
        init = self._parse_expr_list_until("}")
        self._expect("}")

        loop_var = None
        bound = None
        cond_var = None
        if self._peek_text() == "for":
            self._advance()
            var_name = self._expect_word()
            self._expect(":")
            var_type = self._parse_type()
            self._expect("<")
            bound = self._parse_expr()
            loop_var = (var_name, var_type)
        elif self._peek_text() == "while":
            self._advance()
            cond_var = self._expect_word()
        else:
            raise FutharkIRParseError(
                f"Expected 'for' or 'while' in loop but got {self._peek_text()!r}"
            )

        self._expect_word("do")
        body = self._parse_braced_body()
        return Loop(
            pattern=pattern,
            init=init,
            body=body,
            loop_var=loop_var,
            bound=bound,
            cond_var=cond_var,
        )

    def _parse_operand(self) -> Expr:
        t = self._peek()
        if t is None:
            raise FutharkIRParseError("Unexpected end of input")

        if t.text == "{":
            return self._parse_tuple()
        if t.text == "(" or t.text == "[":
            # A bare `(...)`/`[...]` argument, e.g. a `rearrange` permutation
            # tuple, a `reshape` shape-change spec, or a `replicate` shape
            # argument -- see `RawGroup`. A multi-dimensional shape is
            # written as several adjacent `[...]` groups (e.g. `[n][m]`),
            # which together form a single argument.
            self._advance()
            self._skip_balanced_rest()
            while t.text == "[" and self._peek_text() == "[":
                self._advance()
                self._skip_balanced_rest()
            return RawGroup()
        if t.text == "\\":
            return self._parse_lambda()
        if t.text == "apply":
            return self._parse_apply()
        if t.text == "if":
            return self._parse_if()
        if t.text == "loop":
            return self._parse_loop()
        if t.kind == "string":
            self._advance()
            lit = Literal(text=t.text, is_string=True)
            self._maybe_type_annotation(lit)
            return lit
        if t.text == "nilFn":
            self._advance()
            return NilFn()
        if _is_literal_word(t.text):
            self._advance()
            lit = Literal(text=t.text)
            self._maybe_type_annotation(lit)
            return lit

        name = self._advance().text
        nxt = self._peek()

        if nxt is not None and nxt.text == "(":
            self._advance()
            args = self._parse_expr_list_until(")")
            self._expect(")")
            return PrimCall(op=name, args=args)

        if nxt is not None and nxt.text == "[":
            self._advance()
            # Multi-dimensional indexing is written as a single bracket pair
            # with comma-separated indices, e.g. `matrix[i, j]`.
            idxs = self._parse_expr_list_until("]")
            self._expect("]")
            return PrimCall(op="index", args=[VarRef(name=name)] + idxs)

        if nxt is not None and nxt.kind == "word" and _is_type_word(nxt.text):
            from_ty = self._advance().text
            operand = self._parse_expr()
            self._expect_word("to")
            to_ty = self._advance().text
            return PrimCall(op=name, args=[operand], meta={"from": from_ty, "to": to_ty})

        if (
            nxt is not None
            and nxt.kind == "word"
            and nxt.text not in _EXPR_TERMINATORS
        ):
            # A bare `op operand` prefix call, e.g. `neg_bool x`.
            operand = self._parse_expr()
            return PrimCall(op=name, args=[operand])

        var = VarRef(name=name)
        self._maybe_type_annotation(var)
        return var


def parse(text: str) -> Program:
    """Parse the textual Futhark Core IR (as printed by ``futhark dev``)."""
    tokens = tokenize(text)
    parser = _Parser(tokens)
    return parser.parse_program()
