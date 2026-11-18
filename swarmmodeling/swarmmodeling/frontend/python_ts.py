from __future__ import annotations

from typing import List, Optional

import tree_sitter_python
from tree_sitter import Language, Node, Parser

from .. import ir

_LANG = Language(tree_sitter_python.language())
_PARSER = Parser(_LANG)


def parse_python_file(path: str, module_name: str) -> ir.Module:
    with open(path, "rb") as f:
        src = f.read()
    return parse_python_source(src, module_name, path)


def parse_python_source(src: bytes, module_name: str, path: str = "<src>") -> ir.Module:
    tree = _PARSER.parse(src)
    lo = _Lowerer(src, path)
    mod = ir.Module(name=module_name, path=path,
                    body=lo.block(tree.root_node),
                    loc=ir.Loc(path, 1))
    mod.imports = lo.imports
    mod.from_imports = lo.from_imports
    mod.star_imports = lo.star_imports
    return mod


class _Lowerer:
    def __init__(self, src: bytes, path: str):
        self.src = src
        self.path = path
        self.imports = {}
        self.from_imports = {}
        self.star_imports = []

    def text(self, n: Node) -> str:
        return self.src[n.start_byte:n.end_byte].decode("utf-8", "replace")

    def loc(self, n: Node) -> ir.Loc:
        return ir.Loc(self.path, n.start_point[0] + 1)

    def named(self, n: Node) -> List[Node]:
        return [c for c in n.named_children if c.type != "comment"]

    def block(self, n: Node) -> List[ir.Stmt]:
        out: List[ir.Stmt] = []
        for c in n.named_children:
            if c.type == "comment":
                continue
            out.extend(self.stmt(c))
        return out

    def stmt(self, n: Node) -> List[ir.Stmt]:
        t = n.type
        L = self.loc(n)

        if t == "expression_statement":
            inner = self.named(n)
            out: List[ir.Stmt] = []
            for c in inner:
                if c.type == "assignment":
                    out.extend(self._assignment(c))
                elif c.type == "augmented_assignment":
                    opn = c.child_by_field_name("operator")
                    op = self.text(opn) if opn is not None else self.text(c.children[1])
                    out.append(ir.AugAssign(self.expr(c.child_by_field_name("left")),
                                            op.rstrip("="),
                                            self.expr(c.child_by_field_name("right")),
                                            loc=L))
                elif c.type in ("string", "concatenated_string"):
                    pass
                else:
                    out.append(ir.ExprStmt(self.expr(c), loc=L))
            return out

        if t == "function_definition":
            return [self._funcdef(n)]
        if t == "decorated_definition":
            d = n.child_by_field_name("definition")
            return self.stmt(d) if d is not None else []
        if t == "class_definition":
            name = self.text(n.child_by_field_name("name"))
            return [ir.ClassDef(name, self.block(n.child_by_field_name("body")), loc=L)]

        if t == "if_statement":
            return [self._if(n)]
        if t == "for_statement":
            return [ir.For(self.expr(n.child_by_field_name("left")),
                           self.expr(n.child_by_field_name("right")),
                           self.block(n.child_by_field_name("body")), loc=L)]
        if t == "while_statement":
            return [ir.While(self.expr(n.child_by_field_name("condition")),
                             self.block(n.child_by_field_name("body")), loc=L)]
        if t == "return_statement":
            vals = self.named(n)
            v = self.expr(vals[0]) if vals else None
            return [ir.Return(v, loc=L)]
        if t == "break_statement":
            return [ir.Break(loc=L)]
        if t == "continue_statement":
            return [ir.Continue(loc=L)]
        if t == "pass_statement":
            return [ir.Pass(loc=L)]

        if t == "with_statement":
            body = n.child_by_field_name("body")
            out = []
            clause = next((c for c in n.children if c.type == "with_clause"), None)
            if clause is not None:
                for item in clause.named_children:
                    v = item.named_children[0] if item.named_children else None
                    if v is not None and v.type == "as_pattern":
                        tgt = v.child_by_field_name("alias")
                        val = v.named_children[0]
                        if tgt is not None:
                            out.append(ir.Assign(self.expr(tgt.named_children[0]
                                                           if tgt.named_children else tgt),
                                                 self.expr(val), loc=L))
            return out + (self.block(body) if body is not None else [])
        if t == "try_statement":
            body = n.child_by_field_name("body")
            return self.block(body) if body is not None else []

        if t == "import_statement":
            for c in self.named(n):
                if c.type == "dotted_name":
                    name = self.text(c)
                    self.imports[name.split(".")[0]] = name
                elif c.type == "aliased_import":
                    name = self.text(c.child_by_field_name("name"))
                    alias = self.text(c.child_by_field_name("alias"))
                    self.imports[alias] = name
            return []
        if t == "import_from_statement":
            modn = n.child_by_field_name("module_name")
            module = self.text(modn).lstrip(".") if modn is not None else ""
            for c in self.named(n):
                if c is modn:
                    continue
                if c.type == "dotted_name":
                    name = self.text(c)
                    if module:
                        self.from_imports[name] = (module, name)
                    else:
                        self.imports[name] = name
                elif c.type == "aliased_import":
                    name = self.text(c.child_by_field_name("name"))
                    alias = self.text(c.child_by_field_name("alias"))
                    if module:
                        self.from_imports[alias] = (module, name)
                    else:
                        self.imports[alias] = name
                elif c.type == "wildcard_import":
                    if module:
                        self.star_imports.append(module)
            return []

        if t in ("assert_statement", "raise_statement", "delete_statement",
                 "global_statement", "nonlocal_statement", "print_statement",
                 "future_import_statement", "type_alias_statement"):
            return []

        return []

    def _assignment(self, n: Node) -> List[ir.Stmt]:
        left = n.child_by_field_name("left")
        right = n.child_by_field_name("right")
        L = self.loc(n)
        if right is None:
            return []
        if right.type == "assignment":
            inner = self._assignment(right)
            first_target = inner[0].target if inner and isinstance(inner[0], ir.Assign) else None
            out = list(inner)
            if first_target is not None:
                out.append(ir.Assign(self.expr(left), first_target, loc=L))
            return out
        return [ir.Assign(self.expr(left), self.expr(right), loc=L)]

    def _slice(self, n: Node) -> ir.SliceE:
        segs: List[List[Node]] = [[]]
        for c in n.children:
            if c.type == ":":
                segs.append([])
            elif c.is_named and c.type != "comment":
                segs[-1].append(c)
        lo = self.expr(segs[0][0]) if segs and segs[0] else None
        hi = self.expr(segs[1][0]) if len(segs) > 1 and segs[1] else None
        return ir.SliceE(lo, hi, loc=self.loc(n))

    def _funcdef(self, n: Node) -> ir.FuncDef:
        name = self.text(n.child_by_field_name("name"))
        params: List[str] = []
        defaults = {}
        pn = n.child_by_field_name("parameters")
        if pn is not None:
            for p in pn.named_children:
                if p.type == "identifier":
                    params.append(self.text(p))
                elif p.type == "typed_parameter":
                    idn = next((c for c in p.named_children if c.type == "identifier"), None)
                    if idn is not None:
                        params.append(self.text(idn))
                elif p.type in ("default_parameter", "typed_default_parameter"):
                    pname = self.text(p.child_by_field_name("name"))
                    params.append(pname)
                    val = p.child_by_field_name("value")
                    if val is not None:
                        defaults[pname] = self.expr(val)
        return ir.FuncDef(name, params, defaults,
                          self.block(n.child_by_field_name("body")), loc=self.loc(n))

    def _if(self, n: Node) -> ir.If:
        cond = self.expr(n.child_by_field_name("condition"))
        body = self.block(n.child_by_field_name("consequence"))
        orelse: List[ir.Stmt] = []
        for alt in [c for c in n.children if c.type in ("elif_clause", "else_clause")]:
            if alt.type == "elif_clause":
                orelse.append(ir.If(self.expr(alt.child_by_field_name("condition")),
                                    self.block(alt.child_by_field_name("consequence")),
                                    [], loc=self.loc(alt)))
            else:
                b = alt.child_by_field_name("body")
                orelse.extend(self.block(b) if b is not None else [])
        if len(orelse) > 1 and isinstance(orelse[0], ir.If):
            head = orelse[0]
            head.orelse = orelse[1:]
            orelse = [head]
        return ir.If(cond, body, orelse, loc=self.loc(n))

    def expr(self, n: Optional[Node]) -> ir.Expr:
        if n is None:
            return ir.Opaque("missing")
        t = n.type
        L = self.loc(n)

        if t == "identifier":
            return ir.NameE(self.text(n), loc=L)
        if t == "integer":
            return ir.Num(float(int(self.text(n).replace("_", ""), 0)), loc=L)
        if t == "float":
            return ir.Num(float(self.text(n).replace("_", "")), loc=L)
        if t == "true":
            return ir.Num(1.0, loc=L)
        if t == "false":
            return ir.Num(0.0, loc=L)
        if t == "none":
            return ir.NameE("None", loc=L)
        if t in ("string", "concatenated_string"):
            return ir.Str(self.text(n), loc=L)

        if t == "attribute":
            return ir.Attr(self.expr(n.child_by_field_name("object")),
                           self.text(n.child_by_field_name("attribute")), loc=L)
        if t == "subscript":
            value_node = n.child_by_field_name("value")
            base = self.expr(value_node)
            subs = [c for c in n.named_children
                    if c.id != value_node.id and c.type != "comment"]
            e: ir.Expr = base
            for s in subs:
                if s.type == "slice":
                    e = ir.Index(e, self._slice(s), loc=L)
                else:
                    e = ir.Index(e, self.expr(s), loc=L)
            return e
        if t == "call":
            fn = self.expr(n.child_by_field_name("function"))
            args: List[ir.Expr] = []
            kwargs = {}
            an = n.child_by_field_name("arguments")
            if an is not None and an.type == "argument_list":
                for a in self.named(an):
                    if a.type == "keyword_argument":
                        kwargs[self.text(a.child_by_field_name("name"))] = \
                            self.expr(a.child_by_field_name("value"))
                    else:
                        args.append(self.expr(a))
            elif an is not None:
                args.append(self.expr(an))
            return ir.Call(fn, args, kwargs, loc=L)

        if t == "binary_operator":
            return ir.BinOp(self.text(n.child_by_field_name("operator")),
                            self.expr(n.child_by_field_name("left")),
                            self.expr(n.child_by_field_name("right")), loc=L)
        if t == "unary_operator":
            return ir.UnOp(self.text(n.children[0]), self.expr(
                n.child_by_field_name("argument")), loc=L)
        if t == "not_operator":
            return ir.UnOp("not", self.expr(n.child_by_field_name("argument")), loc=L)
        if t == "boolean_operator":
            op = self.text(n.child_by_field_name("operator"))
            left = self.expr(n.child_by_field_name("left"))
            right = self.expr(n.child_by_field_name("right"))
            parts = []
            for p in (left, right):
                if isinstance(p, ir.BoolOp) and p.op == op:
                    parts.extend(p.parts)
                else:
                    parts.append(p)
            return ir.BoolOp(op, parts, loc=L)
        if t == "comparison_operator":
            kids = self.named(n)
            ops = [self.text(c) for c in n.children
                   if not c.is_named and c.type not in (",", "(", ")")]
            if len(kids) == 2 and len(ops) >= 1:
                op = "_".join(ops) if len(ops) > 1 else ops[0]
                op = {"is": "is", "is_not": "is_not", "not_in": "not_in"}.get(op, op)
                return ir.CmpOp(op, self.expr(kids[0]), self.expr(kids[1]), loc=L)
            parts = [ir.CmpOp(ops[i], self.expr(kids[i]), self.expr(kids[i + 1]), loc=L)
                     for i in range(len(kids) - 1)]
            return ir.BoolOp("and", parts, loc=L)

        if t == "parenthesized_expression":
            inner = self.named(n)
            return self.expr(inner[0]) if inner else ir.Opaque("empty_paren", loc=L)
        if t in ("tuple", "expression_list", "pattern_list", "tuple_pattern"):
            return ir.TupleE([self.expr(c) for c in self.named(n)], loc=L)
        if t in ("list", "list_pattern"):
            return ir.ListE([self.expr(c) for c in self.named(n)], loc=L)
        if t == "conditional_expression":
            kids = self.named(n)
            if len(kids) == 3:
                return ir.IfExp(self.expr(kids[1]), self.expr(kids[0]),
                                self.expr(kids[2]), loc=L)
            return ir.Opaque("conditional", self.text(n), loc=L)
        if t in ("list_comprehension", "generator_expression", "set_comprehension"):
            body = n.child_by_field_name("body")
            elt = self.expr(body)
            var = None
            it = None
            conds: List[ir.Expr] = []
            for c in self.named(n):
                if c.type == "for_in_clause":
                    var = self.expr(c.child_by_field_name("left"))
                    it = self.expr(c.child_by_field_name("right"))
                elif c.type == "if_clause":
                    conds.append(self.expr(c.named_children[0]))
            if var is None or it is None:
                return ir.Opaque("comprehension", self.text(n), loc=L)
            kind = "list" if t == "list_comprehension" else "gen"
            return ir.Comp(elt, var, it, conds, kind, loc=L)
        if t in ("list_splat", "list_splat_pattern", "dictionary_splat"):
            inner = n.named_children[0] if n.named_children else None
            if inner is not None:
                return ir.UnOp("*", self.expr(inner), loc=L)
            return ir.Opaque("splat", loc=L)
        if t == "await":
            return self.expr(n.named_children[0]) if n.named_children else \
                ir.Opaque("await", loc=L)

        return ir.Opaque(t, self.text(n)[:80], loc=L)
