from __future__ import annotations

from typing import List, Optional

import tree_sitter_cpp
from tree_sitter import Language, Node, Parser

from .. import ir

_LANG = Language(tree_sitter_cpp.language())
_PARSER = Parser(_LANG)


def parse_cpp_file(path: str, module_name: str) -> ir.Module:
    with open(path, "rb") as f:
        src = f.read()
    return parse_cpp_source(src, module_name, path)


def parse_cpp_source(src: bytes, module_name: str, path: str = "<src>") -> ir.Module:
    tree = _PARSER.parse(src)
    lo = _Lowerer(src, path)
    mod = ir.Module(name=module_name, path=path,
                    body=lo.block(tree.root_node),
                    loc=ir.Loc(path, 1))
    mod.star_imports = lo.star_imports
    return mod


class _Lowerer:
    def __init__(self, src: bytes, path: str):
        self.src = src
        self.path = path
        self.star_imports: List[str] = []

    def text(self, n: Node) -> str:
        return self.src[n.start_byte:n.end_byte].decode("utf-8", "replace")

    def loc(self, n: Node) -> ir.Loc:
        return ir.Loc(self.path, n.start_point[0] + 1)

    def named(self, n: Node) -> List[Node]:
        return [c for c in n.named_children if c.type != "comment"]

    def block(self, n: Optional[Node]) -> List[ir.Stmt]:
        out: List[ir.Stmt] = []
        if n is None:
            return out
        if n.type not in ("compound_statement", "translation_unit",
                          "field_declaration_list", "preproc_ifdef",
                          "preproc_if", "preproc_else"):
            return self.stmt(n)
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
                out.extend(self._expr_effect(c))
            return out

        if t == "declaration":
            return self._declaration(n)
        if t == "function_definition":
            f = self._funcdef(n)
            return [f] if f is not None else []
        if t in ("struct_specifier", "class_specifier"):
            return self._classdef(n)

        if t == "if_statement":
            cond = self._condition(n.child_by_field_name("condition"))
            body = self.block(n.child_by_field_name("consequence"))
            alt = n.child_by_field_name("alternative")
            orelse: List[ir.Stmt] = []
            if alt is not None:
                inner = self.named(alt)
                orelse = self.block(inner[0]) if inner else []
            return [ir.If(cond, body, orelse, loc=L)]

        if t == "for_statement":
            return self._for(n)
        if t == "for_range_loop":
            var = self._decl_name(n.child_by_field_name("declarator"))
            it = self.expr(n.child_by_field_name("right"))
            return [ir.For(ir.NameE(var or "_e", loc=L), it,
                           self.block(n.child_by_field_name("body")), loc=L)]
        if t == "while_statement":
            return [ir.While(self._condition(n.child_by_field_name("condition")),
                             self.block(n.child_by_field_name("body")), loc=L)]
        if t == "do_statement":
            return [ir.While(self._condition(n.child_by_field_name("condition")),
                             self.block(n.child_by_field_name("body")), loc=L)]

        if t == "return_statement":
            vals = self.named(n)
            return [ir.Return(self.expr(vals[0]) if vals else None, loc=L)]
        if t == "break_statement":
            return [ir.Break(loc=L)]
        if t == "continue_statement":
            return [ir.Continue(loc=L)]
        if t == "compound_statement":
            return self.block(n)

        if t == "preproc_def":
            name = n.child_by_field_name("name")
            value = n.child_by_field_name("value")
            if name is not None and value is not None:
                try:
                    v = float(self.text(value).strip())
                except ValueError:
                    return []
                return [ir.Assign(ir.NameE(self.text(name)), ir.Num(v), loc=L)]
            return []
        if t == "preproc_include":
            p = n.child_by_field_name("path")
            if p is not None and p.type == "string_literal":
                base = self.text(p).strip('"')
                base = base.rsplit("/", 1)[-1]
                for ext in (".h", ".hpp", ".hh"):
                    if base.endswith(ext):
                        base = base[: -len(ext)]
                        break
                self.star_imports.append(base)
            return []
        if t in ("preproc_ifdef", "preproc_if", "preproc_else"):
            out = []
            for c in self.named(n):
                if c.type in ("identifier", "preproc_arg"):
                    continue
                out.extend(self.stmt(c))
            return out
        if t == "namespace_definition":
            return self.block(n.child_by_field_name("body"))
        if t == "labeled_statement":
            inner = self.named(n)
            return self.stmt(inner[-1]) if inner else []

        return []

    def _condition(self, n: Optional[Node]) -> ir.Expr:
        if n is None:
            return ir.Opaque("missing_cond")
        if n.type == "condition_clause":
            v = n.child_by_field_name("value")
            return self.expr(v if v is not None else
                             (self.named(n)[0] if self.named(n) else None))
        return self.expr(n)

    def _expr_effect(self, c: Node) -> List[ir.Stmt]:
        L = self.loc(c)
        t = c.type
        if t == "assignment_expression":
            op = self.text(c.child_by_field_name("operator"))
            left = self.expr(c.child_by_field_name("left"))
            right = self.expr(c.child_by_field_name("right"))
            if op == "=":
                return [ir.Assign(left, right, loc=L)]
            return [ir.AugAssign(left, op.rstrip("="), right, loc=L)]
        if t == "update_expression":
            arg = self.expr(c.child_by_field_name("argument"))
            op = "+" if "++" in self.text(c) else "-"
            return [ir.AugAssign(arg, op, ir.Num(1.0), loc=L)]
        if t == "comma_expression":
            items: List[Node] = []
            x = c
            while x is not None and x.type == "comma_expression":
                left = x.child_by_field_name("left")
                if left is not None:
                    items.append(left)
                x = x.child_by_field_name("right")
            if x is not None:
                items.append(x)
            if items and items[0].type == "binary_expression":
                opn = items[0].child_by_field_name("operator")
                if opn is not None and self.text(opn) == "<<":
                    target = self.expr(items[0].child_by_field_name("left"))
                    first = self.expr(items[0].child_by_field_name("right"))
                    rest = [self.expr(i) for i in items[1:]]
                    return [ir.Assign(target, ir.ListE([first] + rest), loc=L)]
            out: List[ir.Stmt] = []
            for i in items:
                out.extend(self._expr_effect(i))
            return out
        return [ir.ExprStmt(self.expr(c), loc=L)]

    def _declaration(self, n: Node) -> List[ir.Stmt]:
        L = self.loc(n)
        is_bool = any(c.type == "primitive_type" and self.text(c) == "bool"
                      for c in n.children)
        out: List[ir.Stmt] = []
        for c in n.children:
            if c.type == "init_declarator":
                name = self._decl_name(c.child_by_field_name("declarator"))
                value = c.child_by_field_name("value")
                if name is None or value is None:
                    continue
                if value.type == "argument_list":
                    tname = self._type_name(n)
                    v: ir.Expr = ir.Call(ir.NameE(tname or "?ctor"),
                                         [self.expr(a) for a in self.named(value)],
                                         loc=L)
                elif value.type == "initializer_list":
                    items = self.named(value)
                    v = ir.ListE([self.expr(i) for i in items], loc=L) if items \
                        else ir.Num(0.0)
                else:
                    v = self.expr(value)
                if is_bool and not isinstance(v, (ir.CmpOp, ir.BoolOp)):
                    v = ir.CmpOp("!=", v, ir.Num(0.0), loc=L)
                out.append(ir.Assign(ir.NameE(name, loc=L), v, loc=L, decl=True))
            elif c.type == "function_declarator":
                name = self._decl_name(c.child_by_field_name("declarator"))
                params = c.child_by_field_name("parameters")
                if name is None or params is None or "::" in (name or ""):
                    continue
                args: List[ir.Expr] = []
                ok = bool(self.named(params))
                for p in self.named(params):
                    if p.type == "parameter_declaration" and len(self.named(p)) == 1:
                        args.append(ir.NameE(self.text(self.named(p)[0]), loc=L))
                    else:
                        ok = False
                        break
                if ok:
                    tname = self._type_name(n)
                    out.append(ir.Assign(ir.NameE(name, loc=L),
                                         ir.Call(ir.NameE(tname or "?ctor"), args, loc=L),
                                         loc=L, decl=True))
            elif c.type in ("identifier", "pointer_declarator",
                            "reference_declarator", "array_declarator"):
                name = self._decl_name(c)
                if name is None:
                    continue
                if self._is_container_type(n):
                    out.append(ir.Assign(ir.NameE(name, loc=L), ir.ListE([]),
                                         loc=L, decl=True))
                else:
                    out.append(ir.Assign(ir.NameE(name, loc=L),
                                         ir.Opaque("decl:" + name, loc=L),
                                         loc=L, decl=True))
        return out

    def _type_name(self, decl: Node) -> Optional[str]:
        tn = decl.child_by_field_name("type")
        if tn is None:
            return None
        t = self.text(tn)
        t = t.split("<")[0]
        return t.split("::")[-1].strip()

    def _is_container_type(self, decl: Node) -> bool:
        tn = decl.child_by_field_name("type")
        if tn is None:
            return False
        head = self.text(tn).split("<")[0].split("::")[-1].strip()
        return head in ("vector", "list", "deque", "set", "map", "array")

    def _decl_name(self, n: Optional[Node]) -> Optional[str]:
        while n is not None and n.type in ("pointer_declarator",
                                           "reference_declarator",
                                           "array_declarator",
                                           "parenthesized_declarator"):
            inner = n.child_by_field_name("declarator")
            if inner is None:
                named = [c for c in n.named_children if c.is_named]
                inner = named[0] if named else None
            n = inner
        if n is None:
            return None
        if n.type in ("identifier", "field_identifier", "type_identifier"):
            return self.text(n)
        return None

    def _funcdef(self, n: Node, cls: Optional[str] = None) -> Optional[ir.FuncDef]:
        decl = n.child_by_field_name("declarator")
        while decl is not None and decl.type in ("pointer_declarator",
                                                 "reference_declarator"):
            decl = decl.child_by_field_name("declarator")
        if decl is None or decl.type != "function_declarator":
            return None
        body = n.child_by_field_name("body")
        if body is None:
            return None
        namen = decl.child_by_field_name("declarator")
        if namen is None:
            return None
        name = self.text(namen)
        if "::" in name:
            parts = name.split("::")
            cls = parts[-2]
            name = parts[-1]
        if name.startswith("~"):
            return None
        is_ctor = cls is not None and name == cls
        if is_ctor:
            name = "__init__"

        params: List[str] = []
        defaults = {}
        if cls is not None:
            params.append("this")
        pl = decl.child_by_field_name("parameters")
        if pl is not None:
            for p in self.named(pl):
                if p.type != "parameter_declaration":
                    if p.type == "optional_parameter_declaration":
                        pname = self._decl_name(p.child_by_field_name("declarator"))
                        dv = p.child_by_field_name("default_value")
                        if pname:
                            params.append(pname)
                            if dv is not None:
                                defaults[pname] = self.expr(dv)
                    continue
                pname = self._decl_name(p.child_by_field_name("declarator"))
                if pname:
                    params.append(pname)

        stmts: List[ir.Stmt] = []
        init_list = next((c for c in n.children
                          if c.type == "field_initializer_list"), None)
        if init_list is not None:
            for fi in self.named(init_list):
                if fi.type != "field_initializer":
                    continue
                kids = self.named(fi)
                if len(kids) >= 2 and kids[1].type in ("argument_list",
                                                       "initializer_list"):
                    args = self.named(kids[1])
                    val = self.expr(args[0]) if args else ir.Num(0.0)
                    stmts.append(ir.Assign(
                        ir.Attr(ir.NameE("this"), self.text(kids[0])), val,
                        loc=self.loc(fi)))
        stmts.extend(self.block(body))
        f = ir.FuncDef(name, params, defaults, stmts, loc=self.loc(n))
        f.cls = cls
        return f

    def _classdef(self, n: Node) -> List[ir.Stmt]:
        namen = n.child_by_field_name("name")
        bodyn = n.child_by_field_name("body")
        if namen is None or bodyn is None:
            return []
        cname = self.text(namen)
        body: List[ir.Stmt] = []
        for c in self.named(bodyn):
            if c.type == "function_definition":
                f = self._funcdef(c, cls=cname)
                if f is not None:
                    body.append(f)
        return [ir.ClassDef(cname, body, loc=self.loc(n))]

    def _for(self, n: Node) -> List[ir.Stmt]:
        L = self.loc(n)
        initn = n.child_by_field_name("initializer")
        condn = n.child_by_field_name("condition")
        updn = n.child_by_field_name("update")
        body = self.block(n.child_by_field_name("body"))

        init_stmts = self.stmt(initn) if initn is not None and initn.type == "declaration" \
            else (self._expr_effect(initn) if initn is not None else [])

        var = None
        start = None
        if len(init_stmts) == 1 and isinstance(init_stmts[0], ir.Assign) \
                and isinstance(init_stmts[0].target, ir.NameE):
            var = init_stmts[0].target.id
            start = init_stmts[0].value

        stop = None
        incl = False
        if condn is not None and condn.type == "binary_expression":
            opn = condn.child_by_field_name("operator")
            op = self.text(opn) if opn is not None else ""
            lhs = condn.child_by_field_name("left")
            if op in ("<", "<=") and lhs is not None and var is not None \
                    and self.text(lhs) == var:
                stop = self.expr(condn.child_by_field_name("right"))
                incl = op == "<="

        step: Optional[ir.Expr] = None
        if updn is not None and var is not None:
            ut = updn.type
            if ut == "update_expression" and self.text(
                    updn.child_by_field_name("argument")) == var:
                step = ir.Num(1.0 if "++" in self.text(updn) else -1.0)
            elif ut == "assignment_expression":
                opn = updn.child_by_field_name("operator")
                if opn is not None and self.text(opn) == "+=" \
                        and self.text(updn.child_by_field_name("left")) == var:
                    step = self.expr(updn.child_by_field_name("right"))

        if var is not None and start is not None and stop is not None \
                and step is not None:
            it = ir.Call(ir.NameE("range_incl" if incl else "range"),
                         [start, stop, step], loc=L)
            return [ir.For(ir.NameE(var, loc=L), it, body, loc=L)]

        upd_stmts = self._expr_effect(updn) if updn is not None else []
        cond = self.expr(condn) if condn is not None else ir.Num(1.0)
        return init_stmts + [ir.While(cond, body + upd_stmts, loc=L)]

    def expr(self, n: Optional[Node]) -> ir.Expr:
        if n is None:
            return ir.Opaque("missing")
        t = n.type
        L = self.loc(n)

        if t == "identifier":
            return ir.NameE(self.text(n), loc=L)
        if t == "this":
            return ir.NameE("this", loc=L)
        if t == "qualified_identifier":
            name = n.child_by_field_name("name")
            return self.expr(name) if name is not None and name.is_named \
                else ir.NameE(self.text(n).split("::")[-1], loc=L)
        if t in ("field_identifier", "type_identifier",
                 "namespace_identifier"):
            return ir.NameE(self.text(n), loc=L)
        if t == "number_literal":
            txt = self.text(n).rstrip("fFlLuU").replace("'", "")
            try:
                return ir.Num(float(int(txt, 0)) if txt.lower().startswith("0x")
                              else float(txt), loc=L)
            except ValueError:
                return ir.Opaque("number", txt, loc=L)
        if t in ("true", "false"):
            return ir.Num(1.0 if t == "true" else 0.0, loc=L)
        if t in ("string_literal", "raw_string_literal", "char_literal"):
            return ir.Str(self.text(n).strip('"'), loc=L)
        if t == "nullptr":
            return ir.NameE("None", loc=L)

        if t == "field_expression":
            base = self.expr(n.child_by_field_name("argument"))
            fieldn = n.child_by_field_name("field")
            fname = self.text(fieldn) if fieldn is not None else "?"
            fname = fname.split("<")[0]
            return ir.Attr(base, fname, loc=L)
        if t == "subscript_expression":
            base = self.expr(n.child_by_field_name("argument"))
            idxn = n.child_by_field_name("indices")
            if idxn is not None:
                subs = self.named(idxn)
                e: ir.Expr = base
                for s in subs:
                    e = ir.Index(e, self.expr(s), loc=L)
                return e
            return ir.Index(base, ir.Opaque("index"), loc=L)
        if t == "call_expression":
            fn = self.expr(n.child_by_field_name("function"))
            args: List[ir.Expr] = []
            an = n.child_by_field_name("arguments")
            if an is not None:
                args = [self.expr(a) for a in self.named(an)]
            return ir.Call(fn, args, {}, loc=L)
        if t == "new_expression":
            tn = n.child_by_field_name("type")
            an = n.child_by_field_name("arguments")
            args = [self.expr(a) for a in self.named(an)] if an is not None else []
            tname = self.text(tn).split("::")[-1] if tn is not None else "?new"
            return ir.Call(ir.NameE(tname, loc=L), args, {}, loc=L)

        if t == "binary_expression":
            opn = n.child_by_field_name("operator")
            op = self.text(opn) if opn is not None else "?"
            left = self.expr(n.child_by_field_name("left"))
            right = self.expr(n.child_by_field_name("right"))
            if op in ("&&", "||"):
                lop = "and" if op == "&&" else "or"
                parts = []
                for p in (left, right):
                    if isinstance(p, ir.BoolOp) and p.op == lop:
                        parts.extend(p.parts)
                    else:
                        parts.append(p)
                return ir.BoolOp(lop, parts, loc=L)
            if op in ("<", "<=", ">", ">=", "==", "!="):
                return ir.CmpOp(op, left, right, loc=L)
            if op in ("+", "-", "*", "/", "%"):
                return ir.BinOp(op, left, right, loc=L)
            return ir.Opaque("binop" + op, self.text(n)[:80], loc=L)
        if t == "unary_expression":
            opn = n.child_by_field_name("operator")
            op = self.text(opn) if opn is not None else "-"
            arg = self.expr(n.child_by_field_name("argument"))
            if op == "!":
                return ir.UnOp("not", arg, loc=L)
            return ir.UnOp(op, arg, loc=L) if op in ("-", "+") else arg
        if t == "pointer_expression":
            return self.expr(n.child_by_field_name("argument"))
        if t == "update_expression":
            return self.expr(n.child_by_field_name("argument"))
        if t == "assignment_expression":
            return self.expr(n.child_by_field_name("right"))

        if t == "parenthesized_expression":
            inner = self.named(n)
            return self.expr(inner[0]) if inner else ir.Opaque("empty_paren", loc=L)
        if t == "conditional_expression":
            return ir.IfExp(self.expr(n.child_by_field_name("condition")),
                            self.expr(n.child_by_field_name("consequence")),
                            self.expr(n.child_by_field_name("alternative")), loc=L)
        if t == "initializer_list":
            return ir.ListE([self.expr(c) for c in self.named(n)], loc=L)
        if t == "cast_expression":
            return self.expr(n.child_by_field_name("value"))
        if t == "comma_expression":
            items = self.named(n)
            return self.expr(items[-1]) if items else ir.Opaque("comma", loc=L)

        return ir.Opaque(t, self.text(n)[:80], loc=L)
