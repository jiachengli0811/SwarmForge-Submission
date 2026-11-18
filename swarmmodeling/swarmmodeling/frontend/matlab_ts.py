from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_matlab

from .. import ir

_LANG = Language(tree_sitter_matlab.language())


def parse_matlab_file(path, module_name=None):
    return parse_matlab_source(Path(path).read_bytes(), module_name or Path(path).stem, str(path))


def parse_matlab_source(src, module_name="matlab", path="<src>"):
    if isinstance(src, str):
        src = src.encode()
    src = src + b"\n"
    root = Parser(_LANG).parse(src).root_node
    if root.has_error:
        pending = [root]
        while pending:
            node = pending.pop(0)
            if node.type == "ERROR" or node.is_missing:
                raise SyntaxError(f"{path}:{node.start_point.row + 1}: MATLAB {node.type}: "
                                  f"{src[node.start_byte:node.end_byte].decode()[:100]}")
            pending[0:0] = node.children
        raise SyntaxError(f"{path}: MATLAB parse error")
    lo = _Lowerer(src, path)
    return ir.Module(module_name, path, lo.block(root), loc=ir.Loc(path, 1), language="matlab")


class _Lowerer:
    def __init__(self, src, path):
        self.src, self.path = src, path
        self.outputs = []

    def text(self, n):
        return self.src[n.start_byte:n.end_byte].decode()

    def loc(self, n):
        return ir.Loc(self.path, n.start_point.row + 1)

    def named(self, n):
        return [c for c in n.named_children if c.type not in ("comment", "line_continuation")]

    def child(self, n, kind):
        return next((c for c in self.named(n) if c.type == kind), None)

    def block(self, n):
        return [s for c in self.named(n) for s in self.stmt(c)] if n else []

    def returns(self, loc):
        values = [ir.NameE(x, loc=loc) for x in self.outputs]
        return ir.Return(values[0] if len(values) == 1 else ir.TupleE(values, loc=loc), loc=loc)

    def stmt(self, n):
        t, L = n.type, self.loc(n)
        if t == "assignment":
            return [ir.Assign(self.expr(n.child_by_field_name("left")),
                              self.expr(n.child_by_field_name("right")), loc=L)]
        if t == "function_definition":
            output = self.child(n, "function_output")
            outs = self.named(output) if output else []
            if outs and outs[0].type == "multioutput_variable":
                outs = self.named(outs[0])
            old, self.outputs = self.outputs, [self.text(c) for c in outs]
            params = self.child(n, "function_arguments")
            f = ir.FuncDef(self.text(n.child_by_field_name("name")),
                           [self.text(c) for c in self.named(params)] if params else [],
                           body=self.block(self.child(n, "block")), loc=L)
            f.body.append(self.returns(L))
            self.outputs = old
            return [f]
        if t == "class_definition":
            name = self.text(n.child_by_field_name("name"))
            body = []
            for c in self.named(n):
                if c.type == "methods":
                    for f in self.block(c):
                        if isinstance(f, ir.FuncDef):
                            f.cls = name
                        body.append(f)
                elif c.type == "properties":
                    for p in self.named(c):
                        if p.type != "property":
                            continue
                        v = self.child(p, "default_value")
                        body.append(ir.Assign(ir.NameE(self.text(p.child_by_field_name("name"))),
                                              self.expr(self.named(v)[0]) if v else ir.MatrixE(), loc=self.loc(p)))
            return [ir.ClassDef(name, body, loc=L)]
        if t in ("if_statement", "elseif_clause"):
            branches = [c for c in self.named(n) if c.type in ("elseif_clause", "else_clause")]
            tail = []
            for c in reversed(branches):
                if c.type == "else_clause":
                    tail = self.block(self.child(c, "block"))
                else:
                    tail = [ir.If(self.expr(c.child_by_field_name("condition")),
                                  self.block(self.child(c, "block")), tail, loc=self.loc(c))]
            return [ir.If(self.expr(n.child_by_field_name("condition")),
                          self.block(self.child(n, "block")), tail, loc=L)]
        if t == "for_statement":
            it = self.named(self.child(n, "iterator"))
            return [ir.For(self.expr(it[0]), self.expr(it[1]), self.block(self.child(n, "block")), loc=L)]
        if t == "while_statement":
            return [ir.While(self.expr(n.child_by_field_name("condition")),
                             self.block(self.child(n, "block")), loc=L)]
        if t == "return_statement":
            return [self.returns(L)]
        if t in ("break_statement", "continue_statement"):
            return [ir.Break(loc=L) if t == "break_statement" else ir.Continue(loc=L)]
        if t in ("attributes", "attribute"):
            return [ir.ExprStmt(ir.Opaque(t, self.text(n), loc=L), loc=L)]
        if t == "block":
            return self.block(n)
        return [ir.ExprStmt(self.expr(n), loc=L)]

    def expr(self, n):
        t, L = n.type, self.loc(n)
        kids = self.named(n)
        if t == "identifier":
            return ir.EndE(loc=L) if self.text(n) == "end" else ir.NameE(self.text(n), loc=L)
        if t == "end_keyword":
            return ir.EndE(loc=L)
        if t == "number":
            try:
                return ir.Num(float(self.text(n)), loc=L)
            except ValueError:
                return ir.Opaque(t, self.text(n), loc=L)
        if t == "string":
            s = self.text(n)
            return ir.Str(s[1:-1].replace(s[0] * 2, s[0]), loc=L)
        if t in ("parenthesis", "parenthesized_expression"):
            return self.expr(kids[0])
        if t in ("binary_operator", "comparison_operator", "boolean_operator"):
            return self.binary(n)
        if t in ("unary_operator", "not_operator", "postfix_operator"):
            return self.binary(n)
        if t == "function_call":
            args = self.child(n, "arguments")
            return ir.Apply(self.expr(n.child_by_field_name("name")),
                            [self.expr(c) for c in self.named(args)] if args else [], loc=L)
        if t == "field_expression":
            base = self.expr(n.child_by_field_name("object"))
            for c in n.children_by_field_name("field"):
                if c.type == "identifier":
                    base = ir.Attr(base, self.text(c), loc=L)
                elif c.type == "function_call":
                    call = self.expr(c)
                    base = ir.Apply(ir.Attr(base, call.base.id, loc=L), call.args, loc=L)
                else:
                    return ir.Opaque(t, self.text(n), loc=L)
            return base
        if t == "matrix":
            return ir.MatrixE([[self.expr(c) for c in self.named(row)] for row in kids], loc=L)
        if t == "multioutput_variable":
            return ir.TupleE([self.expr(c) for c in kids], loc=L)
        if t == "ignored_argument":
            return ir.NameE("~", loc=L)
        if t == "range":
            return ir.RangeE(self.expr(kids[0]), self.expr(kids[-1]),
                             self.expr(kids[1]) if len(kids) == 3 else None, loc=L)
        if t == "spread_operator":
            return ir.AllE(loc=L)
        return ir.Opaque(t, self.text(n), loc=L)

    def binary(self, n):
        precedence = {"||": 1, "&&": 2, "|": 3, "&": 4,
                      "<": 5, ">": 5, "<=": 5, ">=": 5, "==": 5, "~=": 5,
                      "+": 7, "-": 7, "*": 8, "/": 8, "\\": 8,
                      ".*": 8, "./": 8, ".\\": 8, "^": 10, ".^": 10,
                      "u+": 9, "u-": 9, "u~": 9, "post'": 10, "post.'": 10}
        tokens = []

        def flatten(node):
            if node.type in ("binary_operator", "comparison_operator", "boolean_operator"):
                kids = self.named(node)
                flatten(kids[0])
                op = next((self.text(c) for c in node.children if self.text(c) in precedence), None)
                if op is None:
                    raise SyntaxError(f"{self.loc(node)}: unsupported infix operator")
                tokens.append((op, self.loc(node)))
                flatten(kids[-1])
            elif node.type in ("unary_operator", "not_operator"):
                operand = self.named(node)[0]
                op = self.src[node.start_byte:operand.start_byte].decode().strip()
                tokens.append(("u" + op, self.loc(node)))
                flatten(operand)
            elif node.type == "postfix_operator":
                operand = self.named(node)[0]
                flatten(operand)
                op = self.src[operand.end_byte:node.end_byte].decode().strip()
                tokens.append(("post" + op, self.loc(node)))
            else:
                tokens.append(self.expr(node))

        flatten(n)
        values, operators = [], []

        def reduce():
            op, loc = operators.pop()
            if op.startswith("u") or op.startswith("post"):
                values.append(ir.UnOp(op[1:] if op.startswith("u") else op[4:], values.pop(), loc=loc))
                return
            right, left = values.pop(), values.pop()
            if op in ("&&", "||"):
                value = ir.BoolOp("and" if op == "&&" else "or", [left, right], loc=loc)
            elif precedence[op] == 5:
                value = ir.CmpOp("!=" if op == "~=" else op, left, right, loc=loc)
            else:
                value = ir.BinOp(op, left, right, loc=loc)
            values.append(value)

        for token in tokens:
            if isinstance(token, tuple):
                while (not token[0].startswith("u") and operators
                       and precedence[operators[-1][0]] >= precedence[token[0]]):
                    reduce()
                operators.append(token)
            else:
                values.append(token)
        while operators:
            reduce()
        return values[0]
