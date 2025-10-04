
from __future__ import annotations
from typing import Dict, Any
import ast

class SafeEval(ast.NodeVisitor):
    ALLOWED = {
        ast.Expression, ast.Compare, ast.Load, ast.Name, ast.Constant,
        ast.BinOp, ast.BoolOp, ast.UnaryOp,
        ast.And, ast.Or, ast.Not,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
        ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
        ast.USub, ast.UAdd
    }
    def generic_visit(self, node):
        if type(node) not in self.ALLOWED:
            raise ValueError(f"Disallowed node: {type(node).__name__}")
        super().generic_visit(node)

def evaluate_policy(expr: str, ctx: Dict[str, Any]) -> bool:
    tree = ast.parse(expr, mode='eval')
    SafeEval().visit(tree)
    code = compile(tree, "<policy>", "eval")
    return bool(eval(code, {"__builtins__": {}}, dict(ctx)))
