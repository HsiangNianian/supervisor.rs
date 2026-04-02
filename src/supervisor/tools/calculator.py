"""Safe math expression calculator tool."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from supervisor.tools.base import BaseTool

# Allowed operators for safe evaluation.
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> Any:
    """Recursively evaluate an AST node with only safe operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _OPERATORS[op_type](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCTIONS:
            args = [_safe_eval(a) for a in node.args]
            return _SAFE_FUNCTIONS[node.func.id](*args)
        raise ValueError("Unsupported function call")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCTIONS:
            val = _SAFE_FUNCTIONS[node.id]
            if not callable(val):
                return val
        raise ValueError(f"Unsupported name: {node.id}")
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


class CalculatorTool(BaseTool):
    """Safe mathematical expression calculator.

    Evaluates math expressions without using ``eval()``.
    Supports basic arithmetic, common math functions, and constants.
    """

    name = "calculator"
    description = "Evaluate a mathematical expression safely."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate (e.g. '2 + 3 * 4')",
            },
        },
        "required": ["expression"],
    }

    def execute(self, *, expression: str, **_: Any) -> dict:
        """Evaluate a math expression.

        Args:
            expression: The math expression string.

        Returns:
            Dict with 'expression' and 'result' keys.
        """
        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
            return {"expression": expression, "result": result}
        except Exception as exc:
            return {"expression": expression, "error": str(exc)}
