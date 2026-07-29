from __future__ import annotations

import ast


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _negative(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
        or isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value < 0
    )


def static_issues(code: str) -> list[str]:
    issues: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc}"]

    factor_functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (node.name == "factor" or node.name.startswith("F_"))
    ]
    if not factor_functions:
        issues.append("Missing factor function named factor or F_<descriptive_name>.")

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "iloc":
            issues.append(".iloc usage is forbidden; use causal vectorized pandas operations.")
        if isinstance(node, ast.While):
            issues.append("Per-row while loops are forbidden in factor computation.")
        if isinstance(node, ast.For):
            name = _call_name(node.iter.func) if isinstance(node.iter, ast.Call) else ""
            if name in {"range", "enumerate", "iterrows", "itertuples"} or name.endswith((".iterrows", ".itertuples")):
                issues.append("Per-row Python loops are forbidden in factor computation.")
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        method = node.func.attr if isinstance(node.func, ast.Attribute) else ""
        if name in {"np.roll", "numpy.roll"}:
            issues.append("np.roll is circular and can introduce future leakage.")
        if name in {"np.convolve", "numpy.convolve"}:
            issues.append("np.convolve is forbidden; use causal rolling or ewm operations.")
        if name in {"np.polyfit", "numpy.polyfit", "np.linalg.lstsq", "numpy.linalg.lstsq"}:
            issues.append("Per-window regression helpers are forbidden for performance reasons.")
        if method in {"bfill", "backfill"}:
            issues.append("Backward filling introduces future leakage.")
        if method in {"shift", "diff", "pct_change"}:
            periods = list(node.args) + [kw.value for kw in node.keywords if kw.arg == "periods"]
            if any(_negative(value) for value in periods):
                issues.append(f"Negative {method} uses future observations.")
        if method == "rolling":
            if any(kw.arg == "center" and isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in node.keywords):
                issues.append("rolling(center=True) uses future observations.")
        if method == "apply" and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if isinstance(owner, ast.Call) and isinstance(owner.func, ast.Attribute) and owner.func.attr == "rolling":
                issues.append("rolling.apply is forbidden for materialization performance.")
    return list(dict.fromkeys(issues))
