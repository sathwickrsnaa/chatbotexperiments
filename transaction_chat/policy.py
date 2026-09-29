"""Defense in depth for generated analysis code; this is not a security sandbox."""
import ast

from transaction_chat.config import MAX_CODE_CHARS

ALLOWED_IMPORTS = {"pandas", "numpy", "math", "statistics", "decimal", "datetime"}
BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "help", "exit", "quit",
}
BLOCKED_ATTRIBUTES = {
    "to_csv", "to_json", "to_pickle", "to_parquet", "to_excel", "to_sql",
    "to_feather", "to_hdf", "to_clipboard", "to_html", "to_xml", "to_latex",
    "save", "savez", "savez_compressed", "load", "loadtxt", "genfromtxt",
    "memmap", "fromfile", "tofile", "read_pickle", "eval", "query",
    "ctypes", "show", "plot", "hist", "boxplot",
}


def validate_code(code: str):
    if not code.strip() or len(code) > MAX_CODE_CHARS:
        raise ValueError(f"Code must contain 1–{MAX_CODE_CHARS} characters")
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name not in ALLOWED_IMPORTS for alias in node.names):
                raise ValueError("Only data-analysis imports are allowed")
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_IMPORTS:
                raise ValueError("Only data-analysis imports are allowed")
            if any(alias.name.startswith("_") for alias in node.names):
                raise ValueError("Private imports are blocked")
        if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in BLOCKED_NAMES):
            raise ValueError(f"Blocked name: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr.startswith("read_") or node.attr in BLOCKED_ATTRIBUTES:
                raise ValueError(f"Blocked attribute: {node.attr}")
    return tree
