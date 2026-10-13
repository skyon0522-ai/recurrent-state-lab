"""Load selected, untouched symbols from pinned package-vendor source files.

This avoids importing upstream CUDA/Triton and trainer dependencies. Callers
provide the explicit CPU bindings; source paths are fixed in this package.
"""
from __future__ import annotations

import ast
import hashlib
from itertools import count
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterable, Mapping


_VENDOR = Path(__file__).resolve().parent / "_vendor"
_SOURCE_IDS = count()


def _selected_source(relative_path: str, names: Iterable[str]):
    path = (_VENDOR / relative_path).resolve()
    if not path.is_relative_to(_VENDOR.resolve()):
        raise ValueError("source path must stay within the package vendor directory")
    source = path.read_bytes()
    tree = ast.parse(source, filename=str(path))
    requested = tuple(names)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("source names must be nonempty and unique")
    symbols = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    missing = set(requested) - symbols.keys()
    if missing:
        raise ValueError(f"source symbols not found: {sorted(missing)}")
    return path, source, [symbols[name] for name in requested]


def source_record(relative_path: str, names: Iterable[str]) -> dict[str, Any]:
    """Return file and exact selected-source hashes for provenance receipts."""
    _, source, nodes = _selected_source(relative_path, names)
    text = source.decode("utf-8")
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(source).hexdigest(),
        "symbols": {
            node.name: hashlib.sha256(ast.get_source_segment(text, node).encode("utf-8")).hexdigest()
            for node in nodes
        },
    }


def load_symbols(
    relative_path: str, names: Iterable[str], bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute only named definitions, preserving their original AST bodies.

    Only package-owned, pinned files are supported. This is not a loader for
    user-provided code. Future annotations make type-only upstream dependencies
    unnecessary, while ordinary runtime dependencies must be supplied explicitly.
    """
    path, _, nodes = _selected_source(relative_path, names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    selected = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    # Real, separate modules support dataclass annotation resolution without
    # sharing the different reference-operation bindings between callers.
    module_name = f"recurrent_state_lab._pinned_source_{next(_SOURCE_IDS)}"
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__dict__.update(bindings or {})
    sys.modules[module_name] = module
    try:
        exec(compile(selected, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return {node.name: module.__dict__[node.name] for node in nodes}
