#!/usr/bin/env python3
"""Show the recursive local-Python import closure for a script.

This is a static analyzer for repository hygiene. Given a top-level script, it
parses imports, resolves only Python files inside this project, and recursively
prints the local scripts/modules needed by that entry point. It deliberately
omits standard-library and third-party imports.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import os
from pathlib import Path


SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "site-packages",
}


@dataclasses.dataclass(frozen=True)
class ImportEdge:
    target: Path
    line: int
    source: str


@dataclasses.dataclass
class Graph:
    edges: dict[Path, list[ImportEdge]] = dataclasses.field(default_factory=dict)
    missing_localish: dict[Path, list[tuple[int, str]]] = dataclasses.field(default_factory=dict)


def find_project_root(start: Path) -> Path:
    """Prefer the enclosing git root; otherwise use the script's parent."""
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for path in (cur, *cur.parents):
        if (path / ".git").exists():
            return path
    return cur


def iter_project_py(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.endswith(".py"):
                yield Path(dirpath, name).resolve()


def module_candidates(module: str) -> list[Path]:
    parts = [p for p in module.split(".") if p]
    if not parts:
        return []
    rel = Path(*parts)
    return [rel.with_suffix(".py"), rel / "__init__.py"]


def resolve_module(module: str, current: Path, root: Path) -> Path | None:
    """Resolve an import name to a project file, or None for external imports."""
    search_roots = [current.parent, root / "code", root]
    seen_roots: set[Path] = set()
    for base in search_roots:
        base = base.resolve()
        if base in seen_roots:
            continue
        seen_roots.add(base)
        for rel in module_candidates(module):
            candidate = (base / rel).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def resolve_relative(module: str | None, level: int, current: Path,
                     root: Path) -> str | None:
    """Convert a relative import to an approximate dotted project module."""
    if level <= 0:
        return module

    rel_parent = current.parent.resolve().relative_to(root)
    parts = list(rel_parent.parts)
    if current.name == "__init__.py":
        parts.append(current.parent.name)
    if level > len(parts) + 1:
        return module
    base = parts[:len(parts) - level + 1]
    if module:
        base.extend(module.split("."))
    return ".".join(p for p in base if p)


def literal_path_from_node(node: ast.AST, current: Path) -> Path | None:
    """Best-effort extraction of literal .py paths from dynamic import calls."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        value = node.value
    elif isinstance(node, ast.JoinedStr):
        chunks = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                chunks.append(part.value)
            else:
                return None
        value = "".join(chunks)
    elif isinstance(node, ast.Call):
        # os.path.join(code_dir, 'foo.py') or Path(...)/'foo.py' is often too
        # dynamic to solve safely. Keep the useful common case: any literal
        # argument ending in .py is interpreted relative to the current file.
        literals = [
            arg.value for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]
        py_literals = [x for x in literals if x.endswith(".py")]
        if not py_literals:
            return None
        value = py_literals[-1]
    else:
        return None

    if not value.endswith(".py"):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = current.parent / path
    return path.resolve()


def dynamic_import_paths(tree: ast.AST, current: Path, root: Path) -> list[ImportEdge]:
    edges = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        dotted = []
        while isinstance(func, ast.Attribute):
            dotted.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name):
            dotted.append(func.id)
        name = ".".join(reversed(dotted))
        if not name.endswith("spec_from_file_location"):
            continue
        for arg in node.args[1:]:
            path = literal_path_from_node(arg, current)
            if path is None:
                continue
            try:
                path.relative_to(root)
            except ValueError:
                continue
            if path.exists() and path.is_file():
                edges.append(ImportEdge(path, getattr(node, "lineno", 0),
                                        "dynamic spec_from_file_location"))
    return edges


def import_edges_for_file(path: Path, root: Path) -> tuple[list[ImportEdge], list[tuple[int, str]]]:
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        raise SystemExit(f"cannot parse {path}: {exc}") from exc

    edges: list[ImportEdge] = []
    missing: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = resolve_module(alias.name, path, root)
                if target is not None:
                    edges.append(ImportEdge(target, node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            base = resolve_relative(node.module, node.level, path, root)
            if not base:
                continue
            target = resolve_module(base, path, root)
            if target is not None:
                edges.append(ImportEdge(target, node.lineno, f"from {base} import ..."))
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                sub = f"{base}.{alias.name}"
                target = resolve_module(sub, path, root)
                if target is not None:
                    edges.append(ImportEdge(target, node.lineno, f"from {base} import {alias.name}"))

    edges.extend(dynamic_import_paths(tree, path, root))

    # De-duplicate by target while preserving the first line/source encountered.
    dedup: dict[Path, ImportEdge] = {}
    for edge in sorted(edges, key=lambda e: (e.line, str(e.target))):
        dedup.setdefault(edge.target, edge)
    return list(dedup.values()), missing


def build_graph(root_script: Path, project_root: Path) -> Graph:
    graph = Graph()
    stack = [root_script.resolve()]
    visited: set[Path] = set()

    while stack:
        path = stack.pop()
        if path in visited:
            continue
        visited.add(path)
        edges, missing = import_edges_for_file(path, project_root)
        graph.edges[path] = edges
        if missing:
            graph.missing_localish[path] = missing
        for edge in reversed(edges):
            if edge.target not in visited:
                stack.append(edge.target)
    return graph


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def print_tree(path: Path, graph: Graph, root: Path) -> None:
    seen: set[Path] = {path}
    print(rel(path, root))

    def children(parent: Path, prefix: str) -> None:
        edges = sorted(graph.edges.get(parent, []), key=lambda e: rel(e.target, root))
        for i, edge in enumerate(edges):
            last = i == len(edges) - 1
            branch = "`- " if last else "|- "
            marker = "  (already shown)" if edge.target in seen else ""
            print(f"{prefix}{branch}{rel(edge.target, root)}{marker}")
            if edge.target in seen:
                continue
            seen.add(edge.target)
            child_prefix = prefix + ("   " if last else "|  ")
            children(edge.target, child_prefix)

    children(path, "")


def print_flat(root_script: Path, graph: Graph, root: Path,
               include_root: bool = False) -> None:
    files = set(graph.edges)
    for edges in graph.edges.values():
        files.update(edge.target for edge in edges)
    if not include_root:
        files.discard(root_script.resolve())
    for path in sorted(files, key=lambda p: rel(p, root)):
        print(rel(path, root))


def print_edges(graph: Graph, root: Path) -> None:
    for src in sorted(graph.edges, key=lambda p: rel(p, root)):
        for edge in sorted(graph.edges[src], key=lambda e: (rel(e.target, root), e.line)):
            print(f"{rel(src, root)}:{edge.line} -> {rel(edge.target, root)}  [{edge.source}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List project-local Python files imported by a top-level script."
    )
    parser.add_argument("script", help="entry-point Python script, e.g. code/e1_sequential_bandwidth.py")
    parser.add_argument("--root", default=None,
                        help="project root; defaults to enclosing git repository")
    parser.add_argument("--format", choices=("tree", "flat", "edges"), default="tree",
                        help="output format")
    parser.add_argument("--include-root", action="store_true",
                        help="include the entry script in flat output")
    args = parser.parse_args()

    script = Path(args.script).resolve()
    if not script.exists():
        raise SystemExit(f"not found: {script}")
    if script.suffix != ".py":
        raise SystemExit(f"not a Python script: {script}")

    project_root = Path(args.root).resolve() if args.root else find_project_root(script)
    try:
        script.relative_to(project_root)
    except ValueError as exc:
        raise SystemExit(f"{script} is not under project root {project_root}") from exc

    graph = build_graph(script, project_root)
    files = set(graph.edges)
    for edges in graph.edges.values():
        files.update(edge.target for edge in edges)

    print(f"project root: {project_root}")
    print(f"entry script: {rel(script, project_root)}")
    print(f"local Python files in closure: {len(files)}")
    print()

    if args.format == "tree":
        print_tree(script, graph, project_root)
    elif args.format == "flat":
        print_flat(script, graph, project_root, include_root=args.include_root)
    else:
        print_edges(graph, project_root)


if __name__ == "__main__":
    main()
