#!/usr/bin/env python3
"""Basic leakage and cleanliness check for generated student notebooks."""

from __future__ import annotations

from pathlib import Path
import re
import nbformat


STUDENT_NOTEBOOKS = [
    Path("Dispersion_Chromaticity_Local_Lab_Student.ipynb"),
    Path("Quadrupole_Focusing_Xsuite_Student.ipynb"),
]

FORBIDDEN_PATTERNS = [
    re.compile(r'<div\s+class=["\']answer["\']', re.IGNORECASE),
    re.compile(r'\*\*\s*Q\d+\s+answer\.', re.IGNORECASE),
    re.compile(r'\bworked answers are shown in red\b', re.IGNORECASE),
    re.compile(r'\binstructor-development note\b', re.IGNORECASE),
    re.compile(r'\bsolution-red\b', re.IGNORECASE),
    re.compile(r'\bsolution-code\b', re.IGNORECASE),
]

FORBIDDEN_TAGS = {"solution", "solution-code", "solution-red", "instructor-note", "skip-execution"}


def scan(path: Path) -> list[tuple[int, str]]:
    nb = nbformat.read(path, as_version=4)
    hits: list[tuple[int, str]] = []

    for i, cell in enumerate(nb.cells):
        source = cell.get("source", "")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(source):
                hits.append((i, pattern.pattern))

        tags = set(cell.get("metadata", {}).get("tags", []))
        leaked_tags = tags & FORBIDDEN_TAGS
        if leaked_tags:
            hits.append((i, f"metadata tags: {sorted(leaked_tags)}"))

        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                hits.append((i, "code cell still has outputs"))
            if cell.get("execution_count") is not None:
                hits.append((i, "code cell still has execution_count"))

    if "widgets" in nb.metadata:
        hits.append((-1, "notebook metadata still contains widget state"))

    return hits


def main() -> None:
    any_hits = False
    for path in STUDENT_NOTEBOOKS:
        hits = scan(path)
        if hits:
            any_hits = True
            print(f"{path}: possible leakage or uncleared outputs")
            for cell_index, reason in hits:
                print(f"  cell {cell_index}: {reason}")
        else:
            print(f"{path}: no obvious solution leakage; all code outputs cleared")

    if any_hits:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
