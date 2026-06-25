#!/usr/bin/env python3
"""Generate student notebooks from tagged instructor solution notebooks.

Edit the instructor solution notebooks, not the generated student notebooks.
Cells tagged with solution/instructor tags are removed entirely.
Outputs are cleared from the remaining code cells.
"""

from __future__ import annotations

from pathlib import Path
import nbformat


NOTEBOOK_PAIRS = [
    (
        Path("Dispersion_Chromaticity_Local_Lab_Instructor_Solutions.ipynb"),
        Path("Dispersion_Chromaticity_Local_Lab_Student.ipynb"),
    ),
    (
        Path("Quadrupole_Focusing_Xsuite_Instructor_Solutions.ipynb"),
        Path("Quadrupole_Focusing_Xsuite_Student.ipynb"),
    ),
]

REMOVE_CELL_TAGS = {
    "solution",
    "solution-code",
    "solution-red",
    "instructor-note",
    "scratch",
}

DROP_WORKFLOW_TAGS_FROM_STUDENT = True


def cell_tags(cell) -> set[str]:
    return set(cell.get("metadata", {}).get("tags", []))


def should_remove_cell(cell) -> bool:
    return bool(cell_tags(cell) & REMOVE_CELL_TAGS)


def clear_outputs(cell) -> None:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def scrub_tags(cell) -> None:
    if DROP_WORKFLOW_TAGS_FROM_STUDENT:
        cell.setdefault("metadata", {})
        cell["metadata"].pop("tags", None)


def build_student_notebook(instructor_path: Path, student_path: Path) -> None:
    nb = nbformat.read(instructor_path, as_version=4)

    kept_cells = []
    for cell in nb.cells:
        if should_remove_cell(cell):
            continue
        clear_outputs(cell)
        scrub_tags(cell)
        kept_cells.append(cell)

    nb.cells = kept_cells

    # Do not preserve internal workflow metadata or widget-state metadata in the student file.
    nb.metadata.pop("lab_workflow", None)
    nb.metadata.pop("widgets", None)

    nbformat.write(nb, student_path)
    print(f"Wrote {student_path}")


def main() -> None:
    for instructor_path, student_path in NOTEBOOK_PAIRS:
        build_student_notebook(instructor_path, student_path)


if __name__ == "__main__":
    main()
