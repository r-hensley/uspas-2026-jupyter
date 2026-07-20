# USPAS 2026 Jupyter helpers

Helper modules and student notebooks for the USPAS 2026 accelerator physics labs.

## Colab setup

Each notebook installs the helper package from this repository before importing the lab modules:

```ipython
HELPER_VERSION = "main"
HELPER_REPO = "git+https://github.com/r-hensley/uspas-2026-jupyter.git"

%pip install -q --upgrade xsuite
%pip install -q --upgrade --no-cache-dir "{HELPER_REPO}@{HELPER_VERSION}"
```

For a student release, replace `main` with a tag such as `v2026-lab1` after pushing the matching helper code.

## Helper Modules

- `uspas_labs.dispersion_chromaticity` supports the dispersion and chromaticity lab.
- `uspas_labs.quadrupole_focusing` supports the quadrupole focusing Xsuite lab.
- `uspas_labs.shared` contains common notebook/display utilities.

## Student notebooks

- `Dispersion_Chromaticity_Local_Lab_Student.ipynb`
- `Quadrupole_Focusing_Xsuite_Student.ipynb`

Instructor materials and the local notebook-authoring workflow are intentionally
excluded from this public repository.

## Tests

Install the test extras and run pytest from the repository root:

```bash
python3 -m pip install -e ".[test]"
pytest
```
