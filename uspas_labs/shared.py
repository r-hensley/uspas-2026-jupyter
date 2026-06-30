"""Shared utilities for the USPAS 2026 lab helper modules."""

from __future__ import annotations

import os
from collections.abc import Iterable

import pandas as pd


_TRUE_VALUES = {"1", "true", "True", "yes", "on"}


def dependency_table(packages: Iterable[str]) -> pd.DataFrame:
    """Return package availability and version information."""
    rows = []
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "installed")
            status = "available"
        except Exception as exc:
            version = "not installed"
            status = f"missing: {exc.__class__.__name__}"
        rows.append({"package": package, "version": version, "status": status})
    return pd.DataFrame(rows)


def maybe_display(obj) -> None:
    """Display an object in notebooks, falling back to print outside IPython."""
    try:
        from IPython.display import display
    except Exception:
        print(obj)
    else:
        display(obj)


def should_show_plot(suppress_env_var: str | None = None) -> bool:
    """Return whether plot helpers should call ``fig.show()``."""
    if suppress_env_var and os.environ.get(suppress_env_var) in _TRUE_VALUES:
        return False
    return os.environ.get("USPAS_LABS_SUPPRESS_PLOTS", "0") not in _TRUE_VALUES


def show_or_return(fig, show: bool = True):
    """Show a Plotly figure once, or return it for custom handling.

    Returning ``None`` after ``fig.show()`` prevents Jupyter from rendering the
    same Plotly figure a second time when the helper call is the last expression
    in a notebook cell.
    """
    if show:
        fig.show()
        return None
    return fig
