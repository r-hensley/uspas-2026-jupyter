"""Shared utilities for the USPAS 2026 lab helper modules."""

from __future__ import annotations

from dataclasses import dataclass
import os
from collections.abc import Iterable

import numpy as np
import pandas as pd

FODO_HALF_CELL_LENGTH = 2.5      # distance between QF/QD centers [m]
FODO_CELL_LENGTH = 5.0           # full FODO period [m]
FODO_QUAD_LENGTH = 0.5           # full quadrupole length [m]
FODO_BEND_LENGTH = 0.5           # dipole length used in each FODO half-cell [m]
WIDGET_SLIDER_WIDTH = "760px"
WIDGET_DESCRIPTION_WIDTH = "145px"
WIDGET_CONTAINER_WIDTH = "900px"


_TRUE_VALUES = {"1", "true", "True", "yes", "on"}


@dataclass(frozen=True)
class FodoSegment:
    """Representation-independent FODO cell segment."""

    name: str
    kind: str
    length: float
    k1_sign: float = 0.0
    bend_angle_fraction: float = 0.0
    role: str = ""


def fodo_split_quad_lengths(
    quad_length: float = FODO_QUAD_LENGTH,
    half_cell_length: float = FODO_HALF_CELL_LENGTH,
) -> tuple[float, float]:
    """Return the split-QF half length and drift length for one half-cell."""
    quad_length = float(quad_length)
    half_cell_length = float(half_cell_length)
    if quad_length <= 0:
        raise ValueError("quad_length must be positive")
    if quad_length >= half_cell_length:
        raise ValueError("quad_length must be smaller than half_cell_length")
    return 0.5 * quad_length, half_cell_length - quad_length


def fodo_cell_segments(
    *,
    quad_length: float = FODO_QUAD_LENGTH,
    half_cell_length: float = FODO_HALF_CELL_LENGTH,
    with_bends: bool = False,
    bend_length: float = FODO_BEND_LENGTH,
) -> list[FodoSegment]:
    """Return the canonical 5 m split-QF FODO cell geometry.

    The cell starts and ends at the center of a focusing quadrupole:
    QFa, drift, QD, drift, QFb.  When ``with_bends`` is true, a dipole is
    centered in each drift.  The two bend segments each carry half of the
    requested total cell bend angle in lab-specific builders.
    """
    half_quad, drift = fodo_split_quad_lengths(quad_length, half_cell_length)
    if not with_bends:
        return [
            FodoSegment("QFa", "quad", half_quad, k1_sign=1.0, role="focusing quadrupole half"),
            FodoSegment("D1", "drift", drift),
            FodoSegment("QD", "quad", float(quad_length), k1_sign=-1.0, role="defocusing quadrupole"),
            FodoSegment("D2", "drift", drift),
            FodoSegment("QFb", "quad", half_quad, k1_sign=1.0, role="focusing quadrupole half"),
        ]

    bend_length = float(bend_length)
    if bend_length <= 0:
        raise ValueError("bend_length must be positive")
    if bend_length >= drift:
        raise ValueError("bend_length must be smaller than the half-cell drift length")
    split_drift = 0.5 * (drift - bend_length)
    return [
        FodoSegment("QFa", "quad", half_quad, k1_sign=1.0, role="focusing quadrupole half"),
        FodoSegment("D1a", "drift", split_drift),
        FodoSegment("B1", "bend", bend_length, bend_angle_fraction=0.5, role="first FODO bend"),
        FodoSegment("D1b", "drift", split_drift),
        FodoSegment("QD", "quad", float(quad_length), k1_sign=-1.0, role="defocusing quadrupole"),
        FodoSegment("D2a", "drift", split_drift),
        FodoSegment("B2", "bend", bend_length, bend_angle_fraction=0.5, role="second FODO bend"),
        FodoSegment("D2b", "drift", split_drift),
        FodoSegment("QFb", "quad", half_quad, k1_sign=1.0, role="focusing quadrupole half"),
    ]


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


def widget_slider_css() -> str:
    """Return CSS that makes ipywidgets sliders easier to grab in notebooks."""
    return f"""
<style>
.widget-slider, .jupyter-widgets.widget-slider {{
    width: {WIDGET_CONTAINER_WIDTH};
    max-width: 100%;
}}
.widget-slider .noUi-horizontal, .jupyter-widgets.widget-slider .noUi-horizontal {{
    height: 12px;
}}
.widget-slider .noUi-horizontal .noUi-handle,
.jupyter-widgets.widget-slider .noUi-horizontal .noUi-handle {{
    width: 24px;
    height: 24px;
    right: -12px;
    top: -7px;
    border-radius: 50%;
}}
.widget-slider .widget-readout, .jupyter-widgets.widget-slider .widget-readout {{
    min-width: 70px;
}}
</style>
"""


def display_widget_slider_css() -> None:
    """Inject the lab's larger slider CSS in notebook frontends."""
    try:
        from IPython.display import HTML, display
    except Exception:
        return
    display(HTML(widget_slider_css()))


def widget_slider_style(description_width: str = WIDGET_DESCRIPTION_WIDTH) -> dict[str, str]:
    return {"description_width": description_width}


def widget_slider_layout(widgets, width: str = WIDGET_SLIDER_WIDTH):
    return widgets.Layout(width=width, max_width="100%")


def widget_container_layout(widgets, width: str = WIDGET_CONTAINER_WIDTH):
    return widgets.Layout(width=width, max_width="100%")


def lab_float_slider(widgets, **kwargs):
    """Create a larger lab-style FloatSlider."""
    kwargs.setdefault("layout", widget_slider_layout(widgets))
    kwargs.setdefault("style", widget_slider_style())
    kwargs.setdefault("continuous_update", False)
    return widgets.FloatSlider(**kwargs)


def lab_int_slider(widgets, **kwargs):
    """Create a larger lab-style IntSlider."""
    kwargs.setdefault("layout", widget_slider_layout(widgets))
    kwargs.setdefault("style", widget_slider_style())
    kwargs.setdefault("continuous_update", False)
    return widgets.IntSlider(**kwargs)


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


def add_lattice_strip(
    fig,
    layout: pd.DataFrame | None,
    *,
    xref: str = "x",
    yref: str = "paper",
    y: float = 1.045,
    height: float = 0.055,
    show_labels: bool = True,
):
    """Add a compact beamline element strip above an s-axis Plotly figure.

    ``layout`` is expected to contain at least ``name``, ``kind``,
    ``s_start_m``, and ``s_end_m``.  Optional ``k1_m^-2`` values are used to
    draw focusing and defocusing quadrupoles on opposite sides of the baseline.
    The helper mutates and returns ``fig`` so plot functions can call it just
    before showing or returning the figure.
    """
    if layout is None or len(layout) == 0:
        return fig
    required = {"name", "kind", "s_start_m", "s_end_m"}
    if not required.issubset(layout.columns):
        return fig

    rows = layout.copy()
    rows = rows[rows["s_end_m"].astype(float) > rows["s_start_m"].astype(float)]
    if rows.empty:
        return fig

    start = float(rows["s_start_m"].min())
    end = float(rows["s_end_m"].max())
    tick = height * 0.60
    magnet = height * 0.72
    bend = height * 0.90

    # Make room in the top margin for the strip when it is drawn above the plot.
    if yref == "paper" and y + height > 1.0:
        current_top = getattr(getattr(fig.layout, "margin", None), "t", None) or 60
        fig.update_layout(margin=dict(t=max(current_top, 105)))

    fig.add_shape(
        type="line",
        xref=xref,
        yref=yref,
        x0=start,
        x1=end,
        y0=y,
        y1=y,
        line=dict(color="rgba(80, 90, 105, 0.85)", width=2),
        layer="above",
    )

    label_count = 0
    max_labels = 36
    for _, row in rows.iterrows():
        name = str(row["name"])
        kind = str(row["kind"]).lower()
        x0 = float(row["s_start_m"])
        x1 = float(row["s_end_m"])

        fig.add_shape(
            type="line",
            xref=xref,
            yref=yref,
            x0=x0,
            x1=x0,
            y0=y - tick / 2,
            y1=y + tick / 2,
            line=dict(color="rgba(80, 90, 105, 0.55)", width=1),
            layer="above",
        )

        if kind == "drift":
            fig.add_shape(
                type="line",
                xref=xref,
                yref=yref,
                x0=x0,
                x1=x1,
                y0=y,
                y1=y,
                line=dict(color="rgba(120, 130, 145, 0.85)", width=3),
                layer="above",
            )
            continue

        if kind == "quad":
            k1 = row.get("k1_m^-2", np.nan)
            try:
                k1 = float(k1)
            except Exception:
                k1 = np.nan
            if not np.isfinite(k1):
                upper_name = name.upper()
                k1 = -1.0 if "QD" in upper_name else 1.0
            y0, y1 = (y, y + magnet) if k1 >= 0 else (y - magnet, y)
            fill = "rgba(55, 126, 184, 0.78)" if k1 >= 0 else "rgba(228, 95, 86, 0.78)"
            line = "rgba(35, 80, 120, 0.95)" if k1 >= 0 else "rgba(150, 50, 45, 0.95)"
        elif kind == "bend":
            y0, y1 = y - bend / 2, y + bend / 2
            fill = "rgba(245, 158, 11, 0.78)"
            line = "rgba(160, 100, 20, 0.95)"
        else:
            y0, y1 = y - magnet / 2, y + magnet / 2
            fill = "rgba(110, 120, 130, 0.55)"
            line = "rgba(80, 90, 100, 0.95)"

        fig.add_shape(
            type="rect",
            xref=xref,
            yref=yref,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            fillcolor=fill,
            line=dict(color=line, width=1.2),
            layer="above",
        )
        if show_labels and kind != "drift" and label_count < max_labels:
            fig.add_annotation(
                x=0.5 * (x0 + x1),
                y=max(y0, y1) + height * 0.12,
                xref=xref,
                yref=yref,
                text=name,
                showarrow=False,
                font=dict(size=9, color="rgba(45, 55, 72, 0.95)"),
                align="center",
            )
            label_count += 1

    fig.add_shape(
        type="line",
        xref=xref,
        yref=yref,
        x0=end,
        x1=end,
        y0=y - tick / 2,
        y1=y + tick / 2,
        line=dict(color="rgba(80, 90, 105, 0.55)", width=1),
        layer="above",
    )
    return fig
