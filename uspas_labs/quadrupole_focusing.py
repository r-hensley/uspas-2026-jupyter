"""Helper functions for the local Xsuite quadrupole-focusing lab.

The student-facing notebook keeps the important physics knobs visible. This file
contains the repetitive Xsuite line construction, dense Twiss sampling, plotting,
matching optimization, distribution tracking, and widget utilities.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import xtrack as xt

from .shared import (
    FODO_CELL_LENGTH,
    FODO_HALF_CELL_LENGTH,
    FODO_QUAD_LENGTH,
    add_lattice_strip,
    display_widget_slider_css,
    fodo_cell_segments,
    fodo_split_quad_lengths,
    lab_float_slider,
    lab_int_slider,
    should_show_plot,
    show_or_return,
)

GEOMETRIC_EMITTANCE = 6e-6       # 6 mm mrad = 6e-6 m rad
P0C_EV = 1e9                     # 1 GeV/c reference momentum
DEFAULT_K1 = 0.6                 # default quadrupole strength [1/m^2]
HYBRID_SECOND_K1 = 0.5


# -----------------------------------------------------------------------------
# Setup and lattice construction
# -----------------------------------------------------------------------------


def set_plotly_renderer(renderer: str = "notebook_connected") -> None:
    pio.renderers.default = renderer


def print_versions() -> None:
    print(f"xtrack version: {getattr(xt, '__version__', 'unknown')}")
    print(f"plotly version: {getattr(plotly, '__version__', 'unknown')}")
    print(f"plotly renderer: {pio.renderers.default}")


def version_table() -> pd.DataFrame:
    return pd.DataFrame({
        "package": ["xtrack", "plotly"],
        "version": [getattr(xt, "__version__", "unknown"), getattr(plotly, "__version__", "unknown")],
        "note": ["Xsuite tracking/optics", f"renderer: {pio.renderers.default}"],
    })


def electron_ref(p0c: float = P0C_EV) -> xt.Particles:
    return xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV, q0=-1)


def build_line(elements: Sequence[object], names: Sequence[str], *, p0c: float = P0C_EV) -> xt.Line:
    line = xt.Line(elements=list(elements), element_names=list(names))
    line.particle_ref = electron_ref(p0c=p0c)
    line.build_tracker()
    return line


def _drift_lengths(quad_length: float, half_cell_length: float) -> tuple[float, float]:
    """Return half-QF length and drift length for the split-QF FODO cell."""
    return fodo_split_quad_lengths(quad_length, half_cell_length)


def make_fodo_line(
    n_cells: int = 1,
    *,
    k1: float = DEFAULT_K1,
    quad_length: float = FODO_QUAD_LENGTH,
    half_cell_length: float = FODO_HALF_CELL_LENGTH,
    p0c: float = P0C_EV,
    prefix: str = "",
) -> xt.Line:
    """Build the symmetric FODO cell(s) used in the lab.

    Default one-cell sequence: QF half 0.25 m, drift 2.0 m, two adjacent
    QD halves of 0.25 m each, drift 2.0 m, QF half 0.25 m. The line starts
    and ends at the center of a focusing quadrupole, while the split QD
    exposes its center as an element boundary. The full period is 5 m.
    """
    if n_cells < 1:
        raise ValueError("n_cells must be at least 1")
    segments = fodo_cell_segments(quad_length=quad_length, half_cell_length=half_cell_length)
    pre = f"{prefix}_" if prefix else ""
    elements: list[object] = []
    names: list[str] = []
    for i_cell in range(1, n_cells + 1):
        tag = f"c{i_cell:02d}"
        for segment in segments:
            occurrences = [(segment.name, segment.length)]
            if segment.kind == "quad" and segment.name == "QD":
                half_length = 0.5 * segment.length
                occurrences = [("QDa", half_length), ("QDb", half_length)]

            for occurrence_name, occurrence_length in occurrences:
                if segment.kind == "drift":
                    elements.append(xt.Drift(length=occurrence_length))
                elif segment.kind == "quad":
                    elements.append(xt.Quadrupole(length=occurrence_length, k1=segment.k1_sign * k1))
                else:
                    raise ValueError(f"Unsupported FODO segment kind {segment.kind!r}")
                names.append(f"{pre}{occurrence_name}_{tag}")
    return build_line(elements, names, p0c=p0c)


def make_hybrid_fodo_line(
    first_cells: int = 10,
    second_cells: int = 10,
    *,
    k1_first: float = DEFAULT_K1,
    k1_second: float = HYBRID_SECOND_K1,
) -> xt.Line:
    pieces = [
        make_fodo_line(first_cells, k1=k1_first, prefix="strong"),
        make_fodo_line(second_cells, k1=k1_second, prefix="weak"),
    ]
    elements: list[object] = []
    names: list[str] = []
    for piece in pieces:
        for name in piece.element_names:
            if name == "_end_point":
                continue
            elements.append(piece[name].copy())
            names.append(name)
    return build_line(elements, names)


make_hybrid_line = make_hybrid_fodo_line


def compose_labeled_sections(
    pieces: Sequence[tuple[str, str, xt.Line]],
) -> tuple[xt.Line, pd.DataFrame]:
    """Join labeled beamline sections and report their longitudinal spans.

    Each tuple contains a display label, an element-name prefix, and an Xsuite
    line. Elements are copied so the input lines are left unchanged.
    """
    elements: list[object] = []
    names: list[str] = []
    section_rows: list[dict[str, float | str]] = []
    s_position = 0.0

    for section_label, name_prefix, line in pieces:
        s_start = s_position
        for name in line.element_names:
            if name == "_end_point":
                continue
            elements.append(line[name].copy())
            names.append(f"{name_prefix}_{name}" if name_prefix else str(name))
        s_position += float(line.get_length())
        section_rows.append({
            "section": section_label,
            "s_start [m]": s_start,
            "s_end [m]": s_position,
        })

    if len(names) != len(set(names)):
        raise ValueError("Section prefixes do not produce unique element names")
    return build_line(elements, names), pd.DataFrame(section_rows)


def make_match_section(initial_strength: float = DEFAULT_K1) -> xt.Line:
    """Build the four-quadrupole matching section.

    Geometry: 1 m drift, then four 0.5 m quadrupoles, each followed by a
    2 m drift. Default sign pattern is + - + -.
    """
    elements: list[object] = [xt.Drift(length=1.0)]
    names: list[str] = ["DM0"]
    for i in range(1, 5):
        elements.extend([xt.Quadrupole(length=FODO_QUAD_LENGTH, k1=0.0), xt.Drift(length=2.0)])
        names.extend([f"QM{i}", f"DM{i}"])
    line = build_line(elements, names)
    for i, sign in enumerate([1, -1, 1, -1], start=1):
        knob = f"kQM{i}"
        line.vars[knob] = sign * initial_strength
        line.element_refs[f"QM{i}"].k1 = line.vars[knob]
    return line


def set_match_knobs(section: xt.Line, k_values: Sequence[float]) -> None:
    if len(k_values) != 4:
        raise ValueError("Expected four quadrupole strengths")
    for i, value in enumerate(k_values, start=1):
        section.vars[f"kQM{i}"] = float(value)


def make_injection_insertion_line(fodo_cell: xt.Line, match_section: xt.Line) -> xt.Line:
    """Build a simple FODO + matching section + reversed matching section insertion."""
    elements: list[object] = []
    names: list[str] = []
    for prefix, line, iterable in [
        ("fodo", fodo_cell, fodo_cell.element_names),
        ("match", match_section, match_section.element_names),
        ("reverse", match_section, list(reversed(match_section.element_names))),
    ]:
        for name in iterable:
            if name == "_end_point":
                continue
            elements.append(line[name].copy())
            names.append(f"{prefix}_{name}")
    return build_line(elements, names)


# -----------------------------------------------------------------------------
# Twiss calculations and summaries
# -----------------------------------------------------------------------------


def twiss_periodic(line: xt.Line):
    return line.twiss(method="4d")


class _SilentProgress:
    """Minimal iterable wrapper used to hide Xsuite's notebook progress bars."""

    def __init__(self, iterable, **_options):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)


@contextlib.contextmanager
def _without_xtrack_progress():
    """Temporarily suppress Xsuite progress widgets and restore its configuration."""

    config = xt.progress_indicator._config
    indicator = config.default_indicator_cls
    options = dict(config.default_options)
    xt.progress_indicator.set_default_indicator(_SilentProgress)
    try:
        yield
    finally:
        xt.progress_indicator.set_default_indicator(indicator, **options)


@contextlib.contextmanager
def suppress_xsuite_output():
    """Hide Xsuite progress widgets and diagnostic text in a notebook cell."""
    with (
        _without_xtrack_progress(),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        yield


def twiss_dense(line: xt.Line, *, n_points: int | None = None, points_per_meter: float = 200.0, **twiss_kwargs):
    """Compute Twiss functions on a dense uniform s-grid without progress widgets.

    We cut a shallow copy of the line at uniformly spaced s-locations. This is
    slightly heavier than the `at_s` interface, but it is robust both for
    periodic Twiss and for explicitly supplied initial Twiss parameters.
    """
    length = float(line.get_length())
    if n_points is None:
        n_points = max(2, int(np.ceil(length * points_per_meter)) + 1)
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    s_grid = np.linspace(0.0, length, int(n_points))
    sampled = line.copy(shallow=True)
    with (
        _without_xtrack_progress(),
        warnings.catch_warnings(),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        warnings.simplefilter("ignore")
        sampled.cut_at_s(s_grid)
        tw = sampled.twiss(method="4d", **twiss_kwargs)
    return tw


# Compatibility alias used in some drafts.
dense_twiss = twiss_dense


def fodo_end_twiss_dict(tw) -> dict[str, float]:
    return {"betx": float(tw.betx[-1]), "alfx": float(tw.alfx[-1]), "bety": float(tw.bety[-1]), "alfy": float(tw.alfy[-1])}


def initial_from_start(tw) -> dict[str, float]:
    return {"betx": float(tw.betx[0]), "alfx": float(tw.alfx[0]), "bety": float(tw.bety[0]), "alfy": float(tw.alfy[0])}


def initial_from_end(tw) -> dict[str, float]:
    return fodo_end_twiss_dict(tw)


def twiss_dataframe(tw) -> pd.DataFrame:
    df = tw.to_pandas().copy()
    cols = [c for c in ["name", "s", "betx", "bety", "alfx", "alfy", "mux", "muy", "x", "px", "y", "py"] if c in df.columns]
    return df[cols]


def add_beam_sizes(tw, emit_x: float = GEOMETRIC_EMITTANCE, emit_y: float = GEOMETRIC_EMITTANCE) -> pd.DataFrame:
    df = tw.to_pandas().copy()
    df["sigma_x_mm"] = 1e3 * np.sqrt(df["betx"] * emit_x)
    df["sigma_y_mm"] = 1e3 * np.sqrt(df["bety"] * emit_y)
    df["sigma_x_over_y"] = df["sigma_x_mm"] / df["sigma_y_mm"]
    return df


def _line_average(s: Sequence[float], y: Sequence[float]) -> float:
    s_arr = np.asarray(s, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    length = float(s_arr[-1] - s_arr[0]) if len(s_arr) > 1 else 0.0
    if length <= 0:
        return float(np.mean(y_arr))
    return float(np.trapz(y_arr, s_arr) / length)


def phase_advance_table(tw) -> pd.DataFrame:
    return pd.DataFrame({
        "quantity": ["Qx", "Qy", "psi_x [rad]", "psi_y [rad]", "psi_x [deg]", "psi_y [deg]"],
        "value": [float(tw.qx), float(tw.qy), float(2*np.pi*tw.qx), float(2*np.pi*tw.qy), float(360*tw.qx), float(360*tw.qy)],
    })


def beam_size_summary(tw, label: str = "line") -> pd.DataFrame:
    sig = add_beam_sizes(tw)
    qx = getattr(tw, "qx", np.nan)
    qy = getattr(tw, "qy", np.nan)
    rows = [
        ("length [m]", float(sig["s"].iloc[-1])),
        ("Qx", float(qx) if np.isfinite(qx) else np.nan),
        ("Qy", float(qy) if np.isfinite(qy) else np.nan),
        ("phase x [deg]", 360 * float(qx) if np.isfinite(qx) else np.nan),
        ("phase y [deg]", 360 * float(qy) if np.isfinite(qy) else np.nan),
        ("min beta_x [m]", float(sig["betx"].min())),
        ("max beta_x [m]", float(sig["betx"].max())),
        ("min beta_y [m]", float(sig["bety"].min())),
        ("max beta_y [m]", float(sig["bety"].max())),
        ("mean sigma_x [mm]", _line_average(sig["s"], sig["sigma_x_mm"])),
        ("mean sigma_y [mm]", _line_average(sig["s"], sig["sigma_y_mm"])),
        ("min sigma_x [mm]", float(sig["sigma_x_mm"].min())),
        ("max sigma_x [mm]", float(sig["sigma_x_mm"].max())),
        ("min sigma_y [mm]", float(sig["sigma_y_mm"].min())),
        ("max sigma_y [mm]", float(sig["sigma_y_mm"].max())),
        ("max sigma_x/sigma_y", float(sig["sigma_x_over_y"].max())),
    ]
    return pd.DataFrame({"quantity": [r[0] for r in rows], label: [r[1] for r in rows]})


optics_summary = beam_size_summary


def thin_lens_comparison_table(tw_dense, *, length_for_formula: float = FODO_CELL_LENGTH) -> pd.DataFrame:
    psi = 2 * np.pi * float(tw_dense.qx)
    beta_min_tl = length_for_formula * (1 - np.sin(psi/2)) / np.sin(psi)
    beta_max_tl = length_for_formula * (1 + np.sin(psi/2)) / np.sin(psi)
    beta_min_xs = float(np.min(np.asarray(tw_dense.betx, dtype=float)))
    beta_max_xs = float(np.max(np.asarray(tw_dense.betx, dtype=float)))
    return pd.DataFrame({
        "quantity": [
            "beta_min thin lens [m]", "beta_max thin lens [m]",
            "beta_min Xsuite dense [m]", "beta_max Xsuite dense [m]",
            "thin-lens min error [%]", "thin-lens max error [%]",
        ],
        "value": [
            beta_min_tl, beta_max_tl, beta_min_xs, beta_max_xs,
            100*(beta_min_tl-beta_min_xs)/beta_min_xs,
            100*(beta_max_tl-beta_max_xs)/beta_max_xs,
        ],
    })


def _drift_matrix(length: float) -> np.ndarray:
    return np.array([[1.0, float(length)], [0.0, 1.0]])


def _quad_matrix(k1: float, length: float) -> np.ndarray:
    """First-order horizontal thick-lens quadrupole matrix."""
    k1 = float(k1)
    length = float(length)
    if abs(k1) < 1e-14:
        return _drift_matrix(length)
    root = np.sqrt(abs(k1))
    phase = root * length
    if k1 > 0:
        c = np.cos(phase)
        s = np.sin(phase)
        return np.array([[c, s/root], [-root*s, c]])
    c = np.cosh(phase)
    s = np.sinh(phase)
    return np.array([[c, s/root], [root*s, c]])


def _tune_for(k1: float, quad_length: float) -> float:
    """Fast matrix tune used only for the exploratory quadrupole-length scan."""
    half_quad, drift = _drift_lengths(float(quad_length), FODO_HALF_CELL_LENGTH)
    total = (
        _quad_matrix(float(k1), half_quad)
        @ _drift_matrix(drift)
        @ _quad_matrix(-float(k1), float(quad_length))
        @ _drift_matrix(drift)
        @ _quad_matrix(float(k1), half_quad)
    )
    half_trace = 0.5 * float(np.trace(total))
    if abs(half_trace) >= 1.0:
        return np.nan
    return float(np.arccos(half_trace) / (2*np.pi))


def solve_k1_for_tune(target_q: float, quad_length: float) -> float:
    """Choose |k1| that gives approximately the requested one-cell phase advance.

    The solve uses the first-order thick-lens matrix for speed; the table then
    evaluates the resulting lattice with Xsuite. This keeps the notebook
    responsive while retaining Xsuite for the displayed optics.
    """
    target_q = float(target_q)
    lo, hi = 1e-10, 0.1
    for _ in range(60):
        q_hi = _tune_for(hi, quad_length)
        if np.isfinite(q_hi) and q_hi >= target_q:
            break
        hi *= 2.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        q_mid = _tune_for(mid, quad_length)
        if (not np.isfinite(q_mid)) or q_mid >= target_q:
            hi = mid
        else:
            lo = mid
    return float(hi)


def quad_length_fixed_phase_table(
    target_q: float,
    quad_lengths: Sequence[float] = (0.10, 0.20, 0.50, 0.80, 1.10, 1.40, 1.80, 2.20),
) -> pd.DataFrame:
    psi = 2*np.pi*float(target_q)
    beta_min_thin = FODO_CELL_LENGTH * (1 - np.sin(psi/2)) / np.sin(psi)
    beta_max_thin = FODO_CELL_LENGTH * (1 + np.sin(psi/2)) / np.sin(psi)
    rows = []
    for qlen in quad_lengths:
        k1 = solve_k1_for_tune(float(target_q), float(qlen))
        line = make_fodo_line(1, k1=k1, quad_length=float(qlen))
        tw = twiss_dense(line, points_per_meter=300)
        beta_min = float(np.min(tw.betx))
        beta_max = float(np.max(tw.betx))
        rows.append({
            "quad length [m]": float(qlen),
            "|k1| for same phase [1/m^2]": k1,
            "beta_min thick [m]": beta_min,
            "beta_max thick [m]": beta_max,
            "|thick-thin| beta_min [m]": abs(beta_min-beta_min_thin),
            "|thick-thin| beta_max [m]": abs(beta_max-beta_max_thin),
        })
    return pd.DataFrame(rows)


def fodo_region(s: float, *, cell_length: float = FODO_CELL_LENGTH) -> str:
    smod = float(s) % cell_length
    if np.isclose(smod, 0.0, atol=2e-3) or np.isclose(smod, cell_length, atol=2e-3):
        return "cell boundary / focusing quadrupole center"
    if smod <= 0.25 or smod >= 4.75:
        return "focusing quadrupole QF"
    if 2.25 <= smod <= 2.75:
        return "defocusing quadrupole QD"
    return "drift"


def location_table(tw_dense, *, round_tolerance_mm: float = 2e-3) -> pd.DataFrame:
    sig = add_beam_sizes(tw_dense)
    round_metric = np.abs(sig["sigma_x_mm"] - sig["sigma_y_mm"])
    round_s: list[float] = []
    for idx in np.where(round_metric.to_numpy() < round_tolerance_mm)[0]:
        s_val = float(sig.iloc[idx]["s"])
        if not round_s or abs(s_val - round_s[-1]) > 0.05:
            round_s.append(s_val)
    if not round_s:
        round_s = [float(sig.loc[round_metric.idxmin(), "s"])]
    rows = [{
        "condition": "round beam",
        "s [m]": ", ".join(f"{s:.2f}" for s in round_s),
        "lattice region": "mid-drifts / symmetry points",
        "sigma_x [mm]": "same as sigma_y",
        "sigma_y [mm]": "same as sigma_x",
    }]
    for condition, idx in [
        ("max sigma_x", int(sig["sigma_x_mm"].idxmax())),
        ("min sigma_x", int(sig["sigma_x_mm"].idxmin())),
        ("max sigma_y", int(sig["sigma_y_mm"].idxmax())),
        ("min sigma_y", int(sig["sigma_y_mm"].idxmin())),
    ]:
        s_val = float(sig.loc[idx, "s"])
        rows.append({
            "condition": condition,
            "s [m]": s_val,
            "lattice region": fodo_region(s_val),
            "sigma_x [mm]": float(sig.loc[idx, "sigma_x_mm"]),
            "sigma_y [mm]": float(sig.loc[idx, "sigma_y_mm"]),
        })
    return pd.DataFrame(rows)


def cell_start_envelope_table(tw_dense, *, cell_length: float = FODO_CELL_LENGTH) -> pd.DataFrame:
    sig = add_beam_sizes(tw_dense)
    max_cell = int(np.floor(float(sig["s"].max()) / cell_length + 1e-9))
    rows = []
    for i_cell in range(max_cell + 1):
        target_s = i_cell * cell_length
        idx = int(np.argmin(np.abs(sig["s"].to_numpy() - target_s)))
        rows.append({
            "cell index": i_cell,
            "s [m]": float(sig.loc[idx, "s"]),
            "sigma_x [mm]": float(sig.loc[idx, "sigma_x_mm"]),
            "sigma_y [mm]": float(sig.loc[idx, "sigma_y_mm"]),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def _show(fig: go.Figure):
    return show_or_return(fig, should_show_plot("QF_LAB_SUPPRESS_PLOTS"))


def _names(tw) -> np.ndarray:
    try:
        return np.asarray(tw.name, dtype=str)
    except Exception:
        return np.asarray([""] * len(tw.s), dtype=str)


def _strip_base_name(name: str) -> str:
    name = str(name)
    if not name or name in {"_end_point", "START"}:
        return ""
    if ".." in name:
        name = name.split("..", 1)[0]
    for suffix in ("_entry", "_exit"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _kind_from_name(name: str) -> tuple[str, float]:
    upper = name.upper()
    if "BEND" in upper or upper.startswith("B"):
        return "bend", np.nan
    if "QD" in upper:
        return "quad", -1.0
    if "QF" in upper or "QM" in upper or upper.startswith("Q"):
        return "quad", 1.0
    return "drift", np.nan


def _twiss_lattice_layout(tw) -> pd.DataFrame:
    names = _names(tw)
    s_values = np.asarray(tw.s, dtype=float)
    if len(names) < 2 or len(s_values) < 2:
        return pd.DataFrame()

    segments: list[dict[str, float | str]] = []
    for name, s0, s1 in zip(names[:-1], s_values[:-1], s_values[1:]):
        if not np.isfinite(s0) or not np.isfinite(s1) or s1 <= s0:
            continue
        base = _strip_base_name(name)
        if not base:
            continue
        kind, k1 = _kind_from_name(base)
        if segments and segments[-1]["name"] == base and np.isclose(float(segments[-1]["s_end_m"]), float(s0), atol=1e-12):
            segments[-1]["s_end_m"] = float(s1)
        else:
            segments.append({"name": base, "kind": kind, "s_start_m": float(s0), "s_end_m": float(s1), "k1_m^-2": k1})

    if not segments:
        return pd.DataFrame()
    return pd.DataFrame(segments)


def _hover_trace(x, y, text, name, *, mode: str = "lines") -> go.Scatter:
    return go.Scatter(x=x, y=y, mode=mode, name=name, text=text, hovertemplate="%{text}<br>s = %{x:.4g} m<br>value = %{y:.5g}<extra></extra>")


def plot_twiss(tw, title: str = "Twiss functions", show_lattice: bool = True) -> go.Figure:
    names = _names(tw)
    fig = go.Figure()
    fig.add_trace(_hover_trace(tw.s, tw.betx, names, "βx"))
    fig.add_trace(_hover_trace(tw.s, tw.bety, names, "βy"))
    fig.update_layout(title=title, xaxis_title="s [m]", yaxis_title="β [m]", hovermode="closest", template="plotly_white", width=900, height=440)
    if show_lattice:
        add_lattice_strip(fig, _twiss_lattice_layout(tw))
    return _show(fig)


def plot_sigmas(tw, title: str = "RMS beam size", show_lattice: bool = True) -> go.Figure:
    df = add_beam_sizes(tw)
    names = np.asarray(df["name"], dtype=str) if "name" in df.columns else np.asarray([""] * len(df))
    fig = go.Figure()
    fig.add_trace(_hover_trace(df["s"], df["sigma_x_mm"], names, "σx"))
    fig.add_trace(_hover_trace(df["s"], df["sigma_y_mm"], names, "σy"))
    fig.update_layout(title=title, xaxis_title="s [m]", yaxis_title="RMS size [mm]", hovermode="closest", template="plotly_white", width=900, height=440)
    if show_lattice:
        add_lattice_strip(fig, _twiss_lattice_layout(tw))
    return _show(fig)


def plot_beta_and_sigma(tw, title: str = "Twiss functions and RMS beam size", show_lattice: bool = True) -> go.Figure:
    df = add_beam_sizes(tw)
    names = np.asarray(df["name"], dtype=str) if "name" in df.columns else np.asarray([""] * len(df))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Twiss β functions", "RMS beam size"))
    fig.add_trace(_hover_trace(tw.s, tw.betx, names, "βx"), row=1, col=1)
    fig.add_trace(_hover_trace(tw.s, tw.bety, names, "βy"), row=1, col=1)
    fig.add_trace(_hover_trace(df["s"], df["sigma_x_mm"], names, "σx"), row=2, col=1)
    fig.add_trace(_hover_trace(df["s"], df["sigma_y_mm"], names, "σy"), row=2, col=1)
    fig.update_xaxes(title_text="s [m]", row=2, col=1)
    fig.update_yaxes(title_text="β [m]", row=1, col=1)
    fig.update_yaxes(title_text="RMS size [mm]", row=2, col=1)
    fig.update_layout(title=title, hovermode="closest", template="plotly_white", width=900, height=650)
    if show_lattice:
        add_lattice_strip(fig, _twiss_lattice_layout(tw))
    return _show(fig)


def centroid_orbit(line: xt.Line, *, x0: float = 1e-3, px0: float = 0.0, y0: float = 0.0, py0: float = 0.0):
    # Xtrack supports orbit offsets for an explicitly propagated Twiss solution,
    # not for a fully periodic Twiss request. Use the periodic solution only to
    # choose the matched beta/alpha launch envelope.
    base = line.twiss(method="4d")
    return line.twiss(
        method="4d",
        betx=float(base.betx[0]), alfx=float(base.alfx[0]),
        bety=float(base.bety[0]), alfy=float(base.alfy[0]),
        x=x0, px=px0, y=y0, py=py0,
    )


def plot_centroid(centroid, title: str = "Centroid motion and horizontal phase space", show_lattice: bool = True) -> go.Figure:
    """Link the centroid trajectory to phase-space rotation at successive QF centers."""

    s = np.asarray(centroid.s, dtype=float)
    x_mm = 1e3 * np.asarray(centroid.x, dtype=float)
    xp_mrad = 1e3 * np.asarray(centroid.px, dtype=float)
    names = _names(centroid)
    cell_number = s / FODO_CELL_LENGTH
    boundary_mask = np.isclose(cell_number, np.rint(cell_number), atol=2e-3)

    boundary_s = s[boundary_mask]
    boundary_x = x_mm[boundary_mask]
    boundary_xp = xp_mrad[boundary_mask]
    boundary_names = names[boundary_mask]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Centroid trajectory", "Phase space at successive QF centers"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(
            x=s,
            y=x_mm,
            mode="lines",
            line=dict(color="rgba(80, 90, 105, 0.65)", width=1.5),
            name="x(s)",
            text=names,
            hovertemplate="%{text}<br>s = %{x:.4g} m<br>x = %{y:.5g} mm<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=boundary_s,
            y=boundary_x,
            mode="markers",
            marker=dict(size=7, color=boundary_s, coloraxis="coloraxis"),
            name="QF-center samples",
            text=boundary_names,
            customdata=boundary_xp,
            hovertemplate="%{text}<br>s = %{x:.4g} m<br>x = %{y:.5g} mm<br>x' = %{customdata:.5g} mrad<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=boundary_x,
            y=boundary_xp,
            mode="lines",
            line=dict(color="rgba(80, 90, 105, 0.55)", width=1.5),
            name="phase-space path",
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=boundary_x,
            y=boundary_xp,
            mode="markers",
            marker=dict(size=8, color=boundary_s, coloraxis="coloraxis"),
            name="same samples",
            text=boundary_names,
            customdata=boundary_s,
            hovertemplate="%{text}<br>s = %{customdata:.4g} m<br>x = %{x:.5g} mm<br>x' = %{y:.5g} mrad<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="s [m]", dtick=FODO_CELL_LENGTH, row=1, col=1)
    fig.update_yaxes(title_text="x [mm]", row=1, col=1)
    fig.update_xaxes(title_text="x [mm]", row=1, col=2)
    fig.update_yaxes(title_text="x' [mrad]", row=1, col=2)
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1050,
        height=470,
        coloraxis=dict(colorscale="Viridis", colorbar=dict(title="s [m]")),
        legend=dict(orientation="h", yanchor="top", y=-0.18),
    )
    if show_lattice:
        add_lattice_strip(fig, _twiss_lattice_layout(centroid), xref="x")
    return _show(fig)


def plot_mismatch(tw_reference, tw_trial, title: str = "Envelope beating from beta mismatch", show_lattice: bool = True) -> go.Figure:
    ref = add_beam_sizes(tw_reference)
    trial = add_beam_sizes(tw_trial)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ref["s"], y=ref["sigma_x_mm"], mode="lines", name="matched σx"))
    fig.add_trace(go.Scatter(x=trial["s"], y=trial["sigma_x_mm"], mode="lines", name="mismatched σx"))
    fig.add_trace(go.Scatter(x=ref["s"], y=ref["sigma_y_mm"], mode="lines", name="matched σy"))
    fig.add_trace(go.Scatter(x=trial["s"], y=trial["sigma_y_mm"], mode="lines", name="mismatched σy"))
    fig.update_layout(title=title, xaxis_title="s [m]", yaxis_title="RMS size [mm]", template="plotly_white", width=900, height=440)
    if show_lattice:
        add_lattice_strip(fig, _twiss_lattice_layout(tw_reference))
    return _show(fig)


def plot_cell_boundary_envelope(tw_dense, title: str = "Envelope sampled at cell boundaries", show_lattice: bool = True) -> go.Figure:
    table = cell_start_envelope_table(tw_dense)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table["s [m]"], y=table["sigma_x [mm]"], mode="lines+markers", name="σx at cell starts"))
    fig.add_trace(go.Scatter(x=table["s [m]"], y=table["sigma_y [mm]"], mode="lines+markers", name="σy at cell starts"))
    fig.update_layout(title=title, xaxis_title="s [m]", yaxis_title="RMS size [mm]", template="plotly_white", width=900, height=430)
    if show_lattice:
        add_lattice_strip(fig, _twiss_lattice_layout(tw_dense))
    return _show(fig)


def plot_hybrid_transition(tw, *, transition_s: float = 50.0, title: str = "Hybrid line: strong cells followed by weaker cells", show_lattice: bool = True) -> go.Figure:
    fig = go.Figure()
    names = _names(tw)
    fig.add_trace(_hover_trace(tw.s, tw.betx, names, "βx"))
    fig.add_trace(_hover_trace(tw.s, tw.bety, names, "βy"))
    fig.add_vline(x=transition_s, line_dash="dash", annotation_text="transition")
    fig.update_layout(title=title, xaxis_title="s [m]", yaxis_title="β [m]", template="plotly_white", width=900, height=440)
    if show_lattice:
        add_lattice_strip(fig, _twiss_lattice_layout(tw))
    return _show(fig)


def cell_boundary_beta_beat_table(
    transported_twiss,
    matched_twiss_before,
    matched_twiss_after,
    *,
    transition_s: float = 50.0,
    cell_length: float = FODO_CELL_LENGTH,
) -> pd.DataFrame:
    """Compare transported and locally matched beta at equivalent cell boundaries."""
    cell_length = float(cell_length)
    transition_s = float(transition_s)
    if cell_length <= 0:
        raise ValueError("cell_length must be positive")

    line_length = float(np.max(np.asarray(transported_twiss.s, dtype=float)))
    n_cells_float = line_length / cell_length
    if not np.isclose(n_cells_float, np.rint(n_cells_float), atol=1e-9):
        raise ValueError("transported_twiss length must contain a whole number of cells")
    if not np.isclose(transition_s / cell_length, np.rint(transition_s / cell_length), atol=1e-9):
        raise ValueError("transition_s must lie on a cell boundary")

    positions = cell_length * np.arange(int(np.rint(n_cells_float)) + 1, dtype=float)
    reference_before = _twiss_row_near_s(matched_twiss_before, 0.0)
    reference_after = _twiss_row_near_s(matched_twiss_after, 0.0)
    rows: list[dict[str, float]] = []
    for s_value in positions:
        transported = _twiss_row_near_s(transported_twiss, s_value)
        if not np.isclose(float(transported["s"]), s_value, atol=1e-8):
            raise ValueError(f"transported_twiss has no sample at s = {s_value:g} m")
        reference = reference_after if s_value >= transition_s - 1e-12 else reference_before

        betx = float(transported["betx"])
        bety = float(transported["bety"])
        matched_betx = float(reference["betx"])
        matched_bety = float(reference["bety"])
        rows.append({
            "s [m]": s_value,
            "beta_x transported [m]": betx,
            "beta_x matched [m]": matched_betx,
            "beta_x beat [%]": 100.0 * (betx / matched_betx - 1.0),
            "beta_y transported [m]": bety,
            "beta_y matched [m]": matched_bety,
            "beta_y beat [%]": 100.0 * (bety / matched_bety - 1.0),
        })
    return pd.DataFrame(rows)


def plot_cell_boundary_beta_beating(
    transported_twiss,
    matched_twiss_before,
    matched_twiss_after,
    *,
    transition_s: float = 50.0,
    cell_length: float = FODO_CELL_LENGTH,
    title: str = "Beta beating after the lattice-strength change",
) -> go.Figure:
    """Expose mismatch beating by sampling the same optical point each cell."""
    table = cell_boundary_beta_beat_table(
        transported_twiss,
        matched_twiss_before,
        matched_twiss_after,
        transition_s=transition_s,
        cell_length=cell_length,
    )
    fig = go.Figure()
    for column, name, color in [
        ("beta_x beat [%]", "horizontal beta beat", "#1f77b4"),
        ("beta_y beat [%]", "vertical beta beat", "#d62728"),
    ]:
        fig.add_trace(go.Scatter(
            x=table["s [m]"],
            y=table[column],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(color=color, size=7),
            name=name,
            hovertemplate="s = %{x:.4g} m<br>beta deviation = %{y:.4g}%<extra></extra>",
        ))
    fig.add_hline(
        y=0.0,
        line_color="rgba(80, 90, 105, 0.75)",
        line_width=1.5,
        annotation_text="0% = locally matched",
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=float(transition_s),
        line_dash="dash",
        annotation_text="reference changes to weak cell",
        annotation_position="top right",
    )
    fig.update_xaxes(title_text="s [m]", dtick=float(cell_length))
    fig.update_yaxes(title_text="beta deviation from local match [%]", zeroline=False)
    fig.update_layout(
        title=(
            f"{title}<br><sup>sampled at equivalent QF centers: "
            "100 x (beta transported / beta matched - 1)</sup>"
        ),
        template="plotly_white",
        width=900,
        height=440,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    return _show(fig)


def plot_quad_length_scan(
    scan: pd.DataFrame,
    target_q: float,
    title: str = "Finite-length quadrupoles at fixed cell phase advance",
) -> go.Figure:
    """Plot thick-lens beta extrema against fixed thin-lens reference lines."""

    required = {"quad length [m]", "beta_min thick [m]", "beta_max thick [m]"}
    missing = required.difference(scan.columns)
    if missing:
        raise ValueError(f"scan is missing required columns: {sorted(missing)}")

    psi = 2 * np.pi * float(target_q)
    beta_min_thin = FODO_CELL_LENGTH * (1 - np.sin(psi / 2)) / np.sin(psi)
    beta_max_thin = FODO_CELL_LENGTH * (1 + np.sin(psi / 2)) / np.sin(psi)
    lengths = scan["quad length [m]"].to_numpy(dtype=float)

    fig = go.Figure()
    for column, name in [
        ("beta_min thick [m]", "thick-lens βmin"),
        ("beta_max thick [m]", "thick-lens βmax"),
    ]:
        fig.add_trace(go.Scatter(
            x=lengths,
            y=scan[column],
            mode="lines+markers",
            name=name,
            hovertemplate="quadrupole length = %{x:.3g} m<br>β = %{y:.6g} m<extra></extra>",
        ))
    for value, name in [
        (beta_min_thin, "thin-lens βmin"),
        (beta_max_thin, "thin-lens βmax"),
    ]:
        fig.add_trace(go.Scatter(
            x=lengths,
            y=np.full_like(lengths, value),
            mode="lines",
            line=dict(dash="dash"),
            name=name,
            hovertemplate="quadrupole length = %{x:.3g} m<br>β = %{y:.6g} m<extra></extra>",
        ))
    fig.update_layout(
        title=title,
        xaxis_title="quadrupole length [m]",
        yaxis_title="β extremum [m]",
        template="plotly_white",
        width=900,
        height=450,
        hovermode="x unified",
    )
    return _show(fig)


# -----------------------------------------------------------------------------
# Matching and phase-space utilities
# -----------------------------------------------------------------------------


def match_round_section(section: xt.Line, initial_twiss: Mapping[str, float]):
    tw_before = section.twiss(method="4d", **dict(initial_twiss))
    with suppress_xsuite_output():
        opt = section.match(
            method="4d",
            vary=[xt.Vary(f"kQM{i}", step=1e-3, limits=(-3.0, 3.0)) for i in range(1, 5)],
            targets=[
                xt.Target("alfx", 0.0, at="_end_point", tol=1e-8),
                xt.Target("alfy", 0.0, at="_end_point", tol=1e-8),
                xt.Target(lambda tw: tw["betx", "_end_point"] - tw["bety", "_end_point"], 0.0, tol=1e-8),
            ],
            verbose=False,
            n_steps_max=80,
            **dict(initial_twiss),
        )
    tw_after = section.twiss(method="4d", **dict(initial_twiss))
    return opt, tw_before, tw_after


def twiss_endpoint_comparison(
    tw_after,
    target_twiss: Mapping[str, float],
    *,
    keys: Sequence[str] = ("betx", "alfx", "bety", "alfy"),
) -> pd.DataFrame:
    """Compare endpoint Twiss values with a requested matching target."""
    target = dict(target_twiss)
    matched = [float(getattr(tw_after, key)[-1]) for key in keys]
    result = pd.DataFrame({
        "quantity": list(keys),
        "target value": [float(target[key]) for key in keys],
        "matched endpoint": matched,
    })
    result["difference"] = result["matched endpoint"] - result["target value"]
    return result


def matching_summary(section: xt.Line, tw_after) -> pd.DataFrame:
    rows = [
        ("final beta_x [m]", float(tw_after.betx[-1])),
        ("final beta_y [m]", float(tw_after.bety[-1])),
        ("final beta_x - beta_y [m]", float(tw_after.betx[-1] - tw_after.bety[-1])),
        ("final alpha_x", float(tw_after.alfx[-1])),
        ("final alpha_y", float(tw_after.alfy[-1])),
    ]
    for i in range(1, 5):
        rows.append((f"QM{i} k1 [1/m^2]", float(section.vars[f"kQM{i}"]._value)))
    return pd.DataFrame({"quantity": [r[0] for r in rows], "value": [r[1] for r in rows]})


def plot_matching_comparison(tw_before, tw_after, title: str = "Matching section before and after optimization", show_lattice: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(_hover_trace(tw_before.s, tw_before.betx, _names(tw_before), "before βx"))
    fig.add_trace(_hover_trace(tw_before.s, tw_before.bety, _names(tw_before), "before βy"))
    fig.add_trace(_hover_trace(tw_after.s, tw_after.betx, _names(tw_after), "after βx"))
    fig.add_trace(_hover_trace(tw_after.s, tw_after.bety, _names(tw_after), "after βy"))
    fig.update_layout(title=title, xaxis_title="s [m]", yaxis_title="β [m]", template="plotly_white", width=900, height=440)
    if show_lattice:
        add_lattice_strip(fig, _twiss_lattice_layout(tw_after))
    return _show(fig)


def plot_sectioned_twiss(
    tw,
    sections: pd.DataFrame,
    *,
    title: str = "Twiss functions through the composed beamline",
) -> go.Figure:
    """Plot beta functions with labeled shading for composed line sections."""
    required = {"section", "s_start [m]", "s_end [m]"}
    missing = required.difference(sections.columns)
    if missing:
        raise ValueError(f"sections is missing required columns: {sorted(missing)}")

    table = twiss_dataframe(tw)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=table["s"],
        y=table["betx"],
        mode="lines",
        name="βx",
        text=table["name"],
        hovertemplate="%{text}<br>s = %{x:.4g} m<br>βx = %{y:.5g} m<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=table["s"],
        y=table["bety"],
        mode="lines",
        name="βy",
        text=table["name"],
        hovertemplate="%{text}<br>s = %{x:.4g} m<br>βy = %{y:.5g} m<extra></extra>",
    ))

    colors = [
        "rgba(31, 119, 180, 0.10)",
        "rgba(255, 127, 14, 0.12)",
        "rgba(44, 160, 44, 0.10)",
    ]
    for index, (_, row) in enumerate(sections.iterrows()):
        fig.add_vrect(
            x0=row["s_start [m]"],
            x1=row["s_end [m]"],
            fillcolor=colors[index % len(colors)],
            line_width=0,
            layer="below",
            annotation_text=row["section"],
            annotation_position="top left",
        )
    for s_boundary in sections["s_end [m]"].iloc[:-1]:
        fig.add_vline(x=s_boundary, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=title,
        xaxis_title="s [m]",
        yaxis_title="β [m]",
        hovermode="closest",
        template="plotly_white",
        width=950,
        height=460,
    )
    return _show(fig)


def twiss_covariance(beta: float, alpha: float, emit: float = GEOMETRIC_EMITTANCE) -> np.ndarray:
    gamma = (1.0 + alpha**2) / beta
    return np.array([[emit*beta, -emit*alpha], [-emit*alpha, emit*gamma]])


def _sample_distribution(initial_twiss: Mapping[str, float], *, n_particles: int = 3000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x, px = rng.multivariate_normal([0.0, 0.0], twiss_covariance(float(initial_twiss["betx"]), float(initial_twiss["alfx"])), size=n_particles).T
    y, py = rng.multivariate_normal([0.0, 0.0], twiss_covariance(float(initial_twiss["bety"]), float(initial_twiss["alfy"])), size=n_particles).T
    return pd.DataFrame({"x": x, "px": px, "y": y, "py": py})


def _particles_from_dataframe(df: pd.DataFrame) -> xt.Particles:
    return xt.Particles(
        p0c=P0C_EV,
        mass0=xt.ELECTRON_MASS_EV,
        q0=-1,
        x=df["x"].to_numpy(), px=df["px"].to_numpy(),
        y=df["y"].to_numpy(), py=df["py"].to_numpy(),
        delta=np.zeros(len(df)), zeta=np.zeros(len(df)),
    )


def _particles_dataframe(particles: xt.Particles) -> pd.DataFrame:
    return pd.DataFrame({
        "x": np.asarray(particles.x, dtype=float),
        "px": np.asarray(particles.px, dtype=float),
        "y": np.asarray(particles.y, dtype=float),
        "py": np.asarray(particles.py, dtype=float),
    })


def _track_dataframe(line: xt.Line, df: pd.DataFrame) -> pd.DataFrame:
    particles = _particles_from_dataframe(df)
    line.track(particles)
    return _particles_dataframe(particles)


def track_distribution(line: xt.Line, initial_twiss: Mapping[str, float] | pd.DataFrame, *, n_particles: int = 3000, seed: int = 11) -> dict[str, pd.DataFrame]:
    if isinstance(initial_twiss, pd.DataFrame):
        input_df = initial_twiss.copy()
    else:
        input_df = _sample_distribution(initial_twiss, n_particles=n_particles, seed=seed)
    output_df = _track_dataframe(line, input_df)
    return {"input": input_df, "output": output_df}


def track_distribution_along_line(
    line: xt.Line,
    initial_twiss: Mapping[str, float] | pd.DataFrame,
    *,
    n_particles: int = 3000,
    seed: int = 11,
) -> dict[str, object]:
    """Track one sampled bunch and retain the distribution at every element boundary."""

    if isinstance(initial_twiss, pd.DataFrame):
        input_df = initial_twiss.copy()
    else:
        input_df = _sample_distribution(initial_twiss, n_particles=n_particles, seed=seed)

    particles = _particles_from_dataframe(input_df)
    line.track(particles, turn_by_turn_monitor="ONE_TURN_EBE")
    monitor = line.record_last_track

    x = np.asarray(monitor.x, dtype=float)
    px = np.asarray(monitor.px, dtype=float)
    y = np.asarray(monitor.y, dtype=float)
    py = np.asarray(monitor.py, dtype=float)
    s_monitor = np.asarray(monitor.s, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(line.element_names) + 1:
        raise RuntimeError("Unexpected Xsuite element-by-element monitor shape")

    line_table = line.get_table().to_pandas()
    line_table = line_table[line_table["name"] != "_end_point"].reset_index(drop=True)
    if len(line_table) != len(line.element_names):
        raise RuntimeError("Xsuite line table and monitor have inconsistent element counts")

    station_rows = [{
        "station_index": 0,
        "s [m]": float(s_monitor[0, 0]),
        "after_element": "entrance",
        "label": "entrance",
        "is_quad_exit": False,
    }]
    for i_element, row in line_table.iterrows():
        station_index = int(i_element) + 1
        name = str(row["name"])
        is_last = station_index == len(line.element_names)
        station_rows.append({
            "station_index": station_index,
            "s [m]": float(s_monitor[0, station_index]),
            "after_element": name,
            "label": "exit" if is_last else f"after {name}",
            "is_quad_exit": str(row["element_type"]) == "Quadrupole",
        })
    stations = pd.DataFrame(station_rows)

    frames = []
    particle_ids = np.arange(x.shape[0], dtype=int)
    for station in station_rows:
        idx = int(station["station_index"])
        frames.append(pd.DataFrame({
            "station_index": idx,
            "s [m]": float(station["s [m]"]),
            "label": str(station["label"]),
            "after_element": str(station["after_element"]),
            "is_quad_exit": bool(station["is_quad_exit"]),
            "particle_id": particle_ids,
            "x": x[:, idx],
            "px": px[:, idx],
            "y": y[:, idx],
            "py": py[:, idx],
        }))
    snapshots = pd.concat(frames, ignore_index=True)

    final_station = int(stations["station_index"].iloc[-1])
    output_df = snapshots[snapshots["station_index"] == final_station][
        ["x", "px", "y", "py"]
    ].reset_index(drop=True)
    return {
        "input": input_df.reset_index(drop=True),
        "output": output_df,
        "snapshots": snapshots,
        "stations": stations,
    }


def _rms_emittance(df: pd.DataFrame, coordinate: str, angle: str) -> float:
    covariance = np.cov(df[[coordinate, angle]].to_numpy(dtype=float), rowvar=False, bias=True)
    return float(np.sqrt(max(float(np.linalg.det(covariance)), 0.0)))


def rms_emittance_table(tracked: Mapping[str, object]) -> pd.DataFrame:
    """Summarize tracked horizontal and vertical RMS emittance at each station."""

    snapshots = tracked["snapshots"]
    stations = tracked["stations"]
    if not isinstance(snapshots, pd.DataFrame) or not isinstance(stations, pd.DataFrame):
        raise TypeError("tracked data must come from track_distribution_along_line")

    rows = []
    for _, station in stations.iterrows():
        cloud = snapshots[snapshots["station_index"] == int(station["station_index"])]
        rows.append({
            "station_index": int(station["station_index"]),
            "s [m]": float(station["s [m]"]),
            "label": str(station["label"]),
            "epsilon_x [mm mrad]": 1e6 * _rms_emittance(cloud, "x", "px"),
            "epsilon_y [mm mrad]": 1e6 * _rms_emittance(cloud, "y", "py"),
        })
    table = pd.DataFrame(rows)
    table["epsilon_x / entrance"] = table["epsilon_x [mm mrad]"] / table["epsilon_x [mm mrad]"].iloc[0]
    table["epsilon_y / entrance"] = table["epsilon_y [mm mrad]"] / table["epsilon_y [mm mrad]"].iloc[0]
    return table


def _twiss_row_near_s(tw, s_position: float) -> pd.Series:
    table = tw.to_pandas()
    idx = int(np.argmin(np.abs(table["s"].to_numpy(dtype=float) - float(s_position))))
    return table.iloc[idx]


def _filmstrip_stations(tracked: Mapping[str, object]) -> pd.DataFrame:
    stations = tracked["stations"]
    if not isinstance(stations, pd.DataFrame):
        raise TypeError("tracked data must come from track_distribution_along_line")
    last_index = int(stations["station_index"].iloc[-1])
    selected = stations[
        (stations["station_index"] == 0)
        | stations["is_quad_exit"].astype(bool)
        | (stations["station_index"] == last_index)
    ]
    return selected.drop_duplicates("station_index").reset_index(drop=True)


def plot_phase_space_filmstrip(
    tracked: Mapping[str, object],
    tw,
    *,
    plane: str = "x",
    title: str = "Tracked bunch through the matching section",
) -> go.Figure:
    """Show the same particles at the entrance, quadrupole exits, and line exit."""

    if plane not in {"x", "y"}:
        raise ValueError("plane must be 'x' or 'y'")
    coordinate = plane
    angle = f"p{plane}"
    beta_name = f"bet{plane}"
    alpha_name = f"alf{plane}"

    snapshots = tracked["snapshots"]
    if not isinstance(snapshots, pd.DataFrame):
        raise TypeError("tracked data must come from track_distribution_along_line")
    selected = _filmstrip_stations(tracked)

    entrance = snapshots[snapshots["station_index"] == 0].sort_values("particle_id")
    entrance_twiss = _twiss_row_near_s(tw, 0.0)
    phase_color = np.mod(
        np.degrees(np.arctan2(
            -(
                float(entrance_twiss[alpha_name]) * entrance[coordinate].to_numpy()
                + float(entrance_twiss[beta_name]) * entrance[angle].to_numpy()
            ),
            entrance[coordinate].to_numpy(),
        )),
        360.0,
    )

    panel_data = []
    x_extent = []
    xp_extent = []
    for _, station in selected.iterrows():
        station_index = int(station["station_index"])
        cloud = snapshots[snapshots["station_index"] == station_index].sort_values("particle_id")
        twiss_row = _twiss_row_near_s(tw, float(station["s [m]"]))
        ellipse_x, ellipse_xp = _ellipse_points(
            float(twiss_row[beta_name]),
            float(twiss_row[alpha_name]),
        )
        cloud_x = 1e3 * cloud[coordinate].to_numpy(dtype=float)
        cloud_xp = 1e3 * cloud[angle].to_numpy(dtype=float)
        panel_data.append((station, cloud, cloud_x, cloud_xp, ellipse_x, ellipse_xp))
        x_extent.extend([cloud_x, ellipse_x])
        xp_extent.extend([cloud_xp, ellipse_xp])

    x_limit = 1.05 * max(float(np.max(np.abs(values))) for values in x_extent)
    xp_limit = 1.05 * max(float(np.max(np.abs(values))) for values in xp_extent)
    n_panels = len(panel_data)
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))
    subplot_titles = [
        f"{station['label']}<br><sup>s = {float(station['s [m]']):.3g} m</sup>"
        for station, *_ in panel_data
    ]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.07,
        vertical_spacing=0.13,
    )

    for i_panel, (station, cloud, cloud_x, cloud_xp, ellipse_x, ellipse_xp) in enumerate(panel_data):
        row = i_panel // n_cols + 1
        col = i_panel % n_cols + 1
        fig.add_trace(
            go.Scattergl(
                x=cloud_x,
                y=cloud_xp,
                mode="markers",
                marker=dict(
                    size=3,
                    opacity=0.42,
                    color=phase_color,
                    colorscale="HSV",
                    cmin=0.0,
                    cmax=360.0,
                    showscale=i_panel == 0,
                    colorbar=dict(title="initial phase [deg]"),
                ),
                customdata=cloud["particle_id"].to_numpy(dtype=int),
                hovertemplate="particle %{customdata}<br>position = %{x:.5g} mm<br>angle = %{y:.5g} mrad<extra></extra>",
                name="tracked particles",
                showlegend=i_panel == 0,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=ellipse_x,
                y=ellipse_xp,
                mode="lines",
                line=dict(color="black", width=1.5),
                name="local 1-rms ellipse",
                hoverinfo="skip",
                showlegend=i_panel == 0,
            ),
            row=row,
            col=col,
        )

    fig.update_xaxes(range=[-x_limit, x_limit], title_text=f"{plane} [mm]")
    fig.update_yaxes(range=[-xp_limit, xp_limit], title_text=f"{plane}' [mrad]")
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1050,
        height=310 * n_rows,
        legend=dict(orientation="h", yanchor="top", y=-0.08),
    )
    return _show(fig)


def plot_emittance_conservation(
    tracked: Mapping[str, object],
    title: str = "Entrance-normalized RMS emittance through the matching section",
    *,
    transition_s: float | None = None,
) -> go.Figure:
    """Display entrance-normalized horizontal and vertical RMS emittance."""

    table = rms_emittance_table(tracked)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Horizontal", "Vertical"),
    )
    for row, column, name in [
        (1, "epsilon_x / entrance", "epsilon_x / entrance"),
        (2, "epsilon_y / entrance", "epsilon_y / entrance"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=table["s [m]"],
                y=table[column],
                mode="lines+markers",
                name=name,
                text=table["label"],
                hovertemplate="%{text}<br>s = %{x:.4g} m<br>ratio = %{y:.12g}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        deviation = float(np.max(np.abs(table[column].to_numpy(dtype=float) - 1.0)))
        half_span = max(5e-5, 1.5 * deviation)
        fig.update_yaxes(range=[1.0 - half_span, 1.0 + half_span], title_text="epsilon / entrance", row=row, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=row, col=1)

    fig.update_xaxes(title_text="s [m]", row=2, col=1)
    if transition_s is not None:
        for row in (1, 2):
            fig.add_vline(
                x=float(transition_s),
                line_color="#d62728",
                line_dash="dash",
                line_width=1.5,
                row=row,
                col=1,
            )
    fig.update_layout(title=title, template="plotly_white", width=900, height=560)
    return _show(fig)


def plot_tracked_xpx_distribution(tracked: dict[str, pd.DataFrame], title: str = "Tracked input/output horizontal phase space") -> go.Figure:
    input_df, output_df = tracked["input"], tracked["output"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=1e3*input_df["x"], y=1e3*input_df["px"], mode="markers", marker=dict(size=3, opacity=0.35), name="input x-x'", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=1e3*output_df["x"], y=1e3*output_df["px"], mode="markers", marker=dict(size=3, opacity=0.35), name="output x-x'", hoverinfo="skip"))
    fig.update_layout(title=title, xaxis_title="x [mm]", yaxis_title="x' [mrad]", template="plotly_white", width=720, height=600)
    return _show(fig)


def _ellipse_points(beta: float, alpha: float, emit: float = GEOMETRIC_EMITTANCE, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    phase = np.linspace(0.0, 2*np.pi, n)
    x = np.sqrt(emit*beta) * np.cos(phase)
    xp = -np.sqrt(emit/beta) * (alpha*np.cos(phase) + np.sin(phase))
    return 1e3*x, 1e3*xp


def _stations_at_s(
    tracked: Mapping[str, object],
    s_positions: Sequence[float],
    *,
    atol: float = 1e-9,
) -> pd.DataFrame:
    """Select unique tracked stations at requested longitudinal positions."""
    stations = tracked["stations"]
    if not isinstance(stations, pd.DataFrame):
        raise TypeError("tracked must come from track_distribution_along_line")

    station_s = stations["s [m]"].to_numpy(dtype=float)
    selected_rows: list[dict[str, object]] = []
    for requested_s in s_positions:
        nearest_index = int(np.argmin(np.abs(station_s - float(requested_s))))
        nearest = stations.iloc[nearest_index]
        if not np.isclose(float(nearest["s [m]"]), float(requested_s), rtol=0.0, atol=atol):
            raise ValueError(f"No tracked station at s = {float(requested_s):g} m")
        selected_rows.append(nearest.to_dict())

    selected = pd.DataFrame(selected_rows)
    if selected["station_index"].duplicated().any():
        raise ValueError("Requested positions must select distinct tracked stations")
    return selected.reset_index(drop=True)


def plot_phase_space_evolution(
    tracked: Mapping[str, object],
    transported_twiss,
    matched_twiss_before,
    matched_twiss_after,
    *,
    s_positions: Sequence[float] = (45.0, 50.0, 55.0, 60.0, 65.0, 70.0),
    transition_s: float = 50.0,
    plane: str = "x",
    reference_levels: Sequence[float] = (0.75, 1.5, 2.25, 3.0, 3.75),
    transported_level: float = 3.0,
    title: str = "Phase-space evolution across the lattice change",
) -> go.Figure:
    """Show one transported distribution against the locally matched ellipses."""
    if plane not in {"x", "y"}:
        raise ValueError("plane must be 'x' or 'y'")
    if transported_level <= 0 or any(float(level) <= 0 for level in reference_levels):
        raise ValueError("Ellipse levels must be positive")

    snapshots = tracked["snapshots"]
    if not isinstance(snapshots, pd.DataFrame):
        raise TypeError("tracked must come from track_distribution_along_line")
    selected = _stations_at_s(tracked, s_positions)

    coordinate = plane
    angle = f"p{plane}"
    beta_name = f"bet{plane}"
    alpha_name = f"alf{plane}"
    reference_before = _twiss_row_near_s(matched_twiss_before, 0.0)
    reference_after = _twiss_row_near_s(matched_twiss_after, 0.0)
    contour_colors = ("#443983", "#31688e", "#21918c", "#35b779", "#90d743")

    panel_data: list[dict[str, object]] = []
    x_extent: list[np.ndarray] = []
    xp_extent: list[np.ndarray] = []
    for _, station in selected.iterrows():
        station_index = int(station["station_index"])
        s_value = float(station["s [m]"])
        cloud = snapshots[snapshots["station_index"] == station_index].sort_values("particle_id")
        transported_row = _twiss_row_near_s(transported_twiss, s_value)
        reference_row = reference_after if s_value >= float(transition_s) - 1e-12 else reference_before

        cloud_x = 1e3 * cloud[coordinate].to_numpy(dtype=float)
        cloud_xp = 1e3 * cloud[angle].to_numpy(dtype=float)
        transported_x, transported_xp = _ellipse_points(
            float(transported_row[beta_name]),
            float(transported_row[alpha_name]),
        )
        reference_x, reference_xp = _ellipse_points(
            float(reference_row[beta_name]),
            float(reference_row[alpha_name]),
        )
        panel_data.append({
            "station": station,
            "cloud": cloud,
            "cloud_x": cloud_x,
            "cloud_xp": cloud_xp,
            "transported_x": transported_level * transported_x,
            "transported_xp": transported_level * transported_xp,
            "reference_x": reference_x,
            "reference_xp": reference_xp,
        })
        x_extent.extend([cloud_x, transported_level * transported_x])
        xp_extent.extend([cloud_xp, transported_level * transported_xp])
        for level in reference_levels:
            x_extent.append(float(level) * reference_x)
            xp_extent.append(float(level) * reference_xp)

    x_limit = 1.04 * max(float(np.max(np.abs(values))) for values in x_extent)
    xp_limit = 1.04 * max(float(np.max(np.abs(values))) for values in xp_extent)
    n_panels = len(panel_data)
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    subplot_titles = []
    for panel in panel_data:
        s_value = float(panel["station"]["s [m]"])
        suffix = " (new lattice)" if np.isclose(s_value, transition_s, atol=1e-12) else ""
        subplot_titles.append(f"s = {s_value:g} m{suffix}")

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
    for i_panel, panel in enumerate(panel_data):
        row = i_panel // n_cols + 1
        col = i_panel % n_cols + 1
        for i_level, level in enumerate(reference_levels):
            fig.add_trace(
                go.Scatter(
                    x=float(level) * panel["reference_x"],
                    y=float(level) * panel["reference_xp"],
                    mode="lines",
                    line=dict(color=contour_colors[i_level % len(contour_colors)], width=1.2),
                    name="locally matched contours",
                    legendgroup="reference contours",
                    showlegend=i_panel == 0 and i_level == 0,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
        fig.add_trace(
            go.Scattergl(
                x=panel["cloud_x"],
                y=panel["cloud_xp"],
                mode="markers",
                marker=dict(size=3, color="#1f77b4", opacity=0.55),
                customdata=panel["cloud"]["particle_id"].to_numpy(dtype=int),
                name="same transported particles",
                legendgroup="particles",
                showlegend=i_panel == 0,
                hovertemplate="particle %{customdata}<br>position = %{x:.5g} mm<br>angle = %{y:.5g} mrad<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=panel["transported_x"],
                y=panel["transported_xp"],
                mode="lines",
                line=dict(color="crimson", width=2.0, dash="dash"),
                name=f"transported {transported_level:g}σ ellipse",
                legendgroup="transported ellipse",
                showlegend=i_panel == 0,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    fig.update_xaxes(range=[-x_limit, x_limit], title_text=f"{plane} [mm]")
    fig.update_yaxes(range=[-xp_limit, xp_limit], title_text=f"{plane}' [mrad]")
    fig.update_layout(
        title=(
            f"{title}<br><sup>colored: locally matched contours; "
            f"red dashed: constant-area transported {transported_level:g}σ ellipse</sup>"
        ),
        template="plotly_white",
        width=1050,
        height=330 * n_rows,
        legend=dict(orientation="h", yanchor="top", y=-0.08),
    )
    return _show(fig)


def plot_phase_space_ellipses(tw_before, tw_after=None, title: str = "Input/output one-rms phase-space ellipses") -> go.Figure:
    if tw_after is None:
        tw_after = tw_before
    x_in, xp_in = _ellipse_points(float(tw_before.betx[0]), float(tw_before.alfx[0]))
    x_out, xp_out = _ellipse_points(float(tw_after.betx[-1]), float(tw_after.alfx[-1]))
    y_in, yp_in = _ellipse_points(float(tw_before.bety[0]), float(tw_before.alfy[0]))
    y_out, yp_out = _ellipse_points(float(tw_after.bety[-1]), float(tw_after.alfy[-1]))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("horizontal", "vertical"))
    fig.add_trace(go.Scatter(x=x_in, y=xp_in, mode="lines", name="input x"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_out, y=xp_out, mode="lines", name="output x"), row=1, col=1)
    fig.add_trace(go.Scatter(x=y_in, y=yp_in, mode="lines", name="input y"), row=1, col=2)
    fig.add_trace(go.Scatter(x=y_out, y=yp_out, mode="lines", name="output y"), row=1, col=2)
    fig.update_xaxes(title_text="position [mm]", row=1, col=1)
    fig.update_yaxes(title_text="angle [mrad]", row=1, col=1)
    fig.update_xaxes(title_text="position [mm]", row=1, col=2)
    fig.update_yaxes(title_text="angle [mrad]", row=1, col=2)
    fig.update_layout(title=title, template="plotly_white", width=950, height=430)
    return _show(fig)


# -----------------------------------------------------------------------------
# Interactive helpers
# -----------------------------------------------------------------------------


def _ipywidgets_available():
    try:
        from IPython.display import display  # noqa: F401
        from ipywidgets import interact, FloatSlider, IntSlider  # noqa: F401
        return True
    except Exception:
        return False


def interactive_fodo_strength():
    if not _ipywidgets_available():
        return "ipywidgets is not available in this environment."
    from IPython.display import display
    import ipywidgets as widgets
    from ipywidgets import interact
    display_widget_slider_css()

    @interact(
        k1=lab_float_slider(widgets, value=0.6, min=0.05, max=0.95, step=0.05, description="|k1|"),
        n_cells=lab_int_slider(widgets, value=1, min=1, max=8, step=1, description="cells"),
    )
    def _view(k1=0.6, n_cells=1):
        try:
            line = make_fodo_line(n_cells=n_cells, k1=k1)
            tw = twiss_periodic(line)
            twd = twiss_dense(line, points_per_meter=80)
            plot_beta_and_sigma(twd, f"Matched FODO: |k1|={k1:.3g}, cells={n_cells}")
            display(pd.DataFrame({
                "Xsuite output": ["qx", "qy"],
                "oscillations per cell": [float(tw.qx) / n_cells, float(tw.qy) / n_cells],
            }))
        except Exception as exc:
            print(f"No stable periodic solution for this setting: {exc}")


def interactive_quad_length_effect(target_q: float):
    if not _ipywidgets_available():
        return "ipywidgets is not available in this environment."
    from IPython.display import display
    import ipywidgets as widgets
    from ipywidgets import interact
    display_widget_slider_css()

    @interact(quad_length=lab_float_slider(widgets, value=0.5, min=0.10, max=2.20, step=0.05, description="quad L [m]"))
    def _view(quad_length=0.5):
        try:
            k1 = solve_k1_for_tune(float(target_q), float(quad_length))
            line = make_fodo_line(1, k1=k1, quad_length=float(quad_length))
            twd = twiss_dense(line, points_per_meter=150)
            print(f"Retuned |k1| = {k1:.6g} 1/m^2 to keep q_x,cell = {target_q:.6g}")
            plot_twiss(twd, f"Same phase advance, quadrupole length = {quad_length:.2f} m")
        except Exception as exc:
            print(f"Could not compute this setting: {exc}")


def interactive_mismatch(tw_cell):
    if not _ipywidgets_available():
        return "ipywidgets is not available in this environment."
    from IPython.display import display
    import ipywidgets as widgets
    from ipywidgets import interact
    display_widget_slider_css()
    line = make_fodo_line(20, k1=DEFAULT_K1)
    tw_ref = twiss_dense(line, points_per_meter=20)

    @interact(beta_scale=lab_float_slider(widgets, value=1.10, min=0.70, max=1.60, step=0.02, description="β scale"))
    def _view(beta_scale=1.10):
        tw_trial = twiss_dense(
            line,
            points_per_meter=20,
            betx=float(tw_cell.betx[0])*beta_scale,
            alfx=float(tw_cell.alfx[0]),
            bety=float(tw_cell.bety[0])*beta_scale,
            alfy=float(tw_cell.alfy[0]),
        )
        plot_mismatch(tw_ref, tw_trial, f"Envelope with beta scale = {beta_scale:.2f}")


def interactive_hybrid_transition(tw_cell):
    if not _ipywidgets_available():
        return "ipywidgets is not available in this environment."
    from IPython.display import display
    import ipywidgets as widgets
    from ipywidgets import interact
    display_widget_slider_css()

    @interact(k1_second=lab_float_slider(widgets, value=0.5, min=0.2, max=0.8, step=0.05, description="downstream |k1|"))
    def _view(k1_second=0.5):
        line = make_hybrid_fodo_line(10, 10, k1_first=DEFAULT_K1, k1_second=k1_second)
        twd = twiss_dense(
            line,
            points_per_meter=20,
            betx=float(tw_cell.betx[0]), alfx=float(tw_cell.alfx[0]),
            bety=float(tw_cell.bety[0]), alfy=float(tw_cell.alfy[0]),
        )
        plot_hybrid_transition(twd, transition_s=50.0, title=f"Downstream |k1| = {k1_second:.2f} 1/m²")


def interactive_manual_match(initial_twiss: Mapping[str, float]):
    if not _ipywidgets_available():
        return "ipywidgets is not available in this environment."
    from IPython.display import display
    import ipywidgets as widgets
    from ipywidgets import interact
    display_widget_slider_css()

    @interact(
        k1=lab_float_slider(widgets, value=0.6, min=-1.5, max=1.5, step=0.05, description="QM1"),
        k2=lab_float_slider(widgets, value=-0.6, min=-1.5, max=1.5, step=0.05, description="QM2"),
        k3=lab_float_slider(widgets, value=0.6, min=-1.5, max=1.5, step=0.05, description="QM3"),
        k4=lab_float_slider(widgets, value=-0.6, min=-1.5, max=1.5, step=0.05, description="QM4"),
    )
    def _view(k1=0.6, k2=-0.6, k3=0.6, k4=-0.6):
        section = make_match_section()
        set_match_knobs(section, [k1, k2, k3, k4])
        try:
            tw = twiss_dense(section, points_per_meter=40, **dict(initial_twiss))
            plot_twiss(tw, "Manual matching-section trial", show_lattice=False)
            display(pd.DataFrame({
                "target metric": ["beta_x - beta_y [m]", "alpha_x", "alpha_y"],
                "value at end": [float(tw.betx[-1]-tw.bety[-1]), float(tw.alfx[-1]), float(tw.alfy[-1])],
            }))
        except Exception as exc:
            print(f"Could not compute this setting: {exc}")
