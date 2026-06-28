"""Helper functions for the local Xsuite quadrupole-focusing lab.

The student-facing notebook keeps the important physics knobs visible. This file
contains the repetitive Xsuite line construction, dense Twiss sampling, plotting,
matching optimization, distribution tracking, and widget utilities.
"""

from __future__ import annotations

import contextlib
import io
import os
import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import xtrack as xt

GEOMETRIC_EMITTANCE = 6e-6       # 6 mm mrad = 6e-6 m rad
P0C_EV = 1e9                     # 1 GeV/c reference momentum
FODO_HALF_CELL_LENGTH = 2.5      # distance between QF/QD centers [m]
FODO_CELL_LENGTH = 5.0           # full FODO period [m]
FODO_QUAD_LENGTH = 0.5           # quadrupole length [m]
DEFAULT_K1 = 0.6                 # default quadrupole strength [1/m^2]
WEAK_K1 = 0.2
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
    if quad_length <= 0:
        raise ValueError("quad_length must be positive")
    if quad_length >= half_cell_length:
        raise ValueError("quad_length must be smaller than half_cell_length")
    edge = 0.5 * (half_cell_length - quad_length)
    middle = half_cell_length - quad_length
    return edge, middle


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

    Default one-cell sequence: drift 1.0 m, QF 0.5 m, drift 2.0 m,
    QD 0.5 m, drift 1.0 m. The line starts and ends at mid-drift
    symmetry points, so the full period is 5 m.
    """
    if n_cells < 1:
        raise ValueError("n_cells must be at least 1")
    edge, middle = _drift_lengths(quad_length, half_cell_length)
    pre = f"{prefix}_" if prefix else ""
    elements: list[object] = []
    names: list[str] = []
    for i_cell in range(1, n_cells + 1):
        tag = f"c{i_cell:02d}"
        elements.extend([
            xt.Drift(length=edge),
            xt.Quadrupole(length=quad_length, k1=k1),
            xt.Drift(length=middle),
            xt.Quadrupole(length=quad_length, k1=-k1),
            xt.Drift(length=edge),
        ])
        names.extend([f"{pre}D1_{tag}", f"{pre}QF_{tag}", f"{pre}D2_{tag}", f"{pre}QD_{tag}", f"{pre}D3_{tag}"])
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


def twiss_dense(line: xt.Line, *, n_points: int | None = None, points_per_meter: float = 200.0, **twiss_kwargs):
    """Compute Twiss functions on a dense uniform s-grid.

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
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
    edge, middle = _drift_lengths(float(quad_length), FODO_HALF_CELL_LENGTH)
    total = (
        _drift_matrix(edge)
        @ _quad_matrix(-float(k1), float(quad_length))
        @ _drift_matrix(middle)
        @ _quad_matrix(float(k1), float(quad_length))
        @ _drift_matrix(edge)
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
        return "cell boundary / mid-drift symmetry point"
    if np.isclose(smod, 2.5, atol=3e-3):
        return "central drift symmetry point"
    if 1.0 <= smod <= 1.5:
        return "focusing quadrupole QF"
    if 3.5 <= smod <= 4.0:
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


def _show(fig: go.Figure) -> go.Figure:
    if os.environ.get("QF_LAB_SUPPRESS_PLOTS", "0") not in {"1", "true", "True"}:
        fig.show()
    return fig


def _names(tw) -> np.ndarray:
    try:
        return np.asarray(tw.name, dtype=str)
    except Exception:
        return np.asarray([""] * len(tw.s), dtype=str)


def _hover_trace(x, y, text, name, *, mode: str = "lines") -> go.Scatter:
    return go.Scatter(x=x, y=y, mode=mode, name=name, text=text, hovertemplate="%{text}<br>s = %{x:.4g} m<br>value = %{y:.5g}<extra></extra>")


def plot_twiss(tw, title: str = "Twiss functions") -> go.Figure:
    names = _names(tw)
    fig = go.Figure()
    fig.add_trace(_hover_trace(tw.s, tw.betx, names, "βx"))
    fig.add_trace(_hover_trace(tw.s, tw.bety, names, "βy"))
    fig.update_layout(title=title, xaxis_title="s [m]", yaxis_title="β [m]", hovermode="closest", template="plotly_white", width=900, height=440)
    return _show(fig)


def plot_sigmas(tw, title: str = "RMS beam size") -> go.Figure:
    df = add_beam_sizes(tw)
    names = np.asarray(df["name"], dtype=str) if "name" in df.columns else np.asarray([""] * len(df))
    fig = go.Figure()
    fig.add_trace(_hover_trace(df["s"], df["sigma_x_mm"], names, "σx"))
    fig.add_trace(_hover_trace(df["s"], df["sigma_y_mm"], names, "σy"))
    fig.update_layout(title=title, xaxis_title="s [m]", yaxis_title="RMS size [mm]", hovermode="closest", template="plotly_white", width=900, height=440)
    return _show(fig)


def plot_beta_and_sigma(tw, title: str = "Twiss functions and RMS beam size") -> go.Figure:
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


def plot_centroid(centroid, title: str = "Centroid motion from a 1 mm horizontal offset") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=centroid.s,
        y=1e3*np.asarray(centroid.x, dtype=float),
        mode="lines+markers",
        name="x centroid",
        text=_names(centroid),
        hovertemplate="%{text}<br>s = %{x:.4g} m<br>x = %{y:.5g} mm<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="s [m]", yaxis_title="x [mm]", template="plotly_white", width=900, height=430)
    return _show(fig)


def plot_mismatch(tw_reference, tw_trial, title: str = "Envelope beating from beta mismatch") -> go.Figure:
    ref = add_beam_sizes(tw_reference)
    trial = add_beam_sizes(tw_trial)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ref["s"], y=ref["sigma_x_mm"], mode="lines", name="matched σx"))
    fig.add_trace(go.Scatter(x=trial["s"], y=trial["sigma_x_mm"], mode="lines", name="mismatched σx"))
    fig.add_trace(go.Scatter(x=ref["s"], y=ref["sigma_y_mm"], mode="lines", name="matched σy"))
    fig.add_trace(go.Scatter(x=trial["s"], y=trial["sigma_y_mm"], mode="lines", name="mismatched σy"))
    fig.update_layout(title=title, xaxis_title="s [m]", yaxis_title="RMS size [mm]", template="plotly_white", width=900, height=440)
    return _show(fig)


def plot_cell_boundary_envelope(tw_dense, title: str = "Envelope sampled at cell boundaries") -> go.Figure:
    table = cell_start_envelope_table(tw_dense)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table["s [m]"], y=table["sigma_x [mm]"], mode="lines+markers", name="σx at cell starts"))
    fig.add_trace(go.Scatter(x=table["s [m]"], y=table["sigma_y [mm]"], mode="lines+markers", name="σy at cell starts"))
    fig.update_layout(title=title, xaxis_title="s [m]", yaxis_title="RMS size [mm]", template="plotly_white", width=900, height=430)
    return _show(fig)


def plot_hybrid_transition(tw, *, transition_s: float = 50.0, title: str = "Hybrid line: strong cells followed by weaker cells") -> go.Figure:
    fig = go.Figure()
    names = _names(tw)
    fig.add_trace(_hover_trace(tw.s, tw.betx, names, "βx"))
    fig.add_trace(_hover_trace(tw.s, tw.bety, names, "βy"))
    fig.add_vline(x=transition_s, line_dash="dash", annotation_text="transition")
    fig.update_layout(title=title, xaxis_title="s [m]", yaxis_title="β [m]", template="plotly_white", width=900, height=440)
    return _show(fig)


# -----------------------------------------------------------------------------
# Matching and phase-space utilities
# -----------------------------------------------------------------------------


def match_round_section(section: xt.Line, initial_twiss: Mapping[str, float]):
    tw_before = section.twiss(method="4d", **dict(initial_twiss))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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


def plot_matching_comparison(tw_before, tw_after, title: str = "Matching section before and after optimization") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(_hover_trace(tw_before.s, tw_before.betx, _names(tw_before), "before βx"))
    fig.add_trace(_hover_trace(tw_before.s, tw_before.bety, _names(tw_before), "before βy"))
    fig.add_trace(_hover_trace(tw_after.s, tw_after.betx, _names(tw_after), "after βx"))
    fig.add_trace(_hover_trace(tw_after.s, tw_after.bety, _names(tw_after), "after βy"))
    fig.update_layout(title=title, xaxis_title="s [m]", yaxis_title="β [m]", template="plotly_white", width=900, height=440)
    return _show(fig)


def twiss_covariance(beta: float, alpha: float, emit: float = GEOMETRIC_EMITTANCE) -> np.ndarray:
    gamma = (1.0 + alpha**2) / beta
    return np.array([[emit*beta, -emit*alpha], [-emit*alpha, emit*gamma]])


def _sample_distribution(initial_twiss: Mapping[str, float], *, n_particles: int = 3000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x, px = rng.multivariate_normal([0.0, 0.0], twiss_covariance(float(initial_twiss["betx"]), float(initial_twiss["alfx"])), size=n_particles).T
    y, py = rng.multivariate_normal([0.0, 0.0], twiss_covariance(float(initial_twiss["bety"]), float(initial_twiss["alfy"])), size=n_particles).T
    return pd.DataFrame({"x": x, "px": px, "y": y, "py": py})


def _track_dataframe(line: xt.Line, df: pd.DataFrame) -> pd.DataFrame:
    particles = xt.Particles(
        p0c=P0C_EV,
        mass0=xt.ELECTRON_MASS_EV,
        q0=-1,
        x=df["x"].to_numpy(), px=df["px"].to_numpy(),
        y=df["y"].to_numpy(), py=df["py"].to_numpy(),
        delta=np.zeros(len(df)), zeta=np.zeros(len(df)),
    )
    line.track(particles)
    return pd.DataFrame({"x": np.asarray(particles.x), "px": np.asarray(particles.px), "y": np.asarray(particles.y), "py": np.asarray(particles.py)})


def track_distribution(line: xt.Line, initial_twiss: Mapping[str, float] | pd.DataFrame, *, n_particles: int = 3000, seed: int = 11) -> dict[str, pd.DataFrame]:
    if isinstance(initial_twiss, pd.DataFrame):
        input_df = initial_twiss.copy()
    else:
        input_df = _sample_distribution(initial_twiss, n_particles=n_particles, seed=seed)
    output_df = _track_dataframe(line, input_df)
    return {"input": input_df, "output": output_df}


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
    from ipywidgets import interact, FloatSlider, IntSlider

    @interact(k1=FloatSlider(value=0.6, min=0.05, max=0.95, step=0.05, description="|k1|"),
              n_cells=IntSlider(value=1, min=1, max=8, step=1, description="cells"))
    def _view(k1=0.6, n_cells=1):
        try:
            line = make_fodo_line(n_cells=n_cells, k1=k1)
            tw = twiss_periodic(line)
            twd = twiss_dense(line, points_per_meter=80)
            plot_beta_and_sigma(twd, f"Matched FODO: |k1|={k1:.3g}, cells={n_cells}")
            display(phase_advance_table(tw))
            display(beam_size_summary(twd, label="current setting"))
        except Exception as exc:
            print(f"No stable periodic solution for this setting: {exc}")


def interactive_quad_length_effect(target_q: float):
    if not _ipywidgets_available():
        return "ipywidgets is not available in this environment."
    from IPython.display import display
    from ipywidgets import interact, FloatSlider

    @interact(quad_length=FloatSlider(value=0.5, min=0.10, max=2.20, step=0.05, description="quad L [m]"))
    def _view(quad_length=0.5):
        try:
            k1 = solve_k1_for_tune(float(target_q), float(quad_length))
            line = make_fodo_line(1, k1=k1, quad_length=float(quad_length))
            twd = twiss_dense(line, points_per_meter=150)
            print(f"Retuned |k1| = {k1:.6g} 1/m^2 to keep Qx = {target_q:.6g}")
            plot_twiss(twd, f"Same phase advance, quadrupole length = {quad_length:.2f} m")
            display(thin_lens_comparison_table(twd, length_for_formula=FODO_CELL_LENGTH))
        except Exception as exc:
            print(f"Could not compute this setting: {exc}")


def interactive_mismatch(tw_cell):
    if not _ipywidgets_available():
        return "ipywidgets is not available in this environment."
    from IPython.display import display
    from ipywidgets import interact, FloatSlider
    line = make_fodo_line(20, k1=DEFAULT_K1)
    tw_ref = twiss_dense(line, points_per_meter=20)

    @interact(beta_scale=FloatSlider(value=1.10, min=0.70, max=1.60, step=0.02, description="β scale"))
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
        display(cell_start_envelope_table(tw_trial).head(8))


def interactive_weak_cell():
    if not _ipywidgets_available():
        return "ipywidgets is not available in this environment."
    from IPython.display import display
    from ipywidgets import interact, FloatSlider

    @interact(k1=FloatSlider(value=0.2, min=0.05, max=0.8, step=0.05, description="|k1|"))
    def _view(k1=0.2):
        try:
            line = make_fodo_line(1, k1=k1)
            tw = twiss_periodic(line)
            twd = twiss_dense(line, points_per_meter=120)
            plot_beta_and_sigma(twd, f"Matched FODO cell: |k1| = {k1:.3g} 1/m²")
            display(phase_advance_table(tw))
            display(beam_size_summary(twd, label="cell"))
        except Exception as exc:
            print(f"No stable periodic solution for this setting: {exc}")


def interactive_hybrid_transition(tw_cell):
    if not _ipywidgets_available():
        return "ipywidgets is not available in this environment."
    from IPython.display import display
    from ipywidgets import interact, FloatSlider

    @interact(k1_second=FloatSlider(value=0.5, min=0.2, max=0.8, step=0.05, description="downstream |k1|"))
    def _view(k1_second=0.5):
        line = make_hybrid_fodo_line(10, 10, k1_first=DEFAULT_K1, k1_second=k1_second)
        twd = twiss_dense(
            line,
            points_per_meter=20,
            betx=float(tw_cell.betx[0]), alfx=float(tw_cell.alfx[0]),
            bety=float(tw_cell.bety[0]), alfy=float(tw_cell.alfy[0]),
        )
        plot_hybrid_transition(twd, transition_s=50.0, title=f"Downstream |k1| = {k1_second:.2f} 1/m²")
        display(beam_size_summary(twd, label="strong-cell initial match"))


def interactive_manual_match(initial_twiss: Mapping[str, float]):
    if not _ipywidgets_available():
        return "ipywidgets is not available in this environment."
    from IPython.display import display
    from ipywidgets import interact, FloatSlider

    @interact(
        k1=FloatSlider(value=0.6, min=-1.5, max=1.5, step=0.05, description="QM1"),
        k2=FloatSlider(value=-0.6, min=-1.5, max=1.5, step=0.05, description="QM2"),
        k3=FloatSlider(value=0.6, min=-1.5, max=1.5, step=0.05, description="QM3"),
        k4=FloatSlider(value=-0.6, min=-1.5, max=1.5, step=0.05, description="QM4"),
    )
    def _view(k1=0.6, k2=-0.6, k3=0.6, k4=-0.6):
        section = make_match_section()
        set_match_knobs(section, [k1, k2, k3, k4])
        try:
            tw = twiss_dense(section, points_per_meter=40, **dict(initial_twiss))
            plot_twiss(tw, "Manual matching-section trial")
            display(pd.DataFrame({
                "target metric": ["beta_x - beta_y [m]", "alpha_x", "alpha_y"],
                "value at end": [float(tw.betx[-1]-tw.bety[-1]), float(tw.alfx[-1]), float(tw.alfy[-1])],
            }))
        except Exception as exc:
            print(f"Could not compute this setting: {exc}")
