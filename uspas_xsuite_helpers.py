"""Shared Xsuite helpers for the USPAS notebook exercises.

The notebooks intentionally keep most lattice construction and plotting
details here so the visible cells can focus on the physics knobs.
"""

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import xtrack as xt


GEOMETRIC_EMITTANCE = 6e-6  # 6 mm mrad = 6e-6 m rad
SIGMA_DP_DEFAULT = 1e-3


def set_plotly_renderer(renderer="notebook"):
    pio.renderers.default = renderer


def print_versions():
    print(f"xtrack version: {xt.__version__}")
    print(f"plotly renderer: {pio.renderers.default}")


def electron_ref(p0c=1e9):
    return xt.Particles(p0c=p0c, mass0=xt.ELECTRON_MASS_EV, q0=-1)


def build_line(elements, names, p0c=1e9):
    line = xt.Line(elements=elements, element_names=names)
    line.particle_ref = electron_ref(p0c=p0c)
    line.build_tracker()
    return line


def _hover_trace(x, y, text, name, mode="lines+markers"):
    return go.Scatter(
        x=x,
        y=y,
        mode=mode,
        name=name,
        text=text,
        hovertemplate="%{text}<br>s = %{x:.4g} m<br>value = %{y:.5g}<extra></extra>",
    )


# ---------------------------------------------------------------------------
# Quadrupole focusing helpers
# ---------------------------------------------------------------------------


def make_fodo_line(n_cells=1, *, k1=0.6, p0c=1e9, name_prefix=""):
    """Build n FODO cells with the geometry used in the focusing lab."""
    elements = []
    names = []
    prefix = f"{name_prefix}_" if name_prefix else ""

    for i_cell in range(n_cells):
        tag = f"c{i_cell + 1}"
        elements += [
            xt.Drift(length=1.0),
            xt.Quadrupole(length=0.5, k1=k1),
            xt.Drift(length=2.0),
            xt.Quadrupole(length=0.5, k1=-k1),
            xt.Drift(length=1.0),
        ]
        names += [
            f"{prefix}d1_{tag}",
            f"{prefix}qf_{tag}",
            f"{prefix}d2_{tag}",
            f"{prefix}qd_{tag}",
            f"{prefix}d3_{tag}",
        ]

    return build_line(elements, names, p0c=p0c)


def make_hybrid_line(first_cells=10, second_cells=10, *, k1_first=0.6, k1_second=0.5):
    elements = []
    names = []
    for source in [
        make_fodo_line(first_cells, k1=k1_first, name_prefix="strong"),
        make_fodo_line(second_cells, k1=k1_second, name_prefix="weak"),
    ]:
        for name in source.element_names:
            elements.append(source[name].copy())
            names.append(name)
    return build_line(elements, names)


def make_match_section():
    elements = [xt.Drift(length=1.0)]
    names = ["dm0"]
    for i_quad, sign in enumerate([1, -1, 1, -1], start=1):
        elements += [xt.Quadrupole(length=0.5, k1=0), xt.Drift(length=2.0)]
        names += [f"qm{i_quad}", f"dm{i_quad}"]

    line = build_line(elements, names)
    for i_quad, sign in enumerate([1, -1, 1, -1], start=1):
        knob = f"kqm{i_quad}"
        line.vars[knob] = sign * 0.6
        line.element_refs[f"qm{i_quad}"].k1 = line.vars[knob]
    return line


def twiss_dataframe(tw):
    return tw.to_pandas()[["name", "s", "betx", "bety", "alfx", "alfy", "mux", "muy"]]


def add_sigmas(tw, emit_x=GEOMETRIC_EMITTANCE, emit_y=GEOMETRIC_EMITTANCE):
    df = tw.to_pandas()[["name", "s", "betx", "bety", "alfx", "alfy", "mux", "muy"]].copy()
    df["sigma_x_mm"] = 1e3 * np.sqrt(df["betx"] * emit_x)
    df["sigma_y_mm"] = 1e3 * np.sqrt(df["bety"] * emit_y)
    df["sigma_ratio_x_over_y"] = df["sigma_x_mm"] / df["sigma_y_mm"]
    return df


def summarize_twiss(tw, label="line"):
    sig = add_sigmas(tw)
    try:
        qx = tw.qx
        qy = tw.qy
    except AttributeError:
        qx = np.nan
        qy = np.nan
    return pd.DataFrame({
        "quantity": [
            "length [m]", "Qx", "Qy", "phase x [deg]", "phase y [deg]",
            "min betx [m]", "max betx [m]", "min bety [m]", "max bety [m]",
            "mean sigma_x [mm]", "mean sigma_y [mm]",
            "min sigma_x [mm]", "max sigma_x [mm]",
            "min sigma_y [mm]", "max sigma_y [mm]",
            "max sigma_x/sigma_y",
        ],
        label: [
            tw.s[-1], qx, qy, 360 * qx, 360 * qy,
            np.min(tw.betx), np.max(tw.betx), np.min(tw.bety), np.max(tw.bety),
            sig["sigma_x_mm"].mean(), sig["sigma_y_mm"].mean(),
            sig["sigma_x_mm"].min(), sig["sigma_x_mm"].max(),
            sig["sigma_y_mm"].min(), sig["sigma_y_mm"].max(),
            sig["sigma_ratio_x_over_y"].max(),
        ],
    })


def phase_advance_summary(tw):
    return pd.DataFrame({
        "quantity": [
            "Qx per cell", "Qy per cell", "phase x [rad]",
            "phase y [rad]", "phase x [deg]", "phase y [deg]",
        ],
        "value": [tw.qx, tw.qy, 2 * np.pi * tw.qx, 2 * np.pi * tw.qy, 360 * tw.qx, 360 * tw.qy],
    })


def thin_lens_comparison(tw, half_cell_length=2.5):
    psi = 2 * np.pi * tw.qx
    return pd.DataFrame({
        "quantity": [
            "beta_min thin lens [m]", "beta_max thin lens [m]",
            "beta_min Xsuite [m]", "beta_max Xsuite [m]",
        ],
        "value": [
            half_cell_length * (1 - np.sin(psi / 2)) / np.sin(psi),
            half_cell_length * (1 + np.sin(psi / 2)) / np.sin(psi),
            np.min(tw.betx),
            np.max(tw.betx),
        ],
    })


def location_table(tw):
    df = add_sigmas(tw)
    idx_round = (df["sigma_x_mm"] - df["sigma_y_mm"]).abs().idxmin()
    rows = [
        ("round", idx_round),
        ("max sigma_x", df["sigma_x_mm"].idxmax()),
        ("min sigma_x", df["sigma_x_mm"].idxmin()),
        ("max sigma_y", df["sigma_y_mm"].idxmax()),
        ("min sigma_y", df["sigma_y_mm"].idxmin()),
    ]
    return pd.DataFrame({
        "condition": [row[0] for row in rows],
        "s [m]": [df.loc[row[1], "s"] for row in rows],
        "element": [df.loc[row[1], "name"] for row in rows],
    })


def plot_twiss(tw, title="Twiss functions"):
    fig = go.Figure()
    names = np.asarray(tw.name, dtype=str)
    fig.add_trace(_hover_trace(tw.s, tw.betx, names, "βx"))
    fig.add_trace(_hover_trace(tw.s, tw.bety, names, "βy"))
    fig.update_layout(
        title=title,
        xaxis_title="s [m]",
        yaxis_title="β [m]",
        hovermode="closest",
        template="plotly_white",
        width=850,
        height=430,
    )
    fig.show()
    return fig


def plot_sigmas(tw, title="RMS beam size"):
    df = add_sigmas(tw)
    fig = go.Figure()
    fig.add_trace(_hover_trace(df["s"], df["sigma_x_mm"], df["name"], "σx"))
    fig.add_trace(_hover_trace(df["s"], df["sigma_y_mm"], df["name"], "σy"))
    fig.update_layout(
        title=title,
        xaxis_title="s [m]",
        yaxis_title="RMS size [mm]",
        hovermode="closest",
        template="plotly_white",
        width=850,
        height=430,
    )
    fig.show()
    return fig


def plot_centroid(centroid):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=centroid.s,
        y=1e3 * centroid.x,
        mode="lines+markers",
        name="x centroid",
        text=np.asarray(centroid.name, dtype=str),
        hovertemplate="%{text}<br>s = %{x:.4g} m<br>x = %{y:.5g} mm<extra></extra>",
    ))
    fig.update_layout(
        title="Centroid motion from a 1 mm horizontal launch offset",
        xaxis_title="s [m]",
        yaxis_title="x [mm]",
        template="plotly_white",
        width=850,
        height=430,
    )
    fig.show()
    return fig


def plot_mismatch(sig_matched, sig_mismatch):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sig_matched["s"], y=sig_matched["sigma_x_mm"], mode="lines+markers",
        name="matched σx", text=sig_matched["name"],
        hovertemplate="%{text}<br>s = %{x:.4g} m<br>σx = %{y:.5g} mm<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=sig_mismatch["s"], y=sig_mismatch["sigma_x_mm"], mode="lines+markers",
        name="10% mismatched σx", text=sig_mismatch["name"],
        hovertemplate="%{text}<br>s = %{x:.4g} m<br>σx = %{y:.5g} mm<extra></extra>",
    ))
    fig.update_layout(
        title="Envelope beating from a 10% beta mismatch",
        xaxis_title="s [m]",
        yaxis_title="RMS size [mm]",
        template="plotly_white",
        width=850,
        height=430,
    )
    fig.show()
    return fig


def plot_hybrid_transition(tw_hybrid, transition_s=50):
    fig = go.Figure()
    names = np.asarray(tw_hybrid.name, dtype=str)
    fig.add_trace(_hover_trace(tw_hybrid.s, tw_hybrid.betx, names, "start matched to k1=0.6, βx"))
    fig.add_trace(_hover_trace(tw_hybrid.s, tw_hybrid.bety, names, "start matched to k1=0.6, βy"))
    fig.add_vline(x=transition_s, line_dash="dash", line_color="black", annotation_text="transition")
    fig.update_layout(
        title="Injection mismatch at transition to weaker cells",
        xaxis_title="s [m]",
        yaxis_title="β [m]",
        template="plotly_white",
        width=850,
        height=430,
    )
    fig.show()
    return fig


def match_round_section(match_section, initial_twiss):
    tw_before = match_section.twiss(method="4d", **initial_twiss)
    opt = match_section.match(
        method="4d",
        vary=[xt.Vary(f"kqm{i}", step=1e-3, limits=(-3, 3)) for i in range(1, 5)],
        targets=[
            xt.Target("alfx", 0.0, at="_end_point", tol=1e-8),
            xt.Target("alfy", 0.0, at="_end_point", tol=1e-8),
            xt.Target(lambda tw: tw["betx", "_end_point"] - tw["bety", "_end_point"], 0.0, tol=1e-8),
        ],
        **initial_twiss,
    )
    tw_after = match_section.twiss(method="4d", **initial_twiss)
    return opt, tw_before, tw_after


def plot_matching_comparison(tw_before, tw_after):
    fig = go.Figure()
    fig.add_trace(_hover_trace(tw_before.s, tw_before.betx, np.asarray(tw_before.name, dtype=str), "before βx"))
    fig.add_trace(_hover_trace(tw_before.s, tw_before.bety, np.asarray(tw_before.name, dtype=str), "before βy"))
    fig.add_trace(_hover_trace(tw_after.s, tw_after.betx, np.asarray(tw_after.name, dtype=str), "after βx"))
    fig.add_trace(_hover_trace(tw_after.s, tw_after.bety, np.asarray(tw_after.name, dtype=str), "after βy"))
    fig.update_layout(
        title="Matching section before and after optimization",
        xaxis_title="s [m]",
        yaxis_title="β [m]",
        template="plotly_white",
        width=850,
        height=430,
    )
    fig.show()
    return fig


def matching_summary(match_section, tw_after):
    return pd.DataFrame({
        "quantity": ["final betx [m]", "final bety [m]", "final alfx", "final alfy", "QM1 k1", "QM2 k1", "QM3 k1", "QM4 k1"],
        "value": [
            tw_after.betx[-1], tw_after.bety[-1], tw_after.alfx[-1], tw_after.alfy[-1],
            match_section.vars["kqm1"]._value, match_section.vars["kqm2"]._value,
            match_section.vars["kqm3"]._value, match_section.vars["kqm4"]._value,
        ],
    })


def ellipse_points(beta, alpha, emit=GEOMETRIC_EMITTANCE, n=300):
    phase = np.linspace(0, 2 * np.pi, n)
    x = np.sqrt(emit * beta) * np.cos(phase)
    xp = -np.sqrt(emit / beta) * (alpha * np.cos(phase) + np.sin(phase))
    return 1e3 * x, 1e3 * xp


def plot_phase_space_ellipses(tw):
    x_in, xp_in = ellipse_points(tw.betx[0], tw.alfx[0])
    x_out, xp_out = ellipse_points(tw.betx[-1], tw.alfx[-1])
    y_in, yp_in = ellipse_points(tw.bety[0], tw.alfy[0])
    y_out, yp_out = ellipse_points(tw.bety[-1], tw.alfy[-1])

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Horizontal phase-space ellipse", "Vertical phase-space ellipse"))
    fig.add_trace(go.Scatter(x=x_in, y=xp_in, mode="lines", name="x input", hovertemplate="x = %{x:.5g} mm<br>x' = %{y:.5g} mrad<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_out, y=xp_out, mode="lines", name="x output", hovertemplate="x = %{x:.5g} mm<br>x' = %{y:.5g} mrad<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=y_in, y=yp_in, mode="lines", name="y input", hovertemplate="y = %{x:.5g} mm<br>y' = %{y:.5g} mrad<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=y_out, y=yp_out, mode="lines", name="y output", hovertemplate="y = %{x:.5g} mm<br>y' = %{y:.5g} mrad<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text="x [mm]", row=1, col=1)
    fig.update_yaxes(title_text="x' [mrad]", row=1, col=1)
    fig.update_xaxes(title_text="y [mm]", row=1, col=2)
    fig.update_yaxes(title_text="y' [mrad]", row=1, col=2)
    fig.update_layout(template="plotly_white", width=950, height=430)
    fig.show()
    return fig


def make_insertion_line(fodo_cell, match_section):
    reverse_elements = []
    reverse_names = []
    for name in reversed(match_section.element_names):
        if name == "_end_point":
            continue
        reverse_elements.append(match_section[name].copy())
        reverse_names.append(f"rev_{name}")

    insertion_elements = [fodo_cell[name].copy() for name in fodo_cell.element_names]
    insertion_names = [f"fodo_{name}" for name in fodo_cell.element_names]
    for name in match_section.element_names:
        if name == "_end_point":
            continue
        insertion_elements.append(match_section[name].copy())
        insertion_names.append(f"match_{name}")
    insertion_elements += reverse_elements
    insertion_names += reverse_names
    return build_line(insertion_elements, insertion_names)


# ---------------------------------------------------------------------------
# Dispersion and chromaticity helpers
# ---------------------------------------------------------------------------


def make_fodo_with_bend(k1=0.6, bend_angle_deg=20.0):
    bend_length = 0.5
    angle = np.deg2rad(bend_angle_deg)
    h = angle / bend_length
    return build_line(
        elements=[
            xt.Drift(length=1.0),
            xt.Quadrupole(length=0.5, k1=k1),
            xt.Drift(length=1.5),
            xt.RBend(length=bend_length, angle=angle, k0=h, h=h),
            xt.Quadrupole(length=0.5, k1=-k1),
            xt.Drift(length=1.0),
        ],
        names=["d1", "qf", "d2a", "dipo", "qd", "d3"],
    )


def twiss_table(tw):
    cols = ["name", "s", "betx", "bety", "dx", "dpx", "mux", "muy"]
    return tw.to_pandas()[cols]


def add_beam_size(tw, sigma_dp=0.0, emit_x=GEOMETRIC_EMITTANCE, emit_y=GEOMETRIC_EMITTANCE):
    df = tw.to_pandas()[["name", "s", "betx", "bety", "dx", "dpx"]].copy()
    df["sigma_x_mm"] = 1e3 * np.sqrt(df["betx"] * emit_x + (df["dx"] * sigma_dp) ** 2)
    df["sigma_y_mm"] = 1e3 * np.sqrt(df["bety"] * emit_y)
    return df


def plot_optics(tw, title="Optics"):
    names = np.asarray(tw.name, dtype=str)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Beta functions", "Dispersion"),
    )
    fig.add_trace(_hover_trace(tw.s, tw.betx, names, "βx"), row=1, col=1)
    fig.add_trace(_hover_trace(tw.s, tw.bety, names, "βy"), row=1, col=1)
    fig.add_trace(_hover_trace(tw.s, tw.dx, names, "ηx"), row=2, col=1)
    fig.add_trace(_hover_trace(tw.s, tw.dpx, names, "ηx'"), row=2, col=1)
    fig.update_xaxes(title_text="s [m]", row=2, col=1)
    fig.update_yaxes(title_text="β [m]", row=1, col=1)
    fig.update_yaxes(title_text="dispersion [m]", row=2, col=1)
    fig.update_layout(
        title=title,
        hovermode="closest",
        template="plotly_white",
        width=900,
        height=620,
    )
    fig.show()
    return fig


def make_dba_section(q1=0.0, q2=0.0, q3=0.0, bend_angle_deg=18.0):
    bend_length = 1.0
    angle = np.deg2rad(bend_angle_deg)
    h = angle / bend_length
    elements = [
        xt.Drift(length=0.5),
        xt.RBend(length=bend_length, angle=angle, k0=h, h=h),
        xt.Drift(length=0.5),
        xt.Quadrupole(length=0.3, k1=q2),
        xt.Drift(length=0.7),
        xt.Quadrupole(length=0.3, k1=q1),
        xt.Drift(length=0.7),
        xt.Quadrupole(length=0.3, k1=q3),
        xt.Drift(length=0.5),
        xt.RBend(length=bend_length, angle=angle, k0=h, h=h),
        xt.Drift(length=0.5),
    ]
    names = ["d0", "b1", "d1", "q2", "d2", "q1", "d3", "q3", "d4", "b2", "d5"]
    return build_line(elements, names)


def twiss_from_zero_dispersion(line):
    return line.twiss(
        method="4d",
        betx=10.0, alfx=0.0,
        bety=10.0, alfy=0.0,
        dx=0.0, dpx=0.0,
        dy=0.0, dpy=0.0,
    )


def achromat_penalty(q1):
    tw = twiss_from_zero_dispersion(make_dba_section(q1=q1, q2=0.0, q3=0.0))
    return tw.dx[-1] ** 2 + tw.dpx[-1] ** 2


def fodo_dispersion_summary(tw):
    return pd.DataFrame({
        "quantity": [
            "min eta_x [m]", "location of min eta_x", "max eta_x [m]", "location of max eta_x",
            "Qx", "Qy", "chromaticity Cx=dQx/ddelta", "chromaticity Cy=dQy/ddelta",
        ],
        "value": [
            np.min(tw.dx),
            tw.name[np.argmin(tw.dx)],
            np.max(tw.dx),
            tw.name[np.argmax(tw.dx)],
            tw.qx,
            tw.qy,
            tw.dqx,
            tw.dqy,
        ],
    })


def qf_beam_size_table(tw, sigma_dp=SIGMA_DP_DEFAULT):
    beam_qf_delta0 = add_beam_size(tw, sigma_dp=0.0).query("name == 'qf'")
    beam_qf_delta = add_beam_size(tw, sigma_dp=sigma_dp).query("name == 'qf'")
    return pd.DataFrame({
        "": [r"$\sigma_x$ at QF [mm]", r"$\sigma_y$ at QF [mm]"],
        r"$\delta=0$": [beam_qf_delta0["sigma_x_mm"].iloc[0], beam_qf_delta0["sigma_y_mm"].iloc[0]],
        rf"$\delta={sigma_dp:g}$": [beam_qf_delta["sigma_x_mm"].iloc[0], beam_qf_delta["sigma_y_mm"].iloc[0]],
    })


def focused_dba_summary(tw):
    beam = add_beam_size(tw, sigma_dp=0.0)
    return pd.DataFrame({
        "quantity": ["max eta_x [m]", "location", "max beta_x [m]", "max sigma_x for delta=0 [mm]"],
        "value": [
            np.max(tw.dx),
            tw.name[np.argmax(tw.dx)],
            np.max(tw.betx),
            np.max(beam["sigma_x_mm"]),
        ],
    })


def aperture_table(tw, pipe_radius=0.025):
    df = tw.to_pandas()[["name", "s", "betx", "dx"]].copy()
    df["sigma_x_delta0_m"] = np.sqrt(GEOMETRIC_EMITTANCE * df["betx"])
    df["max_sigma_dp_before_25mm"] = np.sqrt(
        np.maximum(pipe_radius ** 2 - df["sigma_x_delta0_m"] ** 2, 0.0)
    ) / np.abs(df["dx"].replace(0, np.nan))
    return df.sort_values("max_sigma_dp_before_25mm")


def make_chromatic_ring(n_bends=20, kq=0.6):
    elements = []
    names = []
    bend_angle = 2 * np.pi / n_bends
    bend_length = 1.0
    h = bend_angle / bend_length
    for i in range(n_bends):
        tag = f"c{i + 1}"
        elements += [
            xt.Drift(length=0.5),
            xt.Quadrupole(length=0.3, k1=kq),
            xt.Drift(length=0.5),
            xt.Bend(length=bend_length, angle=bend_angle, k0=h, h=h),
            xt.Drift(length=0.5),
            xt.Quadrupole(length=0.3, k1=-kq),
            xt.Drift(length=0.5),
        ]
        names += [f"d1_{tag}", f"qf_{tag}", f"d2_{tag}", f"b_{tag}", f"d3_{tag}", f"qd_{tag}", f"d4_{tag}"]
    return build_line(elements, names)


def ring_summary(tw):
    return pd.DataFrame({
        "quantity": ["Qx", "Qy", "chromaticity Cx=dQx/ddelta", "chromaticity Cy=dQy/ddelta", "max eta_x [m]"],
        "value": [tw.qx, tw.qy, tw.dqx, tw.dqy, np.max(tw.dx)],
    })


def chromatic_spread_table(tw, sigma_dp=SIGMA_DP_DEFAULT):
    return pd.DataFrame({
        "quantity": ["Cx", "Cy", f"Delta Qx for sigma_dp={sigma_dp:g}", f"Delta Qy for sigma_dp={sigma_dp:g}"],
        "value": [tw.dqx, tw.dqy, tw.dqx * sigma_dp, tw.dqy * sigma_dp],
    })


def resonance_segments(max_order=3, integer=(0, 0), lines=(1, 1, 1, 1)):
    pval = 40
    p_values = np.arange(0, pval + 1)
    qxmin, qymin = integer[0], integer[1]
    qxmax, qymax = qxmin + 1, qymin + 1
    segments = []

    for order in range(1, max_order + 1):
        m_values = np.linspace(-order, order, 2 * order + 1)
        n1_values = order - np.abs(m_values)
        n2_values = -n1_values
        for m, n1, n2 in zip(m_values, n1_values, n2_values):
            for p in p_values:
                if n1 == 0 and lines[1] and m != 0:
                    x = p / m
                    if qxmin - 0.01 <= x <= qxmax + 0.01:
                        segments.append((order, [x, x], [qymin, qymax], f"{int(m)}νx = {int(p)}"))
                elif m == 0 and lines[0] and n1 != 0:
                    for n in [n1, n2]:
                        y = p / n
                        if qymin - 0.01 <= y <= qymax + 0.01:
                            segments.append((order, [qxmin, qxmax], [y, y], f"{int(n)}νy = {int(p)}"))
                elif n1 != 0 and m != 0:
                    if lines[2]:
                        denom = n2 if np.sign(m) > 0 else n1
                        y0 = p / denom - m * qxmin / denom
                        y1 = p / denom - m * qxmax / denom
                        if max(y0, y1) >= qymin - 0.01 and min(y0, y1) <= qymax + 0.01:
                            segments.append((order, [qxmin, qxmax], [y0, y1], f"{int(m)}νx + {int(denom)}νy = {int(p)}"))
                    if lines[3]:
                        denom = n1 if np.sign(m) > 0 else n2
                        y0 = p / denom - m * qxmin / denom
                        y1 = p / denom - m * qxmax / denom
                        if max(y0, y1) >= qymin - 0.01 and min(y0, y1) <= qymax + 0.01:
                            segments.append((order, [qxmin, qxmax], [y0, y1], f"{int(m)}νx + {int(denom)}νy = {int(p)}"))
    return segments


def plot_tune_footprint(nux, nuy, cx, cy, sigma_dp=0.001, resonance_order=3):
    delta_nux = cx * sigma_dp
    delta_nuy = cy * sigma_dp
    integer = (int(nux), int(nuy))
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    fig = go.Figure()

    for order, xs, ys, label in resonance_segments(max_order=resonance_order, integer=integer):
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=f"order {order}",
            legendgroup=f"order {order}",
            showlegend=not any(tr.name == f"order {order}" for tr in fig.data),
            line=dict(color=colors.get(order, "black"), width=1),
            hovertemplate=f"{label}<br>order {order}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=[nux], y=[nuy], mode="markers", name="ring tune",
        marker=dict(color="black", size=9),
        hovertemplate="ring tune<br>νx = %{x:.6g}<br>νy = %{y:.6g}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[nux - delta_nux, nux + delta_nux],
        y=[nuy - delta_nuy, nuy + delta_nuy],
        mode="lines+markers",
        name="chromatic footprint",
        line=dict(color="black", width=3),
        hovertemplate="chromatic footprint<br>νx = %{x:.6g}<br>νy = %{y:.6g}<extra></extra>",
    ))

    fig.update_layout(
        title="Resonance diagram",
        xaxis_title="horizontal tune",
        yaxis_title="vertical tune",
        xaxis=dict(range=[integer[0] - 0.01, integer[0] + 1.01]),
        yaxis=dict(range=[integer[1] - 0.01, integer[1] + 1.01]),
        template="plotly_white",
        width=720,
        height=650,
    )
    fig.show()
    return delta_nux, delta_nuy

