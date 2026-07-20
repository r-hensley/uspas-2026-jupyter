"""Linked matching-section beam and phase-space displays for the Q10 lab."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .shared import (
    add_lattice_strip,
    should_show_plot,
    show_or_return,
)


DEFAULT_GEOMETRIC_EMITTANCE = 6e-6  # 6 mm mrad = 6e-6 m rad


def _line_lattice_layout(line) -> pd.DataFrame:
    """Describe the actual line geometry and quadrupole signs for the strip."""

    table = line.get_table().to_pandas()
    table = table[table["name"] != "_end_point"].reset_index(drop=True)
    rows: list[dict[str, float | str]] = []
    for _, element_row in table.iterrows():
        name = str(element_row["name"])
        element_type = str(element_row["element_type"])
        s_start = float(element_row["s_start"])
        s_end = float(element_row["s_end"])
        if s_end <= s_start:
            continue

        if element_type == "Quadrupole":
            kind = "quad"
            try:
                k1 = float(line[name].k1)
            except (AttributeError, TypeError, ValueError):
                k1 = np.nan
        elif "Bend" in element_type:
            kind = "bend"
            k1 = np.nan
        else:
            kind = "drift"
            k1 = np.nan

        rows.append({
            "name": name,
            "kind": kind,
            "s_start_m": s_start,
            "s_end_m": s_end,
            "k1_m^-2": k1,
        })
    return pd.DataFrame(rows)


def _ellipse_points(
    beta: float,
    alpha: float,
    *,
    emittance: float,
    sigma_level: float,
    n_points: int = 320,
) -> tuple[np.ndarray, np.ndarray]:
    phase = np.linspace(0.0, 2.0 * np.pi, n_points)
    x = np.sqrt(emittance * beta) * np.cos(phase)
    xp = -np.sqrt(emittance / beta) * (
        alpha * np.cos(phase) + np.sin(phase)
    )
    return 1e3 * sigma_level * x, 1e3 * sigma_level * xp


def _twiss_row_near_s(twiss_table: pd.DataFrame, s_position: float) -> pd.Series:
    index = int(np.argmin(np.abs(
        twiss_table["s"].to_numpy(dtype=float) - float(s_position)
    )))
    return twiss_table.iloc[index]


def _validate_tracking_input(tracked: Mapping[str, object]) -> pd.DataFrame:
    input_df = tracked.get("input")
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("tracked must come from track_distribution_along_line")
    required = {"x", "px", "y", "py"}
    if not required.issubset(input_df.columns):
        raise ValueError("tracked input does not contain transverse phase space")
    if len(input_df) == 0:
        raise ValueError("tracked input contains no particles")
    return input_df.reset_index(drop=True)


def _uniform_s_positions(line_length: float, requested_step: float) -> np.ndarray:
    """Return endpoint-inclusive, exactly equidistant longitudinal samples."""

    if requested_step <= 0.0:
        raise ValueError("sample_step_m must be positive")
    n_intervals = max(1, int(np.ceil(line_length / requested_step)))
    return np.linspace(0.0, line_length, n_intervals + 1)


def _track_selected_particles_at_s(
    line,
    input_df: pd.DataFrame,
    particle_ids: np.ndarray,
    s_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Track selected real particles on a copy cut at the requested positions."""

    # Import at call time to avoid a module-import cycle: this plotting module
    # is re-exported by quadrupole_focusing.
    from .quadrupole_focusing import suppress_xsuite_output

    sampled_line = line.copy()
    with suppress_xsuite_output():
        sampled_line.cut_at_s(s_positions)

    launch = input_df.iloc[particle_ids]
    particles = sampled_line.build_particles(
        x=launch["x"].to_numpy(dtype=float),
        px=launch["px"].to_numpy(dtype=float),
        y=launch["y"].to_numpy(dtype=float),
        py=launch["py"].to_numpy(dtype=float),
        delta=np.zeros(len(launch)),
        zeta=np.zeros(len(launch)),
    )
    with suppress_xsuite_output():
        sampled_line.track(particles, turn_by_turn_monitor="ONE_TURN_EBE")
    monitor = sampled_line.record_last_track

    monitor_s = np.asarray(monitor.s, dtype=float)
    monitor_x = np.asarray(monitor.x, dtype=float)
    monitor_px = np.asarray(monitor.px, dtype=float)
    if monitor_s.ndim != 2 or monitor_x.shape != monitor_s.shape:
        raise RuntimeError("Unexpected Xsuite element-by-element monitor shape")

    reference_s = monitor_s[0]
    monitor_columns = []
    for s_value in s_positions:
        matches = np.flatnonzero(np.isclose(
            reference_s,
            float(s_value),
            rtol=0.0,
            atol=1e-9,
        ))
        if len(matches) == 0:
            raise RuntimeError(f"Xsuite did not record the requested s = {s_value:g} m")
        # Several zero-length entry/exit maps can share an s coordinate. The
        # final one is the state fully transported to that longitudinal point.
        monitor_columns.append(int(matches[-1]))

    return monitor_x[:, monitor_columns].T, monitor_px[:, monitor_columns].T


def plot_linked_matching_section_phase_space(
    tracked: Mapping[str, object],
    tw_dense,
    line,
    *,
    emittance: float = DEFAULT_GEOMETRIC_EMITTANCE,
    sigma_level: float = 3.0,
    sample_step_m: float = 0.1,
    max_display_particles: int = 250,
    frame_duration_ms: int = 80,
    title: str = "Tracked horizontal phase space through the matching section",
    show: bool | None = None,
):
    """Animate one tracked particle subset in x(s) and horizontal phase space.

    The left panel shows actual particle coordinates tracked at fine, uniformly
    spaced longitudinal positions. The shaded envelope is derived from the
    transported ``betx``:
    ``sigma_x(s) = sqrt(emittance * betx(s))``. The right panel shows the same
    particles and the corresponding transported constant-area ellipse.

    The already-sampled input bunch from ``track_distribution_along_line`` is
    retracked through a sliced copy of the line; the original line is unchanged.
    """

    if emittance <= 0.0:
        raise ValueError("emittance must be positive")
    if sigma_level <= 0.0:
        raise ValueError("sigma_level must be positive")
    if max_display_particles < 1:
        raise ValueError("max_display_particles must be at least one")

    input_df = _validate_tracking_input(tracked)
    twiss_table = tw_dense.to_pandas()
    required_twiss = {"s", "betx", "alfx"}
    if not required_twiss.issubset(twiss_table.columns):
        raise ValueError("tw_dense must contain s, betx, and alfx")

    station_s = _uniform_s_positions(float(line.get_length()), sample_step_m)
    all_particle_ids = np.arange(len(input_df), dtype=int)
    if len(all_particle_ids) > max_display_particles:
        selection = np.linspace(
            0,
            len(all_particle_ids) - 1,
            max_display_particles,
            dtype=int,
        )
        particle_ids = all_particle_ids[selection]
    else:
        particle_ids = all_particle_ids

    tracked_x, tracked_px = _track_selected_particles_at_s(
        line,
        input_df,
        particle_ids,
        station_s,
    )
    x_by_station_mm = 1e3 * tracked_x
    px_by_station_mrad = 1e3 * tracked_px

    # One static trace contains all particle paths. NaNs separate particles so
    # Plotly does not join the end of one track to the start of the next.
    n_particles = len(particle_ids)
    trajectory_s = np.concatenate([
        np.tile(station_s, (n_particles, 1)),
        np.full((n_particles, 1), np.nan),
    ], axis=1).ravel()
    trajectory_x_mm = np.concatenate([
        x_by_station_mm.T,
        np.full((n_particles, 1), np.nan),
    ], axis=1).ravel()

    dense_s = twiss_table["s"].to_numpy(dtype=float)
    dense_beta_x = twiss_table["betx"].to_numpy(dtype=float)
    sigma_x_mm = 1e3 * np.sqrt(emittance * dense_beta_x)
    envelope_mm = sigma_level * sigma_x_mm

    ellipse_frames: list[tuple[np.ndarray, np.ndarray]] = []
    for s_value in station_s:
        twiss_row = _twiss_row_near_s(twiss_table, s_value)
        ellipse_frames.append(_ellipse_points(
            float(twiss_row["betx"]),
            float(twiss_row["alfx"]),
            emittance=emittance,
            sigma_level=sigma_level,
        ))

    left_limit = 1.06 * max(
        float(np.max(np.abs(x_by_station_mm))),
        float(np.max(np.abs(envelope_mm))),
    )
    phase_x_limit = 1.06 * max(
        float(np.max(np.abs(x_by_station_mm))),
        max(float(np.max(np.abs(points[0]))) for points in ellipse_frames),
    )
    phase_px_limit = 1.06 * max(
        float(np.max(np.abs(px_by_station_mrad))),
        max(float(np.max(np.abs(points[1]))) for points in ellipse_frames),
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "xy"}, None],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        row_heights=[0.13, 0.87],
        column_widths=[0.60, 0.40],
        shared_xaxes=True,
        horizontal_spacing=0.11,
        vertical_spacing=0.035,
    )

    fig.add_trace(
        go.Scattergl(
            x=trajectory_s,
            y=trajectory_x_mm,
            mode="lines",
            line=dict(color="rgba(70, 95, 125, 0.10)", width=0.7),
            name="tracked particles",
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dense_s,
            y=-envelope_mm,
            mode="lines",
            line=dict(color="rgba(31, 119, 180, 0.75)", width=1.4),
            name=f"±{sigma_level:g}σx from βx(s)",
            legendgroup="beam envelope",
            showlegend=False,
            hovertemplate="s = %{x:.4g} m<br>lower envelope = %{y:.5g} mm<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dense_s,
            y=envelope_mm,
            mode="lines",
            line=dict(color="rgba(31, 119, 180, 0.75)", width=1.4),
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.10)",
            name=f"±{sigma_level:g}σx from βx(s)",
            legendgroup="beam envelope",
            hovertemplate="s = %{x:.4g} m<br>upper envelope = %{y:.5g} mm<extra></extra>",
        ),
        row=2,
        col=1,
    )

    initial_ellipse_x, initial_ellipse_px = ellipse_frames[0]
    station_color = np.full(n_particles, station_s[0])
    fig.add_trace(
        go.Scatter(
            x=np.full(n_particles, station_s[0]),
            y=x_by_station_mm[0],
            mode="markers",
            marker=dict(
                size=4,
                opacity=0.62,
                color=station_color,
                coloraxis="coloraxis",
            ),
            customdata=particle_ids,
            name="current particle slice",
            legendgroup="current particles",
            hovertemplate=(
                "particle %{customdata}<br>s = %{x:.4g} m"
                "<br>x = %{y:.5g} mm<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[station_s[0], station_s[0]],
            y=[-left_limit, left_limit],
            mode="lines",
            line=dict(color="rgba(45, 55, 72, 0.75)", width=1.3, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_by_station_mm[0],
            y=px_by_station_mrad[0],
            mode="markers",
            marker=dict(
                size=5,
                opacity=0.60,
                color=station_color,
                coloraxis="coloraxis",
            ),
            customdata=particle_ids,
            name="same particles in phase space",
            legendgroup="current particles",
            showlegend=False,
            hovertemplate=(
                "particle %{customdata}<br>x = %{x:.5g} mm"
                "<br>x' = %{y:.5g} mrad<extra></extra>"
            ),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=initial_ellipse_x,
            y=initial_ellipse_px,
            mode="lines",
            line=dict(color="crimson", width=2.0),
            name=f"transported {sigma_level:g}σ ellipse",
            hoverinfo="skip",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[-0.94 * phase_x_limit],
            y=[0.91 * phase_px_limit],
            mode="text",
            text=[f"s = {station_s[0]:.2f} m"],
            textposition="middle right",
            textfont=dict(size=12, color="rgba(70, 80, 95, 0.90)"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=2,
    )

    animated_trace_indices = [3, 4, 5, 6, 7]
    frames = []
    for i_station, s_value in enumerate(station_s):
        s_value = float(s_value)
        color_values = np.full(n_particles, s_value)
        ellipse_x, ellipse_px = ellipse_frames[i_station]
        frames.append(go.Frame(
            name=f"sample-{i_station}",
            traces=animated_trace_indices,
            data=[
                go.Scatter(
                    x=np.full(n_particles, s_value),
                    y=x_by_station_mm[i_station],
                    marker=dict(color=color_values),
                    customdata=particle_ids,
                ),
                go.Scatter(
                    x=[s_value, s_value],
                    y=[-left_limit, left_limit],
                ),
                go.Scatter(
                    x=x_by_station_mm[i_station],
                    y=px_by_station_mrad[i_station],
                    marker=dict(color=color_values),
                    customdata=particle_ids,
                ),
                go.Scatter(x=ellipse_x, y=ellipse_px),
                go.Scatter(
                    x=[-0.94 * phase_x_limit],
                    y=[0.91 * phase_px_limit],
                    text=[f"s = {s_value:.2f} m"],
                ),
            ],
        ))
    fig.frames = frames

    slider_steps = []
    actual_step = float(station_s[1] - station_s[0])
    for i_station, s_value in enumerate(station_s):
        frame_name = f"sample-{i_station}"
        show_tick_label = np.isclose(
            float(s_value),
            round(float(s_value)),
            rtol=0.0,
            atol=0.25 * actual_step,
        )
        slider_steps.append({
            "label": f"{float(s_value):.0f}" if show_tick_label else "",
            "method": "animate",
            "args": [[frame_name], {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": False},
                "transition": {"duration": 0},
            }],
        })

    fig.update_xaxes(
        range=[float(station_s.min()), float(station_s.max())],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[-1.05, 1.05],
        visible=False,
        fixedrange=True,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        range=[float(station_s.min()), float(station_s.max())],
        title_text="s [m]",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        range=[-left_limit, left_limit],
        title_text="x [mm]",
        row=2,
        col=1,
    )
    fig.update_xaxes(
        range=[-phase_x_limit, phase_x_limit],
        title_text="x [mm]",
        row=2,
        col=2,
    )
    fig.update_yaxes(
        range=[-phase_px_limit, phase_px_limit],
        title_text="x' [mrad]",
        row=2,
        col=2,
    )

    left_domain = tuple(float(value) for value in fig.layout.xaxis2.domain)
    right_domain = tuple(float(value) for value in fig.layout.xaxis3.domain)

    fig.update_layout(
        title=dict(text=title, x=0.50, xanchor="center"),
        template="plotly_white",
        width=1080,
        height=535,
        margin=dict(l=70, r=45, t=125, b=85),
        coloraxis=dict(
            colorscale="Viridis",
            cmin=float(station_s.min()),
            cmax=float(station_s.max()),
            colorbar=dict(title="s [m]", len=0.60, y=0.42),
        ),
        legend=dict(
            orientation="h",
            x=0.0,
            y=-0.19,
            yanchor="top",
        ),
        sliders=[{
            "active": 0,
            "x": left_domain[0],
            "y": 1.18,
            "len": left_domain[1] - left_domain[0],
            "pad": {"t": 0, "b": 0},
            "currentvalue": {"prefix": "s = ", "suffix": " m"},
            "steps": slider_steps,
        }],
        updatemenus=[{
            "type": "buttons",
            "direction": "left",
            "x": right_domain[0],
            "y": 1.18,
            "xanchor": "left",
            "showactive": False,
            "pad": {"r": 8, "t": 0},
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {
                        "fromcurrent": True,
                        "mode": "immediate",
                        "frame": {
                            "duration": int(frame_duration_ms),
                            "redraw": False,
                        },
                        "transition": {
                            "duration": min(250, int(frame_duration_ms)),
                        },
                    }],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": False},
                        "transition": {"duration": 0},
                    }],
                },
            ],
        }],
    )

    add_lattice_strip(
        fig,
        _line_lattice_layout(line),
        xref="x",
        yref="y",
        y=0.0,
        height=1.0,
        show_labels=True,
    )

    if show is None:
        show = should_show_plot("QF_LAB_SUPPRESS_PLOTS")
    return show_or_return(fig, show=show)
