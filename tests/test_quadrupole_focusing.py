from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from uspas_labs import quadrupole_focusing as qfh


def test_drift_lengths_validate_geometry():
    assert qfh._drift_lengths(0.5, 2.5) == pytest.approx((0.25, 2.0))

    with pytest.raises(ValueError, match="positive"):
        qfh._drift_lengths(0.0, 2.5)
    with pytest.raises(ValueError, match="smaller"):
        qfh._drift_lengths(2.5, 2.5)


def test_make_fodo_line_has_expected_length_names_and_reference_particle():
    line = qfh.make_fodo_line(n_cells=2, k1=0.6, prefix="test")

    assert line.get_length() == pytest.approx(10.0)
    assert "test_QFa_c01" in line.element_names
    assert "test_QFb_c02" in line.element_names
    assert "test_QDa_c02" in line.element_names
    assert "test_QDb_c02" in line.element_names
    assert "test_QD_c02" not in line.element_names
    assert len(line.element_names) == len(set(line.element_names))
    assert line.particle_ref is not None


def test_hybrid_and_insertion_line_lengths_are_composed_from_pieces():
    hybrid = qfh.make_hybrid_fodo_line(first_cells=3, second_cells=2, k1_first=0.6, k1_second=0.5)
    assert hybrid.get_length() == pytest.approx(25.0)
    assert any(name.startswith("strong_") for name in hybrid.element_names)
    assert any(name.startswith("weak_") for name in hybrid.element_names)

    fodo = qfh.make_fodo_line(1, k1=0.6)
    section = qfh.make_match_section()
    insertion = qfh.make_injection_insertion_line(fodo, section)
    assert insertion.get_length() == pytest.approx(27.0)


def test_match_section_knobs_can_be_set_and_validate_length():
    section = qfh.make_match_section(initial_strength=0.4)

    qfh.set_match_knobs(section, [0.1, -0.2, 0.3, -0.4])
    assert [float(section.vars[f"kQM{i}"]._value) for i in range(1, 5)] == pytest.approx([0.1, -0.2, 0.3, -0.4])

    with pytest.raises(ValueError, match="four"):
        qfh.set_match_knobs(section, [1.0, 2.0, 3.0])


def test_strong_to_weak_matching_and_section_plot_helpers(capsys):
    upstream_cell = qfh.make_fodo_line(1, k1=0.6)
    downstream_cell = qfh.make_fodo_line(1, k1=0.5)
    tw_upstream = qfh.twiss_periodic(upstream_cell)
    tw_downstream = qfh.twiss_periodic(downstream_cell)
    initial = qfh.initial_from_end(tw_upstream)
    target = qfh.initial_from_start(tw_downstream)
    section = qfh.make_match_section(initial_strength=0.6)

    tw_before = qfh.twiss_dense(section, points_per_meter=4, **initial)
    capsys.readouterr()
    with qfh.suppress_xsuite_output():
        optimizer = section.match(
            method="4d",
            vary=[
                qfh.xt.Vary(f"kQM{i}", step=1e-3, limits=(-3.0, 3.0))
                for i in range(1, 5)
            ],
            targets=[
                qfh.xt.Target(key, target[key], at="_end_point", tol=1e-8)
                for key in ("betx", "alfx", "bety", "alfy")
            ],
            n_steps_max=100,
            verbose=False,
            **initial,
        )
    captured = capsys.readouterr()
    tw_after = qfh.twiss_dense(section, points_per_meter=4, **initial)

    assert optimizer is not None
    assert captured.out == ""
    assert captured.err == ""
    comparison = qfh.twiss_endpoint_comparison(tw_after, target)
    assert list(comparison.columns) == [
        "quantity", "target value", "matched endpoint", "difference",
    ]
    assert comparison["difference"].abs().max() < 1e-7
    assert float(tw_before.s[-1]) == pytest.approx(11.0)
    assert all(
        np.isfinite(float(section.vars[f"kQM{i}"]._value))
        and abs(float(section.vars[f"kQM{i}"]._value)) <= 3.0
        for i in range(1, 5)
    )

    upstream = qfh.make_fodo_line(4, k1=0.6)
    downstream = qfh.make_fodo_line(4, k1=0.5)
    combined, sections = qfh.compose_labeled_sections([
        ("upstream FODO", "upstream", upstream),
        ("matching section", "match", section),
        ("weaker FODO", "downstream", downstream),
    ])

    assert combined.get_length() == pytest.approx(51.0)
    np.testing.assert_allclose(
        sections[["s_start [m]", "s_end [m]"]].to_numpy(),
        [[0.0, 20.0], [20.0, 31.0], [31.0, 51.0]],
    )
    assert "upstream_QFa_c01" in combined.element_names
    assert "match_QM1" in combined.element_names
    assert "downstream_QFb_c04" in combined.element_names
    assert len(combined.element_names) == len(set(combined.element_names))

    tw_combined = qfh.twiss_dense(combined, points_per_meter=4, **qfh.initial_from_start(tw_upstream))
    combined_table = qfh.twiss_dataframe(tw_combined)
    for s_boundary in [31.0, 36.0, 41.0, 46.0, 51.0]:
        row = combined_table.iloc[
            int(np.argmin(np.abs(combined_table["s"].to_numpy() - s_boundary)))
        ]
        assert float(row["s"]) == pytest.approx(s_boundary)
        for key in ["betx", "alfx", "bety", "alfy"]:
            assert float(row[key]) == pytest.approx(target[key], abs=1e-7)

    fig = qfh.plot_sectioned_twiss(tw_combined, sections)
    assert isinstance(fig, go.Figure)
    assert [trace.name for trace in fig.data] == ["βx", "βy"]
    rectangles = [shape for shape in fig.layout.shapes if shape.type == "rect"]
    lines = [shape for shape in fig.layout.shapes if shape.type == "line"]
    np.testing.assert_allclose(
        [(shape.x0, shape.x1) for shape in rectangles],
        [(0.0, 20.0), (20.0, 31.0), (31.0, 51.0)],
    )
    assert [shape.x0 for shape in lines] == pytest.approx([20.0, 31.0])


def test_periodic_fodo_twiss_matches_reference_phase_advance():
    line = qfh.make_fodo_line(n_cells=1, k1=0.6)
    tw = qfh.twiss_periodic(line)

    assert float(tw.qx) == pytest.approx(0.113502129062, rel=1e-9)
    assert float(tw.qy) == pytest.approx(float(tw.qx), rel=1e-12)
    assert float(tw.betx[0]) == pytest.approx(float(tw.betx[-1]), rel=1e-12)
    assert float(tw.alfx[0]) == pytest.approx(float(tw.alfx[-1]), rel=1e-12)

    qd_center = list(tw.name).index("QDb_c01")
    assert float(tw.s[qd_center]) == pytest.approx(2.5)
    assert float(tw.alfx[qd_center]) == pytest.approx(0.0, abs=1e-12)
    assert float(tw.alfy[qd_center]) == pytest.approx(0.0, abs=1e-12)


def test_twiss_dense_samples_start_and_end_of_line():
    line = qfh.make_fodo_line(n_cells=1, k1=0.6)
    tw = qfh.twiss_dense(line, n_points=11)

    assert len(tw.s) >= 11
    assert float(tw.s[0]) == pytest.approx(0.0)
    assert float(tw.s[-1]) == pytest.approx(line.get_length())
    assert float(tw.qx) == pytest.approx(0.113502129062, rel=1e-9)


def test_twiss_dataframe_and_initial_twiss_extractors_have_expected_fields():
    tw = qfh.twiss_periodic(qfh.make_fodo_line(1, k1=0.6))
    df = qfh.twiss_dataframe(tw)

    assert {"name", "s", "betx", "bety", "alfx", "alfy"}.issubset(df.columns)
    assert qfh.initial_from_start(tw).keys() == {"betx", "alfx", "bety", "alfy"}
    assert qfh.initial_from_end(tw) == qfh.fodo_end_twiss_dict(tw)


def test_twiss_lattice_layout_infers_element_spans_from_sliced_twiss_names():
    tw = qfh.twiss_dense(qfh.make_fodo_line(1, k1=0.6), n_points=21)
    layout = qfh._twiss_lattice_layout(tw)

    assert list(layout["name"]) == ["QFa_c01", "D1_c01", "QDa_c01", "QDb_c01", "D2_c01", "QFb_c01"]
    assert list(layout["kind"]) == ["quad", "drift", "quad", "quad", "drift", "quad"]
    assert layout["s_start_m"].to_list() == pytest.approx([0.0, 0.25, 2.25, 2.5, 2.75, 4.75])
    assert layout["s_end_m"].to_list() == pytest.approx([0.25, 2.25, 2.5, 2.75, 4.75, 5.0])
    assert (layout.loc[layout["name"].isin(["QDa_c01", "QDb_c01"]), "k1_m^-2"] < 0).all()


def test_beam_size_phase_and_thin_lens_tables_are_well_formed():
    tw = qfh.twiss_dense(qfh.make_fodo_line(1, k1=0.6), n_points=31)

    sizes = qfh.add_beam_sizes(tw)
    assert {"sigma_x_mm", "sigma_y_mm", "sigma_x_over_y"}.issubset(sizes.columns)
    assert sizes["sigma_x_mm"].min() > 0

    summary = qfh.beam_size_summary(tw, label="case")
    assert "case" in summary.columns
    assert summary.loc[summary["quantity"] == "length [m]", "case"].iloc[0] == pytest.approx(5.0)

    phase = qfh.phase_advance_table(tw).set_index("quantity")
    assert phase.loc["psi_x [rad]", "value"] == pytest.approx(2 * np.pi * float(tw.qx))

    thin = qfh.thin_lens_comparison_table(tw)
    assert set(thin["quantity"]).issuperset({"beta_min thin lens [m]", "beta_max Xsuite dense [m]"})


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        (0.0, "cell boundary / focusing quadrupole center"),
        (0.125, "focusing quadrupole QF"),
        (2.5, "defocusing quadrupole QD"),
        (4.875, "focusing quadrupole QF"),
        (1.25, "drift"),
    ],
)
def test_fodo_region_classifies_locations(s, expected):
    assert qfh.fodo_region(s) == expected


def test_location_and_cell_boundary_tables_identify_expected_conditions():
    tw = qfh.twiss_dense(qfh.make_fodo_line(2, k1=0.6), points_per_meter=40)

    locations = qfh.location_table(tw)
    assert {"round beam", "max sigma_x", "min sigma_y"}.issubset(set(locations["condition"]))

    boundaries = qfh.cell_start_envelope_table(tw)
    assert list(boundaries["cell index"]) == [0, 1, 2]
    assert boundaries["s [m]"].to_list() == pytest.approx([0.0, 5.0, 10.0])


def test_solve_k1_for_tune_recovers_default_strength():
    target_q = qfh.twiss_periodic(qfh.make_fodo_line(1, k1=0.6)).qx
    solved = qfh.solve_k1_for_tune(target_q, qfh.FODO_QUAD_LENGTH)

    assert solved == pytest.approx(0.6, rel=5e-4)


def test_twiss_covariance_and_distribution_tracking_are_deterministic():
    cov = qfh.twiss_covariance(beta=4.0, alpha=0.5, emit=2e-6)
    assert cov.shape == (2, 2)
    assert np.linalg.det(cov) == pytest.approx((2e-6) ** 2)

    line = qfh.make_fodo_line(1, k1=0.6)
    initial = {"betx": 4.0, "alfx": 0.0, "bety": 5.0, "alfy": 0.0}
    tracked_a = qfh.track_distribution(line, initial, n_particles=16, seed=123)
    tracked_b = qfh.track_distribution(line, initial, n_particles=16, seed=123)

    assert set(tracked_a) == {"input", "output"}
    assert list(tracked_a["input"].columns) == ["x", "px", "y", "py"]
    pd.testing.assert_frame_equal(tracked_a["input"], tracked_b["input"])
    assert len(tracked_a["output"]) == 16




def test_dense_twiss_restores_xsuite_progress_configuration():
    config = qfh.xt.progress_indicator._config
    indicator = config.default_indicator_cls
    options = dict(config.default_options)

    qfh.twiss_dense(qfh.make_fodo_line(1), n_points=7)

    assert config.default_indicator_cls is indicator
    assert config.default_options == options


def test_linked_centroid_plot_uses_same_qf_center_samples_in_both_panels():
    line = qfh.make_fodo_line(2, k1=0.6)
    centroid = qfh.centroid_orbit(line)
    fig = qfh.plot_centroid(centroid)

    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis2 is not None
    assert np.asarray(fig.data[1].x, dtype=float).tolist() == pytest.approx([0.0, 5.0, 10.0])
    assert np.asarray(fig.data[3].marker.color, dtype=float).tolist() == pytest.approx([0.0, 5.0, 10.0])
    assert fig.layout.xaxis.dtick == pytest.approx(qfh.FODO_CELL_LENGTH)
    assert fig.data[1].name == "QF-center samples"
    assert all(shape.xref == "x" for shape in fig.layout.shapes)


def test_along_line_tracking_filmstrip_and_emittance_display():
    fodo = qfh.make_fodo_line(1, k1=0.6)
    initial = qfh.fodo_end_twiss_dict(qfh.twiss_periodic(fodo))
    section = qfh.make_match_section()
    tracked = qfh.track_distribution_along_line(section, initial, n_particles=64, seed=123)

    stations = tracked["stations"]
    selected = qfh._filmstrip_stations(tracked)
    assert selected["label"].to_list() == ["entrance", "after QM1", "after QM2", "after QM3", "after QM4", "exit"]
    assert selected["s [m]"].to_list() == pytest.approx([0.0, 1.5, 4.0, 6.5, 9.0, 11.0])

    snapshots = tracked["snapshots"]
    assert snapshots.groupby("station_index").size().to_list() == [64] * len(stations)
    for _, group in snapshots.groupby("station_index"):
        assert group["particle_id"].to_list() == list(range(64))
    pd.testing.assert_frame_equal(
        tracked["input"],
        snapshots[snapshots["station_index"] == 0][["x", "px", "y", "py"]].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        tracked["output"],
        snapshots[snapshots["station_index"] == stations["station_index"].iloc[-1]][
            ["x", "px", "y", "py"]
        ].reset_index(drop=True),
    )

    tw = section.twiss(method="4d", **initial)
    filmstrip = qfh.plot_phase_space_filmstrip(tracked, tw)
    assert isinstance(filmstrip, go.Figure)
    assert len(filmstrip.data) == 12
    x_ranges = [tuple(axis.range) for axis in filmstrip.select_xaxes()]
    y_ranges = [tuple(axis.range) for axis in filmstrip.select_yaxes()]
    assert len(set(x_ranges)) == 1
    assert len(set(y_ranges)) == 1
    np.testing.assert_allclose(filmstrip.data[0].marker.color, filmstrip.data[2].marker.color)
    colorscale = filmstrip.data[0].marker.colorscale
    assert colorscale[0][1] == colorscale[-1][1]

    emittance = qfh.rms_emittance_table(tracked)
    assert emittance["epsilon_x / entrance"].to_numpy() == pytest.approx(np.ones(len(emittance)), abs=1e-11)
    assert emittance["epsilon_y / entrance"].to_numpy() == pytest.approx(np.ones(len(emittance)), abs=1e-11)
    emittance_figure = qfh.plot_emittance_conservation(tracked)
    assert isinstance(emittance_figure, go.Figure)
    assert all(shape.yref != "paper" for shape in emittance_figure.layout.shapes)


def test_cell_boundary_beta_beating_exposes_mismatch():
    fodo_before = qfh.make_fodo_line(1, k1=0.6)
    fodo_after = qfh.make_fodo_line(1, k1=0.5)
    tw_before = qfh.twiss_periodic(fodo_before)
    tw_after = qfh.twiss_periodic(fodo_after)
    initial = qfh.initial_from_start(tw_before)
    line = qfh.make_hybrid_fodo_line(
        first_cells=10,
        second_cells=10,
        k1_first=0.6,
        k1_second=0.5,
    )
    tw_line = line.twiss(method="4d", **initial)

    table = qfh.cell_boundary_beta_beat_table(
        tw_line,
        tw_before,
        tw_after,
        transition_s=50.0,
    )
    np.testing.assert_allclose(table["s [m]"], np.arange(0.0, 105.0, 5.0))
    before_transition = table["s [m]"] < 50.0
    np.testing.assert_allclose(table.loc[before_transition, "beta_x beat [%]"], 0.0, atol=1e-11)
    np.testing.assert_allclose(table.loc[before_transition, "beta_y beat [%]"], 0.0, atol=1e-11)

    at_transition = table.loc[table["s [m]"] == 50.0].iloc[0]
    assert at_transition["beta_x transported [m]"] == pytest.approx(float(tw_before.betx[0]))
    assert at_transition["beta_y transported [m]"] == pytest.approx(float(tw_before.bety[0]))
    assert at_transition["beta_x matched [m]"] == pytest.approx(float(tw_after.betx[0]))
    assert at_transition["beta_y matched [m]"] == pytest.approx(float(tw_after.bety[0]))

    after_transition = table["s [m]"] >= 50.0
    for column in ["beta_x beat [%]", "beta_y beat [%]"]:
        values = table.loc[after_transition, column]
        assert values.min() < 0.0
        assert values.max() > 0.0

    fig = qfh.plot_cell_boundary_beta_beating(
        tw_line,
        tw_before,
        tw_after,
        transition_s=50.0,
    )
    assert isinstance(fig, go.Figure)
    assert [trace.name for trace in fig.data] == ["horizontal beta beat", "vertical beta beat"]
    for trace in fig.data:
        np.testing.assert_allclose(trace.x, table["s [m]"])
    assert len(fig.layout.shapes) == 2


def test_quad_length_scan_plot_shows_data_against_fixed_references():
    target_q = qfh.twiss_periodic(qfh.make_fodo_line(1, k1=0.6)).qx
    scan = qfh.quad_length_fixed_phase_table(target_q, quad_lengths=(0.1, 0.5, 1.0))
    fig = qfh.plot_quad_length_scan(scan, target_q)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4
    assert list(fig.data[0].x) == pytest.approx([0.1, 0.5, 1.0])
    assert np.ptp(np.asarray(fig.data[2].y, dtype=float)) == pytest.approx(0.0)
    assert np.ptp(np.asarray(fig.data[3].y, dtype=float)) == pytest.approx(0.0)


def test_plot_helpers_return_figures_when_plot_display_is_suppressed():
    tw = qfh.twiss_dense(qfh.make_fodo_line(1, k1=0.6), n_points=11)

    fig = qfh.plot_twiss(tw)
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.shapes) > 0
    assert len(qfh.plot_twiss(tw, show_lattice=False).layout.shapes or []) == 0
    assert isinstance(qfh.plot_sigmas(tw), go.Figure)
    assert isinstance(qfh.plot_beta_and_sigma(tw), go.Figure)
    assert isinstance(qfh.plot_hybrid_transition(tw, transition_s=2.5), go.Figure)
