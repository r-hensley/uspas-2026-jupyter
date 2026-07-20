from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from uspas_labs import dispersion_chromaticity as dch


def test_element_edge_aliases_and_lattice_name_lookup_copy_independently():
    element = dch.Element("B", "bend", 1.0, angle=0.1)
    element.edge_entry_angle = 0.2
    assert element.edge_exit_angle == pytest.approx(0.2)
    element.edge_exit_angle = 0.3
    assert element.edge_entry_angle == pytest.approx(0.3)

    lattice = dch.Lattice([element])
    copied = lattice.copy()
    copied["B"].length = 2.0

    assert lattice["B"].length == pytest.approx(1.0)
    assert copied["B"].length == pytest.approx(2.0)
    with pytest.raises(KeyError):
        lattice["missing"]


def test_basic_transfer_matrices_are_symplectic_and_handle_limits():
    drift = dch.drift_matrix(2.0)
    assert drift.tolist() == [[1.0, 2.0], [0.0, 1.0]]
    np.testing.assert_allclose(dch.quad_matrix(2.0, 0.0), drift)

    focusing = dch.quad_matrix(0.5, 0.7)
    defocusing = dch.quad_matrix(0.5, -0.7)
    assert np.linalg.det(focusing) == pytest.approx(1.0)
    assert np.linalg.det(defocusing) == pytest.approx(1.0)

    bend_matrix, source = dch.sector_bend_matrix(1.0, 0.0)
    np.testing.assert_allclose(bend_matrix, dch.drift_matrix(1.0))
    np.testing.assert_allclose(source, np.zeros(2))


def test_sector_bend_and_edge_matrices_have_expected_signs():
    angle = 0.2
    matrix, source = dch.sector_bend_matrix(1.0, angle)
    h = angle / 1.0

    assert matrix[0, 0] == pytest.approx(math.cos(angle))
    assert source[0] == pytest.approx((1.0 - math.cos(angle)) / h)
    assert source[1] == pytest.approx(math.sin(angle))

    edge_x = dch.edge_matrix(0.2, 0.1, "x")
    edge_y = dch.edge_matrix(0.2, 0.1, "y")
    assert edge_x[1, 0] == pytest.approx(-edge_y[1, 0])
    with pytest.raises(ValueError, match="plane"):
        dch.edge_matrix(0.2, 0.1, "z")


def test_element_maps_validate_delta_kind_and_dispersion_source_toggle():
    bend = dch.Element("B", "bend", 1.0, angle=0.2)
    _, _, source_on = dch.element_maps(bend, include_dispersion_source=True)
    _, _, source_off = dch.element_maps(bend, include_dispersion_source=False)

    assert np.linalg.norm(source_on) > 0
    np.testing.assert_allclose(source_off, np.zeros(2))

    with pytest.raises(ValueError, match="delta"):
        dch.element_maps(bend, delta=-1.0)
    with pytest.raises(ValueError, match="Unknown"):
        dch.element_maps(dch.Element("X", "unknown", 1.0))


def test_split_element_conserves_integrated_bend_and_resets_internal_edges():
    bend = dch.Element("B", "bend", 1.2, angle=0.3, edge_angle=0.05)
    slices = dch.split_element(bend, 3)

    assert sum(item.length for item in slices) == pytest.approx(bend.length)
    assert sum(item.angle for item in slices) == pytest.approx(bend.angle)
    assert all(item.edge_angle == 0.0 for item in slices)

    quad_slices = dch.split_element(dch.Element("Q", "quad", 1.0, k1=0.4), 4)
    assert [item.length for item in quad_slices] == pytest.approx([0.25] * 4)


def test_transfer_map_composes_drifts_and_matched_twiss_handles_stability():
    elements = [dch.Element("D1", "drift", 1.0), dch.Element("D2", "drift", 2.0)]
    matrix_x, matrix_y, source = dch.transfer_map(elements)

    np.testing.assert_allclose(matrix_x, dch.drift_matrix(3.0))
    np.testing.assert_allclose(matrix_y, dch.drift_matrix(3.0))
    np.testing.assert_allclose(source, np.zeros(2))
    assert dch.matched_twiss_from_matrix(matrix_x) is None

    theta = 0.4
    rotation = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
    beta, alpha, tune = dch.matched_twiss_from_matrix(rotation)
    assert beta == pytest.approx(1.0)
    assert alpha == pytest.approx(0.0)
    assert tune == pytest.approx(theta / (2.0 * math.pi))


def test_propagate_twiss_through_drift_matches_formula():
    beta = 3.0
    alpha = -0.4
    length = 2.0
    gamma = (1 + alpha**2) / beta

    beta_new, alpha_new = dch.propagate_twiss(beta, alpha, dch.drift_matrix(length))

    assert beta_new == pytest.approx(beta - 2 * length * alpha + length**2 * gamma)
    assert alpha_new == pytest.approx(alpha - length * gamma)


def test_lattice_construction_layout_and_repetition():
    cell = dch.make_dba_cell()
    assert [element.name for element in cell][:3] == ["D0", "Q2", "D1"]
    assert len(cell) == 11

    repeated = dch.repeat_cell(cell[:2], n_cells=2)
    assert [element.name for element in repeated] == ["D0_c1", "Q2_c1", "D0_c2", "Q2_c2"]
    assert repeated[0] is not cell[0]

    layout = dch.element_layout(cell[:2])
    assert list(layout["name"]) == ["D0", "Q2"]
    assert layout["s_end_m"].iloc[-1] == pytest.approx(0.8)


@pytest.mark.skipif(dch.xt is None, reason="xtrack is required for Xsuite-line conversion")
def test_make_fodo_cell_uses_shared_split_geometry_and_roles():
    cell = dch.make_fodo_cell(kq=0.6, with_bend=True, edge_focusing=True)

    assert [element.name for element in cell] == ["QFa", "D1a", "B1", "D1b", "QD", "D2a", "B2", "D2b", "QFb"]
    assert cell["QFa"].length == pytest.approx(0.25)
    assert cell["QD"].role == "defocusing quadrupole"
    assert cell["B1"].kind == "bend"
    assert math.degrees(cell["B1"].angle) == pytest.approx(10.0)
    assert cell["B1"].edge_angle > 0
    assert cell["B2"].edge_angle > 0


@pytest.mark.skipif(dch.xt is None, reason="xtrack is required for Xsuite-line conversion")
def test_xsuite_conversion_accepts_repeated_environment_components():
    env = dch.xt.Environment()
    env["kq"] = 0.6
    env["bend_angle"] = math.radians(10.0)
    env.new("QFh", dch.xt.Quadrupole, length=0.25, k1="kq")
    env.new("D", dch.xt.Drift, length=0.75)
    env.new("B", dch.xt.Bend, length=0.5, angle="bend_angle", k0="bend_angle / 0.5")
    line = env.new_line(name="repeated", components=["QFh", "D", "B", "D", "QFh"])

    converted = dch.elements_from_xsuite_line(line, occurrence_names=["QFa", "D1a", "B1", "D1b", "QFb"])

    assert [element.name for element in converted] == ["QFa", "D1a", "B1", "D1b", "QFb"]
    assert [element.kind for element in converted] == ["quad", "drift", "bend", "drift", "quad"]
    assert converted["B1"].angle == pytest.approx(math.radians(10.0))


@pytest.mark.skipif(dch.xt is None, reason="xtrack is required for Xsuite-line conversion")
def test_xsuite_display_workflow_helpers_return_notebook_objects():
    env = dch.xt.Environment()
    env["kq"] = 0.6
    env.new("QFh", dch.xt.Quadrupole, length=0.25, k1="kq")
    env.new("D", dch.xt.Drift, length=2.0)
    env.new("QD", dch.xt.Quadrupole, length=0.5, k1="-kq")
    reference_line = env.new_line(name="fodo_reference", components=["QFh", "D", "QD", "D", "QFh"])

    reference, reference_optics = dch.display_fodo_reference_from_xsuite_line(reference_line, show_plot=False)

    assert [element.name for element in reference] == ["QFa", "D1", "QD", "D2", "QFb"]
    assert reference_optics.periodic is True

    env_bend = dch.xt.Environment()
    env_bend["kq"] = 0.6
    env_bend["bend_angle_each"] = math.radians(10.0)
    env_bend["bend_length"] = 0.5
    env_bend.new("QFh", dch.xt.Quadrupole, length=0.25, k1="kq")
    env_bend.new("D", dch.xt.Drift, length=0.75)
    env_bend.new("BEND", dch.xt.Bend, length="bend_length", angle="bend_angle_each", k0="bend_angle_each / bend_length")
    env_bend.new("QD", dch.xt.Quadrupole, length=0.5, k1="-kq")
    bend_line = env_bend.new_line(name="fodo_with_bend", components=["QFh", "D", "BEND", "D", "QD", "D", "BEND", "D", "QFh"])

    bend, bend_optics = dch.display_fodo_bend_from_xsuite_line(bend_line, show_plot=False)

    assert [element.name for element in bend] == ["QFa", "D1a", "B1", "D1b", "QD", "D2a", "B2", "D2b", "QFb"]
    assert bend_optics.periodic is True

    env_dba = dch.xt.Environment()
    env_dba["bend_angle"] = math.radians(18.0)
    env_dba["bend_length"] = 1.0
    env_dba.new("D0", dch.xt.Drift, length=0.5)
    env_dba.new("Q2", dch.xt.Quadrupole, length=0.3, k1=0.0)
    env_dba.new("D1", dch.xt.Drift, length=0.5)
    env_dba.new("B1", dch.xt.Bend, length="bend_length", angle="bend_angle", k0="bend_angle / bend_length")
    env_dba.new("D2", dch.xt.Drift, length=2.3)
    env_dba.new("Q1", dch.xt.Quadrupole, length=0.3, k1=0.0)
    env_dba.new("D3", dch.xt.Drift, length=2.3)
    env_dba.new("B2", dch.xt.Bend, length="bend_length", angle="bend_angle", k0="bend_angle / bend_length")
    env_dba.new("D4", dch.xt.Drift, length=0.5)
    env_dba.new("Q3", dch.xt.Quadrupole, length=0.3, k1=0.0)
    env_dba.new("D5", dch.xt.Drift, length=0.5)
    dba_line = env_dba.new_line(name="two_bend_off", components=["D0", "Q2", "D1", "B1", "D2", "Q1", "D3", "B2", "D4", "Q3", "D5"])

    dba, dba_optics, eta_end, etap_end = dch.display_dba_insert_from_xsuite_line(dba_line, show_plot=False)

    assert dba["Q1"].role == "central dispersion-control quadrupole"
    assert dba_optics.periodic is False
    assert eta_end == pytest.approx(dch.endpoint_dispersion(dba)[0])
    assert etap_end == pytest.approx(dch.endpoint_dispersion(dba)[1])

    central, focused, focused_optics = dch.display_focused_dba_cell(dch.DBA_Q1_DEFAULT, show_plot=False)
    assert central["Q1"].k1 == pytest.approx(dch.DBA_Q1_DEFAULT)
    assert focused["Q2"].k1 == pytest.approx(dch.DBA_Q2_DEFAULT)
    assert focused_optics.periodic is True


def test_compute_transport_and_periodic_optics_contracts():
    drift = [dch.Element("D", "drift", 2.0)]
    initial = {"beta_x_m": 3.0, "alpha_x": 0.0, "beta_y_m": 4.0, "alpha_y": 0.0}
    transport = dch.compute_transport_optics(drift, initial=initial, samples_per_meter=1)

    assert transport.periodic is False
    assert transport.table["s_m"].iloc[0] == pytest.approx(0.0)
    assert transport.table["s_m"].iloc[-1] == pytest.approx(2.0)
    assert transport.tune_x is None

    with pytest.raises(ValueError, match="not stable"):
        dch.compute_periodic_optics(drift)

    periodic = dch.compute_periodic_optics(dch.make_dba_cell(), samples_per_meter=2)
    assert periodic.periodic is True
    assert bool(periodic.stable_x) is True
    assert bool(periodic.stable_y) is True
    assert periodic.tune_x > 0
    assert periodic.tune_y > 0


def test_periodic_sampling_preserves_bend_edge_kicks():
    cell = dch.make_fodo_cell(kq=0.6, with_bend=True, bend_angle_deg=20.0)
    for bend_name in ["B1", "B2"]:
        cell[bend_name].edge_angle = math.radians(5.0)

    result = dch.compute_periodic_optics(cell, samples_per_meter=8)
    start = result.table.iloc[0]
    end = result.table.iloc[-1]

    for column in ["beta_x_m", "alpha_x", "beta_y_m", "alpha_y", "eta_x_m", "eta_xp"]:
        assert end[column] == pytest.approx(start[column], rel=1e-11, abs=1e-11)


def test_display_fodo_edge_focusing_uses_user_supplied_builder():
    reference = dch.compute_periodic_optics(dch.make_fodo_cell(with_bend=False))
    bend = dch.compute_periodic_optics(dch.make_fodo_cell(with_bend=True))

    def build(edge_angle_deg):
        cell = dch.make_fodo_cell(kq=0.6, with_bend=True, bend_angle_deg=20.0)
        edge_angle = math.radians(edge_angle_deg)
        for bend_name in ["B1", "B2"]:
            cell[bend_name].edge_entry_angle = edge_angle
            cell[bend_name].edge_exit_angle = edge_angle
        return cell

    result = dch.display_fodo_edge_focusing(5.0, build, reference, bend, show_plot=False)

    assert result is not None
    assert result.elements[2].edge_angle == pytest.approx(math.radians(5.0))
    assert result.tune_x is not None


def test_summary_and_table_helpers_extract_expected_rows():
    result = dch.compute_periodic_optics(dch.make_dba_cell(), samples_per_meter=2)

    summary = dch.optics_summary(result, label="cell")
    assert "cell" in summary.columns
    assert set(summary["quantity"]).issuperset({"length [m]", "stable x?", "\N{GREEK SMALL LETTER ETA}\N{LATIN SUBSCRIPT SMALL LETTER X} [m]"})

    compact_layout = dch.compact_lattice_table(dch.make_fodo_cell(with_bend=False))
    assert "element" in compact_layout.columns
    assert "θ (deg)" not in compact_layout.columns
    assert "edge θ (deg)" not in compact_layout.columns
    compact_bend_layout = dch.compact_lattice_table(dch.make_fodo_cell(with_bend=True, edge_focusing=False))
    assert "θ (deg)" in compact_bend_layout.columns
    assert "edge θ (deg)" not in compact_bend_layout.columns

    compact_summary = dch.compact_optics_summary(result)
    assert list(compact_summary.columns) == ["quantity", "value"]
    assert not compact_summary.astype(str).apply(lambda column: column.str.contains("nan", case=False)).any().any()
    compact_optics_comparison = dch.compact_optics_comparison({"cell": result, "same cell": result})
    assert list(compact_optics_comparison.columns) == ["quantity", "cell", "same cell"]
    assert "cell tune Qₓ" in set(compact_optics_comparison["quantity"])
    assert not compact_optics_comparison.astype(str).apply(lambda column: column.str.contains("nan", case=False)).any().any()
    compact_stability = dch.compact_stability_report(dch.make_dba_cell())
    assert list(compact_stability["stable"]) == ["yes", "yes"]

    beam = dch.add_beam_size_columns(result.table, sigma_delta=0.001)
    assert {"sigma_x_beta_mm", "sigma_x_dispersion_mm", "sigma_x_mm", "sigma_y_mm"}.issubset(beam.columns)
    assert (beam["sigma_x_mm"] >= beam["sigma_x_beta_mm"]).all()

    q1_row = dch.row_at_element_center(result, "Q1")
    assert q1_row["element"] == "Q1"
    with pytest.raises(KeyError):
        dch.row_at_element_center(result, "not-an-element")

    centers = dch.table_at_element_centers(result, ["Q1", "B1"], sigma_delta=0.001)
    assert list(centers["element"]) == ["Q1", "B1"]

    fodo_reference = dch.compute_periodic_optics(dch.make_fodo_cell(with_bend=False))
    compact_centers = dch.compact_element_center_table(fodo_reference, ["QFa", "QD"], sigma_delta=0.0)
    assert {"βₓ (m)", "βᵧ (m)", "σₓ (mm)", "σᵧ (mm)"}.issubset(compact_centers.columns)
    assert "ηₓ (m)" not in compact_centers.columns

    comparison = dch.beam_size_comparison_at_elements(result, ["Q1"], sigma_delta=0.001)
    assert comparison.loc[0, "element"] == "Q1"
    compact_comparison = dch.compact_beam_size_comparison_at_elements(result, ["Q1"], sigma_delta=0.001)
    assert "σₓ, δ=0.001 (mm)" in compact_comparison.columns

    extrema = dch.dispersion_extrema(result)
    assert list(extrema["condition"]) == ["minimum eta_x", "maximum eta_x", "maximum |eta_x|"]


def test_endpoint_dispersion_and_achromat_scan_are_consistent():
    eta_end, etap_end = dch.endpoint_dispersion(dch.make_dba_cell(q1=dch.DBA_Q1_DEFAULT, q2=0.0, q3=0.0))
    assert eta_end == pytest.approx(0.0, abs=1e-7)
    assert etap_end == pytest.approx(0.0, abs=1e-7)

    table = dch.dba_endpoint_table(dch.DBA_Q1_DEFAULT, q2=0.0, q3=0.0)
    assert table.loc[2, "value"] < 1e-12
    compact_endpoint = dch.compact_dba_endpoint_table(dch.DBA_Q1_DEFAULT, q2=0.0, q3=0.0)
    assert "ηₓ end (m)" in set(compact_endpoint["quantity"])
    assert not compact_endpoint.astype(str).apply(lambda column: column.str.contains("nan", case=False)).any().any()

    scan = dch.scan_q1_for_achromat(qmin=2.0, qmax=2.6, n=7, q2=0.0, q3=0.0)
    assert list(scan.columns) == ["q1_m^-2", "eta_x_end_m", "eta_xp_end", "penalty"]
    assert scan.loc[scan["penalty"].idxmin(), "q1_m^-2"] == pytest.approx(2.3, abs=0.2)
    compact_scan = dch.compact_q1_scan_table(scan, rows=3)
    assert list(compact_scan.columns) == ["Q1 k₁ (m⁻²)", "ηₓ end (m)", "ηₓ′ end", "η penalty"]


def test_aperture_helpers_sort_limits_and_validate_inputs():
    result = dch.compute_periodic_optics(dch.make_dba_cell(), samples_per_meter=2)

    limits = dch.aperture_limit_table(result, pipe_radius_m=0.025, n_sigma=1.0)
    assert limits["max_sigma_delta"].iloc[0] <= limits["max_sigma_delta"].iloc[-1]
    assert {"max_sigma_delta", "max_delta_percent"}.issubset(limits.columns)

    summary = dch.aperture_summary(result, pipe_radius_m=0.025, n_sigma=1.0)
    assert summary.loc[0, "quantity"] == "limiting momentum spread"
    compact_limits = dch.compact_aperture_limit_table(limits.head(3))
    assert {"ηₓ (m)", "δ limit", "δ limit (%)"}.issubset(compact_limits.columns)
    compact_summary = dch.compact_aperture_summary(result, pipe_radius_m=0.025, n_sigma=1.0)
    assert "δ limit" in set(compact_summary["quantity"])

    with pytest.raises(ValueError, match="positive"):
        dch.aperture_limit_table(result, n_sigma=0.0)


def test_tune_chromaticity_and_resonance_helpers_return_sorted_tables():
    cell = dch.make_dba_cell()
    qx_cell, qy_cell = dch.cell_tunes(cell)
    qx_ring, qy_ring = dch.ring_tunes(cell, n_cells=3)

    assert qx_ring == pytest.approx(3 * qx_cell)
    assert qy_ring == pytest.approx(3 * qy_cell)

    cx, cy = dch.chromaticity_finite_difference(cell, n_cells=3, ddelta=1e-4)
    assert np.isfinite(cx)
    assert np.isfinite(cy)

    ring = dch.ring_summary(cell, n_cells=3)
    assert {"Qx ring tune", "Cy = dQy/d(delta)"}.issubset(set(ring["quantity"]))
    compact_ring = dch.compact_ring_summary(cell, n_cells=3)
    assert {"ring tune Qₓ", "Cᵧ = dQᵧ/dδ"}.issubset(set(compact_ring["quantity"]))

    spread = dch.chromatic_spread_table(qx_ring, qy_ring, cx, cy, sigma_delta=0.002)
    assert spread.loc[spread["quantity"] == "Delta Qx for sigma_delta=0.002", "value"].iloc[0] == pytest.approx(cx * 0.002)
    compact_spread = dch.compact_chromatic_spread_table(qx_ring, qy_ring, cx, cy, sigma_delta=0.002)
    assert "ΔQₓ for σδ=0.002" in set(compact_spread["quantity"])

    segments = dch.resonance_lines(2, (0.0, 1.0), (0.0, 1.0))
    assert segments
    assert all(segment["order"] <= 2 for segment in segments)

    crossings = dch.first_resonance_crossing(0.31, 0.28, -2.0, 1.0, max_order=3)
    assert not crossings.empty
    assert crossings["abs_delta_at_crossing"].is_monotonic_increasing
    compact_crossings = dch.compact_resonance_crossing(crossings.head(2))
    assert list(compact_crossings.columns) == ["resonance", "order", "|δ| at crossing", "|δ| (%)"]

    acceptance = dch.acceptance_comparison(0.01, 0.02)
    assert acceptance.loc[2, "note"] == "dispersion/aperture"
    compact_acceptance = dch.compact_acceptance_comparison(0.01, 0.02)
    assert {"mechanism", "δ limit", "δ limit (%)"}.issubset(compact_acceptance.columns)


def test_plot_helpers_return_figures_when_show_false():
    result = dch.compute_periodic_optics(dch.make_dba_cell(), samples_per_meter=2)
    scan = dch.scan_q1_for_achromat(qmin=2.0, qmax=2.6, n=7)
    qx, qy = dch.ring_tunes(dch.make_dba_cell(), n_cells=3)
    cx, cy = dch.chromaticity_finite_difference(dch.make_dba_cell(), n_cells=3)

    optics_fig = dch.plot_optics(result, show=False)
    assert isinstance(optics_fig, go.Figure)
    assert len(optics_fig.layout.shapes) > 0
    assert len(dch.plot_optics(result, show=False, show_lattice=False).layout.shapes or []) == 0
    assert isinstance(dch.plot_beam_size(result, show=False), go.Figure)
    combined_fig = dch.plot_optics_and_beam_size(result, sigma_delta=0.001, show=False)
    assert isinstance(combined_fig, go.Figure)
    assert {trace.name for trace in combined_fig.data}.issuperset({"beta_x", "eta_x", "sigma_x total"})
    assert len(dch.plot_optics_and_beam_size(result, show=False, show_lattice=False).layout.shapes or []) == 0
    assert isinstance(dch.plot_beam_size_with_aperture(result, sigma_delta=0.001, show=False), go.Figure)
    assert isinstance(dch.plot_q1_scan(scan, show=False), go.Figure)
    assert isinstance(dch.plot_tune_footprint(qx, qy, cx, cy, show=False), go.Figure)
