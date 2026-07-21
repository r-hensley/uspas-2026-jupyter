from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from uspas_labs import dispersion_chromaticity as dch


def _hover_templates(trace) -> list[str]:
    raw = trace.hovertemplate
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(value) for value in np.asarray(raw, dtype=object).ravel()]


def _assert_every_data_trace_hovers_its_value(fig: go.Figure) -> None:
    for trace in fig.data:
        assert trace.hoverinfo != "skip", trace.name
        templates = _hover_templates(trace)
        assert templates, trace.name
        role = trace.meta.get("role") if isinstance(trace.meta, dict) else None
        if role == "decorative":
            continue
        assert all("%{y" in template for template in templates), trace.name

    if fig.layout.hovermode == "x unified":
        for trace in fig.data:
            role = trace.meta.get("role") if isinstance(trace.meta, dict) else None
            if role == "decorative":
                continue
            for template in _hover_templates(trace):
                assert "s=%{x" not in template
                assert "s = %{x" not in template


@pytest.fixture(scope="module")
def section_c_xsuite_data():
    q1 = dch.best_q1_for_achromat(qmin=0.0, qmax=6.0)
    cell = dch.make_dba_cell(q1=q1)
    ring = dch.xsuite_ring_from_lattice(cell, n_cells=10)
    twiss = dch.xsuite_ring_twiss(ring, delta_chrom=1e-4)
    deltas = [-0.001, -0.0001, 0.0, 0.0001, 0.001, 0.0030, 0.0035, 0.0040]
    scan = dch.xsuite_tune_scan(ring, delta_values=deltas)
    return cell, ring, twiss, scan


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


def test_xsuite_ring_builder_preserves_fixed_design_bends():
    cell = dch.make_dba_cell()
    ring = dch.xsuite_ring_from_lattice(cell, n_cells=10)

    assert ring.get_length() == pytest.approx(10 * sum(element.length for element in cell))
    assert ring["B1_c1"].angle == pytest.approx(math.radians(18.0))
    assert ring["B1_c1"].k0 == pytest.approx(math.radians(18.0))
    assert ring["B1_c10"].angle == pytest.approx(ring["B1_c1"].angle)


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

    # The notebook shows this DBA construction explicitly once, then later
    # exercises reuse the background builder.  Keep both definitions identical.
    background_dba = dch.make_dba_cell(q1=0.0, q2=0.0, q3=0.0)
    assert [element.name for element in background_dba] == [element.name for element in dba]
    for visible, background in zip(dba, background_dba, strict=True):
        assert background.kind == visible.kind
        assert background.length == pytest.approx(visible.length)
        assert background.k1 == pytest.approx(visible.k1)
        assert background.angle == pytest.approx(visible.angle)
        assert background.edge_angle == pytest.approx(visible.edge_angle)

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
    assert set(summary["quantity"]).issuperset({"length [m]", "stable x?", "Dₓ [m]"})

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
    assert q1_row["s_m"] == pytest.approx(4.75)
    assert q1_row["eta_xp"] == pytest.approx(0.0, abs=1e-7)
    with pytest.raises(KeyError):
        dch.row_at_element_center(result, "not-an-element")

    centers = dch.table_at_element_centers(result, ["Q1", "B1"], sigma_delta=0.001)
    assert list(centers["element"]) == ["Q1", "B1"]
    assert centers.loc[0, "s_center_m"] == pytest.approx(q1_row["s_m"])
    assert centers.loc[0, "eta_x_m"] == pytest.approx(q1_row["eta_x_m"])

    for element in result.elements:
        if element.length > 0:
            assert dch._sample_count(element) % 2 == 0

    fodo_reference = dch.compute_periodic_optics(dch.make_fodo_cell(with_bend=False))
    compact_centers = dch.compact_element_center_table(fodo_reference, ["QFa", "QD"], sigma_delta=0.0)
    assert {"βₓ (m)", "βᵧ (m)", "σₓ (mm)", "σᵧ (mm)"}.issubset(compact_centers.columns)
    assert "Dₓ (m)" not in compact_centers.columns

    comparison = dch.beam_size_comparison_at_elements(result, ["Q1"], sigma_delta=0.001)
    assert comparison.loc[0, "element"] == "Q1"
    compact_comparison = dch.compact_beam_size_comparison_at_elements(result, ["Q1"], sigma_delta=0.001)
    assert "σₓ, σδ=0.001 (mm)" in compact_comparison.columns

    extrema = dch.dispersion_extrema(result)
    assert list(extrema["condition"]) == ["minimum D_x", "maximum D_x", "maximum |D_x|"]


def test_endpoint_dispersion_and_achromat_scan_are_consistent():
    eta_end, etap_end = dch.endpoint_dispersion(dch.make_dba_cell(q1=dch.DBA_Q1_DEFAULT, q2=0.0, q3=0.0))
    assert eta_end == pytest.approx(0.0, abs=1e-7)
    assert etap_end == pytest.approx(0.0, abs=1e-7)

    table = dch.dba_endpoint_table(dch.DBA_Q1_DEFAULT, q2=0.0, q3=0.0)
    assert table.loc[2, "value"] < 1e-12
    assert table.loc[2, "value"] == pytest.approx(
        dch.achromat_closure_merit(eta_end, etap_end)
    )
    compact_endpoint = dch.compact_dba_endpoint_table(dch.DBA_Q1_DEFAULT, q2=0.0, q3=0.0)
    assert "Dₓ end (m)" in set(compact_endpoint["quantity"])
    assert not compact_endpoint.astype(str).apply(lambda column: column.str.contains("nan", case=False)).any().any()

    scan = dch.scan_q1_for_achromat(qmin=2.0, qmax=2.6, n=7, q2=0.0, q3=0.0)
    assert list(scan.columns) == ["q1_m^-2", "eta_x_end_m", "eta_xp_end", "closure_merit"]
    assert scan.loc[scan["closure_merit"].idxmin(), "q1_m^-2"] == pytest.approx(2.3, abs=0.2)
    compact_scan = dch.compact_q1_scan_table(scan, rows=3)
    assert list(compact_scan.columns) == ["Q1 k₁ (m⁻²)", "Dₓ end (m)", "Dₓ′ end (1)", "dimensionless closure merit"]

    midpoint = dch.row_at_element_center(
        dch.compute_transport_optics(
            dch.make_dba_cell(q1=dch.DBA_Q1_DEFAULT, q2=0.0, q3=0.0),
            samples_per_meter=200,
        ),
        "Q1",
    )
    assert midpoint["eta_xp"] == pytest.approx(0.0, abs=1e-7)


def test_aperture_helpers_sort_limits_and_validate_inputs():
    result = dch.compute_periodic_optics(dch.make_dba_cell(), samples_per_meter=2)

    limits = dch.aperture_limit_table(result, pipe_radius_m=0.025, n_sigma=1.0)
    assert limits["max_sigma_delta"].iloc[0] <= limits["max_sigma_delta"].iloc[-1]
    assert {"max_sigma_delta", "max_delta_percent", "aperture_status", "on_momentum_clearance_mm"}.issubset(limits.columns)

    summary = dch.aperture_summary(result, pipe_radius_m=0.025, n_sigma=1.0)
    assert summary.loc[0, "quantity"] == "limiting momentum spread"
    compact_limits = dch.compact_aperture_limit_table(limits.head(3))
    assert {"Dₓ (m)", "σδ limit", "σδ limit (%)", "status"}.issubset(compact_limits.columns)
    compact_summary = dch.compact_aperture_summary(result, pipe_radius_m=0.025, n_sigma=1.0)
    assert "σδ limit" in set(compact_summary["quantity"])

    with pytest.raises(ValueError, match="positive"):
        dch.aperture_limit_table(result, n_sigma=0.0)


def test_aperture_zero_dispersion_distinguishes_clearance_from_infeasibility():
    result = dch.compute_periodic_optics(dch.make_dba_cell())
    synthetic = replace(
        result,
        table=pd.DataFrame(
            {
                "s_m": [0.0, 1.0],
                "element": ["fits", "does_not_fit"],
                "kind": ["marker", "marker"],
                "beta_x_m": [1.0, 200.0],
                "eta_x_m": [0.0, 0.0],
            }
        ),
    )
    limits = dch.aperture_limit_table(synthetic, pipe_radius_m=0.025, n_sigma=1.0)
    by_element = limits.set_index("element")

    assert by_element.loc["fits", "max_sigma_delta"] == np.inf
    assert by_element.loc["fits", "aperture_status"] == "not dispersion-limited locally"
    assert by_element.loc["does_not_fit", "max_sigma_delta"] == pytest.approx(0.0)
    assert by_element.loc["does_not_fit", "aperture_status"] == "on-momentum envelope reaches/exceeds pipe"


def test_dense_section_b_aperture_values_and_crossing_locations_converge():
    q1 = dch.best_q1_for_achromat()
    cell = dch.make_dba_cell(q1=q1)
    result_100 = dch.compute_periodic_optics(cell, samples_per_meter=100)
    result_200 = dch.compute_periodic_optics(cell, samples_per_meter=200)

    limit_100 = float(dch.aperture_limit_table(result_100, n_sigma=1.0).iloc[0]["max_sigma_delta"])
    limit_200_table = dch.aperture_limit_table(result_200, n_sigma=1.0)
    limit_200 = float(limit_200_table.iloc[0]["max_sigma_delta"])
    assert limit_200 == pytest.approx(0.0272322, abs=2e-7)
    assert limit_100 == pytest.approx(limit_200, abs=5e-7)
    assert float(limit_200_table.iloc[0]["s_m"]) == pytest.approx(4.76, abs=0.01)

    for n_sigma in (3.0, 5.0):
        limiting = dch.aperture_limit_table(result_200, n_sigma=n_sigma).iloc[0]
        assert limiting["max_sigma_delta"] == pytest.approx(0.0)
        assert limiting["aperture_status"] == "on-momentum envelope reaches/exceeds pipe"

    diagnostics = dch.aperture_envelope_diagnostics(
        result_200,
        sigma_delta=1.10 * limit_200,
        n_sigma=1.0,
    ).set_index("condition")
    first = diagnostics.loc["first RMS-envelope crossing"]
    worst = diagnostics.loc["largest RMS envelope"]
    assert first["element"] == "D2"
    assert first["s_m"] == pytest.approx(4.4383, abs=0.001)
    assert worst["element"] == "Q1"
    assert worst["s_m"] == pytest.approx(4.756, abs=0.01)
    assert worst["envelope_mm"] == pytest.approx(27.355, abs=0.01)
    assert first["s_m"] < worst["s_m"]


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
    assert spread.loc[spread["quantity"] == "signed Delta Qx at +sigma_delta=0.002", "value"].iloc[0] == pytest.approx(cx * 0.002)
    assert spread.loc[spread["quantity"] == "RMS sigma_Qx for sigma_delta=0.002", "value"].iloc[0] == pytest.approx(abs(cx) * 0.002)
    compact_spread = dch.compact_chromatic_spread_table(qx_ring, qy_ring, cx, cy, sigma_delta=0.002)
    assert "signed ΔQₓ at +σδ=0.002" in set(compact_spread["quantity"])
    assert "RMS σQₓ for σδ=0.002" in set(compact_spread["quantity"])

    segments = dch.resonance_lines(2, (0.0, 1.0), (0.0, 1.0))
    assert segments
    assert all(segment["order"] <= 2 for segment in segments)
    assert all(math.gcd(abs(segment["m"]), abs(segment["n"])) == 1 for segment in segments)
    assert len({(segment["m"], segment["n"], segment["p"]) for segment in segments}) == len(segments)

    crossings = dch.first_resonance_crossing(0.31, 0.28, -2.0, 1.0, max_order=3)
    assert not crossings.empty
    assert crossings["abs_delta_at_crossing"].is_monotonic_increasing
    compact_crossings = dch.compact_resonance_crossing(crossings.head(2))
    assert list(compact_crossings.columns) == ["resonance", "order", "|δ| at crossing", "|δ| (%)"]

    acceptance = dch.acceptance_comparison(0.01, 0.02)
    assert acceptance.loc[2, "note"] == "dispersion/aperture"
    compact_acceptance = dch.compact_acceptance_comparison(0.01, 0.02)
    assert {"mechanism", "σδ proxy", "σδ proxy (%)"}.issubset(compact_acceptance.columns)


def test_plot_helpers_return_figures_when_show_false():
    result = dch.compute_periodic_optics(dch.make_dba_cell(), samples_per_meter=2)
    scan = dch.scan_q1_for_achromat(qmin=2.0, qmax=2.6, n=7)
    qx, qy = dch.ring_tunes(dch.make_dba_cell(), n_cells=3)
    cx, cy = dch.chromaticity_finite_difference(dch.make_dba_cell(), n_cells=3)

    optics_fig = dch.plot_optics(result, show=False, show_eta_prime=True)
    assert isinstance(optics_fig, go.Figure)
    assert len(optics_fig.layout.shapes) > 0
    _assert_every_data_trace_hovers_its_value(optics_fig)
    optics_without_slope = dch.plot_optics(result, show=False, show_eta_prime=False)
    assert "Dₓ′" not in {trace.name for trace in optics_without_slope.data}
    assert len(dch.plot_optics(result, show=False, show_lattice=False).layout.shapes or []) == 0
    beam_fig = dch.plot_beam_size(result, show=False)
    assert isinstance(beam_fig, go.Figure)
    _assert_every_data_trace_hovers_its_value(beam_fig)
    combined_fig = dch.plot_optics_and_beam_size(result, sigma_delta=0.001, show=False, show_eta_prime=True)
    assert isinstance(combined_fig, go.Figure)
    assert {trace.name for trace in combined_fig.data}.issuperset({"βₓ", "Dₓ", "σₓ total"})
    _assert_every_data_trace_hovers_its_value(combined_fig)
    combined_without_slope = dch.plot_optics_and_beam_size(result, show=False, show_eta_prime=False)
    assert "Dₓ′" not in {trace.name for trace in combined_without_slope.data}
    assert len(dch.plot_optics_and_beam_size(result, show=False, show_lattice=False).layout.shapes or []) == 0
    aperture_fig = dch.plot_beam_size_with_aperture(result, sigma_delta=0.001, show=False)
    assert isinstance(aperture_fig, go.Figure)
    _assert_every_data_trace_hovers_its_value(aperture_fig)
    q1_fig = dch.plot_q1_scan(scan, show=False)
    assert isinstance(q1_fig, go.Figure)
    _assert_every_data_trace_hovers_its_value(q1_fig)
    footprint_fig = dch.plot_tune_footprint(qx, qy, cx, cy, show=False)
    assert isinstance(footprint_fig, go.Figure)
    _assert_every_data_trace_hovers_its_value(footprint_fig)


def test_linked_orbit_fan_uses_uniform_stations_and_left_aligned_slider():
    result = dch.compute_periodic_optics(dch.make_fodo_cell())
    sigma_delta = 0.001
    fig = dch.plot_linked_orbit_fan(
        result,
        sigma_delta=sigma_delta,
        delta_values=[-sigma_delta, 0.0, sigma_delta],
        sample_step_m=0.1,
        show=False,
    )

    stations = np.array([float(step.label) for step in fig.layout.sliders[0].steps])
    assert stations[0] == pytest.approx(0.0)
    assert stations[-1] == pytest.approx(sum(element.length for element in result.elements))
    assert np.diff(stations) == pytest.approx(np.full(len(stations) - 1, 0.1))
    left_domain = fig.layout.xaxis2.domain
    assert fig.layout.sliders[0].x == pytest.approx(left_domain[0])
    assert fig.layout.sliders[0].len == pytest.approx(left_domain[1] - left_domain[0])
    assert len(fig.frames) == len(stations)
    assert fig.layout.coloraxis.cmid == pytest.approx(0.0)
    assert len(fig.layout.shapes) > 0
    _assert_every_data_trace_hovers_its_value(fig)


def test_dispersion_state_portrait_keeps_Dx_and_slope_on_distinct_axes():
    result = dch.compute_transport_optics(dch.make_dba_cell(q1=0.0, q2=0.0, q3=0.0))
    fig = dch.plot_dispersion_state_portrait(result, show=False)

    eta_trace = next(trace for trace in fig.data if trace.xaxis == "x3" and trace.yaxis == "y3")
    etap_trace = next(trace for trace in fig.data if trace.xaxis == "x4" and trace.yaxis == "y4")
    exit_trace = next(trace for trace in fig.data if trace.name == "line exit")
    assert eta_trace.yaxis != etap_trace.yaxis
    assert exit_trace.x[0] == pytest.approx(result.table["eta_x_m"].iloc[-1])
    assert exit_trace.y[0] == pytest.approx(result.table["eta_xp"].iloc[-1])
    assert "Dₓ = %{x:.2f} m" in exit_trace.hovertemplate
    assert "Dₓ′ = %{y:.2f}" in exit_trace.hovertemplate
    assert all(trace.hovertemplate for trace in fig.data)
    assert not any(trace.hoverinfo == "skip" for trace in fig.data)
    _assert_every_data_trace_hovers_its_value(fig)

    reference_fig = dch.plot_dispersion_state_portrait(
        result,
        reference=result,
        show=False,
    )
    assert all(trace.hovertemplate for trace in reference_fig.data)
    assert not any(trace.hoverinfo == "skip" for trace in reference_fig.data)
    _assert_every_data_trace_hovers_its_value(reference_fig)
    assert not any("best" in str(annotation.text).lower() for annotation in fig.layout.annotations)
    assert len(fig.layout.shapes) > 0


def test_first_order_particles_are_deterministic_and_match_launch_covariance():
    result = dch.compute_periodic_optics(dch.make_dba_cell())
    tracks_a = dch.first_order_particle_tracks(result, sigma_delta=0.002, n_particles=240, seed=17)
    tracks_b = dch.first_order_particle_tracks(result, sigma_delta=0.002, n_particles=240, seed=17)

    assert np.array_equal(tracks_a.particle_id, tracks_b.particle_id)
    assert np.array_equal(tracks_a.delta, tracks_b.delta)
    assert np.array_equal(tracks_a.x_m, tracks_b.x_m)
    assert np.all(np.diff(tracks_a.s_m) > 0)
    assert np.var(tracks_a.delta) == pytest.approx(0.002**2, rel=1e-12)
    expected_variance = (
        dch.GEOMETRIC_EMITTANCE * result.initial["beta_x_m"]
        + (result.initial["eta_x_m"] * 0.002) ** 2
    )
    assert np.var(tracks_a.x_m[:, 0]) == pytest.approx(expected_variance, rel=1e-12)


def test_aperture_ribbon_clearance_matches_beam_size_decomposition():
    result = dch.compute_periodic_optics(dch.make_dba_cell())
    sigma_delta = 0.001
    radius = 0.025
    n_sigma = 1.5
    tracks = dch.first_order_particle_tracks(result, sigma_delta=sigma_delta, n_particles=32, seed=4)
    fig = dch.plot_aperture_ribbon(
        result,
        sigma_delta=sigma_delta,
        pipe_radius_m=radius,
        n_sigma=n_sigma,
        tracks=tracks,
        show=False,
    )

    sampled = dch._sample_horizontal_optics_at_s(result, tracks.s_m)
    sigma_total_mm = 1e3 * np.sqrt(
        dch.GEOMETRIC_EMITTANCE * sampled["beta_x_m"].to_numpy()
        + (sampled["eta_x_m"].to_numpy() * sigma_delta) ** 2
    )
    clearance = next(trace for trace in fig.data if trace.name == "clearance")
    assert np.asarray(clearance.y) == pytest.approx(1e3 * radius - n_sigma * sigma_total_mm)
    assert sum(trace.name == "pipe wall" for trace in fig.data) == 2
    assert fig.layout.coloraxis.cmid == pytest.approx(0.0)
    assert len(fig.layout.shapes) > 0
    _assert_every_data_trace_hovers_its_value(fig)


def test_section_c_uses_direct_xsuite_twiss_and_tune_scan(section_c_xsuite_data):
    cell, ring, twiss, scan = section_c_xsuite_data

    direct = ring.twiss(method="4d", chrom=True, delta_chrom=1e-4)
    assert twiss.qx == pytest.approx(direct.qx)
    assert twiss.qy == pytest.approx(direct.qy)
    assert twiss.dqx == pytest.approx(direct.dqx)
    assert twiss.dqy == pytest.approx(direct.dqy)
    assert twiss.qx == pytest.approx(5.934679, abs=2e-6)
    assert twiss.qy == pytest.approx(2.599888, abs=2e-6)
    assert twiss.dqx == pytest.approx(-9.60361, abs=2e-4)
    assert twiss.dqy == pytest.approx(-15.03240, abs=2e-4)

    for row in scan.itertuples(index=False):
        expected = ring.twiss(method="4d", delta0=row.delta, chrom=False)
        assert row.Qx == pytest.approx(expected.qx)
        assert row.Qy == pytest.approx(expected.qy)

    crossings = dch.first_resonance_crossing_from_tune_scan(scan, max_order=3)
    first = crossings.iloc[0]
    assert (first["m"], first["n"], first["p"]) == (1, 2, 11)
    assert first["delta_at_crossing"] == pytest.approx(0.003408, abs=2e-6)
    assert first["resonance_residual"] == pytest.approx(0.0, abs=1e-12)

    fig = dch.plot_tune_scan_and_footprint(
        scan,
        sigma_delta=0.001,
        delta_range=(-0.001, 0.001),
        n_delta=11,
        show=False,
    )
    footprint = next(trace for trace in fig.data if trace.name == "chromatic footprint")
    assert footprint.marker.coloraxis == "coloraxis"
    assert fig.layout.coloraxis.cmid == pytest.approx(0.0)
    assert any(str(trace.name).startswith("order ") for trace in fig.data)
    assert not any("cross" in str(annotation.text).lower() for annotation in fig.layout.annotations)
    _assert_every_data_trace_hovers_its_value(fig)

    interactive_fig = dch.plot_tune_footprint_from_scan(
        scan,
        sigma_delta=0.001,
        resonance_order=3,
        show=False,
    )
    _assert_every_data_trace_hovers_its_value(interactive_fig)

    phase_fig = dch.plot_xsuite_accumulated_phase_advance(twiss, cell, show=False)
    phase_x = next(trace for trace in phase_fig.data if trace.name == "ψₓ / 2π")
    phase_y = next(trace for trace in phase_fig.data if trace.name == "ψᵧ / 2π")
    assert phase_x.y[-1] == pytest.approx(twiss.qx / 10.0)
    assert phase_y.y[-1] == pytest.approx(twiss.qy / 10.0)
    _assert_every_data_trace_hovers_its_value(phase_fig)


def test_scan_based_resonance_crossing_retains_nonlinear_tune_path():
    delta = np.linspace(-0.1, 0.1, 81)
    curved_scan = pd.DataFrame(
        {
            "delta": delta,
            "Qx": 0.31 - 2.0 * delta + 3.0 * delta**2,
            "Qy": 0.28 + delta,
            "stable": True,
        }
    )
    crossings = dch.first_resonance_crossing_from_tune_scan(curved_scan, max_order=3)

    assert not crossings.empty
    assert crossings["abs_delta_at_crossing"].is_monotonic_increasing
    first = crossings.iloc[0]
    tangent_crossing = dch.first_resonance_crossing(0.31, 0.28, -2.0, 1.0, max_order=3).iloc[0]
    assert first["abs_delta_at_crossing"] != pytest.approx(tangent_crossing["abs_delta_at_crossing"], abs=1e-5)


def test_phase_advance_table_starts_at_zero_and_recovers_cell_tunes():
    result = dch.compute_periodic_optics(dch.make_dba_cell())
    phase = dch.phase_advance_table(result)
    assert phase.loc[0, "psi_x_turns"] == pytest.approx(0.0)
    assert phase.loc[0, "psi_y_turns"] == pytest.approx(0.0)
    assert np.all(np.diff(phase["psi_x_turns"]) >= 0)
    assert np.all(np.diff(phase["psi_y_turns"]) >= 0)
    assert phase["psi_x_turns"].iloc[-1] == pytest.approx(result.tune_x, abs=1e-4)
    assert phase["psi_y_turns"].iloc[-1] == pytest.approx(result.tune_y, abs=1e-4)

    fig = dch.plot_accumulated_phase_advance(result, show=False)
    assert {trace.name for trace in fig.data}.issuperset(
        {"ψₓ / 2π", "ψᵧ / 2π", "horizontal rate", "vertical rate"}
    )
    assert len(fig.layout.shapes) > 0
    _assert_every_data_trace_hovers_its_value(fig)
