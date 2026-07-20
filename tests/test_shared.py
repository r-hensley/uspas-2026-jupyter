from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from uspas_labs import shared


def test_dependency_table_reports_available_and_missing_packages():
    table = shared.dependency_table(["numpy", "definitely_not_a_real_uspas_package"])

    by_package = table.set_index("package")
    assert by_package.loc["numpy", "status"] == "available"
    assert by_package.loc["definitely_not_a_real_uspas_package", "status"].startswith("missing:")


def test_should_show_plot_respects_global_and_specific_suppression(monkeypatch):
    monkeypatch.delenv("USPAS_LABS_SUPPRESS_PLOTS", raising=False)
    monkeypatch.delenv("LOCAL_SUPPRESS", raising=False)
    assert shared.should_show_plot("LOCAL_SUPPRESS") is True

    monkeypatch.setenv("LOCAL_SUPPRESS", "1")
    assert shared.should_show_plot("LOCAL_SUPPRESS") is False

    monkeypatch.setenv("LOCAL_SUPPRESS", "0")
    monkeypatch.setenv("USPAS_LABS_SUPPRESS_PLOTS", "yes")
    assert shared.should_show_plot("LOCAL_SUPPRESS") is False


def test_show_or_return_returns_figure_when_show_is_false(monkeypatch):
    fig = go.Figure()
    called = {"show": False}

    def fake_show():
        called["show"] = True

    monkeypatch.setattr(fig, "show", fake_show)

    assert shared.show_or_return(fig, show=False) is fig
    assert called["show"] is False


def test_show_or_return_shows_once_and_returns_none(monkeypatch):
    fig = go.Figure()
    called = {"count": 0}

    def fake_show():
        called["count"] += 1

    monkeypatch.setattr(fig, "show", fake_show)

    assert shared.show_or_return(fig, show=True) is None
    assert called["count"] == 1


def test_shared_fodo_geometry_uses_split_focusing_quads_and_two_bends():
    no_bend = shared.fodo_cell_segments(with_bends=False)
    assert [(segment.name, segment.kind, segment.length) for segment in no_bend] == [
        ("QFa", "quad", 0.25),
        ("D1", "drift", 2.0),
        ("QD", "quad", 0.5),
        ("D2", "drift", 2.0),
        ("QFb", "quad", 0.25),
    ]

    with_bends = shared.fodo_cell_segments(with_bends=True)
    assert [segment.name for segment in with_bends] == ["QFa", "D1a", "B1", "D1b", "QD", "D2a", "B2", "D2b", "QFb"]
    assert sum(segment.length for segment in with_bends) == shared.FODO_CELL_LENGTH
    assert [segment.bend_angle_fraction for segment in with_bends if segment.kind == "bend"] == [0.5, 0.5]


def test_add_lattice_strip_draws_element_shapes_and_labels():
    layout = pd.DataFrame(
        [
            {"name": "D1", "kind": "drift", "s_start_m": 0.0, "s_end_m": 1.0, "k1_m^-2": float("nan")},
            {"name": "QF", "kind": "quad", "s_start_m": 1.0, "s_end_m": 1.5, "k1_m^-2": 0.6},
            {"name": "BEND", "kind": "bend", "s_start_m": 1.5, "s_end_m": 2.0, "k1_m^-2": float("nan")},
            {"name": "QD", "kind": "quad", "s_start_m": 2.0, "s_end_m": 2.5, "k1_m^-2": -0.6},
        ]
    )
    fig = go.Figure()
    fig.add_vline(x=1.25, annotation_text="transition")

    returned = shared.add_lattice_strip(fig, layout)

    assert returned is fig
    assert len(fig.layout.shapes) >= len(layout) + 2
    assert {annotation.text for annotation in fig.layout.annotations} == {"transition", "QF", "BEND", "QD"}
    assert any(shape.x0 == 1.25 and shape.x1 == 1.25 for shape in fig.layout.shapes)
    assert fig.layout.margin.t >= 105
