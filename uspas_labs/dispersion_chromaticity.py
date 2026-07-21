"""Local helper functions for the Dispersion and Chromaticity lab.

The notebook keeps most matrix optics, plotting, and widget code here so the
student-facing cells can focus on a small number of physics knobs.  The local
model is intentionally lightweight: it uses first-order transfer matrices for
drifts, thick quadrupoles, and sector bends, plus an optional thin-edge focusing
model.  It is used for the dispersion and achromat exercises.  Ring tunes,
chromaticity, and off-momentum tune footprints are calculated directly with
Xsuite; the local model is not their authoritative backend.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping, Sequence
import contextlib
import io
import math
import warnings

import numpy as np
import pandas as pd

from .shared import (
    add_lattice_strip,
    dependency_table,
    display_widget_slider_css,
    fodo_cell_segments,
    lab_float_slider,
    lab_int_slider,
    maybe_display as _maybe_display,
    show_or_return as _show_or_return,
    widget_container_layout,
)

import xtrack as xt

import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

GEOMETRIC_EMITTANCE = 6e-6        # 6 mm mrad = 6e-6 m rad
SIGMA_DELTA_DEFAULT = 1e-3        # 0.1% fractional momentum spread
PIPE_RADIUS_DEFAULT = 0.025       # 2.5 cm
DBA_BEND_ANGLE_DEG = 18.0
DBA_Q1_DEFAULT = 2.3356332610219  # strength that closes the achromatic boundaries in this model
DBA_Q2_DEFAULT = 2.475            # flanking quad strength used for the stable DBA cell
DBA_Q3_DEFAULT = -2.15            # flanking quad strength used for the stable DBA cell
DBA_N_CELLS_DEFAULT = 10
ETA_CLOSURE_SCALE_M = 1.0         # explicit normalization for the Q1 search merit
ETAP_CLOSURE_SCALE = 1.0          # eta_x' is dimensionless in the paraxial model
DIAGNOSTIC_SAMPLES_PER_METER = 200.0
DELTA_COLORSCALE = "RdBu_r"       # negative momentum blue, positive momentum red


@dataclass
class Element:
    """One first-order lattice element.

    Parameters
    ----------
    name:
        Element name used in tables and hover labels.
    kind:
        One of ``"drift"``, ``"quad"``, or ``"bend"``.
    length:
        Element length in meters.
    k1:
        Quadrupole strength in m^-2. Positive focuses horizontally.
    angle:
        Bend angle in radians. Only used for bends.
    edge_angle:
        Optional pole-face edge angle in radians. The default core exercises use
        zero edge angle to isolate dispersion from edge focusing.
    role:
        Short human-readable description.
    """

    name: str
    kind: str
    length: float
    k1: float = 0.0
    angle: float = 0.0
    edge_angle: float = 0.0
    role: str = ""

    @property
    def edge_entry_angle(self) -> float:
        return self.edge_angle

    @edge_entry_angle.setter
    def edge_entry_angle(self, value: float) -> None:
        self.edge_angle = float(value)

    @property
    def edge_exit_angle(self) -> float:
        return self.edge_angle

    @edge_exit_angle.setter
    def edge_exit_angle(self, value: float) -> None:
        self.edge_angle = float(value)


class Lattice(list):
    """Small named-element list used by the local optics model."""

    def __getitem__(self, key):
        if isinstance(key, str):
            for element in self:
                if element.name == key:
                    return element
            raise KeyError(key)
        return super().__getitem__(key)

    def copy(self):
        return Lattice(replace(element) for element in self)


@dataclass
class OpticsResult:
    """Computed optics sampled along a lattice."""

    elements: list[Element]
    table: pd.DataFrame
    layout: pd.DataFrame
    matrix_x: np.ndarray
    matrix_y: np.ndarray
    dispersion_source: np.ndarray
    stable_x: bool
    stable_y: bool
    tune_x: float | None
    tune_y: float | None
    initial: dict
    periodic: bool
    delta: float = 0.0


@dataclass(frozen=True)
class FirstOrderParticleTracks:
    """Fixed particles propagated through the lab's first-order horizontal model."""

    s_m: np.ndarray
    x_m: np.ndarray
    xp: np.ndarray
    delta: np.ndarray
    particle_id: np.ndarray


# ---------------------------------------------------------------------------
# Environment and small utilities
# ---------------------------------------------------------------------------


def check_environment() -> pd.DataFrame:
    """Return a compact dependency table for the notebook setup cell."""
    return dependency_table(["numpy", "pandas", "plotly", "ipywidgets", "xtrack"])


def _format_element_label(element: Element) -> str:
    if element.kind == "quad":
        return f"{element.name}: quad, k1={element.k1:.4g} m^-2"
    if element.kind == "bend":
        return f"{element.name}: bend, angle={math.degrees(element.angle):.4g} deg"
    return f"{element.name}: drift"


# ---------------------------------------------------------------------------
# First-order element maps
# ---------------------------------------------------------------------------


def drift_matrix(length: float) -> np.ndarray:
    return np.array([[1.0, float(length)], [0.0, 1.0]])


def quad_matrix(length: float, k1: float) -> np.ndarray:
    """Thick-lens quadrupole matrix for one transverse plane.

    Positive ``k1`` is focusing in the plane being represented; negative is
    defocusing.  The opposite plane is obtained by calling this with ``-k1``.
    """
    length = float(length)
    k1 = float(k1)
    if abs(k1) < 1e-14 or abs(length) < 1e-14:
        return drift_matrix(length)

    if k1 > 0:
        root_k = math.sqrt(k1)
        phase = root_k * length
        c = math.cos(phase)
        s = math.sin(phase)
        return np.array([[c, s / root_k], [-root_k * s, c]])

    root_k = math.sqrt(-k1)
    phase = root_k * length
    c = math.cosh(phase)
    s = math.sinh(phase)
    return np.array([[c, s / root_k], [root_k * s, c]])


def sector_bend_matrix(length: float, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal matrix and dispersion source for a sector bend.

    The dispersion vector is the particular solution added to ``(eta, eta')``
    for unit fractional momentum offset in the usual linearized bend model.
    """
    length = float(length)
    angle = float(angle)
    if abs(angle) < 1e-14 or abs(length) < 1e-14:
        return drift_matrix(length), np.zeros(2)

    h = angle / length
    c = math.cos(angle)
    s = math.sin(angle)
    matrix = np.array([[c, s / h], [-h * s, c]])
    source = np.array([(1.0 - c) / h, s])
    return matrix, source


def edge_matrix(curvature: float, edge_angle: float, plane: str) -> np.ndarray:
    """Thin pole-face edge-focusing matrix.

    The default lab exercises set ``edge_angle=0``.  Nonzero edge angles are
    provided as an optional exploration of why dipoles can affect beta functions
    even for on-momentum particles.
    """
    kick = curvature * math.tan(edge_angle)
    if plane == "x":
        return np.array([[1.0, 0.0], [kick, 1.0]])
    if plane == "y":
        return np.array([[1.0, 0.0], [-kick, 1.0]])
    raise ValueError("plane must be 'x' or 'y'")


def element_maps(element: Element, delta: float = 0.0, include_dispersion_source: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``Mx``, ``My``, and the horizontal dispersion source for an element.

    ``delta`` is used for chromatic finite differences: fixed magnets are modeled
    as having effective strengths scaled by ``1/(1+delta)``.  For the on-energy
    optics and dispersion calculations, use ``delta=0``.
    """
    if delta <= -0.99:
        raise ValueError("delta must be greater than -0.99")
    scale = 1.0 / (1.0 + float(delta))

    if element.kind == "drift":
        matrix = drift_matrix(element.length)
        return matrix, matrix, np.zeros(2)

    if element.kind == "quad":
        k_eff = element.k1 * scale
        return quad_matrix(element.length, k_eff), quad_matrix(element.length, -k_eff), np.zeros(2)

    if element.kind == "bend":
        angle_eff = element.angle * scale
        matrix_x, source = sector_bend_matrix(element.length, angle_eff)
        matrix_y = drift_matrix(element.length)

        if abs(element.edge_angle) > 0 and abs(element.length) > 0:
            curvature = angle_eff / element.length
            edge_x = edge_matrix(curvature, element.edge_angle, "x")
            edge_y = edge_matrix(curvature, element.edge_angle, "y")
            matrix_x = edge_x @ matrix_x @ edge_x
            matrix_y = edge_y @ matrix_y @ edge_y
            source = edge_x @ source

        if not include_dispersion_source:
            source = np.zeros(2)
        return matrix_x, matrix_y, source

    raise ValueError(f"Unknown element kind: {element.kind!r}")


def split_element(element: Element, n_slices: int) -> list[Element]:
    """Split an element for plotting/sampling.

    Bend slices represent only the finite-length bend body.  Thin edge kicks are
    applied separately by ``compute_optics`` so the sampled propagation matches
    the full-element map without repeating edge focusing inside the magnet.
    """
    n_slices = max(1, int(n_slices))
    if element.kind == "bend":
        return [
            replace(
                element,
                length=element.length / n_slices,
                angle=element.angle / n_slices,
                edge_angle=0.0,
            )
            for _ in range(n_slices)
        ]
    return [replace(element, length=element.length / n_slices) for _ in range(n_slices)]


def _bend_edge_maps(element: Element, delta: float = 0.0) -> tuple[np.ndarray, np.ndarray] | None:
    if element.kind != "bend" or abs(element.edge_angle) <= 0 or abs(element.length) <= 0:
        return None

    scale = 1.0 / (1.0 + float(delta))
    curvature = element.angle * scale / element.length
    return edge_matrix(curvature, element.edge_angle, "x"), edge_matrix(curvature, element.edge_angle, "y")


def transfer_map(elements: Sequence[Element], delta: float = 0.0, include_dispersion_source: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return total horizontal/vertical maps and total dispersion source."""
    matrix_x = np.eye(2)
    matrix_y = np.eye(2)
    source = np.zeros(2)
    for element in elements:
        mx, my, b = element_maps(element, delta=delta, include_dispersion_source=include_dispersion_source)
        source = mx @ source + b
        matrix_x = mx @ matrix_x
        matrix_y = my @ matrix_y
    return matrix_x, matrix_y, source


def is_stable(matrix: np.ndarray, tolerance: float = 1e-12) -> bool:
    return abs(0.5 * np.trace(matrix)) < 1.0 - tolerance


def matched_twiss_from_matrix(matrix: np.ndarray) -> tuple[float, float, float] | None:
    """Return ``beta``, ``alpha``, and fractional tune for a stable 2x2 matrix."""
    cos_mu = 0.5 * float(np.trace(matrix))
    if abs(cos_mu) >= 1.0:
        return None

    cos_mu = max(-1.0, min(1.0, cos_mu))
    mu = math.acos(cos_mu)
    sin_mu = math.sin(mu)
    beta = matrix[0, 1] / sin_mu
    if beta <= 0:
        mu = 2.0 * math.pi - mu
        sin_mu = math.sin(mu)
        beta = matrix[0, 1] / sin_mu
    alpha = (matrix[0, 0] - matrix[1, 1]) / (2.0 * sin_mu)
    tune = mu / (2.0 * math.pi)
    return float(beta), float(alpha), float(tune)


def propagate_twiss(beta: float, alpha: float, matrix: np.ndarray) -> tuple[float, float]:
    gamma = (1.0 + alpha**2) / beta
    m11, m12, m21, m22 = matrix.ravel()
    beta_new = m11**2 * beta - 2.0 * m11 * m12 * alpha + m12**2 * gamma
    alpha_new = -m11 * m21 * beta + (m11 * m22 + m12 * m21) * alpha - m12 * m22 * gamma
    return float(beta_new), float(alpha_new)


# ---------------------------------------------------------------------------
# Lattice construction
# ---------------------------------------------------------------------------


def make_fodo_cell(kq: float = 0.6, bend_angle_deg: float = 20.0, with_bend: bool = True, edge_focusing: bool = False) -> Lattice:
    """Return the five-meter FODO cell used in Section A.

    The cell starts and ends at the center of a focusing quadrupole.  When
    ``with_bend`` is true, two equal dipoles are placed at the centers of the
    two drifts; ``bend_angle_deg`` is the total bend angle of the cell.
    """
    total_bend_angle = math.radians(float(bend_angle_deg))
    elements = Lattice()
    for segment in fodo_cell_segments(with_bends=with_bend):
        if segment.kind == "drift":
            elements.append(Element(segment.name, "drift", segment.length, role=segment.role))
        elif segment.kind == "quad":
            elements.append(Element(segment.name, "quad", segment.length, k1=segment.k1_sign * float(kq), role=segment.role))
        elif segment.kind == "bend":
            angle = segment.bend_angle_fraction * total_bend_angle
            edge_angle = 0.5 * angle if edge_focusing else 0.0
            bend_degrees = math.degrees(angle)
            elements.append(
                Element(
                    segment.name,
                    "bend",
                    segment.length,
                    angle=angle,
                    edge_angle=edge_angle,
                    role=f"{bend_degrees:.4g}-degree {segment.role}",
                )
            )
        else:
            raise ValueError(f"Unsupported FODO segment kind {segment.kind!r}")
    return elements


def make_dba_cell(
    q1: float = DBA_Q1_DEFAULT,
    q2: float = DBA_Q2_DEFAULT,
    q3: float = DBA_Q3_DEFAULT,
    bend_angle_deg: float = DBA_BEND_ANGLE_DEG,
    edge_focusing: bool = False,
) -> Lattice:
    """Return a compact local double-bend-achromat cell.

    Q1 is inside the two-bend insert and controls endpoint dispersion.  Q2 and
    Q3 are placed outside the insert; when the insert endpoint dispersion is
    zero, changing Q2/Q3 alters periodic focusing without spoiling achromaticity.
    """
    angle = math.radians(bend_angle_deg)
    edge_angle = 0.5 * angle if edge_focusing else 0.0
    return Lattice([
        Element("D0", "drift", 0.5),
        Element("Q2", "quad", 0.3, k1=q2, role="upstream flanking quadrupole"),
        Element("D1", "drift", 0.5),
        Element("B1", "bend", 1.0, angle=angle, edge_angle=edge_angle, role="first DBA bend"),
        Element("D2", "drift", 2.3),
        Element("Q1", "quad", 0.3, k1=q1, role="central dispersion-control quadrupole"),
        Element("D3", "drift", 2.3),
        Element("B2", "bend", 1.0, angle=angle, edge_angle=edge_angle, role="second DBA bend"),
        Element("D4", "drift", 0.5),
        Element("Q3", "quad", 0.3, k1=q3, role="downstream flanking quadrupole"),
        Element("D5", "drift", 0.5),
    ])


def repeat_cell(elements: Sequence[Element], n_cells: int = DBA_N_CELLS_DEFAULT) -> Lattice:
    """Repeat a cell and suffix element names with cell numbers."""
    repeated = Lattice()
    for i_cell in range(1, int(n_cells) + 1):
        for element in elements:
            repeated.append(replace(element, name=f"{element.name}_c{i_cell}"))
    return repeated


def xsuite_line_from_lattice(
    elements: Sequence[Element],
    *,
    particle_ref=None,
):
    """Build an Xsuite line from the lab's named drift/quad/sector-bend list.

    The bend field and reference curvature are specified by the fixed design
    values ``k0=angle/length`` and ``angle``.  Off-momentum optics are left to
    Xsuite; this adapter never rescales a bend angle by hand.
    """
    xsuite_elements = []
    element_names = []
    for element in elements:
        element_names.append(element.name)
        if element.kind == "drift":
            xsuite_elements.append(xt.Drift(length=element.length))
        elif element.kind == "quad":
            xsuite_elements.append(xt.Quadrupole(length=element.length, k1=element.k1))
        elif element.kind == "bend":
            if element.length <= 0:
                raise ValueError(f"Bend {element.name!r} must have positive length")
            xsuite_elements.append(
                xt.Bend(
                    length=element.length,
                    angle=element.angle,
                    k0=element.angle / element.length,
                    edge_entry_angle=element.edge_angle,
                    edge_exit_angle=element.edge_angle,
                )
            )
        else:
            raise ValueError(f"Unsupported Xsuite element kind: {element.kind!r}")

    line = xt.Line(elements=xsuite_elements, element_names=element_names)
    if particle_ref is None:
        particle_ref = xt.Particles(p0c=1e9, mass0=xt.ELECTRON_MASS_EV, q0=-1)
    line.particle_ref = particle_ref.copy()
    return line


def xsuite_ring_from_lattice(
    cell_elements: Sequence[Element],
    *,
    n_cells: int = DBA_N_CELLS_DEFAULT,
    particle_ref=None,
):
    """Repeat a demonstrated cell and return the corresponding Xsuite ring."""
    if int(n_cells) != n_cells or int(n_cells) <= 0:
        raise ValueError("n_cells must be a positive integer")
    return xsuite_line_from_lattice(
        repeat_cell(cell_elements, n_cells=int(n_cells)),
        particle_ref=particle_ref,
    )


def elements_from_xsuite_line(
    line,
    roles: Mapping[str, str] | None = None,
    occurrence_names: Sequence[str] | None = None,
) -> Lattice:
    """Convert a simple Xsuite line into the local first-order model.

    The dispersion/chromaticity lab keeps its transport model local, but the
    notebook can still use modern Xsuite ``Environment``/``Line`` syntax for
    front-facing lattice construction. This adapter is intentionally narrow:
    it supports drifts, quadrupoles, and sector-like bends.
    """
    roles = dict(roles or {})
    table = line.get_table().to_pandas()
    table = table[table["name"] != "_end_point"].reset_index(drop=True)
    component_names = [name for name in line.element_names if name != "_end_point"]
    if len(table) != len(component_names):
        raise ValueError("Xsuite line table and element list have inconsistent lengths")
    if occurrence_names is not None and len(occurrence_names) != len(component_names):
        raise ValueError("occurrence_names must match the number of non-endpoint line elements")
    elements = Lattice()

    for i_item, component_name in enumerate(component_names):
        row = table.iloc[i_item]
        table_name = str(row["name"])
        name = occurrence_names[i_item] if occurrence_names is not None else table_name
        element = line[component_name]
        element_type = str(row["element_type"])
        raw_length = getattr(element, "length", None)
        if raw_length is None:
            s_start = row["s_start"] if "s_start" in row.index else row["s"]
            raw_length = row["s_end"] - s_start
        length = float(raw_length)
        role = roles.get(name, roles.get(table_name, roles.get(component_name, "")))

        if element_type == "Drift":
            elements.append(Element(name, "drift", length, role=role))
        elif element_type == "Quadrupole":
            elements.append(Element(name, "quad", length, k1=float(element.k1), role=role))
        elif element_type in {"Bend", "RBend"}:
            edge_angle = 0.5 * (float(getattr(element, "edge_entry_angle", 0.0)) + float(getattr(element, "edge_exit_angle", 0.0)))
            elements.append(
                Element(
                    name,
                    "bend",
                    length,
                    angle=float(element.angle),
                    edge_angle=edge_angle,
                    role=role,
                )
            )
        else:
            raise ValueError(f"Unsupported Xsuite element type {element_type!r} for element {name!r}")

    return elements


def element_layout(elements: Sequence[Element]) -> pd.DataFrame:
    rows = []
    s = 0.0
    for element in elements:
        start = s
        stop = s + element.length
        rows.append(
            {
                "name": element.name,
                "kind": element.kind,
                "s_start_m": start,
                "s_end_m": stop,
                "length_m": element.length,
                "k1_m^-2": element.k1 if element.kind == "quad" else np.nan,
                "angle_deg": math.degrees(element.angle) if element.kind == "bend" else np.nan,
                "edge_angle_deg": math.degrees(element.edge_angle) if element.kind == "bend" else np.nan,
                "role": element.role,
            }
        )
        s = stop
    return pd.DataFrame(rows)


DISPLAY_COLUMN_LABELS = {
    "name": "element",
    "element": "element",
    "kind": "type",
    "element_type": "type",
    "s_start": "s start (m)",
    "s_end": "s end (m)",
    "s_start_m": "s start (m)",
    "s_end_m": "s end (m)",
    "s_center_m": "s (m)",
    "s_m": "s (m)",
    "length_m": "L (m)",
    "k1_m^-2": "k₁ (m⁻²)",
    "angle_deg": "θ (deg)",
    "edge_angle_deg": "edge θ (deg)",
    "beta_x_m": "βₓ (m)",
    "alpha_x": "αₓ",
    "beta_y_m": "βᵧ (m)",
    "alpha_y": "αᵧ",
    "eta_x_m": "Dₓ (m)",
    "eta_xp": "Dₓ′ (1)",
    "sigma_x_beta_mm": "σₓ betatron (mm)",
    "sigma_x_dispersion_mm": "σₓ dispersion (mm)",
    "sigma_x_mm": "σₓ (mm)",
    "sigma_y_mm": "σᵧ (mm)",
    "plane": "plane",
    "trace(M)/2": "trace(M)/2",
    "stable?": "stable",
    "cell tune if stable": "tune",
    "matched beta at start [m]": "β start (m)",
    "q1_m^-2": "Q1 k₁ (m⁻²)",
    "eta_x_end_m": "Dₓ end (m)",
    "eta_xp_end": "Dₓ′ end (1)",
    "closure_merit": "dimensionless closure merit",
    "pipe_radius_mm": "pipe radius (mm)",
    "n_sigma": "nσ",
    "on_momentum_envelope_mm": "on-momentum nσ envelope (mm)",
    "on_momentum_clearance_mm": "on-momentum clearance (mm)",
    "aperture_status": "status",
    "envelope_mm": "nσ envelope (mm)",
    "clearance_mm": "clearance (mm)",
    "location_method": "location method",
    "max_sigma_delta": "σδ limit",
    "max_delta_percent": "σδ limit (%)",
    "delta_at_crossing": "δ at crossing",
    "abs_delta_at_crossing": "|δ| at crossing",
    "delta_percent": "|δ| (%)",
    "limitation mechanism": "mechanism",
    "momentum spread limit": "σδ proxy",
    "limit [%]": "σδ proxy (%)",
}


DISPLAY_QUANTITY_LABELS = {
    "length [m]": "length (m)",
    "stable x?": "stable x",
    "stable y?": "stable y",
    "cell tune Qₓ": "cell tune Qₓ",
    "cell tune Qᵧ": "cell tune Qᵧ",
    "βₓ [m]": "βₓ (m)",
    "βᵧ [m]": "βᵧ (m)",
    "ηₓ [m]": "Dₓ (m)",
    "Dₓ [m]": "Dₓ (m)",
    "eta_x at end [m]": "Dₓ end (m)",
    "eta_x' at end": "Dₓ′ end (1)",
    "D_x at end [m]": "Dₓ end (m)",
    "D_x' at end": "Dₓ′ end (1)",
    "dimensionless closure merit": "dimensionless closure merit",
    "limiting momentum spread": "σδ limit",
    "limiting momentum spread [%]": "σδ limit (%)",
    "limiting status": "status",
    "limiting element": "limiting element",
    "limiting s [m]": "limiting s (m)",
    "eta_x at limit [m]": "Dₓ at limit (m)",
    "D_x at limit [m]": "Dₓ at limit (m)",
    "beta_x at limit [m]": "βₓ at limit (m)",
    "number of cells": "number of cells",
    "Qx ring tune": "ring tune Qₓ",
    "Qy ring tune": "ring tune Qᵧ",
    "Cx = dQx/d(delta)": "Cₓ = dQₓ/dδ",
    "Cy = dQy/d(delta)": "Cᵧ = dQᵧ/dδ",
    "max eta_x in one cell [m]": "max Dₓ in one cell (m)",
    "min eta_x in one cell [m]": "min Dₓ in one cell (m)",
    "max D_x in one cell [m]": "max Dₓ in one cell (m)",
    "min D_x in one cell [m]": "min Dₓ in one cell (m)",
    "Qx": "Qₓ",
    "Qy": "Qᵧ",
    "Cx": "Cₓ",
    "Cy": "Cᵧ",
    "xi_x = dQx/d(delta)": "ξₓ = dQₓ/dδ",
    "xi_y = dQy/d(delta)": "ξᵧ = dQᵧ/dδ",
}


def _display_quantity_label(quantity: object) -> object:
    if not isinstance(quantity, str):
        return quantity
    if quantity in DISPLAY_QUANTITY_LABELS:
        return DISPLAY_QUANTITY_LABELS[quantity]
    if quantity.startswith("Delta Qx for sigma_delta="):
        delta = quantity.removeprefix("Delta Qx for sigma_delta=")
        return f"ΔQₓ for σδ={delta}"
    if quantity.startswith("Delta Qy for sigma_delta="):
        delta = quantity.removeprefix("Delta Qy for sigma_delta=")
        return f"ΔQᵧ for σδ={delta}"
    if quantity.startswith("signed Delta Qx at +sigma_delta="):
        delta = quantity.removeprefix("signed Delta Qx at +sigma_delta=")
        return f"signed ΔQₓ at +σδ={delta}"
    if quantity.startswith("signed Delta Qy at +sigma_delta="):
        delta = quantity.removeprefix("signed Delta Qy at +sigma_delta=")
        return f"signed ΔQᵧ at +σδ={delta}"
    if quantity.startswith("RMS sigma_Qx for sigma_delta="):
        delta = quantity.removeprefix("RMS sigma_Qx for sigma_delta=")
        return f"RMS σQₓ for σδ={delta}"
    if quantity.startswith("RMS sigma_Qy for sigma_delta="):
        delta = quantity.removeprefix("RMS sigma_Qy for sigma_delta=")
        return f"RMS σQᵧ for σδ={delta}"
    return quantity


def _drop_empty_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for column in df.columns:
        values = df[column]
        if values.isna().all():
            continue
        if values.map(lambda value: isinstance(value, str) and value.strip() == "").all():
            continue
        keep.append(column)
    return df.loc[:, keep]


def compact_table(
    table: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    drop_all_nan: bool = True,
    precision: int = 6,
    labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return a notebook-friendly display table.

    This is intentionally a display-layer helper: calculation helpers keep their
    machine-readable column names, while notebook cells can show concise labels
    and avoid all-NaN columns.
    """
    out = table.copy()
    if columns is not None:
        out = out[[column for column in columns if column in out.columns]]
    if drop_all_nan:
        out = _drop_empty_display_columns(out)

    for column in out.columns:
        non_missing = out[column].dropna()
        if not non_missing.empty and non_missing.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            out[column] = out[column].map(lambda value: "yes" if value else "no")
        elif pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].where(~np.isclose(out[column], 0.0, atol=1e-14), 0.0).round(precision)

    column_labels = dict(DISPLAY_COLUMN_LABELS)
    if labels:
        column_labels.update(labels)
    out = out.rename(columns=column_labels)
    return out.astype(object).where(pd.notna(out), "")


def compact_summary_table(table: pd.DataFrame, *, precision: int = 6) -> pd.DataFrame:
    """Compact a simple quantity/value-style summary table."""
    out = table.copy()
    if "quantity" in out.columns:
        out["quantity"] = out["quantity"].map(_display_quantity_label)
    return compact_table(out, precision=precision)


def compact_lattice_table(elements: Sequence[Element] | pd.DataFrame, *, include_role: bool = False) -> pd.DataFrame:
    layout = elements.copy() if isinstance(elements, pd.DataFrame) else element_layout(elements)
    columns = ["name", "kind", "s_start_m", "s_end_m", "length_m", "k1_m^-2", "angle_deg", "edge_angle_deg"]
    edge_angles = layout["edge_angle_deg"].dropna() if "edge_angle_deg" in layout.columns else pd.Series(dtype=float)
    if edge_angles.empty or np.allclose(edge_angles, 0.0, atol=1e-14):
        columns.remove("edge_angle_deg")
    if include_role:
        columns.append("role")
    return compact_table(layout, columns=columns)


# ---------------------------------------------------------------------------
# Optics calculations
# ---------------------------------------------------------------------------


def _sample_count(element: Element, samples_per_meter: float = 16.0, min_samples: int = 4) -> int:
    if element.length <= 0:
        return 1
    if element.kind == "quad":
        count = max(min_samples, int(math.ceil(samples_per_meter * element.length)))
    elif element.kind == "bend":
        count = max(min_samples, int(math.ceil(samples_per_meter * element.length)))
    else:
        count = max(1, int(math.ceil(samples_per_meter * element.length)))

    # Every positive-length element gets an exact center station.  This keeps
    # element-center tables honest even when the nominal density would produce
    # an odd number of slices (for example, five slices across the 0.3 m Q1).
    return count if count % 2 == 0 else count + 1


def _append_row(rows: list[dict], s: float, element_name: str, element_kind: str, beta_x: float, alpha_x: float, beta_y: float, alpha_y: float, eta: np.ndarray, note: str = "") -> None:
    rows.append(
        {
            "s_m": float(s),
            "element": element_name,
            "kind": element_kind,
            "beta_x_m": float(beta_x),
            "alpha_x": float(alpha_x),
            "beta_y_m": float(beta_y),
            "alpha_y": float(alpha_y),
            "eta_x_m": float(eta[0]),
            "eta_xp": float(eta[1]),
            "note": note,
        }
    )


def compute_optics(
    elements: Sequence[Element],
    *,
    periodic: bool = True,
    initial: dict | None = None,
    delta: float = 0.0,
    samples_per_meter: float = 16.0,
) -> OpticsResult:
    """Compute sampled Twiss and dispersion along a line.

    For ``periodic=True``, the initial Twiss and dispersion are matched to the
    one-cell map.  For ``periodic=False``, pass initial values or use the default
    transport start: beta=10 m, alpha=0, eta=eta'=0.
    """
    elements = list(elements)
    matrix_x, matrix_y, source = transfer_map(elements, delta=delta)
    stable_x = is_stable(matrix_x)
    stable_y = is_stable(matrix_y)

    if periodic:
        tw_x = matched_twiss_from_matrix(matrix_x)
        tw_y = matched_twiss_from_matrix(matrix_y)
        if tw_x is None or tw_y is None:
            raise ValueError(
                "The requested line is not stable in both transverse planes, so a periodic Twiss solution does not exist. "
                "Use stability_report(...) or compute_optics(..., periodic=False) to inspect transport through the line."
            )
        beta_x, alpha_x, tune_x = tw_x
        beta_y, alpha_y, tune_y = tw_y
        try:
            eta = np.linalg.solve(np.eye(2) - matrix_x, source)
        except np.linalg.LinAlgError:
            eta = np.array([np.nan, np.nan])
        initial_used = {
            "beta_x_m": beta_x,
            "alpha_x": alpha_x,
            "beta_y_m": beta_y,
            "alpha_y": alpha_y,
            "eta_x_m": eta[0],
            "eta_xp": eta[1],
        }
    else:
        initial_used = {
            "beta_x_m": 10.0,
            "alpha_x": 0.0,
            "beta_y_m": 10.0,
            "alpha_y": 0.0,
            "eta_x_m": 0.0,
            "eta_xp": 0.0,
        }
        if initial:
            initial_used.update(initial)
        beta_x = float(initial_used["beta_x_m"])
        alpha_x = float(initial_used["alpha_x"])
        beta_y = float(initial_used["beta_y_m"])
        alpha_y = float(initial_used["alpha_y"])
        eta = np.array([float(initial_used["eta_x_m"]), float(initial_used["eta_xp"])])
        tune_x = None
        tune_y = None

    rows: list[dict] = []
    s = 0.0
    _append_row(rows, s, "START", "marker", beta_x, alpha_x, beta_y, alpha_y, eta, note="start")

    for element in elements:
        n_slices = _sample_count(element, samples_per_meter=samples_per_meter)
        slice_elements = split_element(element, n_slices)
        edge_maps = _bend_edge_maps(element, delta=delta)
        if edge_maps is not None:
            edge_x, edge_y = edge_maps
            beta_x, alpha_x = propagate_twiss(beta_x, alpha_x, edge_x)
            beta_y, alpha_y = propagate_twiss(beta_y, alpha_y, edge_y)
            eta = edge_x @ eta

        for slice_index, slice_element_ in enumerate(slice_elements):
            mx, my, b = element_maps(slice_element_, delta=delta)
            beta_x, alpha_x = propagate_twiss(beta_x, alpha_x, mx)
            beta_y, alpha_y = propagate_twiss(beta_y, alpha_y, my)
            eta = mx @ eta + b
            s += slice_element_.length
            if edge_maps is not None and slice_index == len(slice_elements) - 1:
                edge_x, edge_y = edge_maps
                beta_x, alpha_x = propagate_twiss(beta_x, alpha_x, edge_x)
                beta_y, alpha_y = propagate_twiss(beta_y, alpha_y, edge_y)
                eta = edge_x @ eta
            _append_row(rows, s, element.name, element.kind, beta_x, alpha_x, beta_y, alpha_y, eta)

    table = pd.DataFrame(rows)
    return OpticsResult(
        elements=elements,
        table=table,
        layout=element_layout(elements),
        matrix_x=matrix_x,
        matrix_y=matrix_y,
        dispersion_source=source,
        stable_x=stable_x,
        stable_y=stable_y,
        tune_x=tune_x,
        tune_y=tune_y,
        initial=initial_used,
        periodic=periodic,
        delta=delta,
    )


def compute_periodic_optics(elements: Sequence[Element], **kwargs) -> OpticsResult:
    return compute_optics(elements, periodic=True, **kwargs)


def compute_transport_optics(elements: Sequence[Element], **kwargs) -> OpticsResult:
    return compute_optics(elements, periodic=False, **kwargs)


def _uniform_s_positions(elements: Sequence[Element], sample_step_m: float) -> np.ndarray:
    """Return endpoint-inclusive, uniformly spaced longitudinal stations."""
    if sample_step_m <= 0:
        raise ValueError("sample_step_m must be positive")
    length = float(sum(element.length for element in elements))
    if length <= 0:
        return np.array([0.0])
    n_intervals = max(1, int(math.ceil(length / float(sample_step_m))))
    return np.linspace(0.0, length, n_intervals + 1)


def _compose_horizontal_map(
    accumulated_matrix: np.ndarray,
    accumulated_source: np.ndarray,
    next_matrix: np.ndarray,
    next_source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        next_matrix @ accumulated_matrix,
        next_matrix @ accumulated_source + next_source,
    )


def _horizontal_interval_maps(
    elements: Sequence[Element],
    s_positions: Sequence[float],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return first-order maps between consecutive requested ``s`` stations.

    The requested stations may cut through elements. Bend entrance and exit
    edge maps are applied only when those physical boundaries are crossed.
    """
    elements = list(elements)
    stations = np.asarray(s_positions, dtype=float)
    if stations.ndim != 1 or len(stations) == 0:
        raise ValueError("s_positions must be a nonempty one-dimensional sequence")
    if not np.all(np.diff(stations) >= -1e-12):
        raise ValueError("s_positions must be sorted")

    total_length = float(sum(element.length for element in elements))
    if stations[0] < -1e-12 or stations[-1] > total_length + 1e-12:
        raise ValueError("s_positions must lie within the lattice")

    maps: list[tuple[np.ndarray, np.ndarray]] = []
    current_s = 0.0
    element_index = 0
    distance_in_element = 0.0
    tolerance = 1e-12

    for target in stations:
        target = float(np.clip(target, 0.0, total_length))
        interval_matrix = np.eye(2)
        interval_source = np.zeros(2)

        while current_s < target - tolerance:
            while element_index < len(elements) and elements[element_index].length <= tolerance:
                element_index += 1
            if element_index >= len(elements):
                break

            element = elements[element_index]
            edge_maps = _bend_edge_maps(element, delta=0.0)
            if distance_in_element <= tolerance and edge_maps is not None:
                edge_x, _ = edge_maps
                interval_matrix, interval_source = _compose_horizontal_map(
                    interval_matrix,
                    interval_source,
                    edge_x,
                    np.zeros(2),
                )

            remaining_in_element = element.length - distance_in_element
            distance = min(target - current_s, remaining_in_element)
            fraction = distance / element.length
            partial = replace(
                element,
                length=distance,
                angle=element.angle * fraction if element.kind == "bend" else element.angle,
                edge_angle=0.0,
            )
            matrix_x, _, source = element_maps(partial, delta=0.0)
            interval_matrix, interval_source = _compose_horizontal_map(
                interval_matrix,
                interval_source,
                matrix_x,
                source,
            )
            current_s += distance
            distance_in_element += distance

            if remaining_in_element - distance <= tolerance:
                if edge_maps is not None:
                    edge_x, _ = edge_maps
                    interval_matrix, interval_source = _compose_horizontal_map(
                        interval_matrix,
                        interval_source,
                        edge_x,
                        np.zeros(2),
                    )
                element_index += 1
                distance_in_element = 0.0

        maps.append((interval_matrix, interval_source))

    return maps


def _sample_horizontal_optics_at_s(
    result: OpticsResult,
    s_positions: Sequence[float],
) -> pd.DataFrame:
    """Propagate horizontal Twiss and dispersion to exact requested stations."""
    stations = np.asarray(s_positions, dtype=float)
    interval_maps = _horizontal_interval_maps(result.elements, stations)
    beta = float(result.initial["beta_x_m"])
    alpha = float(result.initial["alpha_x"])
    eta = np.array(
        [float(result.initial["eta_x_m"]), float(result.initial["eta_xp"])],
        dtype=float,
    )
    rows = []
    for station, (matrix_x, source) in zip(stations, interval_maps, strict=True):
        beta, alpha = propagate_twiss(beta, alpha, matrix_x)
        eta = matrix_x @ eta + source
        rows.append(
            {
                "s_m": float(station),
                "beta_x_m": beta,
                "alpha_x": alpha,
                "eta_x_m": float(eta[0]),
                "eta_xp": float(eta[1]),
            }
        )
    return pd.DataFrame(rows)


def _propagate_horizontal_particles(
    elements: Sequence[Element],
    s_positions: Sequence[float],
    x0_m: Sequence[float],
    xp0: Sequence[float],
    delta: Sequence[float],
) -> FirstOrderParticleTracks:
    stations = np.asarray(s_positions, dtype=float)
    delta_values = np.asarray(delta, dtype=float)
    coordinates = np.vstack([np.asarray(x0_m, dtype=float), np.asarray(xp0, dtype=float)])
    if coordinates.shape[1] != len(delta_values):
        raise ValueError("x0_m, xp0, and delta must have the same length")

    x_history = np.empty((len(delta_values), len(stations)), dtype=float)
    xp_history = np.empty_like(x_history)
    for station_index, (matrix_x, source) in enumerate(
        _horizontal_interval_maps(elements, stations)
    ):
        coordinates = matrix_x @ coordinates + source[:, None] * delta_values[None, :]
        x_history[:, station_index] = coordinates[0]
        xp_history[:, station_index] = coordinates[1]

    return FirstOrderParticleTracks(
        s_m=stations,
        x_m=x_history,
        xp=xp_history,
        delta=delta_values,
        particle_id=np.arange(len(delta_values), dtype=int),
    )


def _whitened_standard_normal(rng: np.random.Generator, n_modes: int, n_particles: int) -> np.ndarray:
    """Return deterministic Gaussian-like modes with exact zero mean/covariance."""
    if n_particles <= n_modes:
        raise ValueError("n_particles must be larger than the number of sampled modes")
    modes = rng.standard_normal((n_modes, n_particles))
    modes -= modes.mean(axis=1, keepdims=True)
    covariance = modes @ modes.T / n_particles
    return np.linalg.solve(np.linalg.cholesky(covariance), modes)


def first_order_particle_tracks(
    result: OpticsResult,
    *,
    sigma_delta: float,
    emit_x: float = GEOMETRIC_EMITTANCE,
    n_particles: int = 160,
    seed: int = 2026,
    sample_step_m: float = 0.05,
) -> FirstOrderParticleTracks:
    """Launch one matched distribution and propagate the same particles in ``s``.

    The launch covariance uses the result's initial Twiss and dispersion. Each
    particle keeps its identity and momentum offset while the existing
    first-order maps transport ``(x, x')``.
    """
    if sigma_delta < 0:
        raise ValueError("sigma_delta must be nonnegative")
    if emit_x < 0:
        raise ValueError("emit_x must be nonnegative")
    n_particles = int(n_particles)
    rng = np.random.default_rng(seed)
    modes = _whitened_standard_normal(rng, 3, n_particles)
    normalized_x, normalized_px, normalized_delta = modes

    beta0 = float(result.initial["beta_x_m"])
    alpha0 = float(result.initial["alpha_x"])
    eta0 = float(result.initial["eta_x_m"])
    etap0 = float(result.initial["eta_xp"])
    delta_values = float(sigma_delta) * normalized_delta
    x_beta = math.sqrt(emit_x * beta0) * normalized_x
    xp_beta = math.sqrt(emit_x / beta0) * (-alpha0 * normalized_x + normalized_px)
    x0 = x_beta + eta0 * delta_values
    xp0 = xp_beta + etap0 * delta_values
    stations = _uniform_s_positions(result.elements, sample_step_m)
    return _propagate_horizontal_particles(
        result.elements,
        stations,
        x0,
        xp0,
        delta_values,
    )


def stability_report(elements: Sequence[Element]) -> pd.DataFrame:
    mx, my, source = transfer_map(elements)
    tx = matched_twiss_from_matrix(mx)
    ty = matched_twiss_from_matrix(my)
    rows = []
    for plane, matrix, tw in [("x", mx, tx), ("y", my, ty)]:
        half_trace = 0.5 * float(np.trace(matrix))
        rows.append(
            {
                "plane": plane,
                "trace(M)/2": half_trace,
                "stable?": abs(half_trace) < 1.0,
                "cell tune if stable": tw[2] if tw else np.nan,
                "matched beta at start [m]": tw[0] if tw else np.nan,
            }
        )
    rows.append({"plane": "dispersion source", "trace(M)/2": np.nan, "stable?": "n/a", "cell tune if stable": source[0], "matched beta at start [m]": source[1]})
    return pd.DataFrame(rows)


def optics_summary(result: OpticsResult, label: str = "line") -> pd.DataFrame:
    df = result.table
    return pd.DataFrame(
        {
            "quantity": [
                "length [m]",
                "stable x?",
                "stable y?",
                "cell tune Qₓ",
                "cell tune Qᵧ",
                "βₓ [m]",
                "βᵧ [m]",
                "Dₓ [m]",
            ],
            label: [
                result.layout["s_end_m"].iloc[-1] if len(result.layout) else 0.0,
                result.stable_x,
                result.stable_y,
                result.tune_x if result.tune_x is not None else np.nan,
                result.tune_y if result.tune_y is not None else np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "min": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                df["beta_x_m"].min(),
                df["beta_y_m"].min(),
                df["eta_x_m"].min(),
            ],
            "max": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                df["beta_x_m"].max(),
                df["beta_y_m"].max(),
                df["eta_x_m"].max(),
            ],
        }
    )


def _compact_value(value: float | bool | str | None, precision: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_float):
        return ""
    return f"{value_float:.{precision}g}"


def _compact_range(values: pd.Series, precision: int = 6) -> str:
    return f"{_compact_value(values.min(), precision)} to {_compact_value(values.max(), precision)}"


def compact_optics_summary(result: OpticsResult, *, precision: int = 6) -> pd.DataFrame:
    df = result.table
    length = result.layout["s_end_m"].iloc[-1] if len(result.layout) else 0.0
    return pd.DataFrame(
        [
            {"quantity": "length (m)", "value": _compact_value(length, precision)},
            {"quantity": "stable", "value": f"x: {_compact_value(result.stable_x)}, y: {_compact_value(result.stable_y)}"},
            {
                "quantity": "cell tune",
                "value": f"Qₓ: {_compact_value(result.tune_x, precision)}, Qᵧ: {_compact_value(result.tune_y, precision)}",
            },
            {"quantity": "βₓ (m)", "value": _compact_range(df["beta_x_m"], precision)},
            {"quantity": "βᵧ (m)", "value": _compact_range(df["beta_y_m"], precision)},
            {"quantity": "Dₓ (m)", "value": _compact_range(df["eta_x_m"], precision)},
        ]
    )


def compact_optics_comparison(results: Mapping[str, OpticsResult] | Sequence[tuple[str, OpticsResult]], *, precision: int = 6) -> pd.DataFrame:
    """Compare several optics results without the NaN-heavy min/max layout."""
    items = list(results.items()) if isinstance(results, Mapping) else list(results)
    rows = []
    metrics = [
        ("length (m)", lambda result: _compact_value(result.layout["s_end_m"].iloc[-1] if len(result.layout) else 0.0, precision)),
        ("stable", lambda result: f"x: {_compact_value(result.stable_x)}, y: {_compact_value(result.stable_y)}"),
        ("cell tune Qₓ", lambda result: _compact_value(result.tune_x, precision)),
        ("cell tune Qᵧ", lambda result: _compact_value(result.tune_y, precision)),
        ("βₓ (m)", lambda result: _compact_range(result.table["beta_x_m"], precision)),
        ("βᵧ (m)", lambda result: _compact_range(result.table["beta_y_m"], precision)),
        ("Dₓ (m)", lambda result: _compact_range(result.table["eta_x_m"], precision)),
    ]
    for quantity, getter in metrics:
        row = {"quantity": quantity}
        for label, result in items:
            row[str(label)] = getter(result)
        rows.append(row)
    return pd.DataFrame(rows)


def compact_stability_report(elements: Sequence[Element], *, precision: int = 6) -> pd.DataFrame:
    """Return only the transverse stability rows with display-friendly labels."""
    table = stability_report(elements)
    table = table[table["plane"].isin(["x", "y"])]
    return compact_table(
        table,
        columns=["plane", "stable?", "trace(M)/2", "cell tune if stable", "matched beta at start [m]"],
        precision=precision,
    )


def add_beam_size_columns(
    table: pd.DataFrame,
    *,
    sigma_delta: float = 0.0,
    emit_x: float = GEOMETRIC_EMITTANCE,
    emit_y: float = GEOMETRIC_EMITTANCE,
) -> pd.DataFrame:
    df = table.copy()
    df["sigma_x_beta_mm"] = 1e3 * np.sqrt(np.maximum(emit_x * df["beta_x_m"], 0.0))
    df["sigma_x_dispersion_mm"] = 1e3 * np.abs(df["eta_x_m"] * sigma_delta)
    df["sigma_x_mm"] = 1e3 * np.sqrt(np.maximum(emit_x * df["beta_x_m"] + (df["eta_x_m"] * sigma_delta) ** 2, 0.0))
    df["sigma_y_mm"] = 1e3 * np.sqrt(np.maximum(emit_y * df["beta_y_m"], 0.0))
    return df


def _row_at_sampled_s(result: OpticsResult, s_target: float, *, atol: float = 1e-10) -> pd.Series:
    matches = np.isclose(result.table["s_m"].to_numpy(dtype=float), float(s_target), rtol=0.0, atol=atol)
    if not np.any(matches):
        raise ValueError(
            f"No optics sample exists at s={s_target:.12g} m. "
            "Recompute the optics with center-inclusive sampling."
        )
    return result.table.loc[result.table.index[matches][-1]].copy()


def row_at_element_center(result: OpticsResult, element_name: str) -> pd.Series:
    layout = result.layout.query("name == @element_name")
    if layout.empty:
        raise KeyError(f"Element {element_name!r} not found")
    row = layout.iloc[0]
    center = 0.5 * (row["s_start_m"] + row["s_end_m"])
    sampled = _row_at_sampled_s(result, center)
    sampled["s_m"] = float(center)
    return sampled


def table_at_element_centers(result: OpticsResult, element_names: Sequence[str], sigma_delta: float = 0.0) -> pd.DataFrame:
    rows = []
    for name in element_names:
        layout_row = result.layout.query("name == @name").iloc[0]
        center = 0.5 * (layout_row["s_start_m"] + layout_row["s_end_m"])
        row = row_at_element_center(result, name)
        sigma_x_mm = 1e3 * math.sqrt(max(GEOMETRIC_EMITTANCE * row["beta_x_m"] + (row["eta_x_m"] * sigma_delta) ** 2, 0.0))
        sigma_y_mm = 1e3 * math.sqrt(max(GEOMETRIC_EMITTANCE * row["beta_y_m"], 0.0))
        rows.append(
            {
                "element": name,
                "s_center_m": center,
                "beta_x_m": row["beta_x_m"],
                "beta_y_m": row["beta_y_m"],
                "eta_x_m": row["eta_x_m"],
                "eta_xp": row["eta_xp"],
                "sigma_x_mm": sigma_x_mm,
                "sigma_y_mm": sigma_y_mm,
            }
        )
    return pd.DataFrame(rows)


def compact_element_center_table(
    result: OpticsResult,
    element_names: Sequence[str],
    sigma_delta: float = 0.0,
    *,
    keep_zero_dispersion: bool = False,
) -> pd.DataFrame:
    table = table_at_element_centers(result, element_names, sigma_delta=sigma_delta)
    columns = ["element", "s_center_m", "beta_x_m", "beta_y_m", "eta_x_m", "eta_xp", "sigma_x_mm", "sigma_y_mm"]
    if not keep_zero_dispersion:
        if np.allclose(table["eta_x_m"], 0.0, atol=1e-14):
            columns.remove("eta_x_m")
        if np.allclose(table["eta_xp"], 0.0, atol=1e-14):
            columns.remove("eta_xp")
    return compact_table(table, columns=columns)


def beam_size_comparison_at_elements(result: OpticsResult, element_names: Sequence[str], sigma_delta: float = SIGMA_DELTA_DEFAULT) -> pd.DataFrame:
    no_spread = table_at_element_centers(result, element_names, sigma_delta=0.0).set_index("element")
    with_spread = table_at_element_centers(result, element_names, sigma_delta=sigma_delta).set_index("element")
    rows = []
    for name in element_names:
        rows.append(
            {
                "element": name,
                "sigma_x, delta=0 [mm]": no_spread.loc[name, "sigma_x_mm"],
                f"sigma_x, delta={sigma_delta:g} [mm]": with_spread.loc[name, "sigma_x_mm"],
                "sigma_y, delta=0 [mm]": no_spread.loc[name, "sigma_y_mm"],
                f"sigma_y, delta={sigma_delta:g} [mm]": with_spread.loc[name, "sigma_y_mm"],
            }
        )
    return pd.DataFrame(rows)


def compact_beam_size_comparison_at_elements(
    result: OpticsResult,
    element_names: Sequence[str],
    sigma_delta: float = SIGMA_DELTA_DEFAULT,
) -> pd.DataFrame:
    table = beam_size_comparison_at_elements(result, element_names, sigma_delta=sigma_delta)
    labels = {}
    for column in table.columns:
        if column.startswith("sigma_x, delta="):
            delta = column.removeprefix("sigma_x, delta=").removesuffix(" [mm]")
            labels[column] = f"σₓ, σδ={delta} (mm)"
        elif column.startswith("sigma_y, delta="):
            delta = column.removeprefix("sigma_y, delta=").removesuffix(" [mm]")
            labels[column] = f"σᵧ, σδ={delta} (mm)"
    return compact_table(table, labels=labels)


def dispersion_extrema(result: OpticsResult) -> pd.DataFrame:
    df = result.table
    rows = []
    for label, idx in [("minimum D_x", df["eta_x_m"].idxmin()), ("maximum D_x", df["eta_x_m"].idxmax()), ("maximum |D_x|", df["eta_x_m"].abs().idxmax())]:
        row = df.loc[idx]
        rows.append(
            {
                "condition": label,
                "s_m": row["s_m"],
                "element": row["element"],
                "kind": row["kind"],
                "eta_x_m": row["eta_x_m"],
                "eta_xp": row["eta_xp"],
                "beta_x_m": row["beta_x_m"],
                "beta_y_m": row["beta_y_m"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DBA scans and aperture calculations
# ---------------------------------------------------------------------------


def endpoint_dispersion(elements: Sequence[Element]) -> tuple[float, float]:
    _, _, source = transfer_map(elements)
    return float(source[0]), float(source[1])


def achromat_closure_merit(
    eta_x_end_m: float,
    eta_xp_end: float,
    *,
    eta_scale_m: float = ETA_CLOSURE_SCALE_M,
    etap_scale: float = ETAP_CLOSURE_SCALE,
) -> float:
    """Return a dimensionless numerical merit for the two exit residuals.

    The two terms need explicit reference scales because ``eta_x`` has units of
    length while ``eta_x'`` is dimensionless.  This merit only ranks the Q1
    scan; it is not a physical invariant.
    """
    if eta_scale_m <= 0 or etap_scale <= 0:
        raise ValueError("Achromat closure scales must be positive")
    return float((eta_x_end_m / eta_scale_m) ** 2 + (eta_xp_end / etap_scale) ** 2)


def dba_endpoint_table(q1: float, q2: float = 0.0, q3: float = 0.0) -> pd.DataFrame:
    eta_end, etap_end = endpoint_dispersion(make_dba_cell(q1=q1, q2=q2, q3=q3))
    return pd.DataFrame(
        {
            "quantity": ["D_x at end [m]", "D_x' at end", "dimensionless closure merit"],
            "value": [eta_end, etap_end, achromat_closure_merit(eta_end, etap_end)],
        }
    )


def endpoint_dispersion_table(elements: Sequence[Element]) -> pd.DataFrame:
    eta_end, etap_end = endpoint_dispersion(elements)
    return pd.DataFrame(
        {
            "quantity": ["D_x at end [m]", "D_x' at end", "dimensionless closure merit"],
            "value": [eta_end, etap_end, achromat_closure_merit(eta_end, etap_end)],
        }
    )


def compact_endpoint_dispersion_table(elements: Sequence[Element], *, precision: int = 6) -> pd.DataFrame:
    return compact_summary_table(endpoint_dispersion_table(elements), precision=precision)


def compact_dba_endpoint_table(q1: float, q2: float = 0.0, q3: float = 0.0, *, precision: int = 6) -> pd.DataFrame:
    return compact_summary_table(dba_endpoint_table(q1=q1, q2=q2, q3=q3), precision=precision)


def scan_q1_for_achromat(qmin: float = 0.0, qmax: float = 6.0, n: int = 301, q2: float = 0.0, q3: float = 0.0) -> pd.DataFrame:
    q_values = np.linspace(qmin, qmax, int(n))
    rows = []
    for q1 in q_values:
        eta_end, etap_end = endpoint_dispersion(make_dba_cell(q1=q1, q2=q2, q3=q3))
        rows.append(
            {
                "q1_m^-2": q1,
                "eta_x_end_m": eta_end,
                "eta_xp_end": etap_end,
                "closure_merit": achromat_closure_merit(eta_end, etap_end),
            }
        )
    return pd.DataFrame(rows)


def compact_q1_scan_table(scan: pd.DataFrame, *, rows: int | None = 10, precision: int = 6) -> pd.DataFrame:
    table = scan.sort_values("closure_merit")
    if rows is not None:
        table = table.head(rows)
    return compact_table(table, columns=["q1_m^-2", "eta_x_end_m", "eta_xp_end", "closure_merit"], precision=precision)


def best_q1_for_achromat(qmin: float = 0.0, qmax: float = 6.0, rounds: int = 4, n: int = 401, q2: float = 0.0, q3: float = 0.0) -> float:
    """Find Q1 by repeated grid refinement.  No SciPy dependency required."""
    left = float(qmin)
    right = float(qmax)
    best_q = None
    for _ in range(int(rounds)):
        scan = scan_q1_for_achromat(left, right, n=n, q2=q2, q3=q3)
        best = scan.loc[scan["closure_merit"].idxmin()]
        best_q = float(best["q1_m^-2"])
        step = (right - left) / (n - 1)
        left = best_q - 5.0 * step
        right = best_q + 5.0 * step
    return float(best_q)


def aperture_limit_table(
    result: OpticsResult,
    *,
    pipe_radius_m: float = PIPE_RADIUS_DEFAULT,
    n_sigma: float = 1.0,
    emit_x: float = GEOMETRIC_EMITTANCE,
) -> pd.DataFrame:
    """Return the local ``sigma_delta`` allowance from an RMS-envelope proxy.

    A location with no on-momentum clearance has zero momentum-spread
    allowance even when its dispersion is zero.  Infinite allowance is used
    only where the on-momentum envelope fits and ``eta_x`` is locally zero.
    """
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive")
    if pipe_radius_m <= 0:
        raise ValueError("pipe_radius_m must be positive")
    if emit_x < 0:
        raise ValueError("emit_x must be nonnegative")

    df = result.table.copy()
    allowed_rms_radius = pipe_radius_m / n_sigma
    sigma_beta_m = np.sqrt(np.maximum(emit_x * df["beta_x_m"], 0.0))
    eta_abs = np.abs(df["eta_x_m"].to_numpy())
    on_momentum_envelope_m = n_sigma * sigma_beta_m.to_numpy()
    on_momentum_clearance_m = pipe_radius_m - on_momentum_envelope_m
    no_on_momentum_clearance = on_momentum_clearance_m <= max(1e-14, 1e-12 * pipe_radius_m)
    zero_dispersion = eta_abs < 1e-14

    sigma_delta_limit = np.full(len(df), np.inf, dtype=float)
    sigma_delta_limit[no_on_momentum_clearance] = 0.0
    finite_allowance = ~no_on_momentum_clearance & ~zero_dispersion
    numerator = allowed_rms_radius**2 - sigma_beta_m.to_numpy() ** 2
    sigma_delta_limit[finite_allowance] = (
        np.sqrt(np.maximum(numerator[finite_allowance], 0.0))
        / eta_abs[finite_allowance]
    )

    status = np.full(len(df), "not dispersion-limited locally", dtype=object)
    status[finite_allowance] = "finite dispersive allowance"
    status[no_on_momentum_clearance] = "on-momentum envelope reaches/exceeds pipe"

    out = df[["s_m", "element", "kind", "beta_x_m", "eta_x_m"]].copy()
    out["sigma_x_beta_mm"] = 1e3 * sigma_beta_m
    out["on_momentum_envelope_mm"] = 1e3 * on_momentum_envelope_m
    out["on_momentum_clearance_mm"] = 1e3 * on_momentum_clearance_m
    out["pipe_radius_mm"] = 1e3 * pipe_radius_m
    out["n_sigma"] = n_sigma
    out["aperture_status"] = status
    out["max_sigma_delta"] = sigma_delta_limit
    out["max_delta_percent"] = 100.0 * sigma_delta_limit
    return out.sort_values(
        ["max_sigma_delta", "on_momentum_clearance_mm", "s_m"],
        na_position="last",
    ).reset_index(drop=True)


def aperture_summary(result: OpticsResult, **kwargs) -> pd.DataFrame:
    table = aperture_limit_table(result, **kwargs)
    limiting = table.iloc[0]
    return pd.DataFrame(
        {
            "quantity": ["limiting momentum spread", "limiting momentum spread [%]", "limiting status", "limiting element", "limiting s [m]", "D_x at limit [m]", "beta_x at limit [m]"],
            "value": [
                limiting["max_sigma_delta"],
                limiting["max_delta_percent"],
                limiting["aperture_status"],
                limiting["element"],
                limiting["s_m"],
                limiting["eta_x_m"],
                limiting["beta_x_m"],
            ],
        }
    )


def compact_aperture_limit_table(table: pd.DataFrame, *, precision: int = 6) -> pd.DataFrame:
    return compact_table(
        table,
        columns=[
            "element",
            "s_m",
            "kind",
            "beta_x_m",
            "eta_x_m",
            "sigma_x_beta_mm",
            "on_momentum_envelope_mm",
            "on_momentum_clearance_mm",
            "pipe_radius_mm",
            "n_sigma",
            "aperture_status",
            "max_sigma_delta",
            "max_delta_percent",
        ],
        precision=precision,
    )


def compact_aperture_summary(result: OpticsResult, *, precision: int = 6, **kwargs) -> pd.DataFrame:
    return compact_summary_table(aperture_summary(result, **kwargs), precision=precision)


def compact_aperture_limiting_row(table: pd.DataFrame, *, precision: int = 4) -> pd.DataFrame:
    """Return the single controlling row from a sorted aperture-limit table."""
    if table.empty:
        raise ValueError("Aperture-limit table is empty")
    return compact_table(
        table.head(1),
        columns=[
            "element",
            "s_m",
            "eta_x_m",
            "on_momentum_envelope_mm",
            "aperture_status",
            "max_sigma_delta",
            "max_delta_percent",
        ],
        precision=precision,
    )


def compact_aperture_input_row(table: pd.DataFrame, *, precision: int = 4) -> pd.DataFrame:
    """Show the controlling optics inputs without printing the solved limit."""
    if table.empty:
        raise ValueError("Aperture-limit table is empty")
    return compact_table(
        table.head(1),
        columns=["element", "s_m", "beta_x_m", "eta_x_m", "sigma_x_beta_mm"],
        precision=precision,
    )


def aperture_envelope_diagnostics(
    result: OpticsResult,
    *,
    sigma_delta: float,
    pipe_radius_m: float = PIPE_RADIUS_DEFAULT,
    n_sigma: float = 1.0,
    emit_x: float = GEOMETRIC_EMITTANCE,
) -> pd.DataFrame:
    """Separate the first RMS-envelope crossing from the largest envelope.

    The first crossing is linearly interpolated between adjacent diagnostic
    samples.  The largest-envelope location is the sampled minimum clearance;
    notebook prose therefore reports its position only to the sampling scale.
    """
    if sigma_delta < 0:
        raise ValueError("sigma_delta must be nonnegative")
    if pipe_radius_m <= 0:
        raise ValueError("pipe_radius_m must be positive")
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive")
    if emit_x < 0:
        raise ValueError("emit_x must be nonnegative")

    table = result.table[["s_m", "element", "kind", "beta_x_m", "eta_x_m"]].copy()
    sigma_x_m = np.sqrt(
        np.maximum(
            emit_x * table["beta_x_m"].to_numpy()
            + (table["eta_x_m"].to_numpy() * sigma_delta) ** 2,
            0.0,
        )
    )
    envelope_m = n_sigma * sigma_x_m
    clearance_m = pipe_radius_m - envelope_m
    s_values = table["s_m"].to_numpy(dtype=float)

    first_crossing = None
    if clearance_m[0] <= 0:
        first_crossing = {
            "s_m": s_values[0],
            "element": table.iloc[0]["element"],
            "kind": table.iloc[0]["kind"],
            "beta_x_m": table.iloc[0]["beta_x_m"],
            "eta_x_m": table.iloc[0]["eta_x_m"],
            "envelope_mm": 1e3 * envelope_m[0],
            "clearance_mm": 1e3 * clearance_m[0],
            "location_method": "sampled at lattice entrance",
        }
    else:
        brackets = np.flatnonzero((clearance_m[:-1] > 0.0) & (clearance_m[1:] <= 0.0))
        if len(brackets):
            left = int(brackets[0])
            right = left + 1
            fraction = clearance_m[left] / (clearance_m[left] - clearance_m[right])
            first_crossing = {
                "s_m": s_values[left] + fraction * (s_values[right] - s_values[left]),
                "element": table.iloc[right]["element"],
                "kind": table.iloc[right]["kind"],
                "beta_x_m": table.iloc[left]["beta_x_m"] + fraction * (table.iloc[right]["beta_x_m"] - table.iloc[left]["beta_x_m"]),
                "eta_x_m": table.iloc[left]["eta_x_m"] + fraction * (table.iloc[right]["eta_x_m"] - table.iloc[left]["eta_x_m"]),
                "envelope_mm": 1e3 * pipe_radius_m,
                "clearance_mm": 0.0,
                "location_method": "interpolated crossing",
            }

    worst_index = int(np.argmin(clearance_m))
    worst = table.iloc[worst_index]
    rows = []
    if first_crossing is not None:
        rows.append({"condition": "first RMS-envelope crossing", **first_crossing})
    else:
        rows.append(
            {
                "condition": "first RMS-envelope crossing",
                "s_m": np.nan,
                "element": "none in displayed cell",
                "kind": "",
                "beta_x_m": np.nan,
                "eta_x_m": np.nan,
                "envelope_mm": np.nan,
                "clearance_mm": float(np.min(1e3 * clearance_m)),
                "location_method": "no crossing",
            }
        )
    rows.append(
        {
            "condition": "largest RMS envelope",
            "s_m": float(worst["s_m"]),
            "element": worst["element"],
            "kind": worst["kind"],
            "beta_x_m": float(worst["beta_x_m"]),
            "eta_x_m": float(worst["eta_x_m"]),
            "envelope_mm": float(1e3 * envelope_m[worst_index]),
            "clearance_mm": float(1e3 * clearance_m[worst_index]),
            "location_method": "sampled maximum",
        }
    )
    return pd.DataFrame(rows)


def compact_aperture_envelope_diagnostics(table: pd.DataFrame, *, precision: int = 3) -> pd.DataFrame:
    return compact_table(
        table,
        columns=["condition", "element", "s_m", "envelope_mm", "clearance_mm", "location_method"],
        precision=precision,
    )


# ---------------------------------------------------------------------------
# Chromaticity and resonance helpers
# ---------------------------------------------------------------------------


def xsuite_ring_twiss(line, *, delta_chrom: float = 1e-4):
    """Return authoritative 4D Xsuite ring optics and chromatic properties."""
    if delta_chrom <= 0:
        raise ValueError("delta_chrom must be positive")
    return line.twiss(method="4d", chrom=True, delta_chrom=float(delta_chrom))


def xsuite_tune_scan(line, *, delta_values: Sequence[float]) -> pd.DataFrame:
    """Evaluate Xsuite's periodic ring tunes at each particle momentum offset.

    Failed periodic solutions are retained as unstable rows rather than
    terminating a scan.  Xsuite alone handles the off-momentum magnet and orbit
    physics; no manual strength or bend-angle rescaling is applied here.
    """
    rows = []
    for delta in np.asarray(delta_values, dtype=float):
        try:
            twiss = line.twiss(method="4d", delta0=float(delta), chrom=False)
        except (ValueError, RuntimeError) as exc:
            rows.append(
                {
                    "delta": float(delta),
                    "Qx": np.nan,
                    "Qy": np.nan,
                    "stable": False,
                    "error": str(exc),
                }
            )
            continue
        rows.append(
            {
                "delta": float(delta),
                "Qx": float(twiss.qx),
                "Qy": float(twiss.qy),
                "stable": True,
                "error": "",
            }
        )
    return pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)


def _validated_tune_scan(scan: pd.DataFrame) -> pd.DataFrame:
    required = {"delta", "Qx", "Qy", "stable"}
    missing = required.difference(scan.columns)
    if missing:
        raise ValueError(f"Tune scan is missing columns: {sorted(missing)}")
    clean = scan.loc[scan["stable"].astype(bool), ["delta", "Qx", "Qy", "stable"]].copy()
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta", "Qx", "Qy"])
    clean = clean.sort_values("delta").drop_duplicates("delta", keep="last").reset_index(drop=True)
    if clean.empty:
        raise ValueError("Tune scan has no stable Xsuite points")
    return clean


def _interpolate_tune_scan(scan: pd.DataFrame, delta_values: Sequence[float]) -> pd.DataFrame:
    """Interpolate within a prepared direct scan for plotting and slider endpoints."""
    clean = _validated_tune_scan(scan)
    requested = np.asarray(delta_values, dtype=float)
    delta = clean["delta"].to_numpy(dtype=float)
    if requested.size and (requested.min() < delta.min() - 1e-14 or requested.max() > delta.max() + 1e-14):
        raise ValueError("Requested momentum offset lies outside the Xsuite tune scan")
    return pd.DataFrame(
        {
            "delta": requested,
            "Qx": np.interp(requested, delta, clean["Qx"].to_numpy(dtype=float)),
            "Qy": np.interp(requested, delta, clean["Qy"].to_numpy(dtype=float)),
            "stable": True,
        }
    )


def cell_tunes(elements: Sequence[Element], delta: float = 0.0) -> tuple[float, float]:
    """Return local-model cell tunes for dispersion diagnostics.

    Section C uses :func:`xsuite_ring_twiss` and :func:`xsuite_tune_scan`
    instead of this pedagogical matrix approximation.
    """
    mx, my, _ = transfer_map(elements, delta=delta, include_dispersion_source=False)
    tw_x = matched_twiss_from_matrix(mx)
    tw_y = matched_twiss_from_matrix(my)
    if tw_x is None or tw_y is None:
        raise ValueError("Cell is unstable at this momentum offset; tune is undefined.")
    return tw_x[2], tw_y[2]


def ring_tunes(elements: Sequence[Element], n_cells: int = DBA_N_CELLS_DEFAULT, delta: float = 0.0) -> tuple[float, float]:
    qx_cell, qy_cell = cell_tunes(elements, delta=delta)
    return n_cells * qx_cell, n_cells * qy_cell


def chromaticity_finite_difference(elements: Sequence[Element], n_cells: int = DBA_N_CELLS_DEFAULT, ddelta: float = 1e-4) -> tuple[float, float]:
    """Return the legacy local-model finite difference (not Section C's backend)."""
    qx_plus, qy_plus = ring_tunes(elements, n_cells=n_cells, delta=+ddelta)
    qx_minus, qy_minus = ring_tunes(elements, n_cells=n_cells, delta=-ddelta)
    return (qx_plus - qx_minus) / (2.0 * ddelta), (qy_plus - qy_minus) / (2.0 * ddelta)


def tune_scan(
    elements: Sequence[Element],
    *,
    delta_values: Sequence[float],
    n_cells: int = DBA_N_CELLS_DEFAULT,
) -> pd.DataFrame:
    """Evaluate legacy local-model tunes at each requested momentum offset."""
    rows = []
    for delta in np.asarray(delta_values, dtype=float):
        try:
            qx, qy = ring_tunes(elements, n_cells=n_cells, delta=float(delta))
            stable = True
        except ValueError:
            qx, qy = np.nan, np.nan
            stable = False
        rows.append({"delta": float(delta), "Qx": qx, "Qy": qy, "stable": stable})
    return pd.DataFrame(rows)


def ring_summary(elements: Sequence[Element], n_cells: int = DBA_N_CELLS_DEFAULT, ddelta: float = 1e-4) -> pd.DataFrame:
    qx, qy = ring_tunes(elements, n_cells=n_cells, delta=0.0)
    cx, cy = chromaticity_finite_difference(elements, n_cells=n_cells, ddelta=ddelta)
    cell = compute_periodic_optics(elements)
    return pd.DataFrame(
        {
            "quantity": [
                "number of cells",
                "Qx ring tune",
                "Qy ring tune",
                "Cx = dQx/d(delta)",
                "Cy = dQy/d(delta)",
                "max D_x in one cell [m]",
                "min D_x in one cell [m]",
            ],
            "value": [n_cells, qx, qy, cx, cy, cell.table["eta_x_m"].max(), cell.table["eta_x_m"].min()],
        }
    )


def compact_ring_summary(elements: Sequence[Element], n_cells: int = DBA_N_CELLS_DEFAULT, ddelta: float = 1e-4, *, precision: int = 6) -> pd.DataFrame:
    return compact_summary_table(ring_summary(elements, n_cells=n_cells, ddelta=ddelta), precision=precision)


def chromatic_spread_table(nux: float, nuy: float, xi_x: float, xi_y: float, sigma_delta: float = SIGMA_DELTA_DEFAULT) -> pd.DataFrame:
    """Summarize signed +sigma tune shifts and nonnegative RMS tune widths."""
    if sigma_delta < 0:
        raise ValueError("sigma_delta must be nonnegative")
    return pd.DataFrame(
        {
            "quantity": [
                "Qx",
                "Qy",
                "xi_x = dQx/d(delta)",
                "xi_y = dQy/d(delta)",
                f"signed Delta Qx at +sigma_delta={sigma_delta:g}",
                f"signed Delta Qy at +sigma_delta={sigma_delta:g}",
                f"RMS sigma_Qx for sigma_delta={sigma_delta:g}",
                f"RMS sigma_Qy for sigma_delta={sigma_delta:g}",
            ],
            "value": [
                nux,
                nuy,
                xi_x,
                xi_y,
                xi_x * sigma_delta,
                xi_y * sigma_delta,
                abs(xi_x) * sigma_delta,
                abs(xi_y) * sigma_delta,
            ],
        }
    )


def compact_chromatic_spread_table(nux: float, nuy: float, xi_x: float, xi_y: float, sigma_delta: float = SIGMA_DELTA_DEFAULT, *, precision: int = 6) -> pd.DataFrame:
    return compact_summary_table(chromatic_spread_table(nux, nuy, xi_x, xi_y, sigma_delta=sigma_delta), precision=precision)


def resonance_lines(max_order: int, x_range: tuple[float, float], y_range: tuple[float, float]) -> list[dict]:
    """Return line segments for ``m Qx + n Qy = p`` within a rectangle."""
    xmin, xmax = x_range
    ymin, ymax = y_range
    segments: list[dict] = []
    corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]

    for m in range(-max_order, max_order + 1):
        for n in range(-max_order, max_order + 1):
            if m == 0 and n == 0:
                continue
            # Keep one lowest-order representation of each geometric line.
            if m < 0 or (m == 0 and n < 0):
                continue
            if math.gcd(abs(m), abs(n)) != 1:
                continue
            order = abs(m) + abs(n)
            if order == 0 or order > max_order:
                continue
            values = [m * x + n * y for x, y in corners]
            p_min = math.floor(min(values)) - 1
            p_max = math.ceil(max(values)) + 1
            for p in range(p_min, p_max + 1):
                points = []
                if n != 0:
                    for x in (xmin, xmax):
                        y = (p - m * x) / n
                        if ymin - 1e-12 <= y <= ymax + 1e-12:
                            points.append((x, y))
                if m != 0:
                    for y in (ymin, ymax):
                        x = (p - n * y) / m
                        if xmin - 1e-12 <= x <= xmax + 1e-12:
                            points.append((x, y))
                unique = []
                for point in points:
                    if not any(abs(point[0] - q[0]) < 1e-9 and abs(point[1] - q[1]) < 1e-9 for q in unique):
                        unique.append(point)
                if len(unique) >= 2:
                    segments.append(
                        {
                            "m": m,
                            "n": n,
                            "p": p,
                            "order": order,
                            "x0": unique[0][0],
                            "y0": unique[0][1],
                            "x1": unique[1][0],
                            "y1": unique[1][1],
                            "label": f"{m}Qx + {n}Qy = {p}",
                        }
                    )
    return segments


def first_resonance_crossing(nux: float, nuy: float, cx: float, cy: float, max_order: int = 3, sigma_min: float = 1e-9) -> pd.DataFrame:
    """Find the first low-order resonance hit by the chromatic tune line.

    The footprint is parameterized as ``(Qx, Qy) = (nux, nuy) + delta * (cx, cy)``.
    The returned ``abs_delta_at_crossing`` is the smallest absolute momentum
    offset that reaches an order ``<= max_order`` resonance.
    """
    rows = []
    for m in range(-max_order, max_order + 1):
        for n in range(-max_order, max_order + 1):
            if m == 0 and n == 0:
                continue
            order = abs(m) + abs(n)
            if order == 0 or order > max_order:
                continue
            # (m, n, p) and (-m, -n, -p) are the same resonance line.
            # Keep one canonical sign so student-facing tables do not show duplicates.
            if m < 0 or (m == 0 and n < 0):
                continue
            if math.gcd(abs(m), abs(n)) != 1:
                continue
            denominator = m * cx + n * cy
            if abs(denominator) < 1e-14:
                continue
            value0 = m * nux + n * nuy
            p_min = math.floor(value0 - abs(denominator) * 0.2 - 2)
            p_max = math.ceil(value0 + abs(denominator) * 0.2 + 2)
            for p in range(p_min, p_max + 1):
                delta_cross = (p - value0) / denominator
                if abs(delta_cross) >= sigma_min:
                    rows.append(
                        {
                            "m": m,
                            "n": n,
                            "p": p,
                            "order": order,
                            "delta_at_crossing": delta_cross,
                            "abs_delta_at_crossing": abs(delta_cross),
                            "delta_percent": 100.0 * abs(delta_cross),
                            "resonance": f"{m} Qx + {n} Qy = {p}",
                        }
                    )
    if not rows:
        return pd.DataFrame(columns=["m", "n", "p", "order", "delta_at_crossing", "abs_delta_at_crossing", "delta_percent", "resonance"])
    return pd.DataFrame(rows).sort_values("abs_delta_at_crossing").reset_index(drop=True)


def first_resonance_crossing_from_tune_scan(
    scan: pd.DataFrame,
    *,
    max_order: int = 3,
    sigma_min: float = 1e-9,
) -> pd.DataFrame:
    """Find low-order crossings along a directly calculated tune scan.

    Each root is linearly interpolated only between adjacent scan points that
    bracket it.  Unlike :func:`first_resonance_crossing`, the tune trajectory
    itself can be nonlinear in momentum offset.
    """
    clean = _validated_tune_scan(scan)
    delta = clean["delta"].to_numpy(dtype=float)
    qx = clean["Qx"].to_numpy(dtype=float)
    qy = clean["Qy"].to_numpy(dtype=float)
    rows = []

    for m in range(-max_order, max_order + 1):
        for n in range(-max_order, max_order + 1):
            if m == 0 and n == 0:
                continue
            if m < 0 or (m == 0 and n < 0):
                continue
            if math.gcd(abs(m), abs(n)) != 1:
                continue
            order = abs(m) + abs(n)
            if order == 0 or order > max_order:
                continue

            resonance_coordinate = m * qx + n * qy
            p_min = math.floor(float(np.min(resonance_coordinate))) - 1
            p_max = math.ceil(float(np.max(resonance_coordinate))) + 1
            for p in range(p_min, p_max + 1):
                residual = resonance_coordinate - p
                roots = []
                for index in range(len(delta) - 1):
                    d0, d1 = delta[index], delta[index + 1]
                    r0, r1 = residual[index], residual[index + 1]
                    if abs(r0) < 1e-14:
                        roots.append((d0, qx[index], qy[index]))
                    if r0 * r1 < 0.0:
                        fraction = -r0 / (r1 - r0)
                        roots.append(
                            (
                                d0 + fraction * (d1 - d0),
                                qx[index] + fraction * (qx[index + 1] - qx[index]),
                                qy[index] + fraction * (qy[index + 1] - qy[index]),
                            )
                        )
                if abs(residual[-1]) < 1e-14:
                    roots.append((delta[-1], qx[-1], qy[-1]))

                for delta_cross, qx_cross, qy_cross in roots:
                    if abs(delta_cross) < sigma_min:
                        continue
                    if any(abs(delta_cross - row["delta_at_crossing"]) < 1e-12 and m == row["m"] and n == row["n"] and p == row["p"] for row in rows):
                        continue
                    rows.append(
                        {
                            "m": m,
                            "n": n,
                            "p": p,
                            "order": order,
                            "delta_at_crossing": float(delta_cross),
                            "abs_delta_at_crossing": abs(float(delta_cross)),
                            "delta_percent": 100.0 * abs(float(delta_cross)),
                            "Qx_at_crossing": float(qx_cross),
                            "Qy_at_crossing": float(qy_cross),
                            "resonance_residual": float(m * qx_cross + n * qy_cross - p),
                            "resonance": f"{m} Qx + {n} Qy = {p}",
                        }
                    )

    columns = [
        "m",
        "n",
        "p",
        "order",
        "delta_at_crossing",
        "abs_delta_at_crossing",
        "delta_percent",
        "Qx_at_crossing",
        "Qy_at_crossing",
        "resonance_residual",
        "resonance",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("abs_delta_at_crossing").reset_index(drop=True)


def compact_resonance_crossing(table: pd.DataFrame, *, precision: int = 6) -> pd.DataFrame:
    return compact_table(
        table,
        columns=["resonance", "order", "abs_delta_at_crossing", "delta_percent"],
        precision=precision,
    )


def acceptance_comparison(dispersion_limit_delta: float, chromatic_limit_delta: float) -> pd.DataFrame:
    limiting = "dispersion/aperture" if dispersion_limit_delta < chromatic_limit_delta else "chromaticity/resonance"
    return pd.DataFrame(
        {
            "limitation mechanism": ["dispersion/aperture", "chromaticity/resonance", "overall limiting mechanism"],
            "momentum spread limit": [dispersion_limit_delta, chromatic_limit_delta, min(dispersion_limit_delta, chromatic_limit_delta)],
            "limit [%]": [100 * dispersion_limit_delta, 100 * chromatic_limit_delta, 100 * min(dispersion_limit_delta, chromatic_limit_delta)],
            "note": ["n_sigma * sigma_x reaches pipe radius", "footprint reaches order <= 3 resonance", limiting],
        }
    )


def compact_acceptance_comparison(dispersion_limit_delta: float, chromatic_limit_delta: float, *, precision: int = 6) -> pd.DataFrame:
    return compact_table(
        acceptance_comparison(dispersion_limit_delta, chromatic_limit_delta),
        columns=["limitation mechanism", "momentum spread limit", "limit [%]", "note"],
        precision=precision,
    )


def phase_advance_table(result: OpticsResult) -> pd.DataFrame:
    """Integrate ``d psi / ds = 1 / beta`` along a sampled optics result."""
    df = result.table.copy()
    s = df["s_m"].to_numpy(dtype=float)
    beta_x = df["beta_x_m"].to_numpy(dtype=float)
    beta_y = df["beta_y_m"].to_numpy(dtype=float)
    if np.any(np.diff(s) < -1e-12):
        raise ValueError("optics table must be ordered in s")
    if np.any(beta_x <= 0) or np.any(beta_y <= 0):
        raise ValueError("beta functions must be positive")

    ds = np.diff(s)
    dpsi_x = 0.5 * ds * (1.0 / beta_x[:-1] + 1.0 / beta_x[1:])
    dpsi_y = 0.5 * ds * (1.0 / beta_y[:-1] + 1.0 / beta_y[1:])
    psi_x = np.concatenate([[0.0], np.cumsum(dpsi_x)])
    psi_y = np.concatenate([[0.0], np.cumsum(dpsi_y)])
    return pd.DataFrame(
        {
            "s_m": s,
            "element": df["element"].to_numpy(),
            "psi_x_rad": psi_x,
            "psi_y_rad": psi_y,
            "psi_x_turns": psi_x / (2.0 * math.pi),
            "psi_y_turns": psi_y / (2.0 * math.pi),
            "inv_beta_x_m^-1": 1.0 / beta_x,
            "inv_beta_y_m^-1": 1.0 / beta_y,
        }
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _delta_color(delta: float, max_abs_delta: float) -> str:
    scale = max(float(max_abs_delta), 1e-15)
    position = 0.5 + 0.5 * float(delta) / scale
    return sample_colorscale(DELTA_COLORSCALE, [float(np.clip(position, 0.0, 1.0))])[0]


def _padded_range(values: Sequence[float], *, minimum_span: float = 1e-9, fraction: float = 0.08) -> list[float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return [-1.0, 1.0]
    low = float(finite.min())
    high = float(finite.max())
    span = max(high - low, minimum_span)
    return [low - fraction * span, high + fraction * span]


def plot_linked_orbit_fan(
    result: OpticsResult,
    *,
    sigma_delta: float = SIGMA_DELTA_DEFAULT,
    delta_values: Sequence[float] | None = None,
    delta_sigma_extent: float = 2.5,
    n_delta: int = 9,
    sample_step_m: float = 0.1,
    frame_duration_ms: int = 80,
    title: str = "Momentum-dependent orbit response",
    show: bool = True,
):
    """Link a causal off-momentum orbit fan to an ``x``-versus-``delta`` slice."""
    if sigma_delta < 0:
        raise ValueError("sigma_delta must be nonnegative")
    if delta_values is None:
        extent = max(float(delta_sigma_extent) * float(sigma_delta), 1e-4)
        delta_values = np.linspace(-extent, extent, int(n_delta))
    deltas = np.asarray(delta_values, dtype=float)
    if deltas.ndim != 1 or len(deltas) < 3:
        raise ValueError("delta_values must contain at least three offsets")
    deltas = np.sort(deltas)

    # Start the orbit response at x=x'=eta=eta'=0 so students see the bends
    # create the separation rather than entering an already dispersed cell.
    causal_initial = dict(result.initial)
    causal_initial.update({"eta_x_m": 0.0, "eta_xp": 0.0})
    causal_result = compute_transport_optics(result.elements, initial=causal_initial)
    stations = _uniform_s_positions(result.elements, sample_step_m)
    tracks = _propagate_horizontal_particles(
        result.elements,
        stations,
        np.zeros(len(deltas)),
        np.zeros(len(deltas)),
        deltas,
    )
    x_mm = 1e3 * tracks.x_m
    max_abs_delta = max(float(np.max(np.abs(deltas))), 1e-15)
    delta_percent = 100.0 * deltas

    # Begin at the common launch point; the student then scrubs forward to see
    # separation appear in the bends. This also keeps the first slider state,
    # selected line, and selected slice synchronized in every frontend.
    selected_index = 0

    y_range = _padded_range(x_mm.ravel(), minimum_span=0.1)
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, None], [{}, {}]],
        row_heights=[0.16, 0.84],
        column_widths=[0.66, 0.34],
        shared_xaxes="columns",
        vertical_spacing=0.04,
        horizontal_spacing=0.09,
        subplot_titles=("", "", "Momentum-position slice"),
    )

    for particle_index, delta in enumerate(deltas):
        is_reference = abs(delta) < 1e-15
        fig.add_trace(
            go.Scatter(
                x=stations,
                y=x_mm[particle_index],
                mode="lines",
                line=dict(
                    color="rgba(55, 65, 81, 0.95)" if is_reference else _delta_color(delta, max_abs_delta),
                    width=2.4 if is_reference else 1.5,
                ),
                opacity=1.0 if is_reference else 0.68,
                showlegend=False,
                hovertemplate=(
                    f"δ = {100.0 * float(delta):.4f}%<br>"
                    "s = %{x:.3f} m<br>"
                    "x = %{y:.4f} mm<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    selected_line_index = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[stations[selected_index], stations[selected_index]],
            y=y_range,
            mode="lines",
            line=dict(color="rgba(55, 65, 81, 0.8)", dash="dot", width=1.5),
            name="selected s",
            showlegend=False,
            meta={"role": "decorative"},
            hovertemplate="selected s = %{x:.3f} m<extra></extra>",
        ),
        row=2,
        col=1,
    )
    slice_trace_index = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=delta_percent,
            y=x_mm[:, selected_index],
            mode="lines+markers",
            line=dict(color="rgba(80, 90, 105, 0.65)", width=1.5),
            marker=dict(
                size=8,
                color=delta_percent,
                coloraxis="coloraxis",
                line=dict(color="rgba(55, 65, 81, 0.65)", width=0.5),
            ),
            name="selected slice",
            showlegend=False,
            hovertemplate=(
                "δ = %{x:.4f}%<br>"
                "x = %{y:.4f} mm<extra></extra>"
            ),
        ),
        row=2,
        col=2,
    )

    frames = []
    for station_index, station in enumerate(stations):
        frames.append(
            go.Frame(
                name=f"{station_index}",
                data=[
                    go.Scatter(x=[station, station], y=y_range),
                    go.Scatter(x=delta_percent, y=x_mm[:, station_index]),
                ],
                traces=[selected_line_index, slice_trace_index],
            )
        )
    fig.frames = frames

    left_domain = tuple(fig.layout.xaxis2.domain)
    slider_steps = [
        dict(
            method="animate",
            args=[[frame.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}],
        )
        for frame in frames
    ]
    fig.update_layout(
        title=title,
        template="plotly_white",
        autosize=True,
        width=None,
        height=520,
        margin=dict(t=175, r=95),
        showlegend=False,
        coloraxis=dict(
            colorscale=DELTA_COLORSCALE,
            cmin=-100.0 * max_abs_delta,
            cmax=100.0 * max_abs_delta,
            cmid=0.0,
            colorbar=dict(title="δ (%)", len=0.62, y=0.42),
        ),
        sliders=[
            dict(
                active=selected_index,
                x=left_domain[0],
                len=left_domain[1] - left_domain[0],
                y=1.13,
                pad=dict(t=0, b=0),
                font=dict(size=1, color="rgba(0,0,0,0)"),
                ticklen=0,
                minorticklen=0,
                tickwidth=0,
                tickcolor="rgba(0,0,0,0)",
                currentvalue=dict(
                    prefix="selected s = ",
                    suffix=" m",
                    font=dict(size=12, color="rgba(45,55,70,1)"),
                ),
                steps=[dict(step, label=f"{station:.2f}") for step, station in zip(slider_steps, stations, strict=True)],
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=left_domain[0],
                y=1.31,
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"fromcurrent": True, "frame": {"duration": int(frame_duration_ms), "redraw": False}, "transition": {"duration": 0}}],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}],
                    ),
                ],
            )
        ],
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(visible=False, range=[-1.0, 1.0], row=1, col=1)
    fig.update_xaxes(title_text="s [m]", range=[stations[0], stations[-1]], row=2, col=1)
    fig.update_yaxes(title_text="x [mm]", range=y_range, row=2, col=1)
    fig.update_xaxes(title_text="δ [%]", range=_padded_range(delta_percent, minimum_span=0.01), row=2, col=2)
    fig.update_yaxes(title_text="x [mm]", range=y_range, row=2, col=2)
    add_lattice_strip(fig, result.layout, xref="x", yref="y", y=0.0, height=0.50)
    return _show_or_return(fig, show)


def plot_dispersion_state_portrait(
    result: OpticsResult,
    *,
    reference: OpticsResult | None = None,
    current_label: str = "current",
    reference_label: str = "reference",
    eta_range: Sequence[float] | None = None,
    etap_range: Sequence[float] | None = None,
    autorange_portrait: bool = False,
    title: str = "Dispersion state through the line",
    show: bool = True,
):
    """Show ``D_x`` and ``D_x'`` separately and link them to their state portrait.

    ``autorange_portrait`` fits only the right-hand portrait to the displayed
    curves while leaving any supplied longitudinal-panel ranges fixed.
    """
    df = result.table
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[[{}, {"rowspan": 3}], [{}, None], [{}, None]],
        row_heights=[0.15, 0.425, 0.425],
        column_widths=[0.62, 0.38],
        shared_xaxes="columns",
        vertical_spacing=0.04,
        horizontal_spacing=0.10,
        subplot_titles=("", "Dispersion-state portrait", "Dₓ(s)", "Dₓ′(s)"),
    )

    if reference is not None:
        ref = reference.table
        fig.add_trace(
            go.Scatter(
                x=ref["s_m"],
                y=ref["eta_x_m"],
                mode="lines",
                line=dict(color="rgba(115, 125, 140, 0.55)", dash="dash"),
                name=reference_label,
                legendgroup="reference",
                hovertemplate=(
                    f"<b>{reference_label}</b><br>"
                    "s = %{x:.2f} m<br>Dₓ = %{y:.3f} m<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ref["s_m"],
                y=ref["eta_xp"],
                mode="lines",
                line=dict(color="rgba(115, 125, 140, 0.55)", dash="dash"),
                name=reference_label,
                legendgroup="reference",
                showlegend=False,
                hovertemplate=(
                    f"<b>{reference_label}</b><br>"
                    "s = %{x:.2f} m<br>Dₓ′ = %{y:.3f}<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ref["eta_x_m"],
                y=ref["eta_xp"],
                mode="lines",
                line=dict(color="rgba(115, 125, 140, 0.55)", dash="dash"),
                name=reference_label,
                legendgroup="reference",
                showlegend=False,
                hovertemplate=(
                    f"<b>{reference_label}</b><br>"
                    "Dₓ = %{x:.3f} m<br>Dₓ′ = %{y:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    fig.add_trace(
        go.Scatter(
            x=df["s_m"],
            y=df["eta_x_m"],
            mode="lines",
            line=dict(color="#1f77b4", width=2.2),
            name=current_label,
            legendgroup="current",
            hovertemplate=(
                f"<b>{current_label}</b><br>"
                "s = %{x:.2f} m<br>Dₓ = %{y:.3f} m<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["s_m"],
            y=df["eta_xp"],
            mode="lines",
            line=dict(color="#ff7f0e", width=2.2),
            name=current_label,
            legendgroup="current",
            showlegend=False,
            hovertemplate=(
                f"<b>{current_label}</b><br>"
                "s = %{x:.2f} m<br>Dₓ′ = %{y:.3f}<extra></extra>"
            ),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["eta_x_m"],
            y=df["eta_xp"],
            mode="lines",
            line=dict(color="rgba(80, 90, 105, 0.55)", width=1.5),
            name=current_label,
            legendgroup="current",
            showlegend=False,
            hovertemplate=(
                f"<b>{current_label}</b><br>"
                "Dₓ = %{x:.3f} m<br>Dₓ′ = %{y:.3f}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df["eta_x_m"],
            y=df["eta_xp"],
            mode="markers",
            marker=dict(size=5, color=df["s_m"], coloraxis="coloraxis"),
            name="position along line",
            showlegend=False,
            hovertemplate=(
                "<b>Position along line</b><br>"
                "s = %{marker.color:.2f} m<br>"
                "Dₓ = %{x:.3f} m<br>Dₓ′ = %{y:.3f}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0],
            y=[0.0],
            mode="markers",
            marker=dict(
                symbol="circle-open",
                size=12,
                color="rgba(45, 55, 70, 0.9)",
                line=dict(width=2),
            ),
            name="achromat target",
            hovertemplate=(
                "<b>Achromat target</b><br>"
                "Dₓ = %{x:.2f} m<br>Dₓ′ = %{y:.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[df["eta_x_m"].iloc[-1]],
            y=[df["eta_xp"].iloc[-1]],
            mode="markers",
            marker=dict(
                symbol="x",
                size=11,
                color="rgba(45, 55, 70, 0.9)",
                line=dict(width=2),
            ),
            name="line exit",
            hovertemplate=(
                "<b>Line exit</b><br>"
                "Dₓ = %{x:.2f} m<br>"
                "Dₓ′ = %{y:.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        autosize=True,
        width=None,
        height=570,
        margin=dict(t=85, r=120, b=90),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.14, yanchor="top"),
        coloraxis=dict(colorscale="Viridis", cmin=float(df["s_m"].min()), cmax=float(df["s_m"].max()), colorbar=dict(title="s [m]", len=0.62, y=0.56, x=1.03, thickness=14)),
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(visible=False, range=[-1.0, 1.0], row=1, col=1)
    fig.update_xaxes(title_text="s [m]", row=3, col=1)
    fig.update_yaxes(title_text="Dₓ [m]", row=2, col=1)
    fig.update_yaxes(title_text="Dₓ′ [1]", row=3, col=1)
    fig.update_xaxes(title_text="Dₓ [m]", row=1, col=2)
    fig.update_yaxes(title_text="Dₓ′ [1]", row=1, col=2)
    if eta_range is not None:
        eta_range = [float(value) for value in eta_range]
        fig.update_yaxes(range=eta_range, row=2, col=1)
        if not autorange_portrait:
            fig.update_xaxes(range=eta_range, row=1, col=2)
    if etap_range is not None:
        etap_range = [float(value) for value in etap_range]
        fig.update_yaxes(range=etap_range, row=3, col=1)
        if not autorange_portrait:
            fig.update_yaxes(range=etap_range, row=1, col=2)
    if autorange_portrait:
        fig.update_xaxes(autorange=True, row=1, col=2)
        fig.update_yaxes(autorange=True, row=1, col=2)
    fig.add_hline(y=0.0, line=dict(color="rgba(100, 110, 125, 0.45)", width=1), row=2, col=1)
    fig.add_hline(y=0.0, line=dict(color="rgba(100, 110, 125, 0.45)", width=1), row=3, col=1)
    fig.add_hline(y=0.0, line=dict(color="rgba(100, 110, 125, 0.35)", width=1), row=1, col=2)
    fig.add_vline(x=0.0, line=dict(color="rgba(100, 110, 125, 0.35)", width=1), row=1, col=2)
    add_lattice_strip(fig, result.layout, xref="x", yref="y", y=0.0, height=0.50)
    return _show_or_return(fig, show)


def plot_aperture_ribbon(
    result: OpticsResult,
    *,
    sigma_delta: float,
    pipe_radius_m: float = PIPE_RADIUS_DEFAULT,
    n_sigma: float = 1.0,
    emit_x: float = GEOMETRIC_EMITTANCE,
    tracks: FirstOrderParticleTracks | None = None,
    n_particles: int = 120,
    seed: int = 2026,
    sample_step_m: float = 0.05,
    fixed_y_extent_mm: float | None = None,
    fixed_clearance_range_mm: Sequence[float] | None = None,
    title: str = "Horizontal particle ribbon and aperture clearance",
    show: bool = True,
):
    """Show sample particles, beam-size components, pipe walls, and clearance."""
    if pipe_radius_m <= 0:
        raise ValueError("pipe_radius_m must be positive")
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive")
    if tracks is None:
        tracks = first_order_particle_tracks(
            result,
            sigma_delta=sigma_delta,
            emit_x=emit_x,
            n_particles=n_particles,
            seed=seed,
            sample_step_m=sample_step_m,
        )
    sampled = _sample_horizontal_optics_at_s(result, tracks.s_m)
    sigma_beta_mm = 1e3 * np.sqrt(np.maximum(emit_x * sampled["beta_x_m"].to_numpy(), 0.0))
    sigma_dispersion_mm = 1e3 * np.abs(sampled["eta_x_m"].to_numpy() * sigma_delta)
    sigma_total_mm = np.sqrt(sigma_beta_mm**2 + sigma_dispersion_mm**2)
    envelope_beta = n_sigma * sigma_beta_mm
    envelope_total = n_sigma * sigma_total_mm
    pipe_mm = 1e3 * float(pipe_radius_m)
    clearance_mm = pipe_mm - envelope_total
    x_particles_mm = 1e3 * tracks.x_m
    max_abs_delta = max(float(np.max(np.abs(tracks.delta))), float(abs(sigma_delta)), 1e-15)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.13, 0.59, 0.28],
        vertical_spacing=0.045,
        subplot_titles=("", "Particle ribbon, beam envelope, and pipe", "Aperture clearance"),
    )
    fig.add_trace(go.Scatter(
        x=tracks.s_m,
        y=-envelope_total,
        mode="lines",
        line=dict(color="rgba(31, 119, 180, 0.35)", width=1),
        name="total envelope",
        legendgroup="total",
        showlegend=False,
        hovertemplate="s = %{x:.3f} m<br>lower total envelope x = %{y:.4f} mm<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=tracks.s_m,
        y=envelope_total,
        mode="lines",
        line=dict(color="rgba(31, 119, 180, 0.75)", width=1.8),
        fill="tonexty",
        fillcolor="rgba(31, 119, 180, 0.10)",
        name="total envelope",
        legendgroup="total",
        hovertemplate="s = %{x:.3f} m<br>upper total envelope x = %{y:.4f} mm<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=tracks.s_m,
        y=envelope_beta,
        mode="lines",
        line=dict(color="rgba(255, 127, 14, 0.9)", dash="dash", width=1.7),
        name="betatron-only envelope",
        legendgroup="beta",
        hovertemplate="s = %{x:.3f} m<br>upper betatron envelope x = %{y:.4f} mm<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=tracks.s_m,
        y=-envelope_beta,
        mode="lines",
        line=dict(color="rgba(255, 127, 14, 0.9)", dash="dash", width=1.7),
        name="betatron-only envelope",
        legendgroup="beta",
        showlegend=False,
        hovertemplate="s = %{x:.3f} m<br>lower betatron envelope x = %{y:.4f} mm<extra></extra>",
    ), row=2, col=1)
    envelope_dispersion = n_sigma * sigma_dispersion_mm
    fig.add_trace(go.Scatter(
        x=tracks.s_m,
        y=envelope_dispersion,
        mode="lines",
        line=dict(color="rgba(148, 103, 189, 0.9)", dash="dot", width=1.7),
        name="dispersive-only envelope",
        legendgroup="dispersion",
        hovertemplate="s = %{x:.3f} m<br>upper dispersive envelope x = %{y:.4f} mm<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=tracks.s_m,
        y=-envelope_dispersion,
        mode="lines",
        line=dict(color="rgba(148, 103, 189, 0.9)", dash="dot", width=1.7),
        name="dispersive-only envelope",
        legendgroup="dispersion",
        showlegend=False,
        hovertemplate="s = %{x:.3f} m<br>lower dispersive envelope x = %{y:.4f} mm<extra></extra>",
    ), row=2, col=1)

    for particle_index, delta in enumerate(tracks.delta):
        fig.add_trace(
            go.Scatter(
                x=tracks.s_m,
                y=x_particles_mm[particle_index],
                mode="lines",
                line=dict(color=_delta_color(delta, max_abs_delta), width=0.7),
                opacity=0.06,
                showlegend=False,
                hovertemplate=(
                    f"particle {int(tracks.particle_id[particle_index])}<br>"
                    f"δ = {100.0 * float(delta):.4f}%<br>"
                    "s = %{x:.3f} m<br>"
                    "x = %{y:.4f} mm<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=np.full(len(tracks.delta), tracks.s_m[-1]),
            y=x_particles_mm[:, -1],
            mode="markers",
            marker=dict(size=3.5, color=100.0 * tracks.delta, coloraxis="coloraxis", opacity=0.20),
            name="sample particles",
            customdata=tracks.particle_id,
            hovertemplate=(
                "particle %{customdata}<br>"
                "δ = %{marker.color:.4f}%<br>"
                "s = %{x:.3f} m<br>"
                "x = %{y:.4f} mm<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(go.Scatter(
        x=tracks.s_m,
        y=np.full(len(tracks.s_m), pipe_mm),
        mode="lines",
        line=dict(color="rgba(45, 55, 70, 0.9)", width=2.0),
        name="pipe wall",
        legendgroup="pipe",
        hovertemplate="s = %{x:.3f} m<br>upper pipe wall x = %{y:.3f} mm<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=tracks.s_m,
        y=np.full(len(tracks.s_m), -pipe_mm),
        mode="lines",
        line=dict(color="rgba(45, 55, 70, 0.9)", width=2.0),
        name="pipe wall",
        legendgroup="pipe",
        showlegend=False,
        hovertemplate="s = %{x:.3f} m<br>lower pipe wall x = %{y:.3f} mm<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=tracks.s_m,
        y=clearance_mm,
        mode="lines",
        line=dict(color="#1f77b4", width=2.2),
        name="clearance",
        hovertemplate="s = %{x:.3f} m<br>clearance = %{y:.4f} mm<extra></extra>",
    ), row=3, col=1)

    # Keep the view centered on the pipe and analytical envelope. Tracked tails
    # may clip; an explicitly supplied fixed view can also clip very large
    # envelopes so ordinary states remain readable across widget snapshots.
    y_extent = (
        float(fixed_y_extent_mm)
        if fixed_y_extent_mm is not None
        else 1.12 * max(pipe_mm, float(np.max(envelope_total)))
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        autosize=True,
        width=None,
        height=620,
        margin=dict(t=90, r=110, b=100),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.17, yanchor="top"),
        coloraxis=dict(
            colorscale=DELTA_COLORSCALE,
            cmin=-100.0 * max_abs_delta,
            cmax=100.0 * max_abs_delta,
            cmid=0.0,
            colorbar=dict(title="δ (%)", len=0.54, y=0.52, x=1.03, thickness=14),
        ),
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(visible=False, range=[-1.0, 1.0], row=1, col=1)
    fig.update_yaxes(title_text="x [mm]", range=[-y_extent, y_extent], row=2, col=1)
    fig.update_xaxes(title_text="s [m]", range=[tracks.s_m[0], tracks.s_m[-1]], row=3, col=1)
    clearance_range = (
        [float(value) for value in fixed_clearance_range_mm]
        if fixed_clearance_range_mm is not None
        else _padded_range(clearance_mm, minimum_span=1.0)
    )
    fig.update_yaxes(title_text="clearance [mm]", range=clearance_range, row=3, col=1)
    fig.add_hline(y=0.0, line=dict(color="rgba(70, 80, 95, 0.7)", dash="dot", width=1.3), row=3, col=1)
    add_lattice_strip(fig, result.layout, xref="x", yref="y", y=0.0, height=0.50)
    return _show_or_return(fig, show)


def _local_tune_footprint_ranges(
    scan: pd.DataFrame,
    qx0: float,
    qy0: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return fixed local tune-plane ranges for a prepared scan."""
    stable = scan.loc[scan["stable"]].copy()
    qx = np.r_[stable["Qx"].to_numpy(dtype=float), float(qx0)]
    qy = np.r_[stable["Qy"].to_numpy(dtype=float), float(qy0)]
    x_range = tuple(_padded_range(qx, minimum_span=0.16, fraction=0.18))
    y_range = tuple(_padded_range(qy, minimum_span=0.20, fraction=0.18))
    return x_range, y_range


def plot_tune_scan_and_footprint(
    elements: Sequence[Element] | pd.DataFrame,
    *,
    n_cells: int = DBA_N_CELLS_DEFAULT,
    sigma_delta: float = SIGMA_DELTA_DEFAULT,
    delta_range: tuple[float, float] | None = None,
    n_delta: int = 81,
    ddelta: float = 1e-4,
    resonance_order: int = 3,
    title: str = "Momentum offset, tune shift, and resonance footprint",
    show: bool = True,
):
    """Link a tune scan to the same momentum-colored resonance footprint.

    Passing a prepared DataFrame uses those Xsuite scan points throughout the
    display.  Passing local elements retains the legacy plotting API for older
    diagnostics, but Section C supplies an Xsuite DataFrame.
    """
    if sigma_delta < 0 or ddelta <= 0:
        raise ValueError("sigma_delta must be nonnegative and ddelta must be positive")
    if delta_range is None:
        extent = max(4.0 * float(sigma_delta), 10.0 * float(ddelta))
        delta_range = (-extent, extent)
    delta_min, delta_max = map(float, delta_range)
    if not delta_min < delta_max:
        raise ValueError("delta_range must increase")
    requested = np.linspace(delta_min, delta_max, int(n_delta))
    requested = np.unique(np.concatenate([requested, [-ddelta, 0.0, ddelta, -sigma_delta, sigma_delta]]))
    if isinstance(elements, pd.DataFrame):
        source_scan = _validated_tune_scan(elements)
        available_min = float(source_scan["delta"].min())
        available_max = float(source_scan["delta"].max())
        within_range = source_scan["delta"].between(delta_min, delta_max)
        requested_available = requested[(requested >= available_min) & (requested <= available_max)]
        interpolated = _interpolate_tune_scan(source_scan, requested_available)
        scan = pd.concat([source_scan.loc[within_range], interpolated], ignore_index=True)
        scan = scan.sort_values("delta").drop_duplicates("delta", keep="first").reset_index(drop=True)
        nominal = _interpolate_tune_scan(source_scan, [0.0]).iloc[0]
        qx0, qy0 = float(nominal["Qx"]), float(nominal["Qy"])
        range_scan = source_scan
        max_abs_delta = max(
            abs(float(source_scan["delta"].min())),
            abs(float(source_scan["delta"].max())),
            1e-15,
        )
    else:
        scan = tune_scan(elements, delta_values=requested, n_cells=n_cells)
        qx0, qy0 = ring_tunes(elements, n_cells=n_cells, delta=0.0)
        range_scan = scan
        max_abs_delta = max(abs(delta_min), abs(delta_max), 1e-15)
    stable = scan[scan["stable"]].copy()
    if stable.empty:
        raise ValueError("The ring is unstable over the requested momentum range")

    delta_percent = 100.0 * stable["delta"].to_numpy()
    raw_mask = np.isclose(stable["delta"].to_numpy()[:, None], np.array([-ddelta, 0.0, ddelta])[None, :], atol=1e-14).any(axis=1)
    raw = stable.loc[raw_mask]

    x_range, y_range = _local_tune_footprint_ranges(range_scan, qx0, qy0)
    segments = resonance_lines(resonance_order, x_range, y_range)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {"rowspan": 2}], [{}, None]],
        row_heights=[0.5, 0.5],
        column_widths=[0.48, 0.52],
        shared_xaxes="columns",
        vertical_spacing=0.09,
        horizontal_spacing=0.10,
        subplot_titles=("Horizontal tune", "Tune footprint", "Vertical tune"),
    )
    fig.add_trace(go.Scatter(
        x=100.0 * stable["delta"],
        y=stable["Qx"],
        mode="lines",
        line=dict(color="#1f77b4", width=2.2),
        name="Qₓ(δ)",
        showlegend=False,
        hovertemplate="δ = %{x:.4f}%<br>Qₓ = %{y:.8f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=100.0 * stable["delta"],
        y=stable["Qy"],
        mode="lines",
        line=dict(color="#ff7f0e", width=2.2),
        name="Qᵧ(δ)",
        showlegend=False,
        hovertemplate="δ = %{x:.4f}%<br>Qᵧ = %{y:.8f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(x=100.0 * raw["delta"], y=raw["Qx"], mode="markers", marker=dict(size=9, symbol="circle", color=100.0 * raw["delta"], coloraxis="coloraxis", line=dict(color="#1f77b4", width=1.5)), name="Qₓ finite-difference points", showlegend=False, hovertemplate="δ=%{x:.4f}%<br>Qₓ=%{y:.8f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=100.0 * raw["delta"], y=raw["Qy"], mode="markers", marker=dict(size=9, symbol="diamond", color=100.0 * raw["delta"], coloraxis="coloraxis", line=dict(color="#ff7f0e", width=1.5)), name="Qᵧ finite-difference points", showlegend=False, hovertemplate="δ=%{x:.4f}%<br>Qᵧ=%{y:.8f}<extra></extra>"), row=2, col=1)
    raw_sorted = raw.sort_values("delta")
    fig.add_trace(go.Scatter(
        x=100.0 * raw_sorted["delta"],
        y=raw_sorted["Qx"],
        mode="lines",
        line=dict(color="#1f77b4", dash="dot", width=3),
        name="Qₓ secant",
        showlegend=False,
        hovertemplate="δ = %{x:.4f}%<br>Qₓ secant = %{y:.8f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=100.0 * raw_sorted["delta"],
        y=raw_sorted["Qy"],
        mode="lines",
        line=dict(color="#ff7f0e", dash="dot", width=3),
        name="Qᵧ secant",
        showlegend=False,
        hovertemplate="δ = %{x:.4f}%<br>Qᵧ secant = %{y:.8f}<extra></extra>",
    ), row=2, col=1)

    order_styles = {
        1: ("rgba(80, 90, 105, 0.68)", "solid"),
        2: ("rgba(100, 110, 125, 0.52)", "dash"),
        3: ("rgba(120, 130, 145, 0.42)", "dot"),
        4: ("rgba(135, 145, 160, 0.34)", "dashdot"),
    }
    shown_orders: set[int] = set()
    for segment in segments:
        color, dash = order_styles.get(segment["order"], ("rgba(120, 130, 145, 0.35)", "dot"))
        fig.add_trace(
            go.Scatter(
                x=[segment["x0"], segment["x1"]],
                y=[segment["y0"], segment["y1"]],
                mode="lines",
                line=dict(color=color, dash=dash, width=1.2),
                name=f"order {segment['order']}",
                legendgroup=f"resonance-{segment['order']}",
                showlegend=segment["order"] not in shown_orders,
                hovertemplate=(
                    f"{segment['label']}<br>order {segment['order']}<br>"
                    "Qₓ = %{x:.6f}<br>Qᵧ = %{y:.6f}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
        shown_orders.add(segment["order"])

    fig.add_trace(
        go.Scatter(
            x=stable["Qx"],
            y=stable["Qy"],
            mode="lines+markers",
            line=dict(color="rgba(75, 85, 100, 0.55)", width=1.5),
            marker=dict(size=6, color=delta_percent, coloraxis="coloraxis"),
            name="chromatic footprint",
            hovertemplate=(
                "δ = %{marker.color:.4f}%<br>"
                "Qₓ = %{x:.8f}<br>Qᵧ = %{y:.8f}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(go.Scatter(
        x=[qx0],
        y=[qy0],
        mode="markers",
        marker=dict(symbol="star", size=12, color="rgba(45, 55, 70, 0.95)"),
        name="nominal tune",
        hovertemplate="δ = 0.0000%<br>Qₓ = %{x:.8f}<br>Qᵧ = %{y:.8f}<extra></extra>",
    ), row=1, col=2)

    if isinstance(elements, pd.DataFrame):
        source_scan = _validated_tune_scan(elements)
        if -sigma_delta >= source_scan["delta"].min() and sigma_delta <= source_scan["delta"].max():
            sigma_points = _interpolate_tune_scan(source_scan, [-sigma_delta, sigma_delta])
        else:
            sigma_points = pd.DataFrame(columns=["delta", "Qx", "Qy", "stable"])
    else:
        sigma_points = tune_scan(elements, delta_values=[-sigma_delta, sigma_delta], n_cells=n_cells)
        sigma_points = sigma_points[sigma_points["stable"]]
    fig.add_trace(go.Scatter(
        x=sigma_points["Qx"],
        y=sigma_points["Qy"],
        mode="markers",
        marker=dict(symbol="circle-open", size=11, color=100.0 * sigma_points["delta"], coloraxis="coloraxis", line=dict(width=2)),
        name="±σδ",
        hovertemplate=(
            "δ = %{marker.color:.4f}%<br>"
            "Qₓ = %{x:.8f}<br>Qᵧ = %{y:.8f}<extra></extra>"
        ),
    ), row=1, col=2)
    if sigma_delta > 0 and len(sigma_points) == 2:
        positive = sigma_points.loc[sigma_points["delta"].idxmax()]
        fig.add_annotation(x=positive["Qx"], y=positive["Qy"], text="+δ", showarrow=False, xshift=10, yshift=8, row=1, col=2)

    fig.update_layout(
        title=title,
        template="plotly_white",
        autosize=True,
        width=None,
        height=570,
        margin=dict(t=85, r=110, b=105),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18, yanchor="top"),
        coloraxis=dict(
            colorscale=DELTA_COLORSCALE,
            cmin=-100.0 * max_abs_delta,
            cmax=100.0 * max_abs_delta,
            cmid=0.0,
            colorbar=dict(title="δ (%)", len=0.70, y=0.50, x=1.03, thickness=14),
        ),
    )
    fig.update_yaxes(title_text="Qₓ", row=1, col=1)
    fig.update_xaxes(title_text="δ [%]", range=[100.0 * delta_min, 100.0 * delta_max], row=2, col=1)
    fig.update_yaxes(title_text="Qᵧ", row=2, col=1)
    fig.update_xaxes(title_text="Qₓ", range=x_range, row=1, col=2)
    fig.update_yaxes(title_text="Qᵧ", range=y_range, row=1, col=2)
    return _show_or_return(fig, show)


def plot_accumulated_phase_advance(
    result: OpticsResult,
    *,
    show_inverse_beta: bool = True,
    title: str = "Accumulated phase advance through one DBA-like cell",
    show: bool = True,
):
    """Connect the accumulated phase advance to its local ``1/beta`` rate."""
    phase = phase_advance_table(result)
    rows = 3 if show_inverse_beta else 2
    row_heights = [0.14, 0.54, 0.32] if show_inverse_beta else [0.18, 0.82]
    subplot_titles = ("", "Accumulated phase advance", "Local accumulation rate") if show_inverse_beta else ("", "Accumulated phase advance")
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.045, subplot_titles=subplot_titles)
    phase_row = 2
    fig.add_trace(go.Scatter(
        x=phase["s_m"],
        y=phase["psi_x_turns"],
        mode="lines",
        line=dict(color="#1f77b4", width=2.3),
        name="ψₓ / 2π",
        hovertemplate="ψₓ / 2π = %{y:.6f} turns<extra></extra>",
    ), row=phase_row, col=1)
    fig.add_trace(go.Scatter(
        x=phase["s_m"],
        y=phase["psi_y_turns"],
        mode="lines",
        line=dict(color="#ff7f0e", width=2.3),
        name="ψᵧ / 2π",
        hovertemplate="ψᵧ / 2π = %{y:.6f} turns<extra></extra>",
    ), row=phase_row, col=1)
    if show_inverse_beta:
        fig.add_trace(go.Scatter(
            x=phase["s_m"],
            y=phase["inv_beta_x_m^-1"] / (2.0 * math.pi),
            mode="lines",
            line=dict(color="#1f77b4", width=1.8),
            name="horizontal rate",
            showlegend=False,
            hovertemplate="d(ψₓ / 2π)/ds = %{y:.6f} m⁻¹<extra></extra>",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=phase["s_m"],
            y=phase["inv_beta_y_m^-1"] / (2.0 * math.pi),
            mode="lines",
            line=dict(color="#ff7f0e", width=1.8),
            name="vertical rate",
            showlegend=False,
            hovertemplate="d(ψᵧ / 2π)/ds = %{y:.6f} m⁻¹<extra></extra>",
        ), row=3, col=1)
    fig.update_layout(title=title, template="plotly_white", hovermode="x unified", autosize=True, width=None, height=600 if show_inverse_beta else 500, margin=dict(t=85))
    fig.update_xaxes(unifiedhovertitle=dict(text="s = %{x:.3f} m"))
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(visible=False, range=[-1.0, 1.0], row=1, col=1)
    fig.update_yaxes(title_text="phase / 2π", row=phase_row, col=1)
    fig.update_xaxes(title_text="s [m]", row=rows, col=1)
    if show_inverse_beta:
        fig.update_yaxes(title_text="d(phase / 2π)/ds [m⁻¹]", row=3, col=1)
    add_lattice_strip(fig, result.layout, xref="x", yref="y", y=0.0, height=0.50)
    return _show_or_return(fig, show)


def plot_xsuite_accumulated_phase_advance(
    ring_twiss,
    cell_elements: Sequence[Element],
    *,
    show_inverse_beta: bool = True,
    title: str = "Accumulated phase advance through one DBA-like cell",
    show: bool = True,
):
    """Plot one cell's phase advance directly from a full-ring Xsuite Twiss."""
    cell_length = float(sum(element.length for element in cell_elements))
    table = ring_twiss.to_pandas()
    phase = table.loc[table["s"] <= cell_length + 1e-12, ["s", "betx", "bety", "mux", "muy"]].copy()
    if phase.empty or phase["s"].iloc[-1] < cell_length - 1e-9:
        raise ValueError("Xsuite Twiss table does not span one complete cell")
    phase["mux"] -= float(phase["mux"].iloc[0])
    phase["muy"] -= float(phase["muy"].iloc[0])

    rows = 3 if show_inverse_beta else 2
    row_heights = [0.14, 0.54, 0.32] if show_inverse_beta else [0.18, 0.82]
    subplot_titles = ("", "Accumulated phase advance", "Local accumulation rate") if show_inverse_beta else ("", "Accumulated phase advance")
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.045, subplot_titles=subplot_titles)
    phase_row = 2
    fig.add_trace(go.Scatter(
        x=phase["s"],
        y=phase["mux"],
        mode="lines",
        line=dict(color="#1f77b4", width=2.3),
        name="ψₓ / 2π",
        hovertemplate="ψₓ / 2π = %{y:.6f} turns<extra></extra>",
    ), row=phase_row, col=1)
    fig.add_trace(go.Scatter(
        x=phase["s"],
        y=phase["muy"],
        mode="lines",
        line=dict(color="#ff7f0e", width=2.3),
        name="ψᵧ / 2π",
        hovertemplate="ψᵧ / 2π = %{y:.6f} turns<extra></extra>",
    ), row=phase_row, col=1)
    if show_inverse_beta:
        fig.add_trace(go.Scatter(
            x=phase["s"],
            y=1.0 / (2.0 * math.pi * phase["betx"]),
            mode="lines",
            line=dict(color="#1f77b4", width=1.8),
            name="horizontal rate",
            showlegend=False,
            hovertemplate="d(ψₓ / 2π)/ds = %{y:.6f} m⁻¹<extra></extra>",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=phase["s"],
            y=1.0 / (2.0 * math.pi * phase["bety"]),
            mode="lines",
            line=dict(color="#ff7f0e", width=1.8),
            name="vertical rate",
            showlegend=False,
            hovertemplate="d(ψᵧ / 2π)/ds = %{y:.6f} m⁻¹<extra></extra>",
        ), row=3, col=1)
    fig.update_layout(title=title, template="plotly_white", hovermode="x unified", autosize=True, width=None, height=600 if show_inverse_beta else 500, margin=dict(t=85))
    fig.update_xaxes(unifiedhovertitle=dict(text="s = %{x:.3f} m"))
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(visible=False, range=[-1.0, 1.0], row=1, col=1)
    fig.update_yaxes(title_text="phase / 2π", row=phase_row, col=1)
    fig.update_xaxes(title_text="s [m]", range=[0.0, cell_length], row=rows, col=1)
    if show_inverse_beta:
        fig.update_yaxes(title_text="d(phase / 2π)/ds [m⁻¹]", row=3, col=1)
    add_lattice_strip(fig, element_layout(cell_elements), xref="x", yref="y", y=0.0, height=0.50)
    return _show_or_return(fig, show)


def xsuite_dense_twiss_for_first_cell(
    ring_line,
    cell_elements: Sequence[Element],
    *,
    n_points: int = 241,
):
    """Add uniform first-cell observation points and recompute periodic Twiss."""
    if int(n_points) < 2:
        raise ValueError("n_points must be at least 2")
    cell_length = float(sum(element.length for element in cell_elements))
    sampled = ring_line.copy(shallow=True)
    s_grid = np.linspace(0.0, cell_length, int(n_points))
    with (
        warnings.catch_warnings(),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        warnings.simplefilter("ignore")
        sampled.cut_at_s(s_grid)
        return sampled.twiss(method="4d")

def plot_optics(
    result: OpticsResult,
    title: str = "Optics",
    show: bool = True,
    show_lattice: bool = True,
    show_eta_prime: bool = False,
):
    df = result.table
    if show_lattice:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.16, 0.46, 0.46],
            vertical_spacing=0.045,
            subplot_titles=("", "Beta functions", "Horizontal dispersion"),
        )
        beta_row = 2
        dispersion_row = 3
    else:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Beta functions", "Horizontal dispersion"),
        )
        beta_row = 1
        dispersion_row = 2

    element_data = df["element"].astype(str).to_numpy()
    fig.add_trace(go.Scatter(
        x=df["s_m"],
        y=df["beta_x_m"],
        mode="lines",
        name="βₓ",
        customdata=element_data,
        hovertemplate="%{customdata}<br>βₓ = %{y:.5g} m<extra></extra>",
    ), row=beta_row, col=1)
    fig.add_trace(go.Scatter(
        x=df["s_m"],
        y=df["beta_y_m"],
        mode="lines",
        name="βᵧ",
        customdata=element_data,
        hovertemplate="%{customdata}<br>βᵧ = %{y:.5g} m<extra></extra>",
    ), row=beta_row, col=1)
    fig.add_trace(go.Scatter(
        x=df["s_m"],
        y=df["eta_x_m"],
        mode="lines",
        name="Dₓ",
        customdata=element_data,
        hovertemplate="%{customdata}<br>Dₓ = %{y:.5g} m<extra></extra>",
    ), row=dispersion_row, col=1)
    if show_eta_prime:
        fig.add_trace(go.Scatter(
            x=df["s_m"],
            y=df["eta_xp"],
            mode="lines",
            name="Dₓ′",
            customdata=element_data,
            hovertemplate="%{customdata}<br>Dₓ′ = %{y:.5g}<extra></extra>",
        ), row=dispersion_row, col=1)
    fig.update_xaxes(title_text="s [m]", row=dispersion_row, col=1)
    fig.update_yaxes(title_text="β [m]", row=beta_row, col=1)
    fig.update_yaxes(title_text="Dₓ [m]", row=dispersion_row, col=1)
    fig.update_layout(title=title, template="plotly_white", hovermode="x unified", autosize=True, width=None, height=620)
    fig.update_xaxes(unifiedhovertitle=dict(text="s = %{x:.4g} m"))
    if show_lattice:
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
        fig.update_yaxes(visible=False, range=[-1.0, 1.0], row=1, col=1)
        add_lattice_strip(fig, result.layout, yref="y", y=0.0, height=0.48)
    return _show_or_return(fig, show)


def plot_beam_size(result: OpticsResult, sigma_delta: float = SIGMA_DELTA_DEFAULT, title: str = "RMS beam size", show: bool = True, show_lattice: bool = True):
    df = add_beam_size_columns(result.table, sigma_delta=sigma_delta)
    fig = go.Figure()
    element_data = df["element"].astype(str).to_numpy()
    fig.add_trace(go.Scatter(
        x=df["s_m"], y=df["sigma_x_mm"], mode="lines", name="σₓ total",
        customdata=element_data,
        hovertemplate="%{customdata}<br>σₓ total = %{y:.5g} mm<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["s_m"], y=df["sigma_x_beta_mm"], mode="lines", name="σₓ betatron only",
        customdata=element_data,
        hovertemplate="%{customdata}<br>σₓ betatron = %{y:.5g} mm<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["s_m"], y=df["sigma_y_mm"], mode="lines", name="σᵧ",
        customdata=element_data,
        hovertemplate="%{customdata}<br>σᵧ = %{y:.5g} mm<extra></extra>",
    ))
    fig.update_layout(title=f"{title} (σδ={sigma_delta:g})", xaxis_title="s [m]", yaxis_title="rms size [mm]", template="plotly_white", autosize=True, width=None, height=430, hovermode="x unified")
    fig.update_xaxes(unifiedhovertitle=dict(text="s = %{x:.4g} m"))
    if show_lattice:
        add_lattice_strip(fig, result.layout)
    return _show_or_return(fig, show)


def plot_optics_and_beam_size(
    result: OpticsResult,
    sigma_delta: float = SIGMA_DELTA_DEFAULT,
    title: str = "Optics and RMS beam size",
    show: bool = True,
    show_lattice: bool = True,
    show_eta_prime: bool = False,
    fixed_ranges: Mapping[str, Sequence[float]] | None = None,
):
    optics = result.table
    beam = add_beam_size_columns(result.table, sigma_delta=sigma_delta)
    if show_lattice:
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.11, 0.34, 0.20, 0.35],
            vertical_spacing=0.04,
            subplot_titles=("", "Beta functions", "Horizontal dispersion", f"RMS beam size, σδ={sigma_delta:g}"),
        )
        beta_row = 2
        dispersion_row = 3
        beam_row = 4
    else:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.36, 0.32, 0.32],
            vertical_spacing=0.06,
            subplot_titles=("Beta functions", "Horizontal dispersion", f"RMS beam size, σδ={sigma_delta:g}"),
        )
        beta_row = 1
        dispersion_row = 2
        beam_row = 3

    optics_elements = optics["element"].astype(str).to_numpy()
    fig.add_trace(go.Scatter(
        x=optics["s_m"], y=optics["beta_x_m"], mode="lines", name="βₓ",
        customdata=optics_elements,
        hovertemplate="%{customdata}<br>βₓ = %{y:.5g} m<extra></extra>",
    ), row=beta_row, col=1)
    fig.add_trace(go.Scatter(
        x=optics["s_m"], y=optics["beta_y_m"], mode="lines", name="βᵧ",
        customdata=optics_elements,
        hovertemplate="%{customdata}<br>βᵧ = %{y:.5g} m<extra></extra>",
    ), row=beta_row, col=1)
    fig.add_trace(go.Scatter(
        x=optics["s_m"], y=optics["eta_x_m"], mode="lines", name="Dₓ",
        customdata=optics_elements,
        hovertemplate="%{customdata}<br>Dₓ = %{y:.5g} m<extra></extra>",
    ), row=dispersion_row, col=1)
    if show_eta_prime:
        fig.add_trace(go.Scatter(
            x=optics["s_m"], y=optics["eta_xp"], mode="lines", name="Dₓ′",
            customdata=optics_elements,
            hovertemplate="%{customdata}<br>Dₓ′ = %{y:.5g}<extra></extra>",
        ), row=dispersion_row, col=1)

    beam_elements = beam["element"].astype(str).to_numpy()
    show_separate_betatron = not np.allclose(
        beam["sigma_x_mm"].to_numpy(),
        beam["sigma_x_beta_mm"].to_numpy(),
        rtol=1e-10,
        atol=1e-12,
    )
    sigma_x_name = "σₓ total" if show_separate_betatron else "σₓ (betatron = total)"
    fig.add_trace(go.Scatter(
        x=beam["s_m"], y=beam["sigma_x_mm"], mode="lines", name=sigma_x_name,
        customdata=beam_elements,
        hovertemplate="%{customdata}<br>σₓ total = %{y:.5g} mm<extra></extra>",
    ), row=beam_row, col=1)
    if show_separate_betatron:
        fig.add_trace(go.Scatter(
            x=beam["s_m"], y=beam["sigma_x_beta_mm"], mode="lines", name="σₓ betatron only",
            customdata=beam_elements,
            hovertemplate="%{customdata}<br>σₓ betatron = %{y:.5g} mm<extra></extra>",
        ), row=beam_row, col=1)
    fig.add_trace(go.Scatter(
        x=beam["s_m"], y=beam["sigma_y_mm"], mode="lines", name="σᵧ",
        customdata=beam_elements,
        hovertemplate="%{customdata}<br>σᵧ = %{y:.5g} mm<extra></extra>",
    ), row=beam_row, col=1)

    fig.update_xaxes(title_text="s [m]", row=beam_row, col=1)
    fig.update_yaxes(title_text="β [m]", row=beta_row, col=1)
    fig.update_yaxes(title_text="Dₓ [m]", row=dispersion_row, col=1)
    fig.update_yaxes(title_text="rms size [mm]", row=beam_row, col=1)
    if fixed_ranges:
        if "beta" in fixed_ranges:
            fig.update_yaxes(range=[float(value) for value in fixed_ranges["beta"]], row=beta_row, col=1)
        if "dispersion" in fixed_ranges:
            fig.update_yaxes(range=[float(value) for value in fixed_ranges["dispersion"]], row=dispersion_row, col=1)
        if "beam" in fixed_ranges:
            fig.update_yaxes(range=[float(value) for value in fixed_ranges["beam"]], row=beam_row, col=1)
    fig.update_layout(title=title, template="plotly_white", hovermode="x unified", autosize=True, width=None, height=720)
    fig.update_xaxes(unifiedhovertitle=dict(text="s = %{x:.4g} m"))
    if show_lattice:
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
        fig.update_yaxes(visible=False, range=[-1.0, 1.0], row=1, col=1)
        add_lattice_strip(fig, result.layout, yref="y", y=0.0, height=0.48)
    return _show_or_return(fig, show)


def plot_beam_size_with_aperture(
    result: OpticsResult,
    sigma_delta: float,
    pipe_radius_m: float = PIPE_RADIUS_DEFAULT,
    n_sigma: float = 1.0,
    show: bool = True,
    show_lattice: bool = True,
    fixed_y_range: Sequence[float] | None = None,
):
    df = add_beam_size_columns(result.table, sigma_delta=sigma_delta)
    envelope_mm = n_sigma * df["sigma_x_mm"]
    betatron_envelope_mm = n_sigma * df["sigma_x_beta_mm"]
    fig = go.Figure()
    dispersive_envelope_mm = n_sigma * df["sigma_x_dispersion_mm"]
    fig.add_trace(go.Scatter(
        x=df["s_m"], y=betatron_envelope_mm, mode="lines", name=f"{n_sigma:g}σ betatron-only", line=dict(dash="dash"),
        hovertemplate=f"{n_sigma:g}σ betatron envelope = %{{y:.5g}} mm<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["s_m"], y=dispersive_envelope_mm, mode="lines", name=f"{n_sigma:g}σ dispersive-only", line=dict(dash="dot"),
        hovertemplate=f"{n_sigma:g}σ dispersive envelope = %{{y:.5g}} mm<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["s_m"], y=envelope_mm, mode="lines", name=f"{n_sigma:g}σ total envelope",
        hovertemplate=f"{n_sigma:g}σ total envelope = %{{y:.5g}} mm<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["s_m"], y=np.full(len(df), 1e3 * pipe_radius_m), mode="lines", name="pipe radius",
        hovertemplate="pipe radius = %{y:.5g} mm<extra></extra>",
    ))
    fig.update_layout(title=f"Horizontal RMS envelope versus aperture (σδ={sigma_delta:g})", xaxis_title="s [m]", yaxis_title="radius [mm]", template="plotly_white", hovermode="x unified", autosize=True, width=None, height=430)
    fig.update_xaxes(unifiedhovertitle=dict(text="s = %{x:.4g} m"))
    if fixed_y_range is not None:
        fig.update_yaxes(range=[float(value) for value in fixed_y_range])
    if show_lattice:
        add_lattice_strip(fig, result.layout)
    return _show_or_return(fig, show)


def plot_q1_scan(scan: pd.DataFrame, title: str = "Endpoint dispersion versus Q1", show: bool = True):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Exit position response", "Exit angle response"),
    )
    fig.add_trace(go.Scatter(
        x=scan["q1_m^-2"], y=scan["eta_x_end_m"], mode="lines", name="Dₓ at exit", showlegend=False,
        line=dict(color="#1f77b4", width=2.2),
        hovertemplate="Dₓ at exit = %{y:.6f} m<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=scan["q1_m^-2"], y=scan["eta_xp_end"], mode="lines", name="Dₓ′ at exit", showlegend=False,
        line=dict(color="#ff7f0e", width=2.2),
        hovertemplate="Dₓ′ at exit = %{y:.6f}<extra></extra>",
    ), row=2, col=1)
    fig.update_xaxes(title_text="Q1 k1 [m^-2]", row=2, col=1)
    fig.update_yaxes(title_text="Dₓ [m]", row=1, col=1)
    fig.update_yaxes(title_text="Dₓ′ [1]", row=2, col=1)
    fig.add_hline(y=0.0, line=dict(color="rgba(90, 100, 115, 0.45)", width=1), row=1, col=1)
    fig.add_hline(y=0.0, line=dict(color="rgba(90, 100, 115, 0.45)", width=1), row=2, col=1)
    fig.update_layout(title=title, template="plotly_white", hovermode="x unified", autosize=True, width=None, height=540, showlegend=False)
    fig.update_xaxes(unifiedhovertitle=dict(text="Q1 k₁ = %{x:.4g} m⁻²"))
    return _show_or_return(fig, show)


def plot_tune_footprint(nux: float, nuy: float, cx: float, cy: float, sigma_delta: float = SIGMA_DELTA_DEFAULT, resonance_order: int = 3, show: bool = True):
    xmin = math.floor(nux)
    ymin = math.floor(nuy)
    x_range = (xmin - 0.02, xmin + 1.02)
    y_range = (ymin - 0.02, ymin + 1.02)
    segments = resonance_lines(resonance_order, x_range, y_range)
    fig = go.Figure()
    for seg in segments:
        fig.add_trace(
            go.Scatter(
                x=[seg["x0"], seg["x1"]],
                y=[seg["y0"], seg["y1"]],
                mode="lines",
                name=f"order {seg['order']}",
                legendgroup=f"order {seg['order']}",
                showlegend=not any(trace.name == f"order {seg['order']}" for trace in fig.data),
                hovertemplate=(
                    f"{seg['label']}<br>order {seg['order']}<br>"
                    "Qₓ = %{x:.6f}<br>Qᵧ = %{y:.6f}<extra></extra>"
                ),
            )
        )

    deltas = np.array([-sigma_delta, 0.0, sigma_delta])
    footprint_x = nux + cx * deltas
    footprint_y = nuy + cy * deltas
    color_extent = max(100.0 * abs(sigma_delta), 1e-12)
    fig.add_trace(go.Scatter(
        x=[nux], y=[nuy], mode="markers", name="nominal tune",
        marker=dict(symbol="star", size=11, color="rgba(45, 55, 70, 0.95)"),
        hovertemplate="δ = 0.0000%<br>Qₓ = %{x:.8f}<br>Qᵧ = %{y:.8f}<extra></extra>",
    ))
    fig.add_trace(
        go.Scatter(
            x=footprint_x,
            y=footprint_y,
            mode="lines+markers",
            name="chromatic footprint",
            line=dict(color="rgba(75, 85, 100, 0.65)", width=2.5),
            marker=dict(size=8, color=100.0 * deltas, coloraxis="coloraxis"),
            hovertemplate=(
                "δ = %{marker.color:.4f}%<br>"
                "Qₓ = %{x:.8f}<br>Qᵧ = %{y:.8f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Tune footprint, resonance order <= {resonance_order}",
        xaxis_title="Qₓ",
        yaxis_title="Qᵧ",
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
        template="plotly_white",
        width=760,
        height=680,
        coloraxis=dict(colorscale=DELTA_COLORSCALE, cmin=-color_extent, cmax=color_extent, cmid=0.0, colorbar=dict(title="δ (%)")),
    )
    return _show_or_return(fig, show)


def plot_tune_footprint_from_scan(
    scan: pd.DataFrame,
    *,
    sigma_delta: float = SIGMA_DELTA_DEFAULT,
    resonance_order: int = 3,
    show: bool = True,
):
    """Plot the direct tune-scan locus within ``+-sigma_delta``."""
    if sigma_delta < 0:
        raise ValueError("sigma_delta must be nonnegative")
    clean = _validated_tune_scan(scan)
    nominal = _interpolate_tune_scan(clean, [0.0]).iloc[0]
    qx0, qy0 = float(nominal["Qx"]), float(nominal["Qy"])
    selected = clean[clean["delta"].between(-sigma_delta, sigma_delta)].copy()
    endpoints = []
    if -sigma_delta >= clean["delta"].min() and sigma_delta <= clean["delta"].max():
        endpoints = [_interpolate_tune_scan(clean, [-sigma_delta, sigma_delta])]
    footprint = pd.concat([selected, *endpoints], ignore_index=True)
    footprint = footprint.sort_values("delta").drop_duplicates("delta", keep="first")

    x_range, y_range = _local_tune_footprint_ranges(clean, qx0, qy0)
    segments = resonance_lines(resonance_order, x_range, y_range)
    fig = go.Figure()
    order_styles = {
        1: ("rgba(80, 90, 105, 0.68)", "solid"),
        2: ("rgba(100, 110, 125, 0.52)", "dash"),
        3: ("rgba(120, 130, 145, 0.42)", "dot"),
        4: ("rgba(135, 145, 160, 0.34)", "dashdot"),
    }
    for segment in segments:
        color, dash = order_styles.get(segment["order"], ("rgba(120, 130, 145, 0.35)", "dot"))
        fig.add_trace(
            go.Scatter(
                x=[segment["x0"], segment["x1"]],
                y=[segment["y0"], segment["y1"]],
                mode="lines",
                line=dict(color=color, dash=dash, width=1.2),
                name=f"order {segment['order']}",
                legendgroup=f"order {segment['order']}",
                showlegend=not any(trace.name == f"order {segment['order']}" for trace in fig.data),
                hovertemplate=(
                    f"{segment['label']}<br>order {segment['order']}<br>"
                    "Qₓ = %{x:.6f}<br>Qᵧ = %{y:.6f}<extra></extra>"
                ),
            )
        )

    color_extent = max(
        100.0 * abs(float(clean["delta"].min())),
        100.0 * abs(float(clean["delta"].max())),
        1e-12,
    )
    fig.add_trace(go.Scatter(
        x=[qx0], y=[qy0], mode="markers", name="nominal tune",
        marker=dict(symbol="star", size=11, color="rgba(45, 55, 70, 0.95)"),
        hovertemplate="δ = 0.0000%<br>Qₓ = %{x:.8f}<br>Qᵧ = %{y:.8f}<extra></extra>",
    ))
    fig.add_trace(
        go.Scatter(
            x=footprint["Qx"],
            y=footprint["Qy"],
            mode="lines+markers",
            name="chromatic footprint",
            line=dict(color="rgba(75, 85, 100, 0.65)", width=2.5),
            marker=dict(size=8, color=100.0 * footprint["delta"], coloraxis="coloraxis"),
            hovertemplate="δ=%{marker.color:.4f}%<br>Qₓ=%{x:.8f}<br>Qᵧ=%{y:.8f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Xsuite tune footprint, resonance order <= {resonance_order}",
        xaxis_title="Qₓ",
        yaxis_title="Qᵧ",
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
        template="plotly_white",
        autosize=True,
        width=None,
        height=600,
        margin=dict(t=85, r=110, b=105),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18, yanchor="top"),
        coloraxis=dict(
            colorscale=DELTA_COLORSCALE,
            cmin=-color_extent,
            cmax=color_extent,
            cmid=0.0,
            colorbar=dict(title="δ (%)", x=1.03, thickness=14),
        ),
    )
    return _show_or_return(fig, show)


# ---------------------------------------------------------------------------
# Optional interactive widgets
# ---------------------------------------------------------------------------


def _get_widgets():
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except Exception:
        return None, None, None
    display_widget_slider_css()
    return widgets, display, clear_output


def fodo_reference_from_xsuite_line(line) -> Lattice:
    return elements_from_xsuite_line(
        line,
        roles={
            "QFa": "focusing quadrupole half",
            "QD": "defocusing quadrupole",
            "QFb": "focusing quadrupole half",
        },
        occurrence_names=["QFa", "D1", "QD", "D2", "QFb"],
    )


def display_fodo_reference_from_xsuite_line(line, *, show_plot: bool = True) -> tuple[Lattice, OpticsResult]:
    """Convert, solve, and display the reference Xsuite FODO line."""
    elements = fodo_reference_from_xsuite_line(line)
    result = compute_periodic_optics(elements)
    plot_optics_and_beam_size(
        result,
        sigma_delta=0.0,
        title="Figure A1 — Reference FODO cell: periodic optics and beam size",
        show=show_plot,
        show_eta_prime=False,
    )
    centers = table_at_element_centers(result, ["QFa", "QD", "QFb"], sigma_delta=0.0)
    print("Table A1 — Reference FODO beam sizes at quadrupole centers (σδ = 0)")
    _maybe_display(
        compact_table(
            centers,
            columns=["element", "sigma_x_mm", "sigma_y_mm"],
            labels={"sigma_x_mm": "σₓ (mm)", "sigma_y_mm": "σᵧ (mm)"},
        )
    )
    return elements, result


def fodo_bend_from_xsuite_line(line) -> Lattice:
    return elements_from_xsuite_line(
        line,
        roles={
            "QFa": "focusing quadrupole half",
            "B1": "first 10-degree bend",
            "QD": "defocusing quadrupole",
            "B2": "second 10-degree bend",
            "QFb": "focusing quadrupole half",
        },
        occurrence_names=["QFa", "D1a", "B1", "D1b", "QD", "D2a", "B2", "D2b", "QFb"],
    )


def display_fodo_bend_from_xsuite_line(line, *, show_plot: bool = True) -> tuple[Lattice, OpticsResult]:
    """Convert, solve, and plot the symmetric two-bend Xsuite FODO line."""
    elements = fodo_bend_from_xsuite_line(line)
    result = compute_periodic_optics(elements)
    plot_optics(
        result,
        title="Figure A2 — Symmetric two-bend FODO cell: periodic optics and dispersion",
        show=show_plot,
        show_eta_prime=False,
    )
    return elements, result


def dba_insert_from_xsuite_line(line) -> Lattice:
    return elements_from_xsuite_line(
        line,
        roles={
            "Q2": "upstream flanking quadrupole",
            "B1": "first DBA bend",
            "Q1": "central dispersion-control quadrupole",
            "B2": "second DBA bend",
            "Q3": "downstream flanking quadrupole",
        },
    )


def display_dba_insert_from_xsuite_line(line, *, show_plot: bool = True) -> tuple[Lattice, OpticsResult, float, float]:
    """Convert, solve, and plot the Xsuite two-bend insert."""
    elements = dba_insert_from_xsuite_line(line)
    result = compute_transport_optics(elements)
    eta_end, etap_end = endpoint_dispersion(elements)
    plot_optics(
        result,
        title="Figure B0 — Two-bend cell with all quadrupoles off",
        show=show_plot,
        show_eta_prime=False,
    )
    return elements, result, eta_end, etap_end


def display_focused_dba_cell(
    q1: float,
    *,
    q2: float = DBA_Q2_DEFAULT,
    q3: float = DBA_Q3_DEFAULT,
    show_plot: bool = True,
) -> tuple[Lattice, Lattice, OpticsResult]:
    """Build, solve, and show the canonical focused DBA-like cell."""
    central_q1_cell = make_dba_cell(q1=q1, q2=0.0, q3=0.0)
    focused_dba_cell = make_dba_cell(q1=q1, q2=q2, q3=q3)
    optics_focused_dba = compute_periodic_optics(focused_dba_cell)
    plot_optics(
        optics_focused_dba,
        title="Figure B2 — Canonical focused DBA-like periodic cell",
        show=show_plot,
        show_eta_prime=False,
    )
    return central_q1_cell, focused_dba_cell, optics_focused_dba


def display_fodo_edge_focusing(
    edge_angle_deg: float,
    build_fodo_with_edges: Callable[[float], Sequence[Element]],
    optics_fodo_reference: OpticsResult,
    optics_fodo_bend: OpticsResult,
    *,
    show_plot: bool = True,
) -> OpticsResult | None:
    """Display the edge-focusing comparison for a user-supplied FODO builder."""
    fodo_with_edges = build_fodo_with_edges(edge_angle_deg)
    try:
        optics_fodo_edges = compute_periodic_optics(fodo_with_edges)
    except ValueError as exc:
        print(exc)
        _maybe_display(compact_stability_report(fodo_with_edges))
        return None

    plot_optics(
        optics_fodo_edges,
        title=f"FODO cell with 20-degree total bend and {edge_angle_deg:.1f}-degree edge angles",
        show=show_plot,
    )
    _maybe_display(
        compact_optics_comparison(
            {
                "no bend": optics_fodo_reference,
                "bend, no edge": optics_fodo_bend,
                "bend + edge": optics_fodo_edges,
            }
        )
    )

    return optics_fodo_edges


def interactive_fodo_edge_focusing(
    build_fodo_with_edges: Callable[[float], Sequence[Element]],
    optics_fodo_reference: OpticsResult,
    optics_fodo_bend: OpticsResult,
    *,
    default_edge_angle_deg: float = 5.0,
):
    """Show the FODO edge-focusing slider while keeping notebook code minimal."""
    widgets, display, clear_output = _get_widgets()

    def update(edge_angle_deg=default_edge_angle_deg):
        result = display_fodo_edge_focusing(
            edge_angle_deg,
            build_fodo_with_edges,
            optics_fodo_reference,
            optics_fodo_bend,
        )
        return result if widgets is None else None

    if widgets is None:
        print("ipywidgets is not available; showing the default static case instead.")
        return update()

    edge_angle_slider = lab_float_slider(
        widgets,
        value=default_edge_angle_deg,
        min=0.0,
        max=15.0,
        step=1.0,
        description="edge angle [deg]",
    )
    edge_angle_output = widgets.interactive_output(update, {"edge_angle_deg": edge_angle_slider})
    display(widgets.VBox([edge_angle_slider, edge_angle_output], layout=widget_container_layout(widgets)))
    return None


def interactive_fodo_dispersion():
    widgets, display, clear_output = _get_widgets()
    bend_grid = np.linspace(0.0, 30.0, 31)
    range_results = []
    for bend_angle in bend_grid:
        line = make_fodo_cell(kq=0.6, bend_angle_deg=float(bend_angle), with_bend=True, edge_focusing=False)
        try:
            range_results.append(compute_periodic_optics(line))
        except ValueError:
            continue
    beta_values = np.concatenate(
        [
            result.table[["beta_x_m", "beta_y_m"]].to_numpy(dtype=float).ravel()
            for result in range_results
        ]
    )
    dispersion_values = np.concatenate(
        [result.table["eta_x_m"].to_numpy(dtype=float) for result in range_results]
    )
    beam_values = np.concatenate(
        [
            add_beam_size_columns(result.table, sigma_delta=0.005)[
                ["sigma_x_mm", "sigma_x_beta_mm", "sigma_y_mm"]
            ].to_numpy(dtype=float).ravel()
            for result in range_results
        ]
    )
    fixed_ranges = {
        "beta": _padded_range(np.r_[0.0, beta_values], minimum_span=1.0),
        "dispersion": _padded_range(np.r_[0.0, dispersion_values], minimum_span=0.1),
        "beam": _padded_range(np.r_[0.0, beam_values], minimum_span=1.0),
    }

    def update(bend_angle_deg=20.0, sigma_delta=0.001):
        line = make_fodo_cell(kq=0.6, bend_angle_deg=bend_angle_deg, with_bend=True, edge_focusing=False)
        try:
            result = compute_periodic_optics(line)
        except ValueError as exc:
            print(exc)
            _maybe_display(compact_stability_report(line))
            return None
        plot_optics_and_beam_size(
            result,
            sigma_delta=sigma_delta,
            title="Interactive A1 — Bend, dispersion, and beam-size response",
            show_eta_prime=False,
            fixed_ranges=fixed_ranges,
        )
        return result if widgets is None else None

    if widgets is None:
        print("ipywidgets is not available; showing the default static case instead.")
        return update()

    controls = {
        "bend_angle_deg": lab_float_slider(widgets, value=20.0, min=0.0, max=30.0, step=1.0, description="bend deg"),
        "sigma_delta": lab_float_slider(widgets, value=0.001, min=0.0, max=0.005, step=0.00025, readout_format=".4f", description="σδ"),
    }
    widgets.interact(update, **controls)
    return None


def interactive_achromat_q1():
    widgets, display, clear_output = _get_widgets()
    default_q1 = 0.0
    reference = compute_transport_optics(make_dba_cell(q1=0.0, q2=0.0, q3=0.0))
    range_results = [
        compute_transport_optics(make_dba_cell(q1=float(q1), q2=0.0, q3=0.0))
        for q1 in np.linspace(0.0, 6.0, 121)
    ]
    eta_range = _padded_range(
        np.concatenate(
            [np.array([0.0]), *[result.table["eta_x_m"].to_numpy(dtype=float) for result in range_results]]
        ),
        minimum_span=0.1,
    )
    etap_range = _padded_range(
        np.concatenate(
            [np.array([0.0]), *[result.table["eta_xp"].to_numpy(dtype=float) for result in range_results]]
        ),
        minimum_span=0.05,
    )

    def update(q1=default_q1):
        line = make_dba_cell(q1=q1, q2=0.0, q3=0.0)
        result = compute_transport_optics(line)
        plot_dispersion_state_portrait(
            result,
            reference=reference,
            current_label="selected Q1",
            reference_label="Q1 = 0 reference",
            eta_range=eta_range,
            etap_range=etap_range,
            autorange_portrait=True,
            title=f"Interactive B1 — Achromat tuning state, Q1={q1:.2f} m⁻²",
        )
        return result if widgets is None else None

    if widgets is None:
        print("ipywidgets is not available; showing the default static case instead.")
        return update()

    widgets.interact(
        update,
        q1=lab_float_slider(
            widgets,
            value=default_q1,
            min=0.0,
            max=6.0,
            step=0.05,
            readout_format=".2f",
            description="Q1 k₁",
        ),
    )
    return None


def interactive_dba_stability():
    widgets, display, clear_output = _get_widgets()

    def update(q2=DBA_Q2_DEFAULT, q3=DBA_Q3_DEFAULT):
        line = make_dba_cell(q1=DBA_Q1_DEFAULT, q2=q2, q3=q3)
        try:
            result = compute_periodic_optics(line)
        except ValueError as exc:
            print(exc)
            _maybe_display(compact_stability_report(line))
            return None
        plot_optics(result, title=f"Matched DBA cell, Q2={q2:.3f}, Q3={q3:.3f}")
        _maybe_display(compact_stability_report(line))
        _maybe_display(compact_optics_summary(result))
        return result if widgets is None else None

    if widgets is None:
        print("ipywidgets is not available; showing the default static case instead.")
        return update()

    widgets.interact(
        update,
        q2=lab_float_slider(widgets, value=DBA_Q2_DEFAULT, min=-3.0, max=3.0, step=0.025, description="Q2 k1"),
        q3=lab_float_slider(widgets, value=DBA_Q3_DEFAULT, min=-3.0, max=3.0, step=0.025, description="Q3 k1"),
    )
    return None


def interactive_aperture(result: OpticsResult | None = None):
    widgets, display, clear_output = _get_widgets()
    if result is None:
        result = compute_periodic_optics(make_dba_cell())
    # A fixed, pipe-scale view keeps the default and Q6-scale states legible.
    # Extreme 5-sigma/high-spread settings deliberately clip rather than
    # forcing every ordinary state into a few pixels around zero.
    fixed_y_max = 60.0
    fixed_clearance_range = [-60.0, 55.0]

    def update(sigma_delta=0.001, pipe_radius_cm=2.5, n_sigma=1.0):
        radius = pipe_radius_cm / 100.0
        plot_aperture_ribbon(
            result,
            sigma_delta=sigma_delta,
            pipe_radius_m=radius,
            n_sigma=n_sigma,
            n_particles=48,
            seed=2026,
            sample_step_m=0.10,
            fixed_y_extent_mm=fixed_y_max,
            fixed_clearance_range_mm=fixed_clearance_range,
            title="Interactive aperture usage: particles, components, and clearance",
        )

    if widgets is None:
        print("ipywidgets is not available; showing the default static case instead.")
        return update()

    widgets.interact(
        update,
        sigma_delta=lab_float_slider(widgets, value=0.001, min=0.0, max=0.04, step=0.001, readout_format=".3f", description="σδ"),
        pipe_radius_cm=lab_float_slider(widgets, value=2.5, min=0.5, max=5.0, step=0.1, description="pipe cm"),
        n_sigma=lab_float_slider(widgets, value=1.0, min=1.0, max=5.0, step=0.5, description="nσ"),
    )
    return None


def interactive_tune_footprint(nux: float, nuy: float, cx: float, cy: float):
    widgets, display, clear_output = _get_widgets()

    def update(sigma_delta=0.001, resonance_order=3):
        plot_tune_footprint(nux, nuy, cx, cy, sigma_delta=sigma_delta, resonance_order=int(resonance_order))

    if widgets is None:
        print("ipywidgets is not available; showing the default static case instead.")
        return update()

    widgets.interact(
        update,
        sigma_delta=lab_float_slider(widgets, value=0.001, min=0.0, max=0.01, step=0.00025, readout_format=".4f", description="σδ"),
        resonance_order=lab_int_slider(widgets, value=3, min=1, max=4, step=1, description="order"),
    )
    return None


def interactive_tune_footprint_from_scan(scan: pd.DataFrame):
    """Explore a precomputed direct Xsuite tune scan without recalculating it."""
    widgets, display, clear_output = _get_widgets()
    clean = _validated_tune_scan(scan)
    max_available = min(abs(float(clean["delta"].min())), abs(float(clean["delta"].max())), 0.01)
    slider_max = max(0.001, math.floor(max_available / 0.00025) * 0.00025)

    def update(sigma_delta=0.001, resonance_order=3):
        plot_tune_footprint_from_scan(
            clean,
            sigma_delta=sigma_delta,
            resonance_order=int(resonance_order),
        )

    if widgets is None:
        print("ipywidgets is not available; showing the default static case instead.")
        return update()

    widgets.interact(
        update,
        sigma_delta=lab_float_slider(widgets, value=0.001, min=0.0, max=slider_max, step=0.00025, readout_format=".4f", description="σδ"),
        resonance_order=lab_int_slider(widgets, value=3, min=1, max=4, step=1, description="order"),
    )
    return None
