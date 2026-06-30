"""Local helper functions for the Dispersion and Chromaticity lab.

The notebook keeps most matrix optics, plotting, and widget code here so the
student-facing cells can focus on a small number of physics knobs.  The model
is intentionally lightweight: it uses first-order transfer matrices for drifts,
thick quadrupoles, and sector bends, plus an optional thin-edge focusing model.
It is meant for teaching dispersion, achromats, chromaticity, and tune-footprint
ideas; it is not a replacement for a production accelerator code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping, Sequence
import math
import warnings

import numpy as np
import pandas as pd

from .shared import dependency_table, maybe_display as _maybe_display, show_or_return as _show_or_return

try:
    import xtrack as xt
except Exception as exc:  # pragma: no cover - notebook dependency check handles this
    xt = None
    _XTRACK_IMPORT_ERROR = exc
else:
    _XTRACK_IMPORT_ERROR = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as exc:  # pragma: no cover - notebook dependency check handles this
    go = None
    make_subplots = None
    _PLOTLY_IMPORT_ERROR = exc
else:
    _PLOTLY_IMPORT_ERROR = None


GEOMETRIC_EMITTANCE = 6e-6        # 6 mm mrad = 6e-6 m rad
SIGMA_DELTA_DEFAULT = 1e-3        # 0.1% fractional momentum spread
PIPE_RADIUS_DEFAULT = 0.025       # 2.5 cm
DBA_BEND_ANGLE_DEG = 18.0
DBA_Q1_DEFAULT = 2.3356332610219  # strength that closes the zero-dispersion insert in this model
DBA_Q2_DEFAULT = 2.475            # flanking quad strength used for the stable DBA cell
DBA_Q3_DEFAULT = -2.15            # flanking quad strength used for the stable DBA cell
DBA_N_CELLS_DEFAULT = 10


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


# ---------------------------------------------------------------------------
# Environment and small utilities
# ---------------------------------------------------------------------------


def check_environment() -> pd.DataFrame:
    """Return a compact dependency table for the notebook setup cell."""
    return dependency_table(["numpy", "pandas", "plotly", "ipywidgets", "xtrack"])


def _require_plotly() -> None:
    if go is None or make_subplots is None:
        raise ImportError(
            "Plotly is required for plotting in this lab. Install plotly or run the notebook in an environment that includes it."
        ) from _PLOTLY_IMPORT_ERROR


def _require_xtrack() -> None:
    if xt is None:
        raise ImportError(
            "Xtrack is required for the Xsuite Environment lattice examples. Install xtrack or run the notebook in an environment that includes it."
        ) from _XTRACK_IMPORT_ERROR


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

    Edge focusing is intentionally not repeated inside bend slices.  The optional
    edge model is applied only in full-element maps; the core notebook exercises
    use zero edge angle, so this approximation does not affect them.
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

    The public notebook cells use modern Xsuite ``Environment`` syntax. This
    helper follows the same construction path, then converts the simple Xsuite
    line to the local first-order ``Element`` representation used by the optics
    calculations in this lab.
    """
    _require_xtrack()

    env = xt.Environment()
    env["kq"] = float(kq)
    env["bend_angle"] = math.radians(float(bend_angle_deg))
    env["bend_length"] = 0.5
    env["edge_angle"] = 0.5 * env["bend_angle"] if edge_focusing else 0.0

    env.new("D1", xt.Drift, length=1.0)
    env.new("QF", xt.Quadrupole, length=0.5, k1="kq")

    if with_bend:
        env.new("D2", xt.Drift, length=1.5)
        env.new(
            "BEND",
            xt.Bend,
            length="bend_length",
            angle="bend_angle",
            k0="bend_angle / bend_length",
            edge_entry_angle="edge_angle",
            edge_exit_angle="edge_angle",
        )
        components = ["D1", "QF", "D2", "BEND", "QD", "D3"]
        roles = {
            "QF": "focusing quadrupole",
            "BEND": "20-degree bend slot",
            "QD": "defocusing quadrupole",
        }
    else:
        env.new("D2", xt.Drift, length=2.0)
        components = ["D1", "QF", "D2", "QD", "D3"]
        roles = {
            "QF": "focusing quadrupole",
            "QD": "defocusing quadrupole",
        }

    env.new("QD", xt.Quadrupole, length=0.5, k1="-kq")
    env.new("D3", xt.Drift, length=1.0)

    line = env.new_line(name="fodo_cell", components=components)
    line.particle_ref = xt.Particles(p0c=1e9, mass0=xt.ELECTRON_MASS_EV, q0=-1)
    return elements_from_xsuite_line(line, roles=roles)


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


def elements_from_xsuite_line(line, roles: Mapping[str, str] | None = None) -> Lattice:
    """Convert a simple Xsuite line into the local first-order model.

    The dispersion/chromaticity lab keeps its transport model local, but the
    notebook can still use modern Xsuite ``Environment``/``Line`` syntax for
    front-facing lattice construction. This adapter is intentionally narrow:
    it supports drifts, quadrupoles, and sector-like bends.
    """
    roles = dict(roles or {})
    table = line.get_table().to_pandas()
    by_name = table.set_index("name", drop=False)
    elements = Lattice()

    for name in line.element_names:
        if name == "_end_point":
            continue
        element = line[name]
        element_type = str(by_name.loc[name, "element_type"])
        length = float(getattr(element, "length", by_name.loc[name, "s_end"] - by_name.loc[name, "s_start"]))

        if element_type == "Drift":
            elements.append(Element(name, "drift", length, role=roles.get(name, "")))
        elif element_type == "Quadrupole":
            elements.append(Element(name, "quad", length, k1=float(element.k1), role=roles.get(name, "")))
        elif element_type in {"Bend", "RBend"}:
            edge_angle = 0.5 * (float(getattr(element, "edge_entry_angle", 0.0)) + float(getattr(element, "edge_exit_angle", 0.0)))
            elements.append(
                Element(
                    name,
                    "bend",
                    length,
                    angle=float(element.angle),
                    edge_angle=edge_angle,
                    role=roles.get(name, ""),
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


# ---------------------------------------------------------------------------
# Optics calculations
# ---------------------------------------------------------------------------


def _sample_count(element: Element, samples_per_meter: float = 16.0, min_samples: int = 4) -> int:
    if element.length <= 0:
        return 1
    if element.kind == "quad":
        return max(min_samples, int(math.ceil(samples_per_meter * element.length)))
    if element.kind == "bend":
        return max(min_samples, int(math.ceil(samples_per_meter * element.length)))
    return max(1, int(math.ceil(samples_per_meter * element.length)))


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
        for slice_element_ in split_element(element, n_slices):
            mx, my, b = element_maps(slice_element_, delta=delta)
            beta_x, alpha_x = propagate_twiss(beta_x, alpha_x, mx)
            beta_y, alpha_y = propagate_twiss(beta_y, alpha_y, my)
            eta = mx @ eta + b
            s += slice_element_.length
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
                "ηₓ [m]",
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


def _row_nearest_s(result: OpticsResult, s_target: float) -> pd.Series:
    idx = (result.table["s_m"] - s_target).abs().idxmin()
    return result.table.loc[idx]


def row_at_element_center(result: OpticsResult, element_name: str) -> pd.Series:
    layout = result.layout.query("name == @element_name")
    if layout.empty:
        raise KeyError(f"Element {element_name!r} not found")
    row = layout.iloc[0]
    center = 0.5 * (row["s_start_m"] + row["s_end_m"])
    return _row_nearest_s(result, center)


def table_at_element_centers(result: OpticsResult, element_names: Sequence[str], sigma_delta: float = 0.0) -> pd.DataFrame:
    rows = []
    beam = add_beam_size_columns(result.table, sigma_delta=sigma_delta)
    for name in element_names:
        layout_row = result.layout.query("name == @name").iloc[0]
        center = 0.5 * (layout_row["s_start_m"] + layout_row["s_end_m"])
        idx = (beam["s_m"] - center).abs().idxmin()
        row = beam.loc[idx]
        rows.append(
            {
                "element": name,
                "s_center_m": center,
                "beta_x_m": row["beta_x_m"],
                "beta_y_m": row["beta_y_m"],
                "eta_x_m": row["eta_x_m"],
                "eta_xp": row["eta_xp"],
                "sigma_x_mm": row["sigma_x_mm"],
                "sigma_y_mm": row["sigma_y_mm"],
            }
        )
    return pd.DataFrame(rows)


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


def dispersion_extrema(result: OpticsResult) -> pd.DataFrame:
    df = result.table
    rows = []
    for label, idx in [("minimum eta_x", df["eta_x_m"].idxmin()), ("maximum eta_x", df["eta_x_m"].idxmax()), ("maximum |eta_x|", df["eta_x_m"].abs().idxmax())]:
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


def dba_endpoint_table(q1: float, q2: float = 0.0, q3: float = 0.0) -> pd.DataFrame:
    eta_end, etap_end = endpoint_dispersion(make_dba_cell(q1=q1, q2=q2, q3=q3))
    return pd.DataFrame(
        {
            "quantity": ["eta_x at end [m]", "eta_x' at end", "penalty eta_x^2 + eta_x'^2"],
            "value": [eta_end, etap_end, eta_end**2 + etap_end**2],
        }
    )


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
                "penalty": eta_end**2 + etap_end**2,
            }
        )
    return pd.DataFrame(rows)


def best_q1_for_achromat(qmin: float = 0.0, qmax: float = 6.0, rounds: int = 4, n: int = 401, q2: float = 0.0, q3: float = 0.0) -> float:
    """Find Q1 by repeated grid refinement.  No SciPy dependency required."""
    left = float(qmin)
    right = float(qmax)
    best_q = None
    for _ in range(int(rounds)):
        scan = scan_q1_for_achromat(left, right, n=n, q2=q2, q3=q3)
        best = scan.loc[scan["penalty"].idxmin()]
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
    """Return the momentum-spread limit from ``n_sigma * sigma_x <= pipe_radius``.

    The original Sirepo prompt used an rms beam-size formula while saying that
    particles hit the wall.  This notebook makes the convention explicit: the
    default is a 1-rms envelope criterion.  Set ``n_sigma=2`` or ``3`` to make a
    more conservative estimate.
    """
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive")
    df = result.table.copy()
    allowed_rms_radius = pipe_radius_m / n_sigma
    sigma_beta_m = np.sqrt(np.maximum(emit_x * df["beta_x_m"], 0.0))
    numerator = np.maximum(allowed_rms_radius**2 - sigma_beta_m**2, 0.0)
    eta_abs = np.abs(df["eta_x_m"].to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_delta_limit = np.sqrt(numerator) / eta_abs
    sigma_delta_limit[eta_abs < 1e-14] = np.inf
    out = df[["s_m", "element", "kind", "beta_x_m", "eta_x_m"]].copy()
    out["sigma_x_beta_mm"] = 1e3 * sigma_beta_m
    out["pipe_radius_mm"] = 1e3 * pipe_radius_m
    out["n_sigma"] = n_sigma
    out["max_sigma_delta"] = sigma_delta_limit
    out["max_delta_percent"] = 100.0 * sigma_delta_limit
    return out.sort_values("max_sigma_delta", na_position="last").reset_index(drop=True)


def aperture_summary(result: OpticsResult, **kwargs) -> pd.DataFrame:
    table = aperture_limit_table(result, **kwargs)
    limiting = table.iloc[0]
    return pd.DataFrame(
        {
            "quantity": ["limiting momentum spread", "limiting momentum spread [%]", "limiting element", "limiting s [m]", "eta_x at limit [m]", "beta_x at limit [m]"],
            "value": [
                limiting["max_sigma_delta"],
                limiting["max_delta_percent"],
                limiting["element"],
                limiting["s_m"],
                limiting["eta_x_m"],
                limiting["beta_x_m"],
            ],
        }
    )


# ---------------------------------------------------------------------------
# Chromaticity and resonance helpers
# ---------------------------------------------------------------------------


def cell_tunes(elements: Sequence[Element], delta: float = 0.0) -> tuple[float, float]:
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
    qx_plus, qy_plus = ring_tunes(elements, n_cells=n_cells, delta=+ddelta)
    qx_minus, qy_minus = ring_tunes(elements, n_cells=n_cells, delta=-ddelta)
    return (qx_plus - qx_minus) / (2.0 * ddelta), (qy_plus - qy_minus) / (2.0 * ddelta)


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
                "max eta_x in one cell [m]",
                "min eta_x in one cell [m]",
            ],
            "value": [n_cells, qx, qy, cx, cy, cell.table["eta_x_m"].max(), cell.table["eta_x_m"].min()],
        }
    )


def chromatic_spread_table(nux: float, nuy: float, cx: float, cy: float, sigma_delta: float = SIGMA_DELTA_DEFAULT) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "quantity": ["Qx", "Qy", "Cx", "Cy", f"Delta Qx for sigma_delta={sigma_delta:g}", f"Delta Qy for sigma_delta={sigma_delta:g}"],
            "value": [nux, nuy, cx, cy, cx * sigma_delta, cy * sigma_delta],
        }
    )


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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_optics(result: OpticsResult, title: str = "Optics", show: bool = True):
    _require_plotly()
    df = result.table
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Beta functions", "Horizontal dispersion"))
    hover = df["element"] + "<br>s=%{x:.4g} m<br>value=%{y:.5g}"
    fig.add_trace(go.Scatter(x=df["s_m"], y=df["beta_x_m"], mode="lines", name="beta_x", hovertemplate=hover + " m<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["s_m"], y=df["beta_y_m"], mode="lines", name="beta_y", hovertemplate=hover + " m<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["s_m"], y=df["eta_x_m"], mode="lines", name="eta_x", hovertemplate=hover + " m<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["s_m"], y=df["eta_xp"], mode="lines", name="eta_x'", hovertemplate=hover + "<extra></extra>"), row=2, col=1)
    fig.update_xaxes(title_text="s [m]", row=2, col=1)
    fig.update_yaxes(title_text="beta [m]", row=1, col=1)
    fig.update_yaxes(title_text="eta [m]", row=2, col=1)
    fig.update_layout(title=title, template="plotly_white", hovermode="x unified", width=900, height=620)
    return _show_or_return(fig, show)


def plot_beam_size(result: OpticsResult, sigma_delta: float = SIGMA_DELTA_DEFAULT, title: str = "RMS beam size", show: bool = True):
    _require_plotly()
    df = add_beam_size_columns(result.table, sigma_delta=sigma_delta)
    fig = go.Figure()
    hover = df["element"] + "<br>s=%{x:.4g} m<br>size=%{y:.5g} mm"
    fig.add_trace(go.Scatter(x=df["s_m"], y=df["sigma_x_mm"], mode="lines", name="sigma_x total", hovertemplate=hover + "<extra></extra>"))
    fig.add_trace(go.Scatter(x=df["s_m"], y=df["sigma_x_beta_mm"], mode="lines", name="sigma_x betatron only", hovertemplate=hover + "<extra></extra>"))
    fig.add_trace(go.Scatter(x=df["s_m"], y=df["sigma_y_mm"], mode="lines", name="sigma_y", hovertemplate=hover + "<extra></extra>"))
    fig.update_layout(title=f"{title} (sigma_delta={sigma_delta:g})", xaxis_title="s [m]", yaxis_title="rms size [mm]", template="plotly_white", width=900, height=430, hovermode="x unified")
    return _show_or_return(fig, show)


def plot_beam_size_with_aperture(result: OpticsResult, sigma_delta: float, pipe_radius_m: float = PIPE_RADIUS_DEFAULT, n_sigma: float = 1.0, show: bool = True):
    _require_plotly()
    df = add_beam_size_columns(result.table, sigma_delta=sigma_delta)
    envelope_mm = n_sigma * df["sigma_x_mm"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["s_m"], y=envelope_mm, mode="lines", name=f"{n_sigma:g} sigma_x envelope"))
    fig.add_trace(go.Scatter(x=df["s_m"], y=np.full(len(df), 1e3 * pipe_radius_m), mode="lines", name="pipe radius"))
    fig.update_layout(title=f"Horizontal envelope versus aperture (sigma_delta={sigma_delta:g})", xaxis_title="s [m]", yaxis_title="radius [mm]", template="plotly_white", width=900, height=430)
    return _show_or_return(fig, show)


def plot_q1_scan(scan: pd.DataFrame, title: str = "Endpoint dispersion versus Q1", show: bool = True):
    _require_plotly()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Endpoint dispersion", "Merit function"))
    fig.add_trace(go.Scatter(x=scan["q1_m^-2"], y=scan["eta_x_end_m"], mode="lines", name="eta_x end"), row=1, col=1)
    fig.add_trace(go.Scatter(x=scan["q1_m^-2"], y=scan["eta_xp_end"], mode="lines", name="eta_x' end"), row=1, col=1)
    fig.add_trace(go.Scatter(x=scan["q1_m^-2"], y=scan["penalty"], mode="lines", name="eta^2 + eta'^2"), row=2, col=1)
    best = scan.loc[scan["penalty"].idxmin()]
    fig.add_vline(x=best["q1_m^-2"], line_dash="dash", annotation_text=f"best Q1={best['q1_m^-2']:.3f}")
    fig.update_xaxes(title_text="Q1 k1 [m^-2]", row=2, col=1)
    fig.update_yaxes(title_text="eta", row=1, col=1)
    fig.update_yaxes(title_text="penalty", row=2, col=1, type="log")
    fig.update_layout(title=title, template="plotly_white", width=900, height=620)
    return _show_or_return(fig, show)


def plot_tune_footprint(nux: float, nuy: float, cx: float, cy: float, sigma_delta: float = SIGMA_DELTA_DEFAULT, resonance_order: int = 3, show: bool = True):
    _require_plotly()
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
                hovertemplate=f"{seg['label']}<br>order {seg['order']}<extra></extra>",
            )
        )

    x0 = nux - cx * sigma_delta
    x1 = nux + cx * sigma_delta
    y0 = nuy - cy * sigma_delta
    y1 = nuy + cy * sigma_delta
    fig.add_trace(go.Scatter(x=[nux], y=[nuy], mode="markers", name="nominal tune", marker=dict(size=10), hovertemplate="Qx=%{x:.6g}<br>Qy=%{y:.6g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines+markers", name="chromatic footprint", line=dict(width=4), hovertemplate="Qx=%{x:.6g}<br>Qy=%{y:.6g}<extra></extra>"))
    fig.update_layout(title=f"Tune footprint, resonance order <= {resonance_order}, sigma_delta={sigma_delta:g}", xaxis_title="Qx", yaxis_title="Qy", xaxis=dict(range=x_range), yaxis=dict(range=y_range), template="plotly_white", width=760, height=680)
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
    return widgets, display, clear_output


def interactive_fodo_dispersion():
    widgets, display, clear_output = _get_widgets()

    def update(kq=0.6, bend_angle_deg=20.0, sigma_delta=0.001, edge_focusing=False):
        line = make_fodo_cell(kq=kq, bend_angle_deg=bend_angle_deg, with_bend=True, edge_focusing=edge_focusing)
        try:
            result = compute_periodic_optics(line)
        except ValueError as exc:
            print(exc)
            print(stability_report(line))
            return None
        _maybe_display(optics_summary(result, "FODO with bend"))
        _maybe_display(dispersion_extrema(result))
        _maybe_display(beam_size_comparison_at_elements(result, ["QF", "QD"], sigma_delta=sigma_delta))
        plot_optics(result, title="FODO cell with bend")
        plot_beam_size(result, sigma_delta=sigma_delta)
        return result

    if widgets is None:
        print("ipywidgets is not available; showing the default static case instead.")
        return update()

    controls = {
        "kq": widgets.FloatSlider(value=0.6, min=0.1, max=1.2, step=0.05, description="quad k1"),
        "bend_angle_deg": widgets.FloatSlider(value=20.0, min=0.0, max=30.0, step=1.0, description="bend deg"),
        "sigma_delta": widgets.FloatSlider(value=0.001, min=0.0, max=0.005, step=0.00025, readout_format=".4f", description="sigma_delta"),
        "edge_focusing": widgets.Checkbox(value=False, description="edge focusing"),
    }
    return widgets.interact(update, **controls)


def interactive_achromat_q1():
    widgets, display, clear_output = _get_widgets()

    def update(q1=DBA_Q1_DEFAULT):
        line = make_dba_cell(q1=q1, q2=0.0, q3=0.0)
        result = compute_transport_optics(line)
        _maybe_display(dba_endpoint_table(q1))
        plot_optics(result, title=f"Two-bend insert transport, Q1={q1:.3f} m^-2")
        return result

    if widgets is None:
        print("ipywidgets is not available; showing the default static case instead.")
        return update()

    return widgets.interact(update, q1=widgets.FloatSlider(value=DBA_Q1_DEFAULT, min=0.0, max=6.0, step=0.05, description="Q1 k1"))


def interactive_dba_stability():
    widgets, display, clear_output = _get_widgets()

    def update(q2=DBA_Q2_DEFAULT, q3=DBA_Q3_DEFAULT):
        line = make_dba_cell(q1=DBA_Q1_DEFAULT, q2=q2, q3=q3)
        _maybe_display(stability_report(line))
        try:
            result = compute_periodic_optics(line)
        except ValueError as exc:
            print(exc)
            return None
        _maybe_display(optics_summary(result, "DBA cell"))
        plot_optics(result, title=f"Matched DBA cell, Q2={q2:.3f}, Q3={q3:.3f}")
        return result

    if widgets is None:
        print("ipywidgets is not available; showing the default static case instead.")
        return update()

    return widgets.interact(
        update,
        q2=widgets.FloatSlider(value=DBA_Q2_DEFAULT, min=-3.0, max=3.0, step=0.025, description="Q2 k1"),
        q3=widgets.FloatSlider(value=DBA_Q3_DEFAULT, min=-3.0, max=3.0, step=0.025, description="Q3 k1"),
    )


def interactive_aperture(result: OpticsResult | None = None):
    widgets, display, clear_output = _get_widgets()
    if result is None:
        result = compute_periodic_optics(make_dba_cell())

    def update(sigma_delta=0.001, pipe_radius_cm=2.5, n_sigma=1.0):
        radius = pipe_radius_cm / 100.0
        _maybe_display(aperture_summary(result, pipe_radius_m=radius, n_sigma=n_sigma))
        plot_beam_size_with_aperture(result, sigma_delta=sigma_delta, pipe_radius_m=radius, n_sigma=n_sigma)

    if widgets is None:
        print("ipywidgets is not available; showing the default static case instead.")
        return update()

    return widgets.interact(
        update,
        sigma_delta=widgets.FloatSlider(value=0.001, min=0.0, max=0.04, step=0.001, readout_format=".3f", description="sigma_delta"),
        pipe_radius_cm=widgets.FloatSlider(value=2.5, min=0.5, max=5.0, step=0.1, description="pipe cm"),
        n_sigma=widgets.FloatSlider(value=1.0, min=1.0, max=4.0, step=0.5, description="n sigma"),
    )


def interactive_tune_footprint(nux: float, nuy: float, cx: float, cy: float):
    widgets, display, clear_output = _get_widgets()

    def update(sigma_delta=0.001, resonance_order=3):
        plot_tune_footprint(nux, nuy, cx, cy, sigma_delta=sigma_delta, resonance_order=int(resonance_order))
        crossings = first_resonance_crossing(nux, nuy, cx, cy, max_order=int(resonance_order))
        if not crossings.empty:
            _maybe_display(crossings.head(5))

    if widgets is None:
        print("ipywidgets is not available; showing the default static case instead.")
        return update()

    return widgets.interact(
        update,
        sigma_delta=widgets.FloatSlider(value=0.001, min=0.0, max=0.01, step=0.00025, readout_format=".4f", description="sigma_delta"),
        resonance_order=widgets.IntSlider(value=3, min=1, max=4, step=1, description="order"),
    )
