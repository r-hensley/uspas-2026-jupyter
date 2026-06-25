# Exercise Inventory for `Dispersion-Chromaticity.ipynb`

Source: `Dispersion-Chromaticity.ipynb`  
Notebook title: **Computer Lab: Dispersion and Chromaticity in Simulated Beam Transport**  
Authors listed in notebook: K. Ruisard, N. Evans, N. Neveu, and L. Dovlatyan

## Scope and interpretation

This inventory is based on the visible notebook cells only. The original notebook depends on online Sirepo/Elegant simulations that are no longer accessible, so this document does **not** recover hidden Sirepo lattice files, plots, or numeric answers. It describes each visible student activity, what the original activity was meant to simulate or demonstrate, what students were asked to do, and what they were expected to learn.

The notebook contains 12 numbered questions, plus several unnumbered setup/manipulation tasks that function as exercises. Those unnumbered tasks are included because they define the simulation state needed for the numbered questions.

---

## Lab-level intent

The lab was designed to teach how dispersion and chromaticity affect beam transport in simple accelerator lattices. Students used Sirepo running Elegant to modify lattices, generate Twiss and sigma plots, inspect dispersion functions, calculate beam-size growth from momentum spread, construct a double-bend achromat, and estimate momentum acceptance limits from both dispersion and chromaticity.

Core concepts covered:

- Off-momentum particle transport and magnetic rigidity.
- Dispersion, \(\eta_x\), and its contribution to transverse beam size.
- Matched/periodic Twiss solutions in a cell or ring.
- Dipole edge focusing as a dipole effect distinct from dispersion.
- Achromatic transport and double-bend achromat construction.
- Momentum acceptance limits from finite aperture.
- Chromaticity, tune shift with momentum, tune spread, and resonance avoidance.
- Resonance diagrams and tune-footprint interpretation.

---

# Section A — Dispersion in a FODO lattice

Source cells: 1–4

## Section setup

### Visible simulation setup

Students were told to open the Sirepo simulation **“FODObeamline-with-dipole”**, initially configured as a FODO cell equivalent to the earlier FODO-transport lab.

Visible initial conditions:

| Parameter | Value in notebook |
|---|---:|
| Species | Electron, misspelled as “Election” in the table |
| Energy | 1 GeV |
| Horizontal emittance | \(\epsilon_x = 6\) mm-mrad |
| Vertical emittance | \(\epsilon_y = 6\) mm-mrad |
| Quadrupole geometric strength | \(K = 0.6\ \mathrm{m}^{-2}\) |
| FODO cell length | \(L = 5\) m |

### Unnumbered Exercise A0 — Run the baseline matched FODO cell

**What it was trying to simulate or show**  
A reference FODO cell without the added dipole, run over one cell with a matched Twiss solution. This provided baseline beam sizes at the focusing and defocusing quadrupoles before dispersion was introduced.

**What students were asked to do**

- Go to the Visualization tab.
- Run the simulation over one cell, selecting beamline `FODOcell`.
- Verify that the calculated solution is matched.
- If needed, enable matching under `Control -> twiss_output -> Matched = Yes` and rerun.
- Record matched beam sizes `Sx` and `Sy` from the `run_setup.sigma` plot at quadrupoles QF and QD for later comparison.

**What students were supposed to learn**

- How to establish a matched reference solution before perturbing a lattice.
- How to read beam sizes from a sigma plot.
- Why a controlled baseline is needed before attributing later beam-size changes to dispersion or edge focusing.

**Replacement-lab implications**

A replacement notebook should compute and plot the matched Twiss functions and beam sizes for a simple FODO cell before adding any dipole. It should store or display \(\sigma_x\) and \(\sigma_y\) at QF and QD so students can use them later.

---

### Unnumbered Exercise A1 — Replace a drift with a dipole and observe dipole effects

**What it was trying to simulate or show**  
A FODO cell modified by replacing a short drift with a rectangular bending magnet. The point was to introduce horizontal dispersion while also exposing students to edge focusing from a rectangular dipole.

**What students were asked to do**

- Go to the Lattice tab.
- Edit beamline `FODOcell`.
- Delete the 0.5 m drift `D3`.
- Replace it with the RBEN element `DIPO`.
- Note that `DIPO` is a 20-degree rectangular dipole.
- Rerun the modified FODO cell.
- Inspect the `sigma_output` plot and then the `twiss_output` plot.
- Turn off \(\beta_x\) and \(\beta_y\) curves if needed so \(\eta_x\) can be seen clearly.

**What students were supposed to learn**

- A dipole introduces dispersion because off-momentum particles bend differently from the reference particle.
- Dispersion is visible in the Twiss/optics output as \(\eta_x\).
- Beam-size changes in a tracked distribution with no momentum spread are not caused by dispersive beam broadening.
- Rectangular dipoles can introduce edge focusing: in the notebook description, horizontal entrance/exit effects nearly cancel, while the vertical plane experiences net focusing.
- Dipoles affect lattice optics in more than one way, and those mechanisms should not be conflated.

**Replacement-lab implications**

A replacement should separate at least three effects in the computation or plots:

1. On-energy optics before the dipole.
2. On-energy optics after the dipole, showing edge-focusing effects if modeled.
3. Off-energy or finite-energy-spread transport, showing true dispersive beam-size growth.

---

## Q1 — Minimum dispersion and beam-size comparison

Source cell: 2

**Student-facing task**  
Students were asked to find the minimum dispersion in the modified FODO lattice, identify whether it occurs at the focusing quadrupole, defocusing quadrupole, or drift, and compare the expected horizontal and vertical beam sizes to an otherwise identical FODO cell without the dipole.

**What it was trying to simulate or show**  
The spatial variation of the horizontal dispersion function \(\eta_x(s)\) through a FODO cell containing a dipole, and how the location of minimum or maximum dispersion matters for beam size when the beam has momentum spread.

**What students were supposed to learn**

- Dispersion is a lattice function that varies with longitudinal position \(s\), not a single global constant.
- Horizontal beam size can be larger in regions of nonzero \(\eta_x\) when momentum spread is present.
- Vertical beam size is not directly enlarged by horizontal dispersion, though it may change because the dipole also changes vertical focusing through edge effects.
- Students should distinguish dispersive beam-size growth from changes in optical beta functions and edge focusing.

**Data needed from the original Sirepo simulation**

- \(\eta_x(s)\) from the `twiss_output` plot.
- Locations of QF, QD, drifts, and the dipole.
- Baseline \(\sigma_x\), \(\sigma_y\) from the no-dipole FODO cell.

**Replacement-lab implications**

The new lab should generate a plot of \(\eta_x(s)\) with element locations marked. Students should be able to query or read off the minimum dispersion and its lattice location.

---

## Q2 — Beam size with 0.1% momentum spread

Source cells: 3–4

**Student-facing task**  
Students were asked to assume a fractional momentum spread

\[
\frac{\Delta p}{p_0} = 0.001
\]

and calculate the expected horizontal beam size in the focusing quadrupole QF. They were asked to compare it with the beam size without energy spread, then do the same comparison for the vertical beam size.

The notebook gave the formula

\[
\sigma_x^2 = \epsilon_x \beta_x + \eta_{\max}^2 \left(\frac{\Delta p}{p_0}\right)^2.
\]

Students were also told they could check the calculation by setting `Sigma DP` in the Sirepo `bunched_beam` control panel.

A blank table was provided:

| Quantity | \(\delta=0\) | \(\delta=0.001\) |
|---|---:|---:|
| \(\sigma_x\) |  |  |
| \(\sigma_y\) |  |  |

**What it was trying to simulate or show**  
This exercise connected the lattice dispersion function to a measurable beam size increase for a beam with finite momentum spread.

**What students were supposed to learn**

- The horizontal rms beam size has a betatron contribution and a dispersive contribution.
- The dispersive contribution scales with \(\eta_x\) and \(\Delta p/p_0\).
- For horizontal dispersion, energy spread primarily affects \(\sigma_x\), not \(\sigma_y\).
- Simulation and analytic estimates may differ slightly because the formula is idealized and the tracked distribution or lattice model may include additional effects.

**Data needed from the original Sirepo simulation**

- \(\beta_x\) at QF.
- \(\eta_x\) at or near QF, though the notebook says to use maximum \(\eta\).
- Baseline beam sizes at QF for \(\delta=0\).
- Simulated beam sizes after setting `Sigma DP = 0.001`.

**Replacement-lab implications**

The new lab should provide a computed optics table containing \(\beta_x\), \(\beta_y\), \(\eta_x\), and beam sizes at QF. It should then let students add momentum spread and compare analytic and simulated/tracked beam sizes.

---

# Section B — Designing a zero-dispersion insert

Source cells: 5–11

## Section setup

Students were told to open a Sirepo simulation named **“DispersionFree”**. The visible description says the lattice contains two 18-degree bends and five quadrupoles. Initially all quadrupole fields are zero and matching is disabled.

The overall goal of this section was to turn a two-bend transport cell into a double-bend achromat: a cell where dispersion is canceled in part of the lattice, yielding a zero-dispersion insert suitable for devices such as undulators or wigglers.

---

### Unnumbered Exercise B0 — Run the two-bend cell with quadrupoles off

Source cell: 5

**What it was trying to simulate or show**  
The natural dispersion generated by two bending magnets before any quadrupole correction is applied.

**What students were asked to do**

- Run the `DispersionFree` simulation with quadrupole strengths initially set to zero.
- Observe the evolution of horizontal dispersion \(\eta_x\) through the double bend.

**What students were supposed to learn**

- Dipoles generate horizontal dispersion.
- A pair of bends does not automatically cancel dispersion.
- The final values \(\eta_x\) and \(\eta'_x\) are the key quantities for determining whether a cell is achromatic.

**Replacement-lab implications**

A replacement should show \(\eta_x(s)\) and \(\eta'_x(s)\) through a two-bend lattice with all quadrupoles initially off.

---

## Q3 — Final dispersion and dispersion slope before correction

Source cell: 6

**Student-facing task**  
Students were asked to report \(\eta_x\) and \(\eta'_x\) at the end of the cell.

**What it was trying to simulate or show**  
The uncorrected two-bend transport does not return both dispersion and dispersion derivative to zero at the cell exit.

**What students were supposed to learn**

- An achromat requires both \(\eta_x = 0\) and \(\eta'_x = 0\) at the target location or cell boundary.
- Reading the endpoint of the dispersion trajectory is a diagnostic for whether an insert is dispersion-free.
- The dispersion derivative is as important as the dispersion amplitude for matching into a downstream dispersion-free region.

**Data needed from the original Sirepo simulation**

- End-of-cell \(\eta_x\).
- End-of-cell \(\eta'_x\).

**Replacement-lab implications**

The new lab should explicitly report endpoint values of \(\eta_x\) and \(\eta'_x\), not just plot them.

---

### Unnumbered Exercise B1 — Tune the middle quadrupole to cancel dispersion

Source cell: 7

**What it was trying to simulate or show**  
The middle quadrupole in a two-bend cell can be used to alter the dispersion trajectory and find a setting that cancels dispersion after the bends.

**What students were asked to do**

- Go to the Lattice page.
- Turn on the middle quadrupole Q1 with \(k_1 = 1\ \mathrm{m}^{-2}\).
- Observe how Q1 changes the dispersion function.
- Adjust Q1 until dispersion is zero after the two bends.
- Restart the simulation after each change to Q1.

**What students were supposed to learn**

- Quadrupoles affect the evolution of dispersion because the dispersion function obeys an inhomogeneous Hill-type equation through focusing elements.
- Achromatic conditions can be found by scanning or optimizing quadrupole strengths.
- A lattice can be designed to have local zero-dispersion regions even when it contains bends.

**Replacement-lab implications**

The new notebook could replace manual Sirepo scanning with a Python slider, root finder, or parameter scan over Q1. The output should show how endpoint \(\eta_x\) and \(\eta'_x\) vary with Q1.

---

## Q4 — Middle-quadrupole strength for zero dispersion

Source cell: 8

**Student-facing task**  
Students were asked to report the Q1 strength that gives zero dispersion after the two bends, with at least two decimal places.

**What it was trying to simulate or show**  
A practical lattice-matching task: find the quadrupole strength that satisfies an achromatic condition.

**What students were supposed to learn**

- Achromat design can be cast as a parameter-finding problem.
- The quadrupole strength is not arbitrary; a specific optical setting is needed to cancel dispersion at the desired location.
- Numerical precision matters when reporting lattice settings.

**Data needed from the original Sirepo simulation**

- Q1 strength at which endpoint \(\eta_x\) is zero after the two bends. The notebook text does not include the answer.

**Replacement-lab implications**

The new lab should either let students search manually or provide a computational optimization task. A useful replacement exercise would ask students to minimize \(|\eta_x(s_{end})|\), \(|\eta'_x(s_{end})|\), or a combined merit function.

---

### Unnumbered Exercise B2 — Enable matching and stabilize the double-bend achromat

Source cell: 9

**What it was trying to simulate or show**  
After achieving local dispersion cancellation, the cell still may not support a stable matched periodic solution. Additional quadrupoles are used to restore stable optics while preserving achromatic behavior.

**What students were asked to do**

- Enable matching under `Control -> twiss_output -> Matched = Yes`.
- Run the simulation and observe that the beamline is unstable, meaning no periodic solutions are found.
- Turn on flanking quadrupoles with the given strengths:
  - Q2: \(k_1 = 1.33\ \mathrm{m}^{-2}\)
  - Q3: \(k_1 = -1.59\ \mathrm{m}^{-2}\)
- Rerun to find the matched solution.

**What students were supposed to learn**

- Dispersion cancellation alone does not guarantee a stable lattice.
- Stable periodic optics require appropriate focusing in both transverse planes.
- A double-bend achromat is a combined design problem involving dispersion control and beta-function/tune stability.
- Zero dispersion can be local; the notebook notes that dispersion is canceled only in part of the cell, not everywhere.

**Replacement-lab implications**

A replacement should calculate the one-cell transfer matrix or Twiss periodicity condition and identify stable versus unstable optics. It should show the effect of Q2 and Q3 on stability and matched beta functions.

---

## Q5 — Maximum dispersion in the matched double-bend achromat cell

Source cell: 10

**Student-facing task**  
Students were asked to report the maximum horizontal dispersion in the cell after the double-bend achromat settings were established.

**What it was trying to simulate or show**  
Even an achromat that has zero dispersion in an insert can have substantial nonzero dispersion elsewhere, especially inside and around bends.

**What students were supposed to learn**

- “Dispersion-free insert” does not mean \(\eta_x=0\) throughout the entire lattice.
- The maximum dispersion location is important for aperture and momentum-acceptance calculations.
- Achromat design often trades local zero-dispersion regions against larger dispersion elsewhere.

**Data needed from the original Sirepo simulation**

- Maximum value of \(\eta_x(s)\) in the matched DBA cell.

**Replacement-lab implications**

The new notebook should compute `max(abs(eta_x))` or max positive \(\eta_x\), clarify which convention is used, and mark that location in a plot.

---

## Q6 — Momentum acceptance from a 2.5 cm pipe radius

Source cell: 11

**Student-facing task**  
Students were told that the vacuum chamber is a pipe with radius 2.5 cm and asked for the largest momentum spread that can be tolerated before particles hit the chamber wall.

The notebook again pointed to

\[
\sigma_x^2 = \epsilon_x\beta_x + \eta^2\left(\frac{\Delta p}{p_0}\right)^2.
\]

**What it was trying to simulate or show**  
Finite dispersion plus finite momentum spread increases the horizontal beam envelope. A finite aperture therefore imposes a momentum-acceptance limit.

**What students were supposed to learn**

- Momentum acceptance can be limited by physical aperture, not only by RF bucket or longitudinal dynamics.
- The limiting location is usually where the combination of dispersion, beta function, and aperture is most restrictive.
- Students must rearrange the beam-size formula to solve for allowable \(\Delta p/p_0\).
- Aperture comparisons require consistent units between emittance, beta function, dispersion, and chamber radius.

**Data needed from the original Sirepo simulation**

- Relevant \(\beta_x(s)\) and \(\eta_x(s)\), especially near the maximum beam envelope.
- Emittance value from the setup table.
- Aperture radius: 2.5 cm.

**Replacement-lab implications**

The replacement should explicitly define whether the aperture limit is applied to \(1\sigma\), \(2\sigma\), etc. The original prompt says “beam hits the wall” while using an rms beam-size formula, so the intended interpretation may have been simplified to \(\sigma_x = 2.5\) cm. That ambiguity should be resolved in the rewritten lab.

---

## Q7 — Location of beam loss if momentum spread is too large

Source cell: 11

**Student-facing task**  
Students were asked where in the cell beam loss will occur if momentum spread exceeds the aperture-limited value from Q6. They were told they could check this by introducing momentum spread in the simulation as in Q2.

**What it was trying to simulate or show**  
Beam loss from dispersive beam-size growth is localized at the most restrictive point in the lattice, not necessarily uniformly distributed around the cell.

**What students were supposed to learn**

- The loss location corresponds to the maximum beam envelope relative to the aperture.
- High dispersion regions are likely candidates for loss, but beta function and aperture also matter.
- Simulations can be used to validate analytic aperture/momentum-acceptance estimates.

**Data needed from the original Sirepo simulation**

- Beam envelope or \(\sigma_x(s)\) after applying large momentum spread.
- Lattice element locations.
- Maximum of the aperture-normalized horizontal beam size.

**Replacement-lab implications**

The new lab should plot horizontal envelope versus aperture and identify the first or worst loss location. It should explicitly distinguish “first loss along the beamline” from “location of maximum envelope.”

---

# Section C — Chromaticity in a ring

Source cells: 12–20

## Section setup

Students were told to repeat the DBA cell 10 times to create a ring. The section then moved from local dispersion control to chromaticity and tune spread in a periodic ring.

The key conceptual message was that a lattice can have dispersion-free sections but still have chromaticity. Off-momentum particles see different effective focusing and therefore have different tunes.

---

### Unnumbered Exercise C0 — Build a 10-cell DBA ring

Source cell: 12

**What it was trying to simulate or show**  
A periodic ring made by repeating the double-bend achromat cell 10 times. This converted the transport-cell optics problem into a ring tune/chromaticity problem.

**What students were asked to do**

- Under the Lattice tab, create a new beamline.
- Fill it with 10 DBA elements.
- Rerun the simulation under the Visualization tab.
- Select the new beamline from the dropdown menu.

**What students were supposed to learn**

- Rings can be modeled by repeating a stable cell.
- Periodic optics produce tunes \(\nu_x\) and \(\nu_y\).
- Dispersion-free insert regions do not eliminate chromaticity.
- Off-momentum particles experience tune shifts because quadrupole focusing depends on magnetic rigidity.

**Replacement-lab implications**

The new notebook should provide a programmatic way to concatenate the DBA transfer map/cell 10 times and compute ring tunes and chromaticities.

---

## Q8 — Record horizontal and vertical tunes

Source cell: 13

**Student-facing task**  
Students were asked to record the horizontal and vertical tunes, \(\nu_x\) and \(\nu_y\), to three significant figures. The notebook says these are listed as `nux` and `nuy` in the simulation Output Parameters list.

**What it was trying to simulate or show**  
The ring has characteristic betatron tunes determined by the periodic focusing lattice.

**What students were supposed to learn**

- How to identify and interpret ring tunes from optics output.
- The tune is the number of betatron oscillations per turn.
- Tunes are global properties of the periodic ring, not local element properties.

**Data needed from the original Sirepo simulation**

- `nux` and `nuy` from the Sirepo/Elegant output parameters.

**Replacement-lab implications**

The new lab should calculate tunes from the one-turn matrix using, for example,

\[
\cos(2\pi\nu) = \frac{1}{2}\operatorname{Tr}(M).
\]

---

## Q9 — Explain why DBA tunes need not be equal

Source cell: 14

**Student-facing task**  
Students were asked to compare the DBA ring to the earlier FODO cell, where equal focusing strengths in both planes gave \(\nu_x=\nu_y\). They were asked to inspect the beta functions for one DBA cell and explain why equal tunes should not necessarily be expected.

**What it was trying to simulate or show**  
The horizontal and vertical optics in a DBA cell are not symmetric in the same way as the earlier simplified FODO case. Bends, edge focusing, and unequal quadrupole arrangements can make the beta functions and phase advances different in the two planes.

**What students were supposed to learn**

- Equal quadrupole magnitudes do not automatically imply equal tunes in a more complex lattice.
- Tune depends on integrated focusing and phase advance around the full cell/ring.
- Horizontal and vertical beta functions reveal differences in focusing structure.
- Dipoles and edge focusing can break simple symmetry between planes.

**Data needed from the original Sirepo simulation**

- \(\beta_x(s)\) and \(\beta_y(s)\) for one DBA cell.
- Ring tunes from Q8.

**Replacement-lab implications**

The new lab should plot \(\beta_x\) and \(\beta_y\) across one DBA cell and ask students to connect differences in these curves to phase advance and tune.

---

## Q10 — Chromatic tune spread for 0.1% momentum spread

Source cell: 15

**Student-facing task**  
Students were asked to use chromaticity values from the Output Parameters list, labeled `dnux/dp` and `dnuy/dp`, and calculate tune spread for a 0.1% momentum spread:

\[
\Delta\nu = C\frac{\Delta p}{p_0}.
\]

They were asked to fill in:

- \(C_x\)
- \(C_y\)
- \(\Delta\nu_x\)
- \(\Delta\nu_y\)

**What it was trying to simulate or show**  
Chromaticity converts momentum spread into tune spread. Even when dispersion is locally canceled, off-momentum particles can sample different tunes.

**What students were supposed to learn**

- Chromaticity is the derivative of tune with respect to fractional momentum deviation.
- The tune footprint grows with beam momentum spread.
- Horizontal and vertical chromaticity can differ in sign and magnitude.
- Momentum spread can affect beam stability even without causing large dispersive beam-size growth in an insert.

**Data needed from the original Sirepo simulation**

- `dnux/dp` and `dnuy/dp` values from the Elegant output.
- Momentum spread \(\Delta p/p_0 = 0.001\).

**Replacement-lab implications**

The new lab should compute chromaticity either by finite-differencing tunes at positive/negative momentum offsets or by using analytic lattice functions. A finite-difference implementation would be transparent for students.

---

### Unnumbered Exercise C1 — Generate a resonance diagram and tune footprint

Source cells: 16–18

**What it was trying to simulate or show**  
A resonance diagram up to a chosen order, with the ring tune and chromatic tune footprint overlaid. The plot was used to estimate whether the momentum spread causes particles to cross low-order resonance lines.

**What students were asked to do**

- Execute a helper cell defining `tunediagram(...)`.
- Execute a plotting cell after entering values for:
  - `nux`
  - `nuy`
  - `sigma_dp`
  - `Cx`
  - `Cy`
  - `resonance_order`
- Use the black line in the plot as the tune footprint from chromaticity.

**What the code did visibly**

- Plotted resonance lines satisfying relationships of the form

  \[
  m\nu_x + n\nu_y = p
  \]

  up to a user-selected order.
- Computed

  \[
  \Delta\nu_x = C_x\sigma_{dp}, \qquad \Delta\nu_y = C_y\sigma_{dp}.
  \]

- Plotted the nominal ring tune as a point and the chromatic footprint as a line segment through tune space.

**What students were supposed to learn**

- Resonance lines organize tune space into stable and potentially problematic regions.
- Low-order resonances are typically more dangerous than high-order resonances.
- Momentum spread gives the beam a footprint in tune space, not a single tune point.
- Operational momentum acceptance can be limited by tune footprint crossing a resonance.

**Replacement-lab implications**

The resonance-diagram code is already present in the notebook and does not depend on Sirepo. It can likely be retained or refactored. The replacement should clarify the sign convention for chromaticity and whether `sigma_dp` represents rms momentum spread, full half-width, or another acceptance criterion.

---

## Q11 — Momentum acceptance from chromatic resonance crossing

Source cell: 19

**Student-facing task**  
Students were told to assume that resonances of order 4 and higher are tolerable, while resonances of order 3 and below are not. They were asked to set `resonance_order = 3`, enter the previously recorded tunes and chromaticities, and adjust `sigma_dp` until the tune-spread line crosses a resonance line. The requested answer was the tolerable momentum spread to the nearest 0.1%.

**What it was trying to simulate or show**  
Chromaticity can limit momentum acceptance by causing off-momentum particles to cross low-order resonance lines, even if physical aperture and dispersion would allow a larger energy spread.

**What students were supposed to learn**

- Momentum acceptance has both physical-aperture and dynamical-stability limits.
- A tune footprint can be used as a graphical estimate of a chromaticity-driven stability limit.
- Low-order resonances impose practical restrictions on the operating point.
- The allowed momentum spread depends on the nominal tune, chromaticity vector, and resonance structure.

**Data needed from the original Sirepo simulation**

- \(\nu_x\), \(\nu_y\) from Q8.
- \(C_x\), \(C_y\) from Q10.
- Resonance diagram calculation from the provided Python helper.

**Replacement-lab implications**

The new lab should let students vary `sigma_dp` interactively or compute the first intersection of a tune-footprint line with low-order resonance lines. A useful improvement would be to have students compare the visual estimate with an algorithmic intersection calculation.

---

## Q12 — Compare dispersion-limited and chromaticity-limited momentum acceptance

Source cell: 20

**Student-facing task**  
Students were asked to compare their answers to Q6 and Q11 and determine whether the ring momentum acceptance is limited by chromaticity or by dispersion.

**What it was trying to simulate or show**  
The final exercise synthesized the two main limitation mechanisms studied in the lab: aperture loss from dispersive beam-size growth and resonance crossing from chromatic tune spread.

**What students were supposed to learn**

- The limiting momentum acceptance is the smaller of the relevant physical and dynamical limits.
- Different parts of accelerator design impose different constraints on allowed momentum spread.
- A dispersion-free insert solves one local problem but does not remove all off-momentum limitations.
- Students should be able to compare numerical results from separate sections and draw an accelerator-design conclusion.

**Data needed from the original Sirepo simulation**

- Dispersion/aperture momentum limit from Q6.
- Chromatic resonance momentum limit from Q11.

**Replacement-lab implications**

The replacement lab should end with a summary calculation comparing the two limits directly, ideally in a small table:

| Limitation mechanism | Momentum spread limit | Limiting location or condition |
|---|---:|---|
| Dispersion/aperture |  | Beam envelope reaches pipe radius |
| Chromaticity/resonance |  | Tune footprint crosses order \(\le 3\) resonance |

---

# Consolidated exercise map

| ID | Notebook source cells | Exercise type | Main student action | Main learning target | Sirepo-dependent? |
|---|---:|---|---|---|---|
| A0 | 1 | Setup/data collection | Run matched baseline FODO and record QF/QD beam sizes | Establish reference optics and beam sizes | Yes |
| A1 | 1 | Lattice modification/observation | Replace drift with rectangular dipole and inspect sigma/Twiss plots | Separate dispersion from edge focusing | Yes |
| Q1 | 2 | Conceptual + plot reading | Find minimum dispersion and compare beam sizes | Dispersion is position-dependent and affects horizontal size | Yes |
| Q2 | 3–4 | Calculation + simulation check | Compute \(\sigma_x\), \(\sigma_y\) for \(\delta=0.001\) | Dispersive beam-size formula | Yes |
| B0 | 5 | Observation | Run two-bend cell with quads off | Dipoles generate nonzero endpoint dispersion | Yes |
| Q3 | 6 | Plot/output reading | Report endpoint \(\eta_x\), \(\eta'_x\) | Achromat requires both zero dispersion and zero slope | Yes |
| B1 | 7 | Parameter scan | Vary Q1 to cancel dispersion | Quadrupoles can match dispersion | Yes |
| Q4 | 8 | Parameter reporting | Report Q1 strength | Achromat matching as numerical design | Yes |
| B2 | 9 | Stability/matching | Enable matching, observe instability, add Q2/Q3 | Stable optics require more than dispersion cancellation | Yes |
| Q5 | 10 | Plot/output reading | Report maximum dispersion in DBA | Local zero dispersion does not mean global zero dispersion | Yes |
| Q6 | 11 | Calculation | Compute momentum spread before 2.5 cm aperture is hit | Aperture-limited momentum acceptance | Yes |
| Q7 | 11 | Diagnosis | Identify loss location | Loss occurs where beam envelope/aperture ratio is worst | Yes |
| C0 | 12 | Lattice construction | Repeat DBA 10 times to form ring | Cell-to-ring transition and periodic optics | Yes |
| Q8 | 13 | Output reading | Record \(\nu_x\), \(\nu_y\) | Tunes as ring optics quantities | Yes |
| Q9 | 14 | Conceptual explanation | Explain unequal tunes from beta functions | Phase advance depends on full focusing structure | Yes |
| Q10 | 15 | Calculation | Use chromaticities to compute tune spread | Chromaticity maps momentum spread into tune spread | Yes |
| C1 | 16–18 | Python plotting | Plot resonance lines and tune footprint | Resonance avoidance and tune footprint visualization | Partly; plotting code is local, inputs are Sirepo-dependent |
| Q11 | 19 | Graphical estimate | Increase `sigma_dp` until footprint crosses order \(\le 3\) resonance | Chromaticity-limited momentum acceptance | Inputs yes; plotting no |
| Q12 | 20 | Synthesis | Compare Q6 and Q11 limits | Momentum acceptance is set by the stricter mechanism | Yes |

---

# Suggested replacement-lab architecture

The replacement notebook can preserve the original pedagogical sequence while removing Sirepo dependency:

1. **Implement linear elements in Python**: drifts, thin or thick quadrupoles, rectangular/sector bends, and optionally edge focusing.
2. **Compute Twiss and dispersion propagation**: propagate \((\beta, \alpha, \gamma)\) and \((\eta, \eta')\) through a cell; solve matched periodic optics using transfer matrices.
3. **Track sampled particles**: generate a beam distribution with configurable emittance and momentum spread; compare tracked rms sizes with analytic formulas.
4. **Provide lattice-editing exercises in code**: replace the Sirepo GUI steps with functions, sliders, or parameter-scan cells.
5. **Optimize achromat settings**: scan or solve for Q1 and then include Q2/Q3 to restore stability.
6. **Build a ring from repeated cells**: compute one-turn matrices, tunes, and finite-difference chromaticities.
7. **Reuse or improve the resonance diagram**: retain the visible `tunediagram` concept but add clear parameter definitions and automated resonance-crossing checks.
8. **End with a comparison table**: dispersion/aperture limit versus chromaticity/resonance limit.

---

# Ambiguities to resolve when rewriting

- The original notebook uses an rms beam-size formula but asks when particles “hit the chamber wall.” A rewritten lab should specify whether the aperture condition is \(1\sigma\), \(2\sigma\), \(3\sigma\), a maximum particle excursion, or an rms proxy.
- Q2 says to calculate size at QF but also says to use maximum \(\eta\). The rewritten version should specify whether students should use \(\eta_x\) at QF or the lattice maximum.
- The original prompt asks for minimum dispersion in Q1, while Q2 later says to use maximum \(\eta\). The distinction should be made explicit.
- The notebook does not include the Sirepo lattice definitions, so bend lengths, drift lengths, and element ordering must be reconstructed or redesigned.
- The resonance-footprint plot treats `sigma_dp` as the half-length of a tune-footprint line. The rewritten lab should define whether this is rms, full acceptance, or a scanned maximum momentum offset.
