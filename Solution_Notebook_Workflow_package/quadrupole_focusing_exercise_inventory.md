# Quadrupole Focusing Lab: Exercise Inventory and Replacement Map

This document inventories the original Sirepo/Elegant-based `Quadrupole-Focusing.ipynb` lab. It is intended as a design template for a fully local Xsuite replacement.

## Global lab purpose

The lab introduces transverse beam transport through quadrupole focusing systems. The original worksheet used Sirepo/Elegant to let students manipulate a FODO cell, propagate matched and mismatched Twiss parameters, observe single-particle and beam-envelope oscillations, change quadrupole strengths, construct a hybrid lattice with a focusing-strength transition, and optimize a four-quadrupole matching section.

The central learning arc is:

1. A periodic focusing lattice has a matched Twiss solution.
2. The matched solution determines beta functions, phase advance, and RMS beam sizes.
3. Thin-lens formulae are useful but differ from thick-lens tracking.
4. A matched single FODO cell remains matched when repeated.
5. A beam injected with the wrong Twiss parameters exhibits envelope beating.
6. Weaker quadrupole focusing produces larger matched beam sizes and lower phase advance.
7. A change from one periodic cell type to another creates mismatch unless a matching section is used.
8. Matching sections use quadrupoles as knobs to transform Twiss parameters into specified targets.

## Original setup and ungraded orientation tasks

### Setup: Open and inspect `FODObeamline`

**Original Sirepo action:** Students opened the Sirepo/Elegant simulation named `FODObeamline`, copied it into their workspace, and inspected the Control tab. The simulation represented a simple electron FODO beamline using matrix propagation and a 5000-particle bunch.

**What it was showing:** The lab established the beam and lattice parameters: 1 GeV electrons, equal geometric emittances of 6 mm mrad, a 5 m FODO period made from a focusing quadrupole, drift, defocusing quadrupole, and drift, quadrupole length 0.5 m, and quadrupole geometric strength $K_1=0.6\ \mathrm{m}^{-2}$.

**Student task:** Learn where Twiss calculation, bunch generation, and tracking were controlled in Sirepo/Elegant.

**Intended learning:** Recognize the three components of the simulation: optics calculation, beam generation, and particle/centroid tracking. Connect the graphical interface to the accelerator quantities used later.

**Replacement requirement:** The local notebook should make the same parameters explicit in a table and then build the FODO line directly in Xsuite. Sirepo control-panel operations should become visible Python variables, for example `k1`, `n_cells`, `quad_length`, `betx`, and `alfx`.

### Section A: Unmatched beam in one FODO cell

**Original Sirepo action:** Students selected `FODOcell`, ran with `Matched = No`, and used initial values $\beta_x=\beta_y=4\ \mathrm{m}$ and $\alpha_x=\alpha_y=0$.

**What it was showing:** A Twiss solution propagated from arbitrary initial values is generally not periodic through a periodic lattice. The `sigma_output` plot followed the same qualitative pattern as the beta functions but on the millimeter RMS beam-size scale.

**Student task:** Visually compare unmatched beta functions and RMS beam-size plots.

**Intended learning:** A periodic lattice does not automatically make every input beam periodic. The matched condition is a special set of Twiss parameters. RMS beam size is related to beta by $\sigma=\sqrt{\beta\epsilon}$.

**Replacement requirement:** The local notebook should propagate a deliberately unmatched Twiss vector through one Xsuite FODO cell, then plot both beta functions and RMS beam size. This is a conceptual setup for Q0 rather than a numbered deliverable.

## Numbered exercises

### Q0: Phase advance of one matched FODO cell

**Original prompt:** Calculate the X and Y phase advances for a single FODO cell using `nux` and `nuy` from Elegant output parameters, with $\psi=\nu 2\pi$.

**What it was trying to simulate or show:** Elegant's matched Twiss calculation finds the periodic one-cell solution and reports tune-like phase advance values. Since the line is one cell long, the tune per pass is the phase advance per cell in turns.

**What students were asked to do:** Switch `Matched` to `Yes`, rerun `FODOcell`, find `nux` and `nuy`, and report $\psi_x$ and $\psi_y$.

**What students were supposed to learn:** Phase advance is the integral $\int ds/\beta(s)$ and is a primary descriptor of a periodic focusing cell. In a symmetric FODO cell, the horizontal and vertical phase advances are equal.

**Replacement requirement:** Xsuite should call `line.twiss(method="4d")` for the one-cell line and report `tw.qx`, `tw.qy`, $2\pi Q_x$, $2\pi Q_y$, and degrees. The answer block should clarify the unit convention: Xsuite `qx`/`qy` are in turns per pass, while $\psi$ in radians is $2\pi Q$.

### Q1: Thin-lens versus thick-lens beta extrema

**Original prompt:** For the matched cell, calculate $\beta_{\min}$ and $\beta_{\max}$ in two ways: (A) thin-lens prediction and (B) Elegant/Twiss output. The two answers should be close but slightly different.

**What it was trying to simulate or show:** The thin-lens FODO formula gives a useful analytic approximation, while Elegant/Xsuite thick-lens transport includes finite quadrupole length. The difference is a model approximation error, not a numerical error.

**What students were asked to do:** Use the listed phase advance in the formula

\[
\beta_{\max}=L\frac{1+\sin(\psi/2)}{\sin\psi},\qquad
\beta_{\min}=L\frac{1-\sin(\psi/2)}{\sin\psi},
\]

then compare with the simulated beta extrema from the Twiss plot or output parameters.

**What students were supposed to learn:** Analytic thin-lens estimates should be compared against thick-lens lattice functions. The finite quadrupole length matters, but for this cell the approximation is qualitatively and numerically close.

**Replacement requirement:** The notebook should compute both values directly. It should also flag a convention issue in the old worksheet: the parameter table calls $L=2.5\ \mathrm{m}$ the half-cell length, but the comparison described as “quite close” uses the 5 m FODO period in the printed formula. The replacement should expose `length_for_formula=5.0` explicitly so students do not silently inherit the ambiguity.

### Q2: Effect of increasing quadrupole length at fixed cell length and phase advance

**Original prompt:** If quadrupole lengths are increased while holding cell length $L$ and phase advance $\psi$ fixed, does the difference between the thin-lens prediction and Elegant result get larger or smaller? Explain.

**What it was trying to simulate or show:** Thin-lens approximations become less valid as focusing is spread over a longer physical element. Holding phase advance fixed isolates the effect of distributed focusing from a trivial change in net focusing strength.

**What students were asked to do:** Reason qualitatively from the thin-lens assumption and the finite quadrupole model.

**What students were supposed to learn:** Model assumptions have domains of validity. Longer quadrupoles make the thick-lens nature of the lattice more important, so the discrepancy grows.

**Replacement requirement:** The Xsuite notebook should not leave this as only a verbal claim. It should include an interactive or tabulated scan in which quadrupole length changes, `k1` is adjusted to keep the one-cell phase advance fixed, and the thick-lens beta extrema are compared against the same thin-lens values.

### Q3: RMS beam-size statistics in the matched cell

**Original prompt:** Find the average, maximum, and minimum RMS spot sizes for the matched beam using $\sigma_x=\sqrt{\beta_x\epsilon_x}$ and the Twiss parameters. Fill a table for $\langle\sigma_x\rangle_s$, $\langle\sigma_y\rangle_s$, maxima, minima, and maximum aspect ratio.

**What it was trying to simulate or show:** Beta functions are not directly beam sizes; emittance converts optics functions into RMS beam envelopes. Alternating-gradient focusing produces a beam that changes shape and size through the cell.

**What students were asked to do:** Use Twiss values and emittance to compute RMS beam sizes, using plot values or output tables.

**What students were supposed to learn:** The envelope follows $\sqrt{\beta}$. Equal emittances do not imply equal beam sizes everywhere; the beam alternates between horizontally and vertically wider regions.

**Replacement requirement:** The notebook should compute a dense Twiss sample and produce an RMS-size summary table. Dense sampling is preferable to only element-boundary values because the extrema occur inside the quadrupoles.

### Q4: Locations of round beam and size extrema

**Original prompt:** Identify where along the lattice the beam is round and where it is largest/smallest in the horizontal and vertical planes. Give both $s$ in meters and the lattice element or region.

**What it was trying to simulate or show:** The matched FODO beam envelope has symmetry points. The beam is round at mid-drifts/symmetry points and has extrema in the focusing or defocusing quadrupoles.

**What students were asked to do:** Inspect Twiss or sigma plots, click on relevant plot points, and complete a location table.

**What students were supposed to learn:** Beam-size extrema are connected to the focusing role of each quadrupole: QF increases horizontal focusing and vertical defocusing; QD does the opposite. The symmetry of the FODO cell determines where $\sigma_x=\sigma_y$.

**Replacement requirement:** The Xsuite notebook should compute the locations automatically from the dense sampled optics and also ask students to interpret the result physically.

### Q5: Tune consistency between one cell and 20 repeated cells

**Original prompt:** For `FODObeamline`, which contains 20 repeated FODO cells over 100 m, confirm that the tune is consistent with the one-cell solution. Fill in total tune/phase advance and per-cell tune/phase advance.

**What it was trying to simulate or show:** Repeating a periodic cell adds phase advance linearly. The matched envelope remains periodic with the cell length through the full beamline.

**What students were asked to do:** Select the 100 m beamline, run the matched simulation, read total $\nu_x$, $\nu_y$, $\psi_x$, $\psi_y$, and divide by 20.

**What students were supposed to learn:** Total tune over a repeated transport line equals the number of cells multiplied by the cell tune. The matched solution for one cell tiles the full line.

**Replacement requirement:** Build a 20-cell Xsuite line, compute the periodic Twiss solution, and show both the total tune and total tune divided by 20.

### Q6: Single-particle betatron oscillations over 100 m

**Original prompt:** Determine how many oscillations a single particle makes in 100 m. Students could calculate from tune or visualize centroid motion by giving the beam a centroid offset.

**What it was trying to simulate or show:** A bunch centroid offset follows the same linear betatron equation as a single particle. Centroid motion visualizes the betatron phase advance.

**What students were asked to do:** Introduce a centroid offset in Sirepo/Elegant, inspect `run_setup.centroid`, and count oscillations; or compute directly from the tune.

**What students were supposed to learn:** Tune is the number of transverse betatron oscillations over the line period. A centroid error is a practical way to observe single-particle-like motion.

**Replacement requirement:** The Xsuite notebook should launch an initial horizontal offset, plot the centroid/orbit through the 100 m line, and report the number of oscillations from `tw.qx`.

### Q7: Envelope beating from a 10% mismatch

**Original prompt:** Initialize the beam with a 10% beta mismatch relative to the matched solution and count the approximate number of envelope mismatch oscillations over the 100 m beamline. Compare to the smooth-approximation prediction $\nu_{\mathrm{envelope}}=2\nu$.

**What it was trying to simulate or show:** A mismatched beam envelope oscillates about the matched envelope. The mismatch oscillation is slower and conceptually distinct from the local FODO-cell envelope modulation.

**What students were asked to do:** Turn off matched Twiss generation, set initial beta values to 1.1 times the matched beta while keeping alpha fixed, plot the resulting envelope, count the slow beat oscillations, and compare with theory.

**What students were supposed to learn:** Beam envelopes have their own oscillation modes. In smooth focusing, envelope mismatch oscillates at twice the single-particle betatron tune. Students also learn not to count every local FODO-cell maximum as a mismatch oscillation.

**Replacement requirement:** The replacement should include both the raw matched/mismatched RMS-size plot and a cell-boundary or smoothed view that makes the slow envelope beating countable.

### Q8: Matched solution for a weaker FODO cell

**Original prompt:** Construct a new `FODOcell2` with $|k_1|=0.2\ \mathrm{m}^{-2}$, compute its matched solution, and record beam-size statistics, aspect ratio, and phase advances.

**What it was trying to simulate or show:** Reducing quadrupole strength weakens focusing, lowers phase advance, and changes the matched beta functions and beam sizes. With fixed emittance, weaker focusing produces larger beam sizes.

**What students were asked to do:** Copy the original cell and quadrupoles, change only the quadrupole strengths, run matched Twiss, and fill the statistics table.

**What students were supposed to learn:** Matched optics depend on lattice strength. Lower focusing strength increases average beam size and reduces the alternating horizontal/vertical aspect ratio.

**Replacement requirement:** The Xsuite notebook should build the weaker cell directly from a visible `k1_weak` variable and include an interactive strength slider so students can test the trend continuously.

### Q9: Injection mismatch when transitioning from strong to weak FODO cells

**Original prompt:** Build a hybrid 100 m line containing 10 strong cells followed by 10 weaker cells with $|k_1|=0.5\ \mathrm{m}^{-2}$. Start with the matched Twiss parameters of the strong cell and describe what happens at the transition.

**What it was trying to simulate or show:** A beam matched to one periodic lattice section is generally not matched to a different lattice section. The envelope remains well behaved in the first section and then begins beating after the transition.

**What students were asked to do:** Modify the beamline composition, set initial Twiss values to the strong-cell matched values, run with `Matched = No`, and describe the envelope before and after the transition.

**What students were supposed to learn:** Matching is section-specific. Changing lattice strength without a matching section creates injection mismatch into the downstream section.

**Replacement requirement:** The Xsuite replacement should build the hybrid line programmatically and mark the transition location in the plot. An interactive slider for the second-section strength is useful because it shows mismatch amplitude grows as the downstream cell differs more strongly from the upstream matched cell.

### Q10: Matched solution over the full hybrid period versus one-cell matching

**Original prompt:** Ask Elegant for the periodic solution of the entire 100 m hybrid line and determine whether it is matched over $L=100\ \mathrm{m}$, over $L=5\ \mathrm{m}$, and whether any solution can be matched every 5 m in all 20 cells.

**What it was trying to simulate or show:** The word “matched” depends on the period chosen. A periodic solution exists for the full 100 m hybrid line, but because the cell type changes after 50 m it is not periodic every 5 m.

**What students were asked to do:** Turn `Matched = Yes` for the hybrid line and inspect the resulting optics.

**What students were supposed to learn:** Periodicity must match the lattice symmetry. A superperiodic solution can be matched over a long pattern while still being mismatched relative to either local cell period.

**Replacement requirement:** The Xsuite notebook should compute the periodic Twiss solution of the full hybrid line and explicitly compare start/end conditions over 100 m with local 5 m cell periodicity.

### Q11: Input and output phase-space distributions for a matching section

**Original prompt:** Attach graphs of the input and output $xx'$ distributions for `MATCHsection`.

**What it was trying to simulate or show:** The matching section transforms the beam's phase-space ellipse/distribution. The output should be closer to a round, uncorrelated target distribution in configuration space and angular correlation.

**What students were asked to do:** Build the matching section, run optimization, save the source and visualization graphs, and insert them into the notebook.

**What students were supposed to learn:** Twiss parameters are visible in phase space: $\alpha\ne0$ corresponds to a tilted/correlated ellipse, while $\alpha=0$ corresponds to no linear correlation between coordinate and angle.

**Replacement requirement:** The Xsuite notebook should generate the input/output phase-space plots directly. A tracked Gaussian bunch scatter plot is more faithful to the original Sirepo particle distribution than only plotting ideal ellipses, though showing ellipses is also pedagogically useful.

### Q12: Beta functions before and after matching optimization

**Original prompt:** Attach graphs of $\beta_{x,y}$ before and after optimization.

**What it was trying to simulate or show:** The quadrupole optimizer changes the optics through the matching section so that the downstream constraints are satisfied.

**What students were asked to do:** Compare `twiss_output` before optimization and `twiss_output2` after optimization.

**What students were supposed to learn:** Matching is an optics-design problem: quadrupole strengths are varied to reshape beta functions and alphas to meet specified endpoint constraints.

**Replacement requirement:** The Xsuite notebook should compute and plot before/after beta functions on the same axes.

### Q13: Final beta values at the end of `MATCHsection`

**Original prompt:** Report final $\beta_x$ and $\beta_y$ at the end of `MATCHsection`.

**What it was trying to simulate or show:** The matching target includes equal final beta functions.

**What students were asked to do:** Read values from the plot, output parameters, or Elegant log file.

**What students were supposed to learn:** Endpoint constraints can be verified quantitatively, not only visually.

**Replacement requirement:** The Xsuite notebook should print a table containing final $\beta_x$, $\beta_y$, $\alpha_x$, and $\alpha_y` after optimization.

### Q14: Optimized quadrupole strengths in the matching section

**Original prompt:** Report optimized strengths of the four quadrupoles in `MATCHsection`.

**What it was trying to simulate or show:** Matching is achieved by varying physical lattice parameters, here the quadrupole gradients.

**What students were asked to do:** Read the optimized values from the saved lattice file or Elegant log.

**What students were supposed to learn:** The output optics are produced by a concrete set of magnet settings. The optimizer's result should be inspected and assessed physically.

**Replacement requirement:** The Xsuite notebook should use `line.match(...)` with four visible quadrupole knobs and print the optimized values. A manual slider cell should let students try to satisfy the constraints by hand before or after seeing the optimizer result.

### Q15: Extra credit injection insertion

**Original prompt:** Design an entire injection insertion by appending a reversed version of `MATCHsection` and show the resulting optics.

**What it was trying to simulate or show:** A matching section can be used as an insertion: match out of a regular FODO lattice into a desired special region, then use the reversed section to return to the regular lattice.

**What students were asked to do:** Reverse the matching-section elements, append them to the injection beamline, rerun the simulation, and attach the beta-function plot.

**What students were supposed to learn:** Accelerator insertions are constructed by composing optics modules. Symmetry and reversed sections can be used to recover the original lattice conditions after a special region.

**Replacement requirement:** The Xsuite notebook should include an optional cell that programmatically builds `FODO + MATCH + reversed MATCH` and plots the resulting beta functions.

## Important ambiguities or repair points for the replacement

1. **Thin-lens length convention in Q1:** The original parameter table states that $L=2.5\ \mathrm{m}$ is the half-length of the FODO cell, but the claim that the thin-lens values are close to the thick-lens values is consistent with using the 5.0 m cell period in the printed formula. The replacement should state the convention explicitly and preferably show it as a variable.

2. **Element-boundary versus dense extrema:** Xsuite, like many optics tools, naturally reports values at element boundaries unless the line is sliced or sampled more densely. The original Sirepo plots visually represented continuous behavior through elements. The replacement should use dense sampling for extrema, averages, and location questions.

3. **Counting mismatch oscillations:** The raw $\sigma_x(s)$ plot contains both local FODO-period variation and the slower mismatch beat. The replacement should include a cell-boundary or smoothed plot to focus the students on the intended envelope mismatch oscillation.

4. **Distribution plots in Q11:** The prior local attempt used ideal phase-space ellipses. That is useful, but the original asked for particle-distribution plots. The replacement should preferably show both a tracked Gaussian bunch and ideal one-rms ellipses.

5. **Matching-section solution visibility:** The old Sirepo activity taught students to define variables and optimization constraints. The Xsuite replacement should not hide the conceptual target: $\beta_x=\beta_y$, $\alpha_x=0$, and $\alpha_y=0$ at the endpoint.
