# Black Holes, Holography, and the Fine Structure Constant

**Merkabit Research Series, Papers 27 and 28**

Selina Stenberg with Claude Anthropic, April 2026

Simulation code, figures, and numerical output for two papers deriving black hole physics and the fine structure constant from the Eisenstein lattice with zero free parameters.

---

## Paper 27 — Black Holes as Torsion Envelopes

Singularity resolution, the stretched horizon, the holographic principle as a counting theorem, and the Hawking temperature from Berry phase.

### Five Results

1. **The black hole mechanism.** The envelope energy M_env = G_eff K M^2 becomes self-gravitating above M_crit. The black hole is a self-sustaining field-energy loop, not torsion saturation.

2. **Singularity resolution.** A finite T_75-saturated core of radius r_p = (a_0/2) x r_s replaces the classical singularity. The ratio r_p/r_s = 0.154 is a geometric invariant of the lattice.

3. **The stretched horizon.** One lattice shell thick. The membrane paradigm is derived from the stratum structure of PSL(2,7).

4. **The holographic principle.** The Laplace equation suppresses angular modes above l_c ~ r_p. Surviving modes scale as area. S = [ln(1/G)^3/(3pi)] x (A/4). The area law is a theorem about elliptic PDEs on discrete substrates.

5. **Hawking temperature.** T_H = |gamma_H|/(h x gamma x r_s) with gamma = 47/50 = (78-31)/(78-28) matches the standard 1/(4pi r_s) to 0.15%. The same gamma = 47/50 gives the cosmological constant to 0.2%.

### Paper 27 Simulations

| Script | Sections | Result |
|--------|----------|--------|
| `sim1_schwarzschild_radius.py` | 3.1 | Torsion potential profiles; confirms Laplace linearity; measures a_0 = 0.3077 |
| `sim1b_envelope_black_hole.py` | 3.1-3.3 | Envelope M^2 scaling; closed-form Schwarzschild; divergence at M_div = 4.87 |
| `sim1c_bootstrap_stabilisation.py` | 4.1 | Planck cutoff; first identification of saturation plateau |
| `sim1c_2_3_black_hole_interior.py` | 4.1-5.2, 6.1 | Obstacle solver; T_75 stratum classification; unconstrained entropy |
| `sim3b_holographic_entropy.py` | 6.2-6.5 | Laplace-constrained mode counting; area law (exponent 0.991) |
| `sim_fano_test.py` | 6.3, 9.2 | Overcounting convergence to K=10; filter ratio 36.5; alpha connection |
| `sim_hawking_berry.py` | 5.2, 9.4 | Berry phase at horizon; T_H to 0.15% |

### Paper 27 Figures

| File | Description |
|------|-------------|
| `fig1_interior_profile.png` | Torsion potential: flat core + transition + 1/r tail |
| `fig2_transfer_spectrum.png` | Angular mode suppression by Laplace equation |
| `fig3_entropy_vs_area.png` | S vs A/4: the area law |
| `fig4_overcounting.png` | Reduction from ~10 to 0.273 to 1.0 (Bekenstein-Hawking) |
| `fig5_schematic.png` | Three-zone structure: core / horizon / exterior |

---

## Paper 28 — The Fine Structure Constant as the Information Density of the Planck Horizon

Alpha from black hole boundary counting: two inputs, zero free parameters.

### The Formula

```
alpha^-1 = 2*h*pi * S_node / ln(1/G)
```

where h = 12 (Coxeter number), S_node = Shannon entropy of one Planck node, ln(1/G) = Planck depth.

- G_eff = 0.2542 (simulation): alpha^-1 = 136.918 (0.086%)
- G_eff = 0.25479 (exact solution): alpha^-1 = 137.036 exactly

The formula was not obtained by targeting alpha. It emerged from Paper 27's holographic filter ratio.

### Paper 28 Simulations

| Script | Purpose | Key Output |
|--------|---------|------------|
| `paper_28_alpha_holographic.py` | Evaluates formula at all G_eff values | Precision across G values |
| `paper_28_filter_ratio.py` | S_unc/S_Lap convergence | Convergence to 36.5; decomposition |
| `paper_28_running_alpha.py` | Energy-dependent filter ratio | Qualitative QED running comparison |
| `paper28_figures.py` | Publication figures | Figures 1-2 for Paper 28 |

---

## Key Constants (zero free parameters)

| Constant | Value | Source |
|----------|-------|--------|
| G_eff | 0.2542 | Paper 20 (lattice simulation) |
| a_0 | 0.3077 | Green's function amplitude |
| phi_max | 1/G_eff = 3.934 | Planck cutoff |
| K | 9.98 ~ 10 | Pentachoric edge count |
| alpha_K | 0.327 | Envelope energy slope K(r) ~ alpha_K r |
| r_p/r_s | a_0/2 = 0.154 | Geometric invariant |
| S/(A/4) | ln(1/G)^3/(3pi) = 0.273 | Holographic entropy |
| gamma | 47/50 = (78-31)/(78-28) | Non-matter / non-triality E_6 ratio |
| T_H match | 0.15% | Hawking temperature |
| alpha^-1 match | 0.086% (sim G) | Fine structure constant |
| M_div | 1/sqrt(2G^2 alpha_K) = 4.87 | Bootstrap divergence mass |

## The Connection

The algebraic fraction gamma = 47/50 determines:
- The cosmological constant to 0.2% (Papers 22-23)
- The Hawking temperature to 0.15% (Paper 27)

The holographic formula alpha^-1 = 2*h*pi * S_node / ln(1/G) uses the same G and h that underlie these connections. Cosmology, black hole thermodynamics, and the fine structure constant are connected through the same two primitive quantities.

## Requirements

- Python 3.10+
- NumPy, SciPy (gammaln, brentq), Matplotlib
- All simulations use `np.random.seed(42)` for reproducibility

## License

MIT
