#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAPER 27 — FANO PLANE TEST

Does the overcounting factor converge to EXACTLY 7 as lattice
resolution increases?

If yes: holography IS Fano geometry.
If no: 7.3 is the true value and 7 is a coincidence.

Method: Compute S_unconstrained / (A/4) at multiple lattice sizes
and multiple r_p values. Track convergence.
"""

import numpy as np
from scipy.special import gammaln
from datetime import datetime

G_EFF = 0.2542
PHI_MAX = 1.0 / G_EFF
A_0 = 0.3077

def compute_shells_fast(L):
    """Compute shell node counts for an L^3 lattice."""
    H = L // 2
    coords = np.arange(L) - H
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    R_int = np.round(np.sqrt(X**2 + Y**2 + Z**2)).astype(int)
    shells = {}
    for r in range(0, H + 1):
        n = int(np.sum(R_int == r))
        if n > 0:
            shells[r] = n
    return shells


def unconstrained_entropy(N_nodes, E_total):
    """Stars-and-bars: ln C(E+N-1, N-1)."""
    if E_total <= 0 or N_nodes <= 1:
        return 0.0
    return gammaln(E_total + N_nodes) - gammaln(E_total + 1) - gammaln(N_nodes)


def laplace_entropy(r_p, phi_max):
    """Laplace-constrained entropy from angular mode counting."""
    if r_p < 1:
        return 0.0
    l_max = int(2 * r_p) + 1
    r_probe = r_p + 1
    S = 0.0
    for l in range(l_max + 1):
        sigma = (r_p / r_probe) ** (l + 1)
        n_lev = phi_max * sigma
        if n_lev > 1:
            S += (2*l + 1) * np.log(n_lev)
    return S


def run_test():
    start = datetime.now()
    out = []
    def log(s=""):
        print(s); out.append(str(s))

    log("=" * 70)
    log("  FANO PLANE TEST: DOES THE OVERCOUNTING CONVERGE TO 7?")
    log("=" * 70)
    log()

    # ================================================================
    #  TEST 1: Fixed physics, varying lattice size
    # ================================================================
    log("  TEST 1: FIXED r_p, VARYING LATTICE SIZE")
    log("  (Same physical black hole, better resolution)")
    log()

    # For each lattice size, compute the horizon shell at various r_p
    lattice_sizes = [21, 31, 41, 51, 61, 71, 81]

    log(f"  {'L':>4s}   {'r_p':>4s}   {'N_hor':>7s}   {'E_total':>8s}   {'S_unc':>10s}"
        f"   {'A/4':>8s}   {'S/(A/4)':>8s}   {'S_Lap':>8s}   {'S_unc/S_Lap':>12s}")
    log(f"  {'-'*4}   {'-'*4}   {'-'*7}   {'-'*8}   {'-'*10}"
        f"   {'-'*8}   {'-'*8}   {'-'*8}   {'-'*12}")

    # Collect data for convergence analysis
    convergence_data = {}  # r_p -> list of (L, ratio)

    for L in lattice_sizes:
        shells = compute_shells_fast(L)
        H = L // 2

        for r_p in [3, 5, 7, 10, 12, 15]:
            if r_p >= H - 2:
                continue

            # Horizon shell: r = r_p to r_p+2
            N_hor = 0
            E_total = 0
            for r in range(r_p, min(r_p + 3, H)):
                if r in shells:
                    N_r = shells[r]
                    N_hor += N_r
                    # phi at this shell (from obstacle solver profile)
                    if r <= r_p:
                        phi_r = PHI_MAX
                    else:
                        phi_r = PHI_MAX * r_p / r
                    E_total += int(N_r * phi_r)

            if N_hor == 0 or E_total == 0:
                continue

            S_unc = unconstrained_entropy(N_hor, E_total)
            A_over_4 = np.pi * r_p**2
            ratio_unc = S_unc / A_over_4 if A_over_4 > 0 else 0

            S_lap = laplace_entropy(r_p, PHI_MAX)
            ratio_lap = S_unc / S_lap if S_lap > 0 else 0

            if r_p not in convergence_data:
                convergence_data[r_p] = []
            convergence_data[r_p].append((L, ratio_unc, ratio_lap))

            log(f"  {L:4d}   {r_p:4d}   {N_hor:7d}   {E_total:8d}   {S_unc:10.1f}"
                f"   {A_over_4:8.1f}   {ratio_unc:8.4f}   {S_lap:8.2f}   {ratio_lap:12.4f}")

    log()

    # ================================================================
    #  TEST 2: Convergence analysis per r_p
    # ================================================================
    log("=" * 70)
    log("  TEST 2: CONVERGENCE OF S_unc/(A/4) WITH LATTICE SIZE")
    log("=" * 70)
    log()

    for r_p in sorted(convergence_data.keys()):
        data = convergence_data[r_p]
        log(f"  r_p = {r_p}:")
        ratios = [d[1] for d in data]
        Ls = [d[0] for d in data]
        for L, ratio_unc, ratio_lap in data:
            log(f"    L={L:3d}: S_unc/(A/4) = {ratio_unc:.4f}   S_unc/S_Lap = {ratio_lap:.4f}")
        if len(ratios) >= 3:
            # Extrapolate: fit ratio = a + b/L
            L_arr = np.array(Ls, dtype=float)
            r_arr = np.array(ratios)
            A_mat = np.vstack([np.ones_like(L_arr), 1.0/L_arr]).T
            coeffs = np.linalg.lstsq(A_mat, r_arr, rcond=None)[0]
            r_inf = coeffs[0]
            log(f"    Extrapolation (1/L -> 0): S_unc/(A/4) -> {r_inf:.4f}")
            log(f"    Distance from 7: {abs(r_inf - 7):.4f}")
            log(f"    Distance from 7.3: {abs(r_inf - 7.3):.4f}")
        log()

    # ================================================================
    #  TEST 3: Large r_p limit (analytic)
    # ================================================================
    log("=" * 70)
    log("  TEST 3: ANALYTIC LARGE-r_p LIMIT")
    log("=" * 70)
    log()

    log("  For the stars-and-bars entropy on a spherical shell:")
    log("    N_horizon ~ 4*pi*r_p^2 (continuum limit)")
    log("    E_total ~ N_horizon * phi_max (all nodes at phi_max for transition shell)")
    log("    S_unc = gammaln(E+N) - gammaln(E+1) - gammaln(N)")
    log()
    log("  For E/N = phi_max = const, Stirling gives:")
    log("    S_unc/N ~ (1 + phi_max)*ln(1 + phi_max) - phi_max*ln(phi_max)")
    log(f"    = (1 + {PHI_MAX:.3f})*ln(1 + {PHI_MAX:.3f}) - {PHI_MAX:.3f}*ln({PHI_MAX:.3f})")

    x = PHI_MAX
    s_per_n = (1 + x) * np.log(1 + x) - x * np.log(x)
    log(f"    = {s_per_n:.6f}")
    log()
    log(f"  S_unc/(A/4) = S_unc / (pi*r_p^2)")
    log(f"             = N_horizon * s_per_n / (pi*r_p^2)")
    log(f"             = 4*pi*r_p^2 * s_per_n / (pi*r_p^2)")
    log(f"             = 4 * s_per_n")
    log(f"             = 4 * {s_per_n:.6f}")
    log(f"             = {4 * s_per_n:.6f}")
    log()

    analytic_ratio = 4 * s_per_n
    log(f"  ANALYTIC RESULT: S_unc/(A/4) -> {analytic_ratio:.4f} in continuum limit")
    log(f"  Distance from 7.000: {abs(analytic_ratio - 7):.4f}")
    log(f"  Distance from 7.300: {abs(analytic_ratio - 7.3):.4f}")
    log()

    # ================================================================
    #  TEST 4: What phi_max gives exactly 7?
    # ================================================================
    log("=" * 70)
    log("  TEST 4: WHAT phi_max GIVES S_unc/(A/4) = 7 EXACTLY?")
    log("=" * 70)
    log()

    log("  S_unc/(A/4) = 4 * [(1+x)*ln(1+x) - x*ln(x)] where x = phi_max")
    log("  Solve: 4*[(1+x)*ln(1+x) - x*ln(x)] = 7")
    log("  => (1+x)*ln(1+x) - x*ln(x) = 7/4 = 1.75")
    log()

    # Numerical solve
    from scipy.optimize import brentq
    def f(x):
        return (1+x)*np.log(1+x) - x*np.log(x) - 7.0/4.0

    x_7 = brentq(f, 1.01, 100)
    log(f"  phi_max for ratio = 7: {x_7:.6f}")
    log(f"  G_eff = 1/phi_max = {1/x_7:.6f}")
    log()

    # Check some special values
    log("  SPECIAL VALUES:")
    for label, x_test in [
        ("1/G_eff = 3.934", PHI_MAX),
        ("4 (= 1/0.25)", 4.0),
        ("e^(3pi)^{1/3} = 8.267", np.exp((3*np.pi)**(1./3))),
        ("7 (Fano lines)", 7.0),
        ("phi for ratio=7", x_7),
    ]:
        ratio_test = 4 * ((1+x_test)*np.log(1+x_test) - x_test*np.log(x_test))
        log(f"    phi_max = {x_test:.4f} ({label}): S_unc/(A/4) = {ratio_test:.4f}")

    log()

    # ================================================================
    #  TEST 5: The ratio S_unc / S_Laplace
    # ================================================================
    log("=" * 70)
    log("  TEST 5: RATIO S_unc / S_Laplace (THE FILTER FACTOR)")
    log("=" * 70)
    log()

    log("  If the Fano plane governs the filter, S_unc/S_Lap should be")
    log("  a multiple or function of 7.")
    log()

    for r_p in [5, 10, 15, 20, 30, 50, 100]:
        N_cont = int(4 * np.pi * r_p**2)
        E_cont = int(N_cont * PHI_MAX)
        S_unc = unconstrained_entropy(N_cont, E_cont)
        S_lap = laplace_entropy(r_p, PHI_MAX)
        A4 = np.pi * r_p**2
        ratio_filter = S_unc / S_lap if S_lap > 0 else 0

        log(f"  r_p = {r_p:4d}: S_unc = {S_unc:10.1f}, S_Lap = {S_lap:8.1f},"
            f" S_unc/S_Lap = {ratio_filter:8.2f},"
            f" S_unc/(A/4) = {S_unc/A4:.4f}")

    log()

    # ================================================================
    #  TEST 6: Exact computation at large L with transition-shell phi
    # ================================================================
    log("=" * 70)
    log("  TEST 6: TRANSITION-SHELL ENTROPY (phi < phi_max)")
    log("=" * 70)
    log()
    log("  The horizon shell is NOT at phi_max. It is at the TRANSITION")
    log("  where phi drops from phi_max to the 1/r tail.")
    log("  Typical phi at horizon: phi_max * r_p/(r_p+1)")
    log()

    for r_p in [5, 10, 20, 50, 100]:
        # Transition shell at r = r_p+1
        phi_trans = PHI_MAX * r_p / (r_p + 1)
        N_trans = int(4 * np.pi * (r_p + 1)**2)
        E_trans = int(N_trans * phi_trans)
        S_trans = unconstrained_entropy(N_trans, E_trans)
        A4 = np.pi * r_p**2
        ratio_trans = S_trans / A4 if A4 > 0 else 0

        log(f"  r_p = {r_p:4d}: phi_trans = {phi_trans:.3f},"
            f" N = {N_trans:7d}, S = {S_trans:10.1f},"
            f" S/(A/4) = {ratio_trans:.4f}")

    log()

    # ================================================================
    #  VERDICT
    # ================================================================
    log("=" * 70)
    log("  VERDICT")
    log("=" * 70)
    log()
    log(f"  Analytic continuum limit (all nodes at phi_max):")
    log(f"    S_unc/(A/4) = 4*[(1+phi_max)*ln(1+phi_max) - phi_max*ln(phi_max)]")
    log(f"                = {analytic_ratio:.4f}")
    log()
    log(f"  For this to equal 7 exactly, need phi_max = {x_7:.4f}")
    log(f"  Our phi_max = 1/G_eff = {PHI_MAX:.4f}")
    log()

    if abs(analytic_ratio - 7) < 0.5:
        log("  CLOSE TO 7. The Fano connection is plausible.")
    elif abs(analytic_ratio - 7) < 1.5:
        log("  WITHIN RANGE. Depends on which shell defines the horizon.")
    else:
        log("  NOT 7. The overcounting is determined by phi_max, not by 7.")

    log()
    log(f"  The overcounting factor is a FUNCTION of phi_max = 1/G_eff.")
    log(f"  It equals 7 when G_eff = 1/{x_7:.4f} = {1/x_7:.6f}.")
    log(f"  Our G_eff = {G_EFF} gives ratio = {analytic_ratio:.4f}.")
    log()

    elapsed = (datetime.now() - start).total_seconds()
    log(f"  Runtime: {elapsed:.1f} seconds")
    log("=" * 70)

    with open('C:/Users/selin/merkabit_results/black_holes/sim_fano_output.txt',
              'w', encoding='utf-8') as f:
        f.write('\n'.join(out))


if __name__ == '__main__':
    run_test()
