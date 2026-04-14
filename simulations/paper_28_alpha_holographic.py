#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAPER 28 — ALPHA FROM HOLOGRAPHIC BOUNDARY COUNTING

α⁻¹ = 2hπ × S_node / ln(1/G)

where:
  h = 12 (Coxeter number)
  S_node = (1+1/G)ln(1+1/G) - (1/G)ln(1/G)  (Shannon entropy per node)
  G = G_eff (lattice gravitational constant)

Evaluates the formula at all G_eff values from the series.
Finds the exact G_eff that gives α⁻¹ = 137.035999084.

Merkabit Research Program — Selina Stenberg, 2026
"""

import numpy as np
from scipy.optimize import brentq

H = 12
ALPHA_INV_PDG = 137.035999084  # PDG 2024


def s_node(G):
    """Shannon entropy of one Planck-boundary node at max occupation."""
    x = 1.0 / G
    return (1 + x) * np.log(1 + x) - x * np.log(x)


def alpha_inv(G):
    """α⁻¹ = 2hπ × S_node / ln(1/G)"""
    x = 1.0 / G
    return 2 * H * np.pi * s_node(G) / np.log(x)


def main():
    print("=" * 70)
    print("  PAPER 28: α FROM HOLOGRAPHIC BOUNDARY COUNTING")
    print("=" * 70)
    print()
    print(f"  Formula: α⁻¹ = 2hπ × S_node / ln(1/G)")
    print(f"  h = {H}, PDG α⁻¹ = {ALPHA_INV_PDG}")
    print()

    # Table 3: precision across G_eff values
    print(f"  {'G_eff':>10s}   {'Source':>30s}   {'α⁻¹':>10s}   {'Deviation':>10s}")
    print(f"  {'-'*10}   {'-'*30}   {'-'*10}   {'-'*10}")

    cases = [
        (0.2542,  "Lattice simulation (Paper 20)"),
        (0.2500,  "Algebraic G = 1/4 (Paper 23)"),
        (0.1210,  "Holographic S=A/4 (Paper 27)"),
    ]

    for G, source in cases:
        a = alpha_inv(G)
        dev = abs(a - ALPHA_INV_PDG) / ALPHA_INV_PDG * 100
        print(f"  {G:10.4f}   {source:>30s}   {a:10.3f}   {dev:9.3f}%")

    # Find exact G
    def f(G):
        return alpha_inv(G) - ALPHA_INV_PDG

    G_exact = brentq(f, 0.01, 0.99)
    a_exact = alpha_inv(G_exact)

    print(f"  {G_exact:10.5f}   {'Exact solution':>30s}   {a_exact:10.6f}   {'exact':>10s}")
    print()

    # Detailed breakdown
    G = 0.2542
    x = 1.0 / G
    sn = s_node(G)
    ln_x = np.log(x)

    print("  FACTOR DECOMPOSITION (G = 0.2542):")
    print(f"    2hπ        = 2 × {H} × π = {2*H*np.pi:.4f}")
    print(f"    S_node     = (1+{x:.3f})ln({1+x:.3f}) - {x:.3f}ln({x:.3f}) = {sn:.6f}")
    print(f"    ln(1/G)    = ln({x:.3f}) = {ln_x:.6f}")
    print(f"    α⁻¹        = {2*H*np.pi:.4f} × {sn:.4f} / {ln_x:.4f} = {alpha_inv(G):.3f}")
    print()

    # Distance from exact
    print(f"  G_eff (simulation) = {0.2542}")
    print(f"  G_eff (exact)      = {G_exact:.6f}")
    print(f"  G_eff (algebraic)  = 0.2500")
    print(f"  Simulation vs exact: {abs(0.2542 - G_exact)/G_exact*100:.3f}%")
    print(f"  Algebraic vs exact:  {abs(0.25 - G_exact)/G_exact*100:.3f}%")
    print()

    # The 47/50 connection
    gamma = 47.0 / 50.0
    print("  CONNECTION TO γ = 47/50:")
    print(f"    γ = 47/50 = (78-31)/(78-28) = {gamma}")
    print(f"    Same γ gives Λ to 0.2% (Papers 22-23)")
    print(f"    Same γ gives T_H to 0.15% (Paper 27 §5.2)")
    print(f"    The filter ratio S_unc/S_Lap ≈ α⁻¹ / (2hπ/ln(1/G))")
    print(f"    = {ALPHA_INV_PDG} / {2*H*np.pi/np.log(1/0.2542):.4f} = {ALPHA_INV_PDG/(2*H*np.pi/np.log(1/0.2542)):.4f}")
    print()

    print("=" * 70)


if __name__ == '__main__':
    main()
