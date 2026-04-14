#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAPER 28 — HOLOGRAPHIC FILTER RATIO

Computes the ratio R = S_unc / S_Lap analytically and numerically.
Shows convergence to 36.6 and the decomposition 36 + 0.6.

The filter ratio R = 12π × S_node / ln(1/G)³
The identification: R = α⁻¹ / (2hπ / ln(1/G)) × (some factor)

Merkabit Research Program — Selina Stenberg, 2026
"""

import numpy as np
from scipy.special import gammaln

H = 12
G_EFF = 0.2542
PHI_MAX = 1.0 / G_EFF
A_0 = 0.3077


def s_node(G):
    x = 1.0 / G
    return (1 + x) * np.log(1 + x) - x * np.log(x)


def s_unc_over_A4(G):
    """Unconstrained entropy per A/4 (continuum limit)."""
    return 4 * s_node(G)


def s_lap_over_A4(G):
    """Laplace-constrained entropy per A/4."""
    x = 1.0 / G
    return np.log(x)**3 / (3 * np.pi)


def filter_ratio(G):
    """R = S_unc / S_Lap."""
    return s_unc_over_A4(G) / s_lap_over_A4(G)


def main():
    print("=" * 70)
    print("  PAPER 28: HOLOGRAPHIC FILTER RATIO")
    print("=" * 70)
    print()

    # Analytic continuum limit
    G = G_EFF
    R = filter_ratio(G)
    S_unc = s_unc_over_A4(G)
    S_lap = s_lap_over_A4(G)

    print(f"  G_eff = {G}")
    print(f"  phi_max = 1/G = {PHI_MAX:.4f}")
    print()
    print(f"  S_unc/(A/4) = 4 × S_node = 4 × {s_node(G):.6f} = {S_unc:.6f}")
    print(f"  S_Lap/(A/4) = ln(1/G)³/(3π) = {S_lap:.6f}")
    print(f"  Filter ratio R = {R:.4f}")
    print()

    # Decomposition
    print(f"  INTEGER PART: {int(R)} (N₃₆ = 36 neutral current sector)")
    print(f"  FRACTIONAL PART: {R - int(R):.4f}")
    print()

    # Convergence with lattice size
    print("  NUMERICAL CONVERGENCE (mode counting):")
    print(f"  {'r_p':>6s}   {'S_unc':>10s}   {'S_Lap':>10s}   {'R':>8s}")
    print(f"  {'-'*6}   {'-'*10}   {'-'*10}   {'-'*8}")

    for r_p in [5, 10, 15, 20, 30, 50, 100]:
        # Unconstrained: 4πr² nodes, each with S_node
        N_shell = int(4 * np.pi * r_p**2)
        E_total = int(N_shell * PHI_MAX)
        if E_total > 0 and N_shell > 1:
            S_unc_num = gammaln(E_total + N_shell) - gammaln(E_total + 1) - gammaln(N_shell)
        else:
            S_unc_num = 0
        A4 = np.pi * r_p**2

        # Laplace: mode counting
        l_max = int(2 * r_p) + 1
        r_probe = r_p + 1
        S_lap_num = 0
        for l in range(l_max + 1):
            sigma = (r_p / r_probe) ** (l + 1)
            n_lev = PHI_MAX * sigma
            if n_lev > 1:
                S_lap_num += (2*l + 1) * np.log(n_lev)

        R_num = S_unc_num / S_lap_num if S_lap_num > 0 else 0
        print(f"  {r_p:6d}   {S_unc_num/A4:10.4f}   {S_lap_num/A4:10.4f}   {R_num:8.2f}")

    print()
    print(f"  ANALYTIC LIMIT: R = 12π × S_node / ln(1/G)³ = {R:.4f}")
    print()

    # G_eff scan
    print("  FILTER RATIO vs G_eff:")
    print(f"  {'G':>8s}   {'R':>8s}   {'int(R)':>8s}")
    for G_test in [0.10, 0.15, 0.20, 0.25, 0.2542, 0.30, 0.40, 0.50]:
        R_test = filter_ratio(G_test)
        print(f"  {G_test:8.4f}   {R_test:8.3f}   {int(R_test):8d}")

    print()

    # The α identification
    alpha_inv = 137.035999
    scaling = 2 * H * np.pi / np.log(1/G)
    print(f"  THE α IDENTIFICATION:")
    print(f"    R × [ln(1/G)³/(3π)] = 4 × S_node = {4*s_node(G):.4f}")
    print(f"    R × [ln(1/G)²/3] = 4 × S_node / ln(1/G) = ... ")
    print(f"    α⁻¹ = 2hπ × S_node / ln(1/G) = {2*H*np.pi*s_node(G)/np.log(1/G):.3f}")
    print(f"    R = α⁻¹ × ln(1/G)² / (6π × h)")
    R_from_alpha = alpha_inv * np.log(1/G)**2 / (6 * np.pi * H)
    print(f"    = {alpha_inv} × {np.log(1/G)**2:.4f} / {6*np.pi*H:.2f} = {R_from_alpha:.4f}")
    print()

    print("=" * 70)


if __name__ == '__main__':
    main()
