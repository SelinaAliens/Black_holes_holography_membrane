#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAPER 28 — RUNNING OF α WITH ENERGY

At energy scale E, the effective horizon size changes.
The mode cutoff l_c shifts, the filter ratio changes, α runs.

This script computes the qualitative running of α from the
holographic formula and compares to the QED beta function.

Merkabit Research Program — Selina Stenberg, 2026
"""

import numpy as np

H = 12
G_EFF = 0.2542
ALPHA_INV_LOW = 137.036     # α⁻¹ at low energy (Thomson limit)
ALPHA_INV_MZ = 127.952      # α⁻¹ at M_Z = 91.2 GeV (PDG)
M_PLANCK_GEV = 1.22089e19   # Planck mass in GeV
M_Z_GEV = 91.1876           # Z boson mass


def s_node(G):
    x = 1.0 / G
    return (1 + x) * np.log(1 + x) - x * np.log(x)


def alpha_inv_formula(G):
    x = 1.0 / G
    return 2 * H * np.pi * s_node(G) / np.log(x)


def main():
    print("=" * 70)
    print("  PAPER 28: RUNNING OF α WITH ENERGY SCALE")
    print("=" * 70)
    print()

    # At low energy: the full lattice contributes
    # At high energy E: the effective horizon size shrinks
    # The mode cutoff l_c ~ r_eff, and r_eff ~ M_Planck/E
    #
    # So G_eff(E) = G_eff × (1 + correction from energy scale)
    # In QED: α⁻¹(E) = α⁻¹(0) - (1/3π) × Σ_f Q_f² × ln(E/m_f)
    # The holographic version: α⁻¹(E) = 2hπ × S_node(G_eff(E)) / ln(1/G_eff(E))
    # where G_eff(E) encodes the running

    # Approach: find G_eff(M_Z) such that α⁻¹(G_eff(M_Z)) = 127.952
    from scipy.optimize import brentq

    def f(G):
        return alpha_inv_formula(G) - ALPHA_INV_MZ

    G_at_MZ = brentq(f, 0.01, 0.99)

    print(f"  LOW ENERGY:")
    print(f"    α⁻¹ = {ALPHA_INV_LOW}")
    print(f"    G_eff = {G_EFF} (lattice simulation)")
    print(f"    Formula gives: {alpha_inv_formula(G_EFF):.3f}")
    print()
    print(f"  AT M_Z = {M_Z_GEV} GeV:")
    print(f"    α⁻¹ = {ALPHA_INV_MZ} (PDG)")
    print(f"    G_eff needed: {G_at_MZ:.6f}")
    print(f"    Shift: ΔG = {G_at_MZ - G_EFF:.6f} ({(G_at_MZ - G_EFF)/G_EFF*100:.2f}%)")
    print()

    # The running: how G_eff changes with energy
    # If G_eff(E) = G_eff × (1 + β × ln(E/E_0)):
    # Then ΔG/G = β × ln(M_Z/m_e) where m_e is the low-energy reference
    m_e_GeV = 0.000511
    ln_ratio = np.log(M_Z_GEV / m_e_GeV)
    delta_G = G_at_MZ - G_EFF
    beta_G = delta_G / (G_EFF * ln_ratio)

    print(f"  RUNNING COEFFICIENT:")
    print(f"    ln(M_Z/m_e) = ln({M_Z_GEV}/{m_e_GeV}) = {ln_ratio:.3f}")
    print(f"    β_G = ΔG/(G × ln(E/E_0)) = {beta_G:.6f}")
    print()

    # Compare to QED beta function
    # QED: Δα⁻¹ = -(1/3π) × Σ Q² × ln(E/m_f)
    # For e,μ,τ + u,d,s,c,b quarks (below M_Z):
    # Σ Q² = 3×(4/9 + 1/9 + 4/9 + 1/9 + 4/9) + (1 + 1 + 1) = 3×14/9 + 3 = 42/9 + 27/9 = 69/9
    # Wait, let me be more careful:
    # leptons: e(1), μ(1), τ(1) → ΣQ² = 3
    # quarks (×3 colors): u(4/9), d(1/9), s(1/9), c(4/9), b(1/9) → 3×(4+1+1+4+1)/9 = 3×11/9 = 33/9
    # Total: 3 + 33/9 = 27/9 + 33/9 = 60/9 = 20/3
    sum_Q2 = 20.0 / 3.0
    delta_alpha_QED = -(1.0 / (3 * np.pi)) * sum_Q2 * ln_ratio
    alpha_inv_MZ_QED = ALPHA_INV_LOW + delta_alpha_QED

    print(f"  QED COMPARISON:")
    print(f"    Σ Q² (5 quarks + 3 leptons) = {sum_Q2:.4f}")
    print(f"    Δα⁻¹(QED) = -(1/3π) × {sum_Q2:.3f} × {ln_ratio:.3f} = {delta_alpha_QED:.3f}")
    print(f"    α⁻¹(M_Z, QED) = {ALPHA_INV_LOW} + ({delta_alpha_QED:.3f}) = {alpha_inv_MZ_QED:.3f}")
    print(f"    PDG value: {ALPHA_INV_MZ}")
    print(f"    QED prediction error: {abs(alpha_inv_MZ_QED - ALPHA_INV_MZ)/ALPHA_INV_MZ*100:.2f}%")
    print()

    # Energy scan
    print(f"  α⁻¹ AT VARIOUS ENERGY SCALES:")
    print(f"  {'E (GeV)':>12s}   {'α⁻¹(holo)':>12s}   {'G_eff(E)':>10s}   {'α⁻¹(QED)':>12s}")
    print(f"  {'-'*12}   {'-'*12}   {'-'*10}   {'-'*12}")

    energies = [0.000511, 0.1057, 1.0, 10.0, 91.2, 200.0, 1000.0, 14000.0]
    for E in energies:
        # Holographic: interpolate G_eff(E)
        G_E = G_EFF + beta_G * G_EFF * np.log(max(E, m_e_GeV) / m_e_GeV)
        if G_E > 0.01 and G_E < 0.99:
            a_holo = alpha_inv_formula(G_E)
        else:
            a_holo = float('nan')

        # QED
        a_qed = ALPHA_INV_LOW - (1.0/(3*np.pi)) * sum_Q2 * np.log(max(E, m_e_GeV)/m_e_GeV)

        label = ""
        if abs(E - 0.000511) < 0.001: label = " (m_e)"
        elif abs(E - 91.2) < 1: label = " (M_Z)"
        elif abs(E - 14000) < 100: label = " (LHC)"

        print(f"  {E:12.4f}   {a_holo:12.3f}   {G_E:10.6f}   {a_qed:12.3f}{label}")

    print()
    print("  NOTE: The holographic running is QUALITATIVE in this paper.")
    print("  The exact β_G requires the energy-dependent mode cutoff l_c(E),")
    print("  which depends on how the effective horizon radius scales with")
    print("  probe energy. This is deferred to a companion paper.")
    print("  The QED running is shown for comparison — both give logarithmic")
    print("  running with the correct sign.")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
