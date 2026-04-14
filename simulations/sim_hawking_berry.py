#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAWKING TEMPERATURE FROM BERRY PHASE AT THE HORIZON

The insight: the surface gravity is not the spatial gradient of phi.
It is the Berry phase accumulation rate at r_s — how fast the
ouroboros cycle accumulates geometric phase at the horizon radius.

Route C gave alpha from Berry phase oscillation.
This simulation tests whether the same mechanism gives T_H.

Sim T1: Berry phase at the horizon
  Place a merkabit at r_s in the torsion field.
  Run one ouroboros cycle. Measure gamma(r_s).

Sim T2: Temperature from Berry phase rate
  kappa_Berry = gamma(r_s) / (h * tau_L)
  T_H = kappa_Berry / (2*pi)
  Compare to standard Hawking.

Merkabit Research Program — Selina Stenberg, 2026
"""

import numpy as np
from datetime import datetime

# Constants
COXETER_H = 12
STEP_PHASE = 2 * np.pi / COXETER_H
NUM_GATES = 5
CROSS_STRENGTH = 0.3
G_EFF = 0.2542
PHI_MAX = 1.0 / G_EFF
A_0 = 0.3077
DELTA = 1.0 / 24.0

# ============================================================
#  4-SPINOR MERKABIT WITH TORSION COUPLING
# ============================================================

def make_Rx4(theta):
    c, s = np.cos(theta/2), -1j*np.sin(theta/2)
    R2 = np.array([[c,s],[s,c]], dtype=complex)
    R4 = np.zeros((4,4), dtype=complex); R4[:2,:2] = R2; R4[2:,2:] = R2
    return R4

def make_Rz4(theta):
    return np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2),
                    np.exp(-1j*theta/2), np.exp(1j*theta/2)])

def make_P4_fwd(phi):
    return np.diag([np.exp(1j*phi/2), np.exp(-1j*phi/2),
                    np.exp(1j*phi/2), np.exp(-1j*phi/2)])

def make_P4_inv(phi):
    return np.diag([np.exp(-1j*phi/2), np.exp(1j*phi/2),
                    np.exp(-1j*phi/2), np.exp(1j*phi/2)])

def make_cross_fwd(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c,0,-s,0],[0,c,0,-s],[s,0,c,0],[0,s,0,c]], dtype=complex)

def make_cross_inv(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c,0,s,0],[0,c,0,s],[-s,0,c,0],[0,-s,0,c]], dtype=complex)

def ouroboros_step_in_field(u, v, step_index, phi_local):
    """
    One ouroboros step with the gate angles MODULATED by the
    local torsion potential phi_local.

    The torsion field affects the P gate phase and the cross-coupling
    strength — the same way Paper 20's coupling works.

    phi_local / PHI_MAX = fraction of maximum torsion at this radius.
    """
    k = step_index; absent = k % NUM_GATES

    # Torsion modulation factor
    torsion_frac = phi_local / PHI_MAX  # 0 at infinity, 1 at core

    p_angle = STEP_PHASE * (1.0 + torsion_frac)  # P gate enhanced by torsion
    sym_base = STEP_PHASE / 3
    omega_k = 2*np.pi*k/12
    rx_angle = sym_base*(1.0+0.5*np.cos(omega_k))
    rz_angle = sym_base*(1.0+0.5*np.cos(omega_k+2*np.pi/3))
    cross_angle = CROSS_STRENGTH*STEP_PHASE*(1.0+0.5*np.cos(omega_k+4*np.pi/3))

    # Gate-specific modifications (same as Paper 20)
    label = ['S','R','T','F','P'][absent]
    if label=='S': rz_angle*=0.4; rx_angle*=1.3; cross_angle*=1.2
    elif label=='R': rx_angle*=0.4; rz_angle*=1.3; cross_angle*=0.8
    elif label=='T': rx_angle*=0.7; rz_angle*=0.7; cross_angle*=1.5
    elif label=='P': p_angle*=0.6; rx_angle*=1.8; rz_angle*=1.5; cross_angle*=0.5

    # Torsion also modulates cross-coupling
    cross_angle *= (1.0 + 0.5 * torsion_frac)

    u = make_P4_fwd(p_angle) @ u; v = make_P4_inv(p_angle) @ v
    u = make_cross_fwd(cross_angle) @ u; v = make_cross_inv(cross_angle) @ v
    Rz = make_Rz4(rz_angle); Rx = make_Rx4(rx_angle)
    u = Rx @ Rz @ u; v = Rx @ Rz @ v
    u /= np.linalg.norm(u); v /= np.linalg.norm(v)
    return u, v

def berry_phase_at_radius(r, M, n_settle=50):
    """
    Compute Berry phase of one ouroboros cycle for a merkabit
    sitting at radius r in the torsion field of mass M.

    phi(r) = a_0 * M / r  (magnitude, from Paper 20)
    """
    # Local torsion potential at this radius
    if r <= 0.5:
        phi_local = PHI_MAX
    else:
        r_p = A_0 * G_EFF * M  # plateau radius
        if r <= r_p:
            phi_local = PHI_MAX
        else:
            phi_local = PHI_MAX * r_p / r

    # Settle the spinor in this field
    u = np.array([1,1,1,1], dtype=complex)/2.0
    v = np.array([1,-1,-1,1], dtype=complex)/2.0
    for _ in range(n_settle):
        for step in range(COXETER_H):
            u, v = ouroboros_step_in_field(u, v, step, phi_local)

    # Now measure Berry phase over one cycle
    states_u = [u.copy()]
    states_v = [v.copy()]
    for step in range(COXETER_H):
        u, v = ouroboros_step_in_field(u, v, step, phi_local)
        states_u.append(u.copy())
        states_v.append(v.copy())

    gamma = 0.0
    for k in range(COXETER_H):
        ou = np.vdot(states_u[k], states_u[k+1])
        ov = np.vdot(states_v[k], states_v[k+1])
        gamma += np.angle(ou * ov)

    return -gamma, phi_local

# ============================================================
#  MAIN
# ============================================================

def main():
    np.random.seed(42)
    start = datetime.now()

    print("="*70)
    print("  HAWKING TEMPERATURE FROM BERRY PHASE AT THE HORIZON")
    print("="*70)
    print()

    # First: Berry phase in vacuum (no torsion field)
    gamma_vac, _ = berry_phase_at_radius(1e6, 1, n_settle=100)  # r >> everything
    print(f"  Vacuum Berry phase gamma_0 = {gamma_vac:.6f} rad")
    print(f"  |gamma_0| = {abs(gamma_vac):.6f} rad")
    print(f"  |gamma_0|/pi = {abs(gamma_vac)/np.pi:.6f}")
    print()

    # ================================================================
    #  SIM T1: BERRY PHASE vs RADIUS
    # ================================================================
    print("="*70)
    print("  SIM T1: BERRY PHASE AT VARIOUS RADII")
    print("="*70)
    print()

    M_test = 100
    r_s = 2 * G_EFF * M_test
    r_p = A_0 * G_EFF * M_test

    print(f"  M = {M_test}, r_s = {r_s:.2f}, r_p = {r_p:.2f}")
    print()

    radii = [r_p/2, r_p, r_p+1, r_p+2, r_s/2, r_s, r_s*1.5, r_s*2, r_s*5, r_s*10, 1000]

    print(f"  {'r':>8s}   {'r/r_s':>8s}   {'phi(r)':>10s}   {'phi/phi_max':>12s}   {'gamma(r)':>10s}   {'gamma/gamma_0':>14s}   {'region':>10s}")
    print(f"  {'-'*8}   {'-'*8}   {'-'*10}   {'-'*12}   {'-'*10}   {'-'*14}   {'-'*10}")

    gamma_at_rs = None
    results = {}

    for r in radii:
        gamma_r, phi_r = berry_phase_at_radius(r, M_test, n_settle=80)
        ratio_gamma = gamma_r / gamma_vac if abs(gamma_vac) > 1e-10 else 0

        if r <= r_p:
            region = "CORE"
        elif r <= r_p + 3:
            region = "HORIZON"
        elif r <= r_s:
            region = "INTERIOR"
        else:
            region = "EXTERIOR"

        results[r] = {'gamma': gamma_r, 'phi': phi_r, 'ratio': ratio_gamma}

        if abs(r - r_s) < 0.01:
            gamma_at_rs = gamma_r

        print(f"  {r:8.2f}   {r/r_s:8.4f}   {phi_r:10.4f}   {phi_r/PHI_MAX:12.4f}   {gamma_r:10.6f}   {ratio_gamma:14.6f}   {region:>10s}")

    print()

    # ================================================================
    #  SIM T2: HAWKING TEMPERATURE FROM BERRY PHASE RATE
    # ================================================================
    print("="*70)
    print("  SIM T2: HAWKING TEMPERATURE FROM BERRY PHASE RATE")
    print("="*70)
    print()

    print(f"  The Berry phase rate at r_s is the lattice surface gravity:")
    print(f"    kappa_Berry = |gamma(r_s)| / h")
    print(f"  The Hawking temperature is:")
    print(f"    T_H = kappa_Berry / (2*pi)")
    print()

    M_values = [10, 20, 50, 100, 200, 500, 1000]

    print(f"  {'M':>6s}   {'r_s':>8s}   {'gamma(r_s)':>12s}   {'kappa_Berry':>12s}   {'T_Berry':>12s}   {'T_Hawking':>12s}   {'ratio':>8s}")
    print(f"  {'-'*6}   {'-'*8}   {'-'*12}   {'-'*12}   {'-'*12}   {'-'*12}   {'-'*8}")

    for M in M_values:
        r_s_M = 2 * G_EFF * M
        gamma_rs, phi_rs = berry_phase_at_radius(r_s_M, M, n_settle=80)

        # Berry phase rate (surface gravity)
        kappa_Berry = abs(gamma_rs) / COXETER_H

        # Temperature from Berry phase
        T_Berry = kappa_Berry / (2 * np.pi)

        # Standard Hawking temperature in lattice units
        # T_H = 1/(8*pi*G_eff*M)
        T_Hawking = 1.0 / (8 * np.pi * G_EFF * M)

        ratio = T_Berry / T_Hawking if T_Hawking > 0 else 0

        print(f"  {M:6d}   {r_s_M:8.2f}   {gamma_rs:12.6f}   {kappa_Berry:12.6f}   {T_Berry:12.8f}   {T_Hawking:12.8f}   {ratio:8.4f}")

    print()

    # ================================================================
    #  ANALYSIS: WHAT IS gamma(r_s) / gamma_0?
    # ================================================================
    print("="*70)
    print("  ANALYSIS: BERRY PHASE RATIO AT HORIZON")
    print("="*70)
    print()

    # The potential at r_s
    # phi(r_s) = PHI_MAX * r_p/r_s = PHI_MAX * (a_0*G*M)/(2*G*M) = PHI_MAX * a_0/2
    phi_at_rs = PHI_MAX * A_0 / 2
    print(f"  phi(r_s) = PHI_MAX * a_0/2 = {phi_at_rs:.6f}")
    print(f"  phi(r_s)/PHI_MAX = a_0/2 = {A_0/2:.6f}")
    print(f"  This is the SAME ratio as r_p/r_s — the geometric invariant.")
    print()

    # The prediction: gamma(r_s) = gamma_0 * f(phi(r_s)/PHI_MAX)
    # where f is the torsion modulation function
    if gamma_at_rs is not None:
        ratio_at_rs = gamma_at_rs / gamma_vac
        print(f"  gamma(r_s) = {gamma_at_rs:.6f}")
        print(f"  gamma_0 = {gamma_vac:.6f}")
        print(f"  gamma(r_s)/gamma_0 = {ratio_at_rs:.6f}")
        print()

        # Check: is this related to a_0/2?
        print(f"  a_0/2 = {A_0/2:.6f}")
        print(f"  (1 + a_0/2) = {1 + A_0/2:.6f}")
        print(f"  gamma_ratio / (1+a_0/2) = {ratio_at_rs / (1+A_0/2):.6f}")
        print()

    # ================================================================
    #  THE KEY: phi(r_s) IS M-INDEPENDENT
    # ================================================================
    print("="*70)
    print("  KEY OBSERVATION: phi(r_s) IS M-INDEPENDENT")
    print("="*70)
    print()
    print(f"  phi(r_s) = a_0 * M / r_s = a_0 * M / (2*G_eff*M) = a_0/(2*G_eff)")
    print(f"           = {A_0/(2*G_EFF):.6f}")
    print(f"  This is FIXED. Does not depend on M.")
    print(f"  Therefore gamma(r_s) does not depend on M either.")
    print()
    print(f"  But T_H = 1/(8*pi*G*M) DOES depend on M.")
    print(f"  So T_Berry = |gamma(r_s)|/(h*2*pi) is M-INDEPENDENT.")
    print(f"  This seems wrong...")
    print()
    print(f"  RESOLUTION: the Berry phase gives the TEMPERATURE SCALE,")
    print(f"  but the M-dependence comes from the NUMBER OF MODES")
    print(f"  at the horizon. T_H = T_scale / N_modes(r_s).")
    print(f"  N_modes ~ r_s^2 ~ M^2. But T_H ~ 1/M, not 1/M^2.")
    print()
    print(f"  Better: kappa_Berry should be per unit AREA, not per cycle.")
    print(f"  kappa per unit area = |gamma(r_s)| / (h * 4*pi*r_s^2)")
    print()

    for M in [10, 100, 1000]:
        r_s_M = 2 * G_EFF * M
        gamma_rs, _ = berry_phase_at_radius(r_s_M, M, n_settle=80)

        A_horizon = 4 * np.pi * r_s_M**2
        kappa_per_area = abs(gamma_rs) / (COXETER_H * A_horizon)
        T_per_area = kappa_per_area / (2 * np.pi)
        T_Hawking = 1.0 / (8 * np.pi * G_EFF * M)

        # Alternative: kappa per unit circumference
        C_horizon = 2 * np.pi * r_s_M
        kappa_per_circ = abs(gamma_rs) / (COXETER_H * C_horizon)
        T_per_circ = kappa_per_circ / (2 * np.pi)

        print(f"  M={M:5d}: T_Berry/(4pi*r_s^2) = {T_per_area:.8f}, T_Berry/(2pi*r_s) = {T_per_circ:.8f}, T_H = {T_Hawking:.8f}, ratio_circ = {T_per_circ/T_Hawking:.4f}")

    print()

    # ================================================================
    #  ALTERNATIVE: DIFFERENTIAL BERRY PHASE
    # ================================================================
    print("="*70)
    print("  ALTERNATIVE: DIFFERENTIAL BERRY PHASE d(gamma)/dr AT r_s")
    print("="*70)
    print()
    print(f"  Surface gravity = rate of change of Berry phase WITH RADIUS")
    print(f"  kappa_Berry = |d(gamma)/dr| at r = r_s")
    print()

    for M in [10, 50, 100, 500, 1000]:
        r_s_M = 2 * G_EFF * M
        dr = 0.5
        gamma_minus, _ = berry_phase_at_radius(r_s_M - dr, M, n_settle=80)
        gamma_plus, _ = berry_phase_at_radius(r_s_M + dr, M, n_settle=80)

        dgamma_dr = (gamma_plus - gamma_minus) / (2 * dr)
        kappa_dgamma = abs(dgamma_dr)
        T_dgamma = kappa_dgamma / (2 * np.pi)
        T_Hawking = 1.0 / (8 * np.pi * G_EFF * M)

        ratio = T_dgamma / T_Hawking if T_Hawking > 0 else 0

        print(f"  M={M:5d}: d(gamma)/dr = {dgamma_dr:.6f}, T_dgamma = {T_dgamma:.8f}, T_H = {T_Hawking:.8f}, ratio = {ratio:.4f}")

    print()

    # ================================================================
    #  FINAL: WHAT COMBINATION GIVES T_H?
    # ================================================================
    print("="*70)
    print("  WHAT COMBINATION GIVES T_H EXACTLY?")
    print("="*70)
    print()

    # T_H = 1/(8*pi*G*M) = 1/(4*pi*r_s)
    # If gamma(r_s) is M-independent, call it gamma_H
    # We need: T_H = gamma_H * f(r_s) / (2*pi)
    # 1/(4*pi*r_s) = gamma_H * f(r_s) / (2*pi)
    # f(r_s) = 1/(2*r_s*gamma_H)

    # Measure gamma_H
    M_ref = 100
    r_s_ref = 2 * G_EFF * M_ref
    gamma_H, _ = berry_phase_at_radius(r_s_ref, M_ref, n_settle=100)

    print(f"  gamma_H (Berry phase at r_s, M={M_ref}) = {gamma_H:.6f}")
    print(f"  |gamma_H| = {abs(gamma_H):.6f}")
    print()

    # Check: T_H = |gamma_H| / (h * 2*pi * r_s)?
    for M in [10, 50, 100, 500]:
        r_s_M = 2 * G_EFF * M
        # Recompute gamma at this r_s
        g, _ = berry_phase_at_radius(r_s_M, M, n_settle=80)

        T_test = abs(g) / (COXETER_H * 2 * np.pi * r_s_M)
        T_H = 1.0 / (8 * np.pi * G_EFF * M)

        print(f"  M={M:5d}: |gamma|/(h*2pi*r_s) = {T_test:.8f}, T_H = {T_H:.8f}, ratio = {T_test/T_H:.4f}")

    print()

    # Also test: T_H = Delta * |dgamma/dr| / (2*pi)?
    print("  Test: T_H = Delta * |d(gamma)/dr| / (2*pi)?")
    for M in [10, 50, 100, 500, 1000]:
        r_s_M = 2 * G_EFF * M
        dr = 0.5
        gm, _ = berry_phase_at_radius(r_s_M - dr, M, n_settle=80)
        gp, _ = berry_phase_at_radius(r_s_M + dr, M, n_settle=80)
        dgdr = abs((gp - gm) / (2*dr))

        T_test = DELTA * dgdr / (2*np.pi)
        T_H = 1.0 / (8 * np.pi * G_EFF * M)
        ratio = T_test / T_H if T_H > 0 else 0

        print(f"  M={M:5d}: Delta*|dg/dr|/(2pi) = {T_test:.8f}, T_H = {T_H:.8f}, ratio = {ratio:.4f}")

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n  Runtime: {elapsed:.1f} seconds")
    print("="*70)

if __name__ == '__main__':
    main()
