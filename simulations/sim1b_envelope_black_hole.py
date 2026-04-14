#!/usr/bin/env python3
"""
PAPER 27 — SIMULATION 1b: BLACK HOLE AS ENVELOPE

Merkabit Research Program — Selina Stenberg, 2026

KEY INSIGHT:
  The torsion potential phi(r) = M * f(r) is LINEAR in M — no saturation.
  But the ENVELOPE (field energy density rho = phi^2) is QUADRATIC in M:
    V(M) = sum phi^2 = M^2 * sum f(r)^2

  This means the envelope's self-energy grows faster than the source mass.
  At some critical M, the envelope mass EXCEEDS the source mass:
    M_envelope > M_source

  This is the lattice analogue of "gravity gravitates" — the field energy
  itself is a source. The black hole forms when the envelope becomes
  self-sustaining: its own mass generates a well deep enough to contain
  itself.

  EXPLORATION:
  1. Envelope energy density rho(r) = phi(r)^2 as function of r and M
  2. Enclosed envelope mass M_env(r) = sum_{r'<r} rho(r') * N_shell(r')
  3. Self-gravitation threshold: M_env(r) > M_source within some r
  4. The horizon = the radius where enclosed envelope mass = enclosed source mass
  5. Does r_s = 2*G_eff*M emerge from the self-gravitation condition?
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
#  CONSTANTS
# ============================================================
G_EFF = 0.2542
L = 41
HALF = L // 2
N_ITER = 8000

# ============================================================
#  LAPLACE SOLVER (from Sim 1)
# ============================================================

def laplace_solver(L, M, n_iter=N_ITER):
    H = L // 2
    phi = np.zeros((L, L, L), dtype=np.float64)
    phi[H, H, H] = float(M)
    coords = np.arange(L) - H
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    boundary = (np.abs(X) == H) | (np.abs(Y) == H) | (np.abs(Z) == H)
    for _ in range(n_iter):
        phi_new = (
            np.roll(phi, 1, 0) + np.roll(phi, -1, 0) +
            np.roll(phi, 1, 1) + np.roll(phi, -1, 1) +
            np.roll(phi, 1, 2) + np.roll(phi, -1, 2)
        ) / 6.0
        phi_new[H, H, H] = float(M)
        phi_new[boundary] = 0.0
        phi = phi_new
    return phi

def compute_shells(L):
    H = L // 2
    coords = np.arange(L) - H
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    R_int = np.round(R).astype(int)
    shells = {}
    for r in range(0, H + 1):
        mask = (R_int == r)
        if np.sum(mask) > 0:
            shells[r] = mask
    return shells, R_int

# ============================================================
#  ENVELOPE ANALYSIS
# ============================================================

def envelope_analysis(phi, shells, M):
    """
    Compute envelope properties shell by shell.

    Returns dict with:
      r -> {phi_mean, rho_mean, N_shell, V_shell, M_env_enclosed,
            ratio_env_to_source, self_grav_ratio}
    """
    H = len(phi) // 2
    results = {}
    M_env_cumulative = 0.0
    V_cumulative = 0.0

    for r in sorted(shells.keys()):
        mask = shells[r]
        N_shell = int(np.sum(mask))

        # Torsion potential (shell average)
        phi_mean = np.mean(phi[mask])

        # Envelope energy density: rho = phi^2
        rho_shell = np.mean(phi[mask]**2)

        # Shell volume (number of lattice sites)
        V_shell = rho_shell * N_shell

        # Cumulative enclosed envelope energy
        M_env_cumulative += V_shell
        V_cumulative += N_shell

        # Self-gravitation ratio: M_env_enclosed / M_source
        # When this exceeds 1, the envelope dominates
        ratio = M_env_cumulative / M if M > 0 else 0

        # Local self-gravitation: envelope energy density vs
        # the gravitational binding at this radius
        # Binding energy per unit volume ~ G * M / r
        # Self-grav when rho > G * M / r
        binding = G_EFF * M / max(r, 0.5)
        self_grav = rho_shell / binding if binding > 0 else 0

        results[r] = {
            'phi_mean': phi_mean,
            'rho_mean': rho_shell,
            'N_shell': N_shell,
            'V_shell': V_shell,
            'M_env_enclosed': M_env_cumulative,
            'V_enclosed': V_cumulative,
            'ratio_env_to_source': ratio,
            'self_grav_ratio': self_grav,
            'binding_energy': binding,
        }

    return results


def find_self_grav_radius(env_results, M):
    """
    Find the radius where M_env_enclosed(r) = M.
    This is the self-gravitation horizon.
    """
    radii = sorted(env_results.keys())
    for i in range(len(radii) - 1):
        r = radii[i]
        r_next = radii[i + 1]
        ratio_here = env_results[r]['ratio_env_to_source']
        ratio_next = env_results[r_next]['ratio_env_to_source']

        if ratio_here <= 1.0 and ratio_next > 1.0:
            # Interpolate
            frac = (1.0 - ratio_here) / (ratio_next - ratio_here)
            return r + frac * (r_next - r)

    # Check if it's already > 1 at r=0 or never reaches 1
    if env_results.get(0, {}).get('ratio_env_to_source', 0) > 1.0:
        return 0.0
    return None  # Never reaches self-gravitation


def find_local_self_grav_radius(env_results):
    """
    Find outermost radius where local envelope density exceeds
    local binding energy: rho(r) > G*M/r.
    """
    radii = sorted(env_results.keys(), reverse=True)
    for r in radii:
        if r == 0:
            continue
        if env_results[r]['self_grav_ratio'] > 1.0:
            return float(r)
    return 0.0


# ============================================================
#  ITERATIVE SELF-CONSISTENT ENVELOPE (gravity gravitates)
# ============================================================

def self_consistent_envelope(M_source, L, max_iterations=20, tol=1e-3):
    """
    Iteratively solve for the self-consistent mass distribution
    where the envelope's own mass acts as an additional source.

    Step 1: Solve Laplace for M_source at center
    Step 2: Compute envelope energy rho(r) = phi^2(r)
    Step 3: Use rho(r) as distributed source in next Laplace solve
    Step 4: Repeat until convergence

    This is the lattice analogue of solving Einstein's equation
    where T_mu_nu includes the field energy.
    """
    H = L // 2
    coords = np.arange(L) - H
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    boundary = (np.abs(X) == H) | (np.abs(Y) == H) | (np.abs(Z) == H)
    R = np.sqrt(X**2 + Y**2 + Z**2)

    # Initial solve: point source only
    phi = laplace_solver(L, M_source, n_iter=N_ITER)

    total_masses = [M_source]
    convergence = []
    phi_history = [phi.copy()]

    for iteration in range(max_iterations):
        # Envelope energy density
        rho = phi**2

        # Normalise: total envelope mass in units of source mass
        # The coupling constant determines how strongly the envelope
        # gravitates. In GR this is G/c^4.
        # On the lattice: envelope mass = G_eff * sum(phi^2) / V_lattice
        envelope_mass_density = G_EFF * rho

        # Total effective mass: source + envelope
        # Source is point mass at center; envelope is distributed
        M_total_eff = M_source + G_EFF * np.sum(rho)
        total_masses.append(M_total_eff)

        # New source: point mass + distributed envelope
        # Solve Poisson: nabla^2 phi_new = -rho_source
        # On lattice: Jacobi with both point and distributed sources
        phi_new = np.zeros((L, L, L), dtype=np.float64)
        phi_new[H, H, H] = float(M_source)

        for it in range(N_ITER):
            p = (
                np.roll(phi_new, 1, 0) + np.roll(phi_new, -1, 0) +
                np.roll(phi_new, 1, 1) + np.roll(phi_new, -1, 1) +
                np.roll(phi_new, 1, 2) + np.roll(phi_new, -1, 2)
            ) / 6.0
            # Add distributed envelope source
            # Poisson: nabla^2 phi = -4*pi*rho  =>  phi_new = avg + h^2/6 * source
            # On unit lattice: phi_new = avg + source/6
            p += G_EFF * envelope_mass_density / 6.0
            p[H, H, H] = float(M_source) + G_EFF * rho[H, H, H] / 6.0
            p[boundary] = 0.0
            phi_new = p

        # Convergence check
        delta = np.max(np.abs(phi_new - phi)) / np.max(np.abs(phi))
        convergence.append(delta)

        phi = phi_new
        phi_history.append(phi.copy())

        if delta < tol:
            break

    return phi, total_masses, convergence, phi_history


# ============================================================
#  MAIN
# ============================================================

def main():
    np.random.seed(42)
    start = datetime.now()
    out = []
    def log(s=""):
        print(s); out.append(str(s))

    log("=" * 70)
    log("  PAPER 27 — SIM 1b: BLACK HOLE AS ENVELOPE")
    log("  Merkabit Research Program — Selina Stenberg, 2026")
    log("=" * 70)
    log()
    log(f"  Lattice: {L}x{L}x{L},  G_eff = {G_EFF}")
    log()

    shells, R_int = compute_shells(L)

    # ================================================================
    #  PART 1: ENVELOPE ENERGY DENSITY PROFILES
    # ================================================================
    log("=" * 70)
    log("  PART 1: ENVELOPE ENERGY DENSITY rho(r) = phi(r)^2")
    log("=" * 70)
    log()
    log("  Key: phi(r) = M * f(r)  =>  rho(r) = M^2 * f(r)^2")
    log("  Envelope mass M_env = G_eff * sum rho = G_eff * M^2 * sum f^2")
    log("  Self-gravitation when M_env > M_source: G_eff * M^2 * K > M")
    log("  => M > 1 / (G_eff * K)  where K = sum f(r)^2")
    log()

    # First: compute the geometric constant K from M=1
    phi_1 = laplace_solver(L, 1.0, n_iter=N_ITER)
    K_total = np.sum(phi_1**2)
    log(f"  Geometric constant K = sum f(r)^2 = {K_total:.6f}")
    log(f"  Critical mass M_crit = 1/(G_eff * K) = {1.0/(G_EFF * K_total):.6f}")
    log()

    # Shell-resolved K(r)
    K_cumulative = 0.0
    log(f"  {'r':>4s}   {'f(r)':>10s}   {'f(r)^2':>10s}   {'N_shell':>8s}   {'K_cum(r)':>12s}   {'M_crit(r)':>12s}")
    log(f"  {'-'*4}   {'-'*10}   {'-'*10}   {'-'*8}   {'-'*12}   {'-'*12}")

    K_of_r = {}
    M_crit_of_r = {}
    for r in sorted(shells.keys()):
        if r == 0:
            continue
        mask = shells[r]
        N_shell = int(np.sum(mask))
        f_mean = np.mean(phi_1[mask])
        f2_mean = np.mean(phi_1[mask]**2)
        K_shell = np.sum(phi_1[mask]**2)
        K_cumulative += K_shell
        K_of_r[r] = K_cumulative
        M_crit = 1.0 / (G_EFF * K_cumulative) if K_cumulative > 0 else np.inf
        M_crit_of_r[r] = M_crit
        if r <= 20:
            log(f"  {r:4d}   {f_mean:10.6f}   {f2_mean:10.6f}   {N_shell:8d}   {K_cumulative:12.6f}   {M_crit:12.4f}")

    log()
    log(f"  Total K = {K_total:.6f}")
    log(f"  Global M_crit = {1.0/(G_EFF*K_total):.4f}")
    log()

    # ================================================================
    #  PART 2: ENVELOPE PROFILES FOR VARIOUS M
    # ================================================================
    log("=" * 70)
    log("  PART 2: ENCLOSED ENVELOPE MASS vs RADIUS")
    log("=" * 70)
    log()

    M_values = [1, 2, 5, 10, 20, 50, 100, 200, 500]

    all_env_results = {}
    r_s_self_grav = {}
    r_s_local = {}

    for M in M_values:
        # Linearity: phi_M = M * phi_1, so rho_M = M^2 * rho_1
        # No need to re-solve!
        env = {}
        M_env_cum = 0.0
        V_cum = 0

        for r in sorted(shells.keys()):
            if r == 0:
                continue
            mask = shells[r]
            N_shell = int(np.sum(mask))

            phi_mean = M * np.mean(phi_1[mask])
            rho_mean = M**2 * np.mean(phi_1[mask]**2)
            rho_total_shell = M**2 * np.sum(phi_1[mask]**2)

            # Envelope mass = G_eff * rho (converts field energy to mass)
            M_env_shell = G_EFF * rho_total_shell
            M_env_cum += M_env_shell
            V_cum += N_shell

            ratio = M_env_cum / M
            binding = G_EFF * M / max(r, 0.5)
            local_ratio = G_EFF * rho_mean / binding if binding > 0 else 0
            # local_ratio = rho_mean * r / M  (simplifies)

            env[r] = {
                'phi_mean': phi_mean,
                'rho_mean': rho_mean,
                'N_shell': N_shell,
                'M_env_enclosed': M_env_cum,
                'ratio_env_to_source': ratio,
                'local_ratio': local_ratio,
            }

        all_env_results[M] = env

        # Find self-gravitation radius
        r_sg = find_self_grav_radius(env, M)
        r_s_self_grav[M] = r_sg

        # Find local self-gravitation radius
        r_loc = 0.0
        for r in sorted(env.keys(), reverse=True):
            if env[r]['local_ratio'] > 1.0:
                r_loc = float(r)
                break
        r_s_local[M] = r_loc

    # Print envelope profiles
    for M_show in [1, 10, 100, 500]:
        env = all_env_results[M_show]
        log(f"  M = {M_show}:")
        log(f"    {'r':>4s}   {'phi(r)':>10s}   {'rho(r)':>12s}   {'M_env(<r)':>12s}   {'M_env/M':>10s}   {'local':>8s}")
        log(f"    {'-'*4}   {'-'*10}   {'-'*12}   {'-'*12}   {'-'*10}   {'-'*8}")
        for r in sorted(env.keys()):
            if r > 20:
                continue
            e = env[r]
            log(f"    {r:4d}   {e['phi_mean']:10.4f}   {e['rho_mean']:12.4f}   {e['M_env_enclosed']:12.4f}   {e['ratio_env_to_source']:10.4f}   {e['local_ratio']:8.4f}")
        log()

    # ================================================================
    #  PART 3: SELF-GRAVITATION HORIZON
    # ================================================================
    log("=" * 70)
    log("  PART 3: SELF-GRAVITATION HORIZON (M_env = M)")
    log("=" * 70)
    log()

    log(f"  {'M':>6s}   {'r_s(env)':>10s}   {'r_s=2GM':>10s}   {'ratio':>8s}   {'M_env_total':>12s}   {'M_env/M':>10s}")
    log(f"  {'-'*6}   {'-'*10}   {'-'*10}   {'-'*8}   {'-'*12}   {'-'*10}")

    for M in M_values:
        r_sg = r_s_self_grav[M]
        r_pred = 2 * G_EFF * M
        env = all_env_results[M]
        max_r = max(env.keys())
        M_env_total = env[max_r]['M_env_enclosed']
        ratio_total = M_env_total / M

        r_sg_str = f"{r_sg:.2f}" if r_sg is not None else "N/A"
        ratio_str = f"{r_sg/r_pred:.4f}" if r_sg is not None and r_pred > 0.01 else "N/A"

        log(f"  {M:6d}   {r_sg_str:>10s}   {r_pred:10.2f}   {ratio_str:>8s}   {M_env_total:12.2f}   {ratio_total:10.4f}")

    log()
    log("  INTERPRETATION:")
    log(f"    M_env_total = G_eff * M^2 * K = {G_EFF:.4f} * M^2 * {K_total:.4f}")
    log(f"    M_env/M = G_eff * K * M = {G_EFF * K_total:.6f} * M")
    log(f"    Self-gravitation when M > M_crit = 1/(G_eff*K) = {1.0/(G_EFF*K_total):.4f}")
    log()

    # ================================================================
    #  PART 4: ENVELOPE SCHWARZSCHILD FORMULA
    # ================================================================
    log("=" * 70)
    log("  PART 4: ENVELOPE SCHWARZSCHILD RADIUS DERIVATION")
    log("=" * 70)
    log()

    # The enclosed envelope mass within radius r:
    #   M_env(r) = G_eff * M^2 * K(r)
    # where K(r) = sum_{r'<=r} f(r')^2 * N_shell(r')
    #
    # The self-gravitation horizon: M_env(r_s) = M
    #   G_eff * M^2 * K(r_s) = M
    #   G_eff * M * K(r_s) = 1
    #   K(r_s) = 1 / (G_eff * M)
    #
    # For the 1/r potential: f(r) ~ a_0/r, so f^2 ~ a_0^2/r^2
    # K(r) = sum_{r'=1}^{r} (a_0/r')^2 * 4*pi*r'^2
    #       = 4*pi*a_0^2 * sum_{r'=1}^{r} 1
    #       = 4*pi*a_0^2 * r
    #
    # So K(r) ~ 4*pi*a_0^2 * r  (LINEAR in r for 1/r potential!)
    #
    # Self-gravitation: 4*pi*a_0^2 * r_s = 1/(G_eff * M)
    #   r_s = 1 / (4*pi*a_0^2 * G_eff * M)
    #
    # This gives r_s ~ 1/M — WRONG sign! The horizon SHRINKS with mass.
    # Wait — this is the INNER edge where envelope dominates.
    # For large M, M_env/M > 1 at ALL radii, so the entire lattice
    # is inside the "horizon".
    #
    # Let me think differently. The OUTER horizon is where the
    # TOTAL enclosed mass (source + envelope) creates escape-velocity
    # conditions:
    #   v_escape(r) = sqrt(2*G*M_total(r)/r)
    # Horizon when v_escape = c (= 1 in lattice units):
    #   2*G*M_total(r_s)/r_s = 1
    #   r_s = 2*G*M_total(r_s)
    # where M_total(r_s) = M_source + M_env(r_s) = M + G_eff*M^2*K(r_s)
    #
    # Self-consistent horizon equation:
    #   r_s = 2*G_eff*(M + G_eff*M^2*K(r_s))
    #   r_s = 2*G_eff*M*(1 + G_eff*M*K(r_s))
    #
    # For M << M_crit: G_eff*M*K << 1, so r_s ~ 2*G_eff*M  (Schwarzschild)
    # For M >> M_crit: G_eff*M*K >> 1, so r_s ~ 2*G_eff^2*M^2*K(r_s)
    #   With K(r_s) ~ 4*pi*a_0^2*r_s:
    #   r_s ~ 2*G_eff^2*M^2*4*pi*a_0^2*r_s
    #   1 ~ 8*pi*G_eff^2*a_0^2*M^2
    #   This has NO r_s dependence — it's a critical mass condition!

    # Measure K(r) and verify linear scaling
    log("  K(r) = cumulative sum f(r')^2 * N_shell(r'):")
    log()
    r_vals = sorted(K_of_r.keys())
    K_vals = [K_of_r[r] for r in r_vals]

    # Fit K(r) = alpha * r + beta
    r_arr = np.array(r_vals[2:], dtype=float)  # skip r=1,2 (discretisation)
    K_arr = np.array(K_vals[2:])
    coeffs = np.polyfit(r_arr, K_arr, 1)
    alpha_K = coeffs[0]
    beta_K = coeffs[1]

    log(f"  Fit: K(r) = {alpha_K:.6f} * r + {beta_K:.6f}")
    log(f"  (Expected: K(r) ~ 4*pi*a_0^2 * r if phi ~ a_0/r)")
    log()

    # From Sim 1: a_0 = 0.3077
    a_0 = 0.3077
    expected_alpha = 4 * np.pi * a_0**2
    log(f"  a_0 = {a_0:.4f}")
    log(f"  4*pi*a_0^2 = {expected_alpha:.6f}")
    log(f"  Measured alpha_K = {alpha_K:.6f}")
    log(f"  Ratio = {alpha_K / expected_alpha:.4f}")
    log()

    # Self-consistent horizon equation
    log("  SELF-CONSISTENT HORIZON: r_s = 2*G_eff*M_total(r_s)")
    log("  where M_total(r) = M + G_eff*M^2*K(r)")
    log()

    # Solve numerically for each M
    r_s_sc = {}
    log(f"  {'M':>6s}   {'r_s(SC)':>10s}   {'r_s=2GM':>10s}   {'M_env(r_s)':>12s}   {'M_tot(r_s)':>12s}   {'enhancement':>12s}")
    log(f"  {'-'*6}   {'-'*10}   {'-'*10}   {'-'*12}   {'-'*12}   {'-'*12}")

    for M in M_values:
        # Solve r_s = 2*G_eff*(M + G_eff*M^2*K(r_s)) iteratively
        r_s = 2 * G_EFF * M  # Initial guess (Schwarzschild)
        for _ in range(100):
            # Interpolate K(r_s)
            K_rs = alpha_K * r_s + beta_K
            if K_rs < 0:
                K_rs = 0
            M_env_rs = G_EFF * M**2 * K_rs
            M_total_rs = M + M_env_rs
            r_s_new = 2 * G_EFF * M_total_rs
            if abs(r_s_new - r_s) < 1e-6:
                break
            r_s = r_s_new

        r_s_sc[M] = r_s
        r_s_bare = 2 * G_EFF * M
        M_env_rs = G_EFF * M**2 * (alpha_K * r_s + beta_K)
        M_total_rs = M + M_env_rs
        enhancement = r_s / r_s_bare if r_s_bare > 0 else 0

        log(f"  {M:6d}   {r_s:10.2f}   {r_s_bare:10.2f}   {M_env_rs:12.2f}   {M_total_rs:12.2f}   {enhancement:12.4f}")

    log()

    # ================================================================
    #  PART 5: REGIME ANALYSIS
    # ================================================================
    log("=" * 70)
    log("  PART 5: TWO REGIMES OF BLACK HOLE PHYSICS")
    log("=" * 70)
    log()

    M_crit = 1.0 / (G_EFF * alpha_K * 2 * G_EFF)  # When enhancement ~ 2
    log(f"  Regime boundary: G_eff * alpha_K * 2*G_eff * M ~ 1")
    log(f"  M_crossover ~ 1/(2*G_eff^2*alpha_K) = {M_crit:.2f}")
    log()
    log("  REGIME I (M << M_crossover): SCHWARZSCHILD")
    log("    Envelope mass negligible: M_env << M")
    log("    r_s = 2*G_eff*M  (standard)")
    log("    Event horizon = torsion geometry")
    log()
    log("  REGIME II (M >> M_crossover): ENVELOPE-DOMINATED")
    log("    Envelope mass dominates: M_env >> M")
    log("    r_s grows FASTER than linear in M")
    log("    The black hole IS the envelope — self-sustaining torsion well")
    log("    Gravity gravitates: the field energy is the dominant source")
    log()

    # ================================================================
    #  PART 6: LOCAL ENVELOPE DOMINANCE — WHERE IS THE HOLE?
    # ================================================================
    log("=" * 70)
    log("  PART 6: LOCAL ENVELOPE DOMINANCE")
    log("=" * 70)
    log()
    log("  Where does rho_envelope(r) > rho_binding(r)?")
    log("  rho_env = M^2 * f(r)^2,  rho_bind = G_eff * M / r")
    log("  Dominance when: M * f(r)^2 * r / G_eff > 1")
    log("  For f(r) ~ a_0/r: M * a_0^2/r > G_eff")
    log("  => r < M * a_0^2 / G_eff")
    log()

    r_local_pred = lambda M: M * a_0**2 / G_EFF

    log(f"  Prediction: r_local = M * a_0^2 / G_eff = M * {a_0**2/G_EFF:.6f}")
    log(f"  Compare: 2*G_eff = {2*G_EFF:.6f}")
    log(f"  Ratio: a_0^2/G_eff / (2*G_eff) = {a_0**2/(G_EFF * 2 * G_EFF):.6f}")
    log()

    log(f"  {'M':>6s}   {'r_local(pred)':>14s}   {'r_local(meas)':>14s}   {'r_s=2GM':>10s}")
    log(f"  {'-'*6}   {'-'*14}   {'-'*14}   {'-'*10}")

    for M in M_values:
        r_pred = r_local_pred(M)
        r_meas = r_s_local[M]
        r_schw = 2 * G_EFF * M
        log(f"  {M:6d}   {r_pred:14.2f}   {r_meas:14.1f}   {r_schw:10.2f}")

    log()

    # ================================================================
    #  FIGURE
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # --- Panel 1: Envelope energy density rho(r) ---
    ax1 = axes[0, 0]
    for M_show in [1, 10, 100, 500]:
        env = all_env_results[M_show]
        r_arr = np.array(sorted(env.keys()), dtype=float)
        rho_arr = np.array([env[r]['rho_mean'] for r in sorted(env.keys())])
        ax1.loglog(r_arr[1:], rho_arr[1:], 'o-', markersize=3, label=f'M={M_show}')
    # 1/r^2 reference
    r_ref = np.linspace(1, HALF, 100)
    ax1.loglog(r_ref, 0.1/r_ref**2, 'k--', alpha=0.3, label=r'$\sim 1/r^2$')
    ax1.set_xlabel('r'); ax1.set_ylabel(r'$\rho(r) = \phi^2(r)$')
    ax1.set_title('Envelope Energy Density')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # --- Panel 2: Enclosed envelope mass M_env(<r) / M ---
    ax2 = axes[0, 1]
    for M_show in [1, 10, 50, 100, 500]:
        env = all_env_results[M_show]
        r_arr = np.array(sorted(env.keys()), dtype=float)
        ratio_arr = np.array([env[r]['ratio_env_to_source'] for r in sorted(env.keys())])
        ax2.semilogy(r_arr[1:], ratio_arr[1:], 'o-', markersize=3, label=f'M={M_show}')
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, label='M_env = M')
    ax2.set_xlabel('r'); ax2.set_ylabel(r'$M_{env}(<r) / M$')
    ax2.set_title('Enclosed Envelope Mass / Source Mass')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # --- Panel 3: K(r) and linear fit ---
    ax3 = axes[0, 2]
    r_k = np.array(sorted(K_of_r.keys()), dtype=float)
    K_k = np.array([K_of_r[r] for r in sorted(K_of_r.keys())])
    ax3.plot(r_k, K_k, 'bo-', markersize=3, label='K(r) measured')
    ax3.plot(r_k, alpha_K * r_k + beta_K, 'r--', label=f'Fit: {alpha_K:.4f}r + {beta_K:.4f}')
    ax3.set_xlabel('r'); ax3.set_ylabel('K(r)')
    ax3.set_title(r'Cumulative $\sum f^2 N_{shell}$')
    ax3.legend(); ax3.grid(True, alpha=0.3)

    # --- Panel 4: Self-consistent Schwarzschild radius ---
    ax4 = axes[1, 0]
    M_arr = np.array(M_values, dtype=float)
    r_sc_arr = np.array([r_s_sc[M] for M in M_values])
    r_bare_arr = 2 * G_EFF * M_arr
    ax4.plot(M_arr, r_sc_arr, 'rs-', markersize=7, label=r'$r_s$ (self-consistent)')
    ax4.plot(M_arr, r_bare_arr, 'b--', linewidth=2, label=r'$r_s = 2G_{eff}M$ (bare)')
    ax4.set_xlabel('M'); ax4.set_ylabel(r'$r_s$')
    ax4.set_title('Self-Consistent vs Bare Schwarzschild Radius')
    ax4.legend(); ax4.grid(True, alpha=0.3)

    # --- Panel 5: Enhancement factor ---
    ax5 = axes[1, 1]
    enhancement = r_sc_arr / r_bare_arr
    ax5.semilogx(M_arr, enhancement, 'go-', markersize=7)
    ax5.axhline(1.0, color='gray', linestyle='--')
    ax5.axvline(M_crit, color='red', linestyle=':', label=f'$M_{{cross}}$ = {M_crit:.1f}')
    ax5.set_xlabel('M'); ax5.set_ylabel('Enhancement factor')
    ax5.set_title(r'$r_s^{SC} / r_s^{bare}$')
    ax5.legend(); ax5.grid(True, alpha=0.3)

    # --- Panel 6: Local dominance radius ---
    ax6 = axes[1, 2]
    r_loc_arr = np.array([r_s_local[M] for M in M_values])
    M_line = np.linspace(1, max(M_values), 100)
    ax6.plot(M_arr, r_loc_arr, 'ms', markersize=7, label=r'$r_{local}$ (measured)')
    ax6.plot(M_line, r_local_pred(M_line), 'r--', linewidth=2,
             label=f'$r = Ma_0^2/G_{{eff}}$ = {a_0**2/G_EFF:.4f}M')
    ax6.plot(M_line, 2*G_EFF*M_line, 'b:', linewidth=2,
             label=f'$r_s = 2G_{{eff}}M$ = {2*G_EFF:.4f}M')
    ax6.set_xlabel('M'); ax6.set_ylabel(r'$r_{local}$')
    ax6.set_title('Local Envelope Dominance Radius')
    ax6.legend(); ax6.grid(True, alpha=0.3)

    plt.suptitle('Paper 27 — Black Hole as Envelope', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('C:/Users/selin/merkabit_results/black_holes/sim1b_envelope_black_hole.png',
                dpi=150, bbox_inches='tight')
    plt.savefig('C:/Users/selin/merkabit_results/black_holes/sim1b_envelope_black_hole.pdf',
                bbox_inches='tight')
    log("  Saved: sim1b_envelope_black_hole.png/pdf")
    log()

    # ================================================================
    #  SUMMARY
    # ================================================================
    log("=" * 70)
    log("  SUMMARY: BLACK HOLE AS ENVELOPE")
    log("=" * 70)
    log()
    log("  1. ENVELOPE ENERGY is QUADRATIC in M:")
    log(f"     M_env = G_eff * M^2 * K,  K = {K_total:.4f}")
    log()
    log("  2. SELF-GRAVITATION THRESHOLD:")
    log(f"     M_crit = 1/(G_eff * K) = {1.0/(G_EFF*K_total):.4f}")
    log(f"     Below M_crit: envelope is negligible (Regime I)")
    log(f"     Above M_crit: envelope mass > source mass (Regime II)")
    log()
    log("  3. SELF-CONSISTENT SCHWARZSCHILD RADIUS:")
    log("     r_s = 2*G_eff*(M + M_env(r_s))  [iterative equation]")
    log("     Regime I:  r_s ~ 2*G_eff*M       [standard Schwarzschild]")
    log("     Regime II: r_s >> 2*G_eff*M       [envelope-dominated]")
    log()
    log("  4. THE BLACK HOLE IS THE ENVELOPE:")
    log("     Above M_crit, the envelope's own gravitational field")
    log("     generates more envelope energy than the original source.")
    log("     The system is self-sustaining: the field IS the mass.")
    log("     This is not torsion saturation — it is torsion BOOTSTRAP.")
    log()
    log("  5. K(r) ~ alpha*r (LINEAR in r for 1/r potential):")
    log(f"     alpha_K = {alpha_K:.6f}")
    log(f"     This means M_env(<r) grows linearly with r,")
    log(f"     giving M_total(r) = M + const*M^2*r")
    log(f"     The self-consistent equation r_s = 2G*(M + cM^2*r_s)")
    log(f"     has the unique solution:")
    log(f"       r_s = 2GM / (1 - 2G^2*alpha_K*M^2)")
    log(f"     This DIVERGES at M_div = 1/sqrt(2*G^2*alpha_K)")
    log(f"                            = {1.0/np.sqrt(2*G_EFF**2*alpha_K):.2f}")
    log(f"     Beyond this mass, no static horizon exists —")
    log(f"     the envelope engulfs the entire lattice.")
    log()

    # Check the closed-form solution
    log("  CLOSED-FORM SCHWARZSCHILD RADIUS:")
    log("    r_s = 2*G*M / (1 - 2*G^2*alpha_K*M^2)")
    log()
    log(f"  {'M':>6s}   {'r_s(closed)':>12s}   {'r_s(iter)':>12s}   {'r_s=2GM':>10s}   {'match':>8s}")
    log(f"  {'-'*6}   {'-'*12}   {'-'*12}   {'-'*10}   {'-'*8}")

    M_div = 1.0 / np.sqrt(2 * G_EFF**2 * alpha_K)
    for M in M_values:
        denom = 1 - 2 * G_EFF**2 * alpha_K * M**2
        if denom > 0:
            r_closed = 2 * G_EFF * M / denom
        else:
            r_closed = np.inf
        r_iter = r_s_sc[M]
        r_bare = 2 * G_EFF * M
        match = abs(r_closed - r_iter) / max(r_iter, 1e-10)
        log(f"  {M:6d}   {r_closed:12.2f}   {r_iter:12.2f}   {r_bare:10.2f}   {match:8.4f}")

    log()
    log(f"  DIVERGENCE MASS: M_div = {M_div:.2f}")
    log(f"  (Envelope engulfs the lattice beyond this mass)")
    log()

    elapsed = (datetime.now() - start).total_seconds()
    log(f"  Runtime: {elapsed:.1f} seconds")
    log("=" * 70)

    with open('C:/Users/selin/merkabit_results/black_holes/sim1b_envelope_output.txt', 'w') as f:
        f.write('\n'.join(out))
    log("\n  Output saved to sim1b_envelope_output.txt")


if __name__ == '__main__':
    main()
