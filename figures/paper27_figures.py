#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper 27 — Publication Figures
Merkabit Research Program — Selina Stenberg, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Arc
from scipy.special import gammaln

# Style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'figure.facecolor': 'white',
})

G_EFF = 0.2542
PHI_MAX = 1.0 / G_EFF
A_0 = 0.3077
OUTDIR = 'C:/Users/selin/merkabit_results/black_holes/'

# ============================================================
#  FIGURE 1: The Interior Profile (Core + Transition + Tail)
# ============================================================

def fig1_interior_profile():
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    r = np.linspace(0.5, 20, 500)

    for M, color, ls in [(20, '#2166AC', '-'), (100, '#D6604D', '-'), (500, '#4DAF4A', '-')]:
        r_p = A_0 * G_EFF * M
        phi = np.where(r <= r_p, PHI_MAX, PHI_MAX * r_p / r)
        ax.plot(r, phi, color=color, linewidth=2.2, linestyle=ls, label=f'M = {M}')
        # Mark plateau edge
        ax.plot(r_p, PHI_MAX, 'o', color=color, markersize=6, zorder=5)

    # Regions
    ax.axhline(PHI_MAX, color='#888888', linewidth=1, linestyle='--', alpha=0.5)
    ax.text(0.8, PHI_MAX + 0.15, r'$\phi_{max} = 1/G_{eff}$', fontsize=10, color='#555555')

    # Annotate regions for M=100
    r_p_100 = A_0 * G_EFF * 100
    ax.annotate('', xy=(0.5, 0.3), xytext=(r_p_100, 0.3),
                arrowprops=dict(arrowstyle='<->', color='#2166AC', lw=1.5))
    ax.text((0.5 + r_p_100)/2, 0.55, 'Core\n(Planck density)', ha='center',
            fontsize=9, color='#2166AC', fontstyle='italic')

    ax.annotate('', xy=(r_p_100, 0.3), xytext=(r_p_100 + 3, 0.3),
                arrowprops=dict(arrowstyle='<->', color='#D6604D', lw=1.5))
    ax.text(r_p_100 + 1.5, 0.55, 'Horizon', ha='center',
            fontsize=9, color='#D6604D', fontstyle='italic')

    ax.annotate('', xy=(r_p_100 + 3, 0.3), xytext=(19.5, 0.3),
                arrowprops=dict(arrowstyle='<->', color='#4DAF4A', lw=1.5))
    ax.text(14, 0.55, r'Exterior ($\phi \sim 1/r$)', ha='center',
            fontsize=9, color='#4DAF4A', fontstyle='italic')

    ax.set_xlabel('Radius  $r$  (lattice units)', fontsize=12)
    ax.set_ylabel(r'Torsion potential  $\phi(r)$', fontsize=12)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 5)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUTDIR + 'fig1_interior_profile.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTDIR + 'fig1_interior_profile.pdf', bbox_inches='tight')
    print('Saved fig1')


# ============================================================
#  FIGURE 2: Transfer Spectrum — Mode Suppression
# ============================================================

def fig2_transfer_spectrum():
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    r_p = 15  # M=200
    r_probe = r_p + 1

    l_arr = np.arange(0, 35)
    sigma_l = (r_p / r_probe) ** (l_arr + 1)
    phi_sigma = PHI_MAX * sigma_l

    # Bar chart colored by visibility
    colors = ['#D6604D' if ps > 1 else '#CCCCCC' for ps in phi_sigma]
    bars = ax.bar(l_arr, phi_sigma, color=colors, edgecolor='none', width=0.8)

    # Planck threshold
    ax.axhline(1.0, color='#2166AC', linewidth=2, linestyle='--', zorder=3)
    ax.text(28, 1.15, 'Planck threshold', fontsize=10, color='#2166AC', fontweight='bold')

    # Critical l
    l_c = int(r_p * np.log(PHI_MAX))
    ax.axvline(l_c, color='#4DAF4A', linewidth=1.5, linestyle=':', alpha=0.8)
    ax.text(l_c + 0.5, 2.5, f'$l_c = {l_c}$\n(cutoff)', fontsize=10,
            color='#4DAF4A', fontweight='bold')

    # Labels
    ax.text(5, 3.2, 'Visible modes\n(contribute to entropy)',
            fontsize=9, color='#D6604D', ha='center', fontstyle='italic')
    ax.text(28, 0.3, 'Invisible\n(suppressed)',
            fontsize=9, color='#999999', ha='center', fontstyle='italic')

    ax.set_xlabel('Angular mode number  $l$', fontsize=12)
    ax.set_ylabel(r'$\phi_{max} \times \sigma_l$', fontsize=12)
    ax.set_yscale('log')
    ax.set_ylim(0.01, 6)
    ax.set_xlim(-0.5, 34.5)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(OUTDIR + 'fig2_transfer_spectrum.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTDIR + 'fig2_transfer_spectrum.pdf', bbox_inches='tight')
    print('Saved fig2')


# ============================================================
#  FIGURE 3: Entropy vs Area — The Area Law
# ============================================================

def fig3_entropy_vs_area():
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    M_fine = np.arange(10, 501, 5)
    S_fine = []
    A4_fine = []

    for M in M_fine:
        r_p = max(1, int(A_0 * G_EFF * M))
        l_max = int(2 * r_p) + 1
        r_probe = r_p + 1
        S = 0
        for l in range(l_max + 1):
            sigma = (r_p / r_probe) ** (l + 1)
            n_lev = PHI_MAX * sigma
            if n_lev > 1:
                S += (2*l + 1) * np.log(n_lev)
        S_fine.append(S)
        A4_fine.append(np.pi * r_p**2)

    S_arr = np.array(S_fine)
    A4_arr = np.array(A4_fine)

    ax.plot(A4_arr, S_arr, 'o', color='#D6604D', markersize=4, alpha=0.7, label='Lattice measurement')

    # Fit line
    c = np.sum(S_arr * A4_arr) / np.sum(A4_arr**2)
    A4_line = np.linspace(0, max(A4_arr)*1.1, 100)
    ax.plot(A4_line, c * A4_line, '-', color='#D6604D', linewidth=2,
            label=f'$S = {c:.3f} \\times A/4$')

    # BH line
    ax.plot(A4_line, A4_line, '--', color='#2166AC', linewidth=2,
            label='$S = A/4$  (Bekenstein-Hawking)')

    ax.set_xlabel('$A/4$  (lattice units)', fontsize=12)
    ax.set_ylabel('$S$  (Laplace-constrained entropy)', fontsize=12)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, max(A4_arr)*1.05)
    ax.set_ylim(0, max(S_arr)*1.15)

    plt.tight_layout()
    plt.savefig(OUTDIR + 'fig3_entropy_vs_area.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTDIR + 'fig3_entropy_vs_area.pdf', bbox_inches='tight')
    print('Saved fig3')


# ============================================================
#  FIGURE 4: The Overcounting — Three Bars
# ============================================================

def fig4_overcounting():
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))

    labels = ['Unconstrained\n(nodes independent)', 'Laplace-\nconstrained', 'Bekenstein-\nHawking']
    values = [7.3, 0.273, 1.0]
    colors = ['#DDDDDD', '#D6604D', '#2166AC']
    edge_colors = ['#999999', '#B0413E', '#1A5276']

    bars = ax.bar(labels, values, color=colors, edgecolor=edge_colors, linewidth=2, width=0.6)

    # Value labels
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., v + 0.2,
                f'{v:.3f}' if v < 1 else f'{v:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=13)

    # Arrow showing reduction
    ax.annotate('', xy=(1, 1.5), xytext=(0, 6.5),
                arrowprops=dict(arrowstyle='->', color='#D6604D', lw=2.5,
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(0.15, 4.2, 'Laplace\nconstraint', fontsize=10, color='#D6604D',
            fontweight='bold', rotation=0)

    ax.axhline(1.0, color='#2166AC', linewidth=1.5, linestyle='--', alpha=0.4)

    ax.set_ylabel('$S \\, / \\, (A/4)$', fontsize=13)
    ax.set_ylim(0, 9)
    ax.grid(True, alpha=0.15, axis='y')

    plt.tight_layout()
    plt.savefig(OUTDIR + 'fig4_overcounting.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTDIR + 'fig4_overcounting.pdf', bbox_inches='tight')
    print('Saved fig4')


# ============================================================
#  FIGURE 5: Three-Zone Schematic (Core / Horizon / Exterior)
# ============================================================

def fig5_schematic():
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.set_aspect('equal')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5.5, 5.5)
    ax.axis('off')

    # Core (filled circle)
    core = plt.Circle((0, 0), 1.5, color='#D6604D', alpha=0.85, zorder=3)
    ax.add_patch(core)
    ax.text(0, 0, '$T_{75}$\nsaturated\ncore', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=4)

    # Horizon shell (ring)
    horizon = plt.Circle((0, 0), 1.8, fill=False, edgecolor='#E8A838',
                          linewidth=4, linestyle='-', zorder=3)
    ax.add_patch(horizon)

    # Exterior (faint concentric rings for 1/r field)
    for r_ring in [2.5, 3.3, 4.3, 5.3]:
        ring = plt.Circle((0, 0), r_ring, fill=False, edgecolor='#2166AC',
                          linewidth=0.8, linestyle='-', alpha=0.3 / (r_ring/2.5))
        ax.add_patch(ring)

    # Labels
    ax.annotate('Horizon\n(1 pixel thick)', xy=(1.8 * 0.707, 1.8 * 0.707),
                xytext=(4.0, 3.8),
                fontsize=10, color='#E8A838', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#E8A838', lw=1.5),
                ha='center')

    ax.text(0, -2.8, r'Exterior:  $\phi \sim 1/r$', ha='center',
            fontsize=10, color='#2166AC', fontstyle='italic')

    # r_p and r_s labels
    ax.annotate('', xy=(1.5, -4.2), xytext=(0, -4.2),
                arrowprops=dict(arrowstyle='<->', color='#D6604D', lw=1.5))
    ax.text(0.75, -4.6, '$r_p$', ha='center', fontsize=11, color='#D6604D', fontweight='bold')

    ax.annotate('', xy=(4.0, -4.2), xytext=(0, -4.2),
                arrowprops=dict(arrowstyle='<->', color='#2166AC', lw=1.5))
    ax.text(2.0, -4.9, '$r_s = 2G_{eff}M$', ha='center', fontsize=10, color='#2166AC')

    ax.text(0.75, -3.7, '$= (a_0/2) \\times r_s$', ha='center',
            fontsize=9, color='#D6604D', fontstyle='italic')

    # The ratio
    ax.text(0, 5.0, '$r_p \\, / \\, r_s = a_0/2 = 0.154$',
            ha='center', fontsize=12, fontweight='bold', color='#333333',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F0F0', edgecolor='#CCCCCC'))

    plt.tight_layout()
    plt.savefig(OUTDIR + 'fig5_schematic.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTDIR + 'fig5_schematic.pdf', bbox_inches='tight')
    print('Saved fig5')


# ============================================================
#  MAIN
# ============================================================

if __name__ == '__main__':
    fig1_interior_profile()
    fig2_transfer_spectrum()
    fig3_entropy_vs_area()
    fig4_overcounting()
    fig5_schematic()
    print('\nAll figures saved to', OUTDIR)
