#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper 28 — Publication Figures"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 11,
    'axes.linewidth': 1.2, 'figure.facecolor': 'white',
})

H = 12
ALPHA_INV = 137.035999

def alpha_formula(G):
    x = 1.0 / G
    s_node = (1 + x) * np.log(1 + x) - x * np.log(x)
    return 2 * H * np.pi * s_node / np.log(x)

OUT = 'C:/Users/selin/merkabit_results/black_holes/'

# ================================================================
#  FIGURE 1: alpha^-1 vs G_eff
# ================================================================

def fig1():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    G_arr = np.linspace(0.05, 0.5, 500)
    alpha_arr = np.array([alpha_formula(g) for g in G_arr])

    ax.plot(G_arr, alpha_arr, '-', color='#2166AC', linewidth=2.5, label='$\\alpha^{-1} = 2h\\pi \\cdot S_{node} \\,/\\, \\ln(1/G)$')
    ax.axhline(ALPHA_INV, color='#999999', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(0.42, ALPHA_INV + 0.8, '$\\alpha^{-1} = 137.036$', fontsize=10, color='#999', ha='right')

    # Key points
    points = [
        (0.2542, 'Lattice simulation\n$G = 0.2542$', '#D6604D', 136.918, 's', 10),
        (0.2500, 'Algebraic $G = 1/4$', '#4DAF4A', alpha_formula(0.25), 'D', 9),
        (0.25479, 'Exact solution\n$G = 0.25479$', '#E8A838', 137.036, '*', 14),
        (0.1210, 'Holographic\n$G = 0.1210$', '#9B5FC0', alpha_formula(0.1210), 'o', 8),
    ]

    for g, label, color, a_val, marker, ms in points:
        ax.plot(g, a_val, marker=marker, color=color, markersize=ms,
                markeredgecolor='black', markeredgewidth=0.8, zorder=5)

    # Annotations with offset to avoid overlap
    ax.annotate('Lattice simulation\n$G_{eff} = 0.2542$\n$\\alpha^{-1} = 136.918$\n(0.086%)',
                xy=(0.2542, 136.918), xytext=(0.33, 133),
                fontsize=9, color='#D6604D', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#D6604D', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#D6604D', alpha=0.08))

    ax.annotate('Exact: $G = 0.25479$\n$\\alpha^{-1} = 137.036$',
                xy=(0.25479, 137.036), xytext=(0.33, 139.5),
                fontsize=9, color='#E8A838', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#E8A838', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8A838', alpha=0.08))

    ax.annotate('Algebraic $G = 1/4$\n$\\alpha^{-1} = 136.50$\n(0.39%)',
                xy=(0.25, alpha_formula(0.25)), xytext=(0.12, 133),
                fontsize=9, color='#4DAF4A', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#4DAF4A', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#4DAF4A', alpha=0.08))

    ax.annotate('Holographic\n$G = 0.1210$\n$\\alpha^{-1} = 128.4$\n(6.3%)',
                xy=(0.1210, alpha_formula(0.1210)), xytext=(0.07, 123),
                fontsize=9, color='#9B5FC0', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#9B5FC0', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#9B5FC0', alpha=0.08))

    ax.set_xlabel('$G_{eff}$ (lattice gravitational constant)', fontsize=12)
    ax.set_ylabel('$\\alpha^{-1}$', fontsize=13)
    ax.set_title('Figure 1.  The Fine Structure Constant from Holographic Boundary Counting',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0.05, 0.45)
    ax.set_ylim(115, 145)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUT + 'paper28_fig1.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUT + 'paper28_fig1.pdf', bbox_inches='tight')
    print('Saved paper28_fig1')


# ================================================================
#  FIGURE 2: The three factors decomposition
# ================================================================

def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.1, 1]})

    # --- Left panel: factor breakdown as stacked bar concept ---
    ax1 = axes[0]
    ax1.axis('off')
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
    ax1.set_title('(a)  The Three Factors', fontsize=13, fontweight='bold')

    G = 0.2542
    x = 1.0/G
    s_node = (1+x)*np.log(1+x) - x*np.log(x)
    ln_G = np.log(x)
    two_h_pi = 2 * H * np.pi

    # The formula as a visual multiplication
    y_start = 8.5
    factors = [
        ('$2h\\pi$', f'{two_h_pi:.1f}', '#2166AC',
         'Full angular sweep\n$h = 12$ steps $\\times$ $2\\pi$ rad'),
        ('$S_{node}$', f'{s_node:.3f}', '#D6604D',
         'Shannon entropy per node\nat maximum occupation'),
        ('$1\\,/\\,\\ln(1/G)$', f'{1/ln_G:.4f}', '#4DAF4A',
         'Inverse Planck depth\n(logarithmic torsion cutoff)'),
    ]

    for i, (symbol, value, color, desc) in enumerate(factors):
        y = y_start - i * 2.8

        # Symbol box
        ax1.text(1.5, y, symbol, fontsize=18, fontweight='bold', color=color,
                 ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.1,
                           edgecolor=color, linewidth=1.5))

        # = value
        ax1.text(3.5, y, f'= {value}', fontsize=14, va='center', color='#333')

        # Description
        ax1.text(6.5, y, desc, fontsize=10, va='center', color='#666', fontstyle='italic')

        # Multiplication signs between
        if i < len(factors) - 1:
            ax1.text(1.5, y - 1.4, '$\\times$', fontsize=16, ha='center',
                     color='#999', fontweight='bold')

    # Result
    ax1.plot([0.5, 9.5], [1.0, 1.0], '-', color='#333', linewidth=1.5)
    ax1.text(1.5, 0.3, '$\\alpha^{-1}$', fontsize=20, fontweight='bold',
             color='#333', ha='center')
    ax1.text(3.5, 0.3, f'= {two_h_pi:.1f} $\\times$ {s_node:.3f} $\\times$ {1/ln_G:.4f}',
             fontsize=12, va='center', color='#333')
    ax1.text(8.5, 0.3, '= 136.918', fontsize=14, fontweight='bold',
             color='#D6604D', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E1',
                       edgecolor='#E8A838', linewidth=1.5))

    # --- Right panel: S_node as Shannon entropy visual ---
    ax2 = axes[1]
    ax2.set_title('(b)  Shannon Entropy of a Planck Node', fontsize=13, fontweight='bold')

    # Plot S_node = (1+x)ln(1+x) - x*ln(x) as function of x = 1/G
    x_arr = np.linspace(1.5, 10, 200)
    s_arr = (1 + x_arr) * np.log(1 + x_arr) - x_arr * np.log(x_arr)

    ax2.plot(x_arr, s_arr, '-', color='#D6604D', linewidth=2.5,
             label='$S_{node} = (1+x)\\ln(1+x) - x\\ln(x)$')

    # Mark the key point
    x_key = 1.0/G
    s_key = (1 + x_key) * np.log(1 + x_key) - x_key * np.log(x_key)
    ax2.plot(x_key, s_key, 's', color='#D6604D', markersize=10,
             markeredgecolor='black', markeredgewidth=0.8, zorder=5)
    ax2.annotate(f'$x = 1/G = {x_key:.2f}$\n$S_{{node}} = {s_key:.3f}$',
                xy=(x_key, s_key), xytext=(5.5, 3.2),
                fontsize=10, fontweight='bold', color='#D6604D',
                arrowprops=dict(arrowstyle='->', color='#D6604D', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#D6604D', alpha=0.08))

    ax2.set_xlabel('$x = 1/G_{eff} = \\phi_{max}$', fontsize=12)
    ax2.set_ylabel('$S_{node}$ (nats)', fontsize=12)
    ax2.legend(fontsize=9.5, loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.2)

    # Add interpretation
    ax2.text(7, 1.5, 'Maximum torsion\noccupation per node:\nhow many distinguishable\nstates fit in one pixel',
             fontsize=9, fontstyle='italic', color='#666',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', edgecolor='#CCC'))

    plt.suptitle('Figure 2.  Decomposition of $\\alpha^{-1} = 2h\\pi \\times S_{node}\\,/\\,\\ln(1/G)$',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUT + 'paper28_fig2.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUT + 'paper28_fig2.pdf', bbox_inches='tight')
    print('Saved paper28_fig2')


if __name__ == '__main__':
    fig1()
    fig2()
    print('All Paper 28 figures saved.')
