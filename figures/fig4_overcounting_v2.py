#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper 27 — Figure 4 (updated): The overcounting reduction."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'figure.facecolor': 'white',
})

fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))

labels = ['Unconstrained\n(nodes independent)', 'Laplace-\nconstrained', 'Bekenstein-\nHawking']
values = [9.95, 0.273, 1.0]
colors = ['#DDDDDD', '#D6604D', '#2166AC']
edge_colors = ['#999999', '#B0413E', '#1A5276']

bars = ax.bar(labels, values, color=colors, edgecolor=edge_colors, linewidth=2, width=0.55)

# Value labels
ax.text(bars[0].get_x() + bars[0].get_width()/2., values[0] + 0.25,
        r'$\approx 10$', ha='center', va='bottom', fontweight='bold', fontsize=15)
ax.text(bars[0].get_x() + bars[0].get_width()/2., values[0] - 0.6,
        '(pentachoric\nedge count $K$)', ha='center', va='top', fontsize=9,
        color='#666666', fontstyle='italic')

ax.text(bars[1].get_x() + bars[1].get_width()/2., values[1] + 0.25,
        '0.273', ha='center', va='bottom', fontweight='bold', fontsize=14)

ax.text(bars[2].get_x() + bars[2].get_width()/2., values[2] + 0.25,
        '1.0', ha='center', va='bottom', fontweight='bold', fontsize=14)

# Arrow showing Laplace reduction
ax.annotate('', xy=(1, 2.0), xytext=(0.15, 9.0),
            arrowprops=dict(arrowstyle='->', color='#D6604D', lw=2.5,
                            connectionstyle='arc3,rad=-0.3'))
ax.text(0.2, 5.8, 'Laplace\nconstraint', fontsize=11, color='#D6604D',
        fontweight='bold')

# Filter ratio annotation
ax.annotate('', xy=(0.15, 10.8), xytext=(0.95, 1.2),
            arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.2,
                            connectionstyle='bar,fraction=-0.25'))
ax.text(1.35, 8.5, r'Filter ratio $\approx\, 36.5$' + '\n' + r'$\approx\, \alpha^{-1}\, /\, 2\ln(1/G)^2$',
        fontsize=10, color='#333333', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFDE7', edgecolor='#CCCCCC'))

ax.axhline(1.0, color='#2166AC', linewidth=1.5, linestyle='--', alpha=0.4)

ax.set_ylabel('$S \\; / \\; (A/4)$', fontsize=14)
ax.set_ylim(0, 12.5)
ax.grid(True, alpha=0.15, axis='y')

plt.tight_layout()
plt.savefig('C:/Users/selin/merkabit_results/black_holes/fig4_overcounting.png',
            dpi=300, bbox_inches='tight')
plt.savefig('C:/Users/selin/merkabit_results/black_holes/fig4_overcounting.pdf',
            bbox_inches='tight')
print('Saved fig4 (updated)')
