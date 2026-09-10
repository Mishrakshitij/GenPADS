"""Regenerate release figures from the dataset manifest."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'datasets/manifest.json').read_text())
domains = ['flights', 'food-ordering', 'hotels', 'movies', 'music', 'restaurant-search', 'sports']
labels = ['Impolite', 'Somewhat impolite', 'Somewhat polite', 'Polite']
colors = ['#b6465f', '#e69b68', '#80b5be', '#287c8e']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.spines.left': False})
output = root / 'docs/assets'
output.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 5.4), layout='constrained')
left = np.zeros(7)
for i, (label, color) in enumerate(zip(labels, colors)):
    counts = np.array([manifest['files'][f'padd/{domain}.csv']['labels'].get(str(i), 0) for domain in domains])
    total = np.array([manifest['files'][f'padd/{domain}.csv']['rows'] for domain in domains])
    widths = counts / total * 100
    ax.barh(domains, widths, left=left, label=label, color=color, height=.64)
    for y, (x, width) in enumerate(zip(left, widths)):
        if width > 4:
            ax.text(x + width/2, y, f'{width:.1f}%', ha='center', va='center', color='white' if i in (0, 3) else '#183139', fontsize=9)
    left += widths
ax.invert_yaxis()
ax.set(xlabel='Share of utterances (%)', xlim=(0, 100))
fig.suptitle('PADD · Politeness across seven domains', fontsize=16)
ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=4, frameon=False, fontsize=9)
ax.tick_params(axis='y', length=0)
fig.savefig(output / 'politeness-distribution.png', dpi=180, bbox_inches='tight')
plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 4.7), layout='constrained')
values = [manifest['files'][f'padd/{domain}.csv']['rows'] for domain in domains]
ax.barh(domains, values, color='#287c8e', height=.64)
for i, count in enumerate(values):
    ax.text(count + 600, i, f'{count:,}', va='center', fontsize=10)
ax.invert_yaxis()
ax.set(xlabel='Utterances', xlim=(0, max(values)*1.17), title='Taskmaster-2 · Dataset coverage')
ax.tick_params(axis='y', length=0)
fig.savefig(output / 'dataset-coverage.png', dpi=180, bbox_inches='tight')
plt.close(fig)
