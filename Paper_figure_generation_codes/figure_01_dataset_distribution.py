import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parents[1]
counts = pd.read_csv(root / 'outputs' / 'class_analysis' / 'class_counts.csv')
counts['total'] = counts[['train', 'val', 'test']].sum(axis=1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(counts['class'], counts['total'], color='#4C72B0')
ax.set_title('ISIC 2019 class distribution')
ax.set_ylabel('Sample count')
ax.set_xlabel('Class')
ax.tick_params(axis='x', rotation=45)
fig.tight_layout()
fig.savefig(root / 'paper_assets' / 'figure_01_dataset_distribution.png', dpi=200)
plt.close(fig)
