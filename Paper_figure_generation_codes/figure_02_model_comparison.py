import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parents[1]
comp = pd.read_csv(root / 'Phase_4_Dynamic_Selection' / 'outputs' / 'pairwise_v3_4_eval' / 'model_comparison.csv')
fig, ax = plt.subplots(figsize=(10, 6))
methods = comp['method'].tolist()
values = comp['accuracy'].tolist()
colors = ['#2ca02c' if m == 'V3.4 Selector' else '#4C72B0' for m in methods]
ax.bar(methods, values, color=colors)
ax.set_title('Model comparison: accuracy')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Method')
ax.tick_params(axis='x', rotation=45)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(root / 'paper_assets' / 'figure_02_model_comparison.png', dpi=200)
plt.close(fig)
