import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parents[1]
class_df = pd.read_csv(root / 'Phase_4_Dynamic_Selection' / 'outputs' / 'pairwise_v3_4_eval' / 'class_performance.csv')
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(class_df['class'], class_df['v3_4_accuracy'], color='#55A868')
ax.set_title('V3.4 per-class accuracy')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Class')
ax.tick_params(axis='x', rotation=45)
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(root / 'paper_assets' / 'figure_03_v34_class_performance.png', dpi=200)
plt.close(fig)
