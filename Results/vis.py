import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 18

# ===================== Load & Clean Data =====================
df = pd.read_csv("training_lr0001_bs16_log.csv",
                 encoding="utf-8-sig", on_bad_lines='skip',
                 names=["emb_type", "epoch", "acc", "precision", "recall", "f1"])

# valid_models = ['word2vec', 'glove', 'fasttext', 'bert', 'sbert', 'simcse', 'llm']
valid_models = ['glove', 'fasttext', 'bert', 'sbert', 'simcse', 'llm']
df = df[df['emb_type'].isin(valid_models)].copy()
df['epoch'] = df['epoch'].astype(int)
df['acc'] = df['acc'].astype(float)
df['precision'] = df['precision'].astype(float)
df['recall'] = df['recall'].astype(float)
df['f1'] = df['f1'].astype(float)

model_display = {
    'word2vec': 'Word2Vec',
    'glove': 'GloVe',
    'fasttext': 'FastText',
    'bert': 'BERT',
    'sbert': 'SBERT',
    'simcse': 'SimCSE',
    'llm': 'LLM (BGE)'
}

# ===================== Color Scheme =====================
colors = {
    'word2vec': '#1f77b4',  # 蓝
    'glove': '#ff7f0e',     # 橙
    'fasttext': '#2ca02c',  # 绿
    'bert': '#d62728',      # 红
    'sbert': '#9467bd',     # 紫
    'simcse': '#8c564b',    # 棕
    'llm': '#ff69b4'        # 亮粉
}

# ===================== 2x2 Subplot =====================
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

# metrics = [
#     ('acc', 'Accuracy (%)', ax1, (55, 90)),
#     ('precision', 'Precision', ax2, (0.5, 0.9)),
#     ('recall', 'Recall', ax3, (0.5, 0.9)),
#     ('f1', 'F1 Score', ax4, (0.5, 0.9))
# ]

metrics = [
    ('acc', 'Accuracy (%)', ax1, (75, 90)),
    ('precision', 'Precision', ax2, (0.75, 0.9)),
    ('recall', 'Recall', ax3, (0.75, 0.9)),
    ('f1', 'F1 Score', ax4, (0.73, 0.9))
]

for key, title, ax, ylim in metrics:
    for model in valid_models:
        sub = df[df['emb_type'] == model].sort_values('epoch')
        ax.plot(
            sub['epoch'], sub[key],
            label=model_display[model],
            linewidth=2.5, markersize=5
        )
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    ax.set_ylim(ylim)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig("training_curves_4metrics.pdf", dpi=300, bbox_inches='tight')
plt.show()

print("✅ 4-metric English training curve saved: training_curves_4metrics_en.pdf")