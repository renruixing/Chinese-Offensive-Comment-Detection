import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 18

df = pd.read_csv("train_lr0001_bs16_loss_log.csv",
                 names=["emb_type", "epoch", "train_loss"])

# 颜色 & 名称
colors = {
    "word2vec": "#1f77b4",
    "glove": "#ff7f0e",
    "fasttext": "#2ca02c",
    "bert": "#d62728",
    "sbert": "#9467bd",
    "simcse": "#8c564b",
    "llm": "#e377c2"
}
model_names = {
    "word2vec": "Word2Vec",
    "glove": "GloVe",
    "fasttext": "FastText",
    "bert": "BERT",
    "sbert": "SBERT",
    "simcse": "SimCSE",
    "llm": "LLM (BGE)"
}

plt.figure(figsize=(10, 6))

for emb in df['emb_type'].unique():
    sub = df[df['emb_type'] == emb]
    plt.plot(sub['epoch'], sub['train_loss'],
             label=model_names.get(emb, emb),
             color=colors.get(emb, "#000000"),
             linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(range(1, 52, 2))
plt.tight_layout()

plt.savefig("train_loss_curve.pdf", dpi=300)
plt.show()