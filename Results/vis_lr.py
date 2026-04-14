import pandas as pd
import matplotlib.pyplot as plt

# 中文显示
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 18


# ---------------------- 读取并统一数据 ----------------------
def load_data(path, lr_value):
    df = pd.read_csv(path)
    df.columns = ['emb_type', 'epoch', 'acc', 'precision', 'recall', 'f1']

    # 转数字
    for col in ['epoch', 'acc', 'precision', 'recall', 'f1']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()
    df['lr'] = lr_value
    return df


# 读取三个学习率文件
df_lr1 = load_data("training_lr00001_bs16_log.csv", 0.0001)
df_lr2 = load_data("training_lr0001_bs16_log.csv", 0.001)
df_lr3 = load_data("training_lr001_bs16_log.csv", 0.01)

# 合并
df_all = pd.concat([df_lr1, df_lr2, df_lr3])

# 获取所有7种方案
all_embs = df_all['emb_type'].unique()

# ---------------------- 画图：每个方案一张图 ----------------------
colors = {0.0001: 'blue', 0.001: 'red', 0.01: 'orange'}
labels = {0.0001: 'lr=0.0001', 0.001: 'lr=0.001', 0.01: 'lr=0.01'}

for emb in all_embs:
    plt.figure(figsize=(10, 5))

    # 筛选当前方案
    sub_df = df_all[df_all['emb_type'] == emb]

    # 画3个学习率
    for lr in [0.0001, 0.001, 0.01]:
        data = sub_df[sub_df['lr'] == lr].sort_values('epoch')
        plt.plot(data['epoch'], data['f1'],
                 color=colors[lr],
                 label=labels[lr],
                 linewidth=2.5)

    # plt.title(f"Scheme: {emb} | F1", fontsize=14)
    plt.xlabel("Epoch",)
    plt.ylabel("F1 Score")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./diff_lr_results/{emb}'s F1.pdf", dpi=300, bbox_inches='tight')
    plt.show()