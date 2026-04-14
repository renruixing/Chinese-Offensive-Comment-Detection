import pandas as pd
import matplotlib.pyplot as plt

# 中文显示（Times New Roman 论文字体）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 18


# ---------------------- 读取并统一数据 ----------------------
def load_data(path, batch_value):
    df = pd.read_csv(path)
    df.columns = ['emb_type', 'epoch', 'acc', 'precision', 'recall', 'f1']

    # 转数字
    for col in ['epoch', 'acc', 'precision', 'recall', 'f1']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()
    df['batch_size'] = batch_value  # 改成 batch_size
    return df


# ===================== 读取 3 个 BATCH SIZE 文件（你自己改文件名） =====================
df_bs8 = load_data("training_lr0001_bs8_log.csv", 8)  # 最优 lr=0.001
df_bs16 = load_data("training_lr0001_bs16_log.csv", 16)
df_bs32 = load_data("training_lr0001_bs32_log.csv", 32)

# 合并
df_all = pd.concat([df_bs8, df_bs16, df_bs32])

# 获取所有7种方案
all_embs = df_all['emb_type'].unique()

# ---------------------- 画图：每个方案一张图 ----------------------
colors = {8: 'blue', 16: 'red', 32: 'orange'}
labels = {8: 'batch=8', 16: 'batch=16', 32: 'batch=32'}

for emb in all_embs:
    plt.figure(figsize=(10, 5))

    # 筛选当前模型
    sub_df = df_all[df_all['emb_type'] == emb]

    # 画 3 个 batch size
    for bs in [8, 16, 32]:
        data = sub_df[sub_df['batch_size'] == bs].sort_values('epoch')
        plt.plot(data['epoch'], data['f1'],
                 color=colors[bs],
                 label=labels[bs],
                 linewidth=2.5)

    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 输出 PDF（论文格式）
    plt.savefig(f"./diff_bs_results/{emb}'s F1.pdf", dpi=300, bbox_inches='tight')
    plt.show()