"""
所有词嵌入方法在第 1 个训练轮次即取得较高性能，这源于本文采用的两阶段训练范式：
预训练词嵌入模型在训练过程中保持冻结，仅对轻量分类器进行微调。预训练模型（尤其
是 LLM 与 SBERT）已在海量语料中学习到丰富的语义知识，仅需 1 轮训练即可使分
类器完成适配，从而达到高准确率；后续轮次仅带来边际性能提升，证明该范式下模型可快速收敛。
"""
from prepro_data.prepro import *
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from network import *
from config import *
import random
import numpy as np
import pandas as pd
import warnings
import h5py
import os
import time
warnings.filterwarnings('ignore')

from gensim.models import KeyedVectors, fasttext
from transformers import (
    BertModel, BertTokenizer
)
from sentence_transformers import SentenceTransformer

# ===================== 全局配置 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 32
SEED = 0
DIM_MAP = {
    'word2vec': 300,
    'glove': 50,
    'fasttext': 300,
    'bert': 768,
    'sbert': 768,
    'simcse': 768,
    'llm': 1024,  # LLM Embedding 大模型维度
}

embedding_time_list = []  # 全局记录嵌入时间


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ===================== 全系列词嵌入加载器 =====================
def load_word2vec(path):
    return loadWeiboModel(path, 1)


def load_glove(glove_path):
    model = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            model[word] = vector
    return model


def load_fasttext(path):
    return fasttext.load_facebook_model(path)


def load_bert():
    bert_local_path = r".\model\BERTModel\bert-base-chinese"
    tk = BertTokenizer.from_pretrained(bert_local_path)
    m = BertModel.from_pretrained(bert_local_path).to(device).eval()
    return tk, m


def load_sbert():
    sbert_local_path = r".\model\SBERTModel\sbert_base_chinese"
    model = SentenceTransformer(sbert_local_path).to(device).eval()
    return model


def load_simcse():
    from transformers import AutoModel, AutoTokenizer
    simcse_local_path = r".\model\SimCSEModel"
    tokenizer = AutoTokenizer.from_pretrained(simcse_local_path)
    model = AutoModel.from_pretrained(simcse_local_path).to(device).eval()
    return tokenizer, model


def load_llm_embedding():
    # 这里换成你自己的本地路径
    model_path = r".\model\BGEModel"
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path).to(device).eval()
    return model


# ===================== 统一句子向量化 =====================
def get_embedding(text_tokens, raw_text, emb_type, models):
    if emb_type == 'word2vec':
        vec = np.zeros(300)
        cnt = 0
        for w in text_tokens:
            if w in models['word2vec']:
                vec += models['word2vec'][w]
                cnt += 1
        return vec / cnt if cnt else vec

    elif emb_type == 'glove':
        vec = np.zeros(50)
        cnt = 0
        for w in text_tokens:
            if w in models['glove']:
                vec += models['glove'][w]
                cnt += 1
        return vec / cnt if cnt else vec

    elif emb_type == 'fasttext':
        vec = np.zeros(300)
        cnt = 0
        for w in text_tokens:
            vec += models['fasttext'].wv[w]
            cnt += 1
        return vec / cnt if cnt else vec

    elif emb_type == 'bert':
        tk, m = models['bert']
        inputs = tk(raw_text, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)
        with torch.no_grad():
            o = m(**inputs)
        return o.last_hidden_state[:,0,:].squeeze(0).cpu().numpy()

    elif emb_type == 'sbert':
        m = models['sbert']
        return m.encode(raw_text, convert_to_numpy=True)

    elif emb_type == 'simcse':
        tk, _ = models['bert']
        inputs = tk(raw_text, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            o = models['simcse'](**inputs)
        return o.last_hidden_state[:,0,:].squeeze(0).cpu().numpy()

    elif emb_type == 'llm':
        # 🔥 大模型嵌入（BGE-Large）
        m = models['llm']
        return m.encode(raw_text, convert_to_numpy=True)


# ===================== 数据集 =====================
class MyDataset(Dataset):
    def __init__(self, d, l):
        self.d = torch.tensor(d, dtype=torch.float32).to(device)
        self.l = torch.tensor(l, dtype=torch.long).to(device)
    def __getitem__(self, i): return self.d[i], self.l[i]
    def __len__(self): return len(self.d)


def process_all(filename, emb_type, models):
    df = pd.read_csv(filename)
    texts = df['text'].astype(str).values
    labels = df['label'].values
    tokens = [Sent2Word(t) for t in texts]
    embs = []

    # ===================== ✅ 嵌入开始计时 =====================
    start_emb = time.time()
    print(f"[{emb_type}] 开始文本嵌入...")
    if emb_type == "bert":
        print("🔹 BERT 批量处理（小显存安全版）...")
        tk, m = models['bert']
        batch_size = 32  # 小显卡友好！不会爆显存
        all_emb = []

        # 分批次送入 BERT → 速度快 + 不爆显存
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size].tolist()

            inputs = tk(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=32,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = m(**inputs)

            # 取出 CLS 句向量
            embs_batch = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_emb.append(embs_batch)

            # 立刻释放显存
            del inputs, outputs
            torch.cuda.empty_cache()

        embs = np.concatenate(all_emb, axis=0)

    elif emb_type == "sbert":
        print("🔹 SBERT 批量处理（小显存安全版）...")
        m = models['sbert']
        batch_size = 32
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size].tolist()
            embs_batch = m.encode(batch_texts, convert_to_numpy=True)
            all_emb.append(embs_batch)
            torch.cuda.empty_cache()
        embs = np.concatenate(all_emb, axis=0)

    elif emb_type == "simcse":
        print("🔹 SimCSE 批量处理（小显存安全版）...")
        tk, m = models['simcse']
        batch_size = 32
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size].tolist()
            inputs = tk(batch_texts, padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = m(**inputs)
            # 取 句向量  ](https://www.zhihu.com/question/446475624/answer/2787113763)
            embs_batch = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_emb.append(embs_batch)
            del inputs, outputs
            torch.cuda.empty_cache()
        embs = np.concatenate(all_emb, axis=0)

    elif emb_type == "llm":
        print("🔹 LLM Embedding (BGE-Large) 批量处理...")
        m = models['llm']
        batch_size = 32  # 大模型稍大，batch小一点
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size].tolist()
            embs_batch = m.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True  # LLM 嵌入推荐归一化
            )
            all_emb.append(embs_batch)
            torch.cuda.empty_cache()
        embs = np.concatenate(all_emb, axis=0)

    else:
        for t, raw in zip(tokens, texts):
            emb = get_embedding(t, raw, emb_type, models)
            embs.append(emb)

    # ===================== ✅ 嵌入结束计时 =====================
    emb_time = time.time() - start_emb
    embedding_time_list.append({
        "emb_type": emb_type,
        "embedding_time_sec": round(emb_time, 2)
    })
    print(f"[{emb_type}] 文本嵌入完成！耗时：{emb_time:.2f}s\n")

    from sklearn.model_selection import train_test_split
    return train_test_split(np.array(embs), labels, test_size=0.2, random_state=SEED, stratify=labels)


def run_nn(filename, emb_type, models, model_type='FCN'):
    X_train, X_test, y_train, y_test = process_all(filename, emb_type, models)
    train_loader = DataLoader(MyDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(MyDataset(X_test, y_test), batch_size=16, shuffle=False)

    input_dim = DIM_MAP[emb_type]
    if model_type == 'FCN':
        model = nnNet(input_dim=input_dim).to(device)
    elif model_type == 'CNN':
        model = cnnNet(input_dim=input_dim).to(device)
    elif model_type == 'LSTM':
        model = LSTM(input_dim=input_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ===================== ✅ 日志保存 =====================
    epoch_logs = []
    loss_logs = []
    time_logs = []

    for epoch in range(50):
        # ===================== ✅ 单 epoch 计时开始 =====================
        start_time = time.time()

        model.train()
        loss_sum, corr, total = 0,0,0
        for d, l in train_loader:
            opt.zero_grad()
            o = model(d, is_training=True)
            loss = criterion(o, l)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            corr += (o.argmax(1)==l).sum().item()
            total += l.size(0)

        # 平均训练 loss
        avg_train_loss = loss_sum / len(train_loader)

        # ===================== ✅ 单 epoch 计时结束 =====================
        epoch_time = time.time() - start_time

        model.eval()
        ap, al = [], []
        with torch.no_grad():
            for d, l in test_loader:
                o = model(d, is_training=False)
                ap.extend(o.argmax(1).cpu().numpy())
                al.extend(l.cpu().numpy())

        acc = 100 * np.mean(np.array(ap)==np.array(al))
        pre = precision_score(al, ap, average='macro', zero_division=0)
        rec = recall_score(al, ap, average='macro', zero_division=0)
        f1 = f1_score(al, ap, average='macro', zero_division=0)

        if epoch % 1 == 0:
            print(f"[{emb_type}] E{epoch+1:2d} | Loss={avg_train_loss:.4f} | Time={epoch_time:.2f}s | Acc={acc:.2f} | F1={f1:.4f}")

        # ===================== ✅ 保存每一轮数据 =====================
        epoch_logs.append({
            "emb_type": emb_type,
            "epoch": epoch + 1,
            "acc": round(acc, 4),
            "precision": round(pre, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        })

        # ✅ 保存训练 loss
        loss_logs.append({
            "emb_type": emb_type,
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4)
        })

        # ✅ 保存单 epoch 训练时长
        time_logs.append({
            "emb_type": emb_type,
            "epoch": epoch + 1,
            "time_sec": round(epoch_time, 2)
        })

    # ===================== ✅ 训练结束 → 写入 CSV =====================
    log_df = pd.DataFrame(epoch_logs)
    file_exists = os.path.exists("./Results/training_lr001_bs16_log.csv")
    # 第一次写入创建文件，后面追加
    log_df.to_csv(
        "./Results/training_lr001_bs16_log.csv",
        mode="a",            # 追加模式（所有模型的日志都存在一起）
        header=not file_exists,
        index=False,
        encoding="utf-8-sig"
    )

    # ===================== ✅ 保存 LOSS CSV（新增） =====================
    loss_df = pd.DataFrame(loss_logs)
    loss_file_exists = os.path.exists("./Results/train_lr001_bs16_loss_log.csv")
    loss_df.to_csv(
        "./Results/train_lr001_bs16_loss_log.csv",
        mode="a",
        header=not loss_file_exists,
        index=False,
        encoding="utf-8-sig"
    )

    # ===================== ✅ 保存 训练时长 CSV（新增） =====================
    time_df = pd.DataFrame(time_logs)
    time_file_exists = os.path.exists("./Results/epoch_time_lr001_bs16_log.csv")
    time_df.to_csv(
        "./Results/epoch_time_lr001_bs16_log.csv",
        mode="a",
        header=not time_file_exists,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"✅ {emb_type} 全部保存完成！")

    # 返回最后一轮结果（不破坏原代码）
    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1}


# ===================== 传统机器学习 =====================
def run_traditional(filename, emb_type, models, clf_type='svm'):
    X_train, X_test, y_train, y_test = process_all(filename, emb_type, models)
    if clf_type == 'svm':
        m = svm.SVC()
    elif clf_type == 'rf':
        m = RandomForestClassifier()
    elif clf_type == 'lr':
        m = LogisticRegression(max_iter=2000)
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    acc = 100*np.mean(p == y_test)
    pre = precision_score(y_test, p, 'macro', zero_division=0)
    rec = recall_score(y_test, p, 'macro', zero_division=0)
    f1 = f1_score(y_test, p, 'macro', zero_division=0)
    print(f"[{emb_type}+{clf_type}] Acc={acc:.2f} F1={f1:.4f}")
    return {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1}


# ===================== 主实验：全词嵌入对比 =====================
if __name__ == '__main__':
    setup_seed(SEED)

    # ========== 你的路径 ==========
    filename = r'.\COLDataset\Data_Comments.csv'
    w2v_path = r'.\model\Word2VecModel\new_word2vec.bigram-char'
    glove_path = r'.\model\GloVeModel\vectors.txt'
    ft_path = r'.\model\FastTextModel\wiki.zh.bin'

    # 加载所有模型
    models = {
        'word2vec': load_word2vec(w2v_path),
        'glove': load_glove(glove_path),
        'fasttext': load_fasttext(ft_path),
        'bert': load_bert(),
        'sbert': load_sbert(),
        'simcse': load_simcse(), # https://huggingface.co/cyclone/simcse-chinese-roberta-wwm-ext
        'llm': load_llm_embedding()  # 🔥 大模型嵌入
    }

    # 所有要对比的词嵌入
    EMB_TYPES = ['word2vec', 'glove', 'fasttext', 'bert', 'sbert', 'simcse', 'llm']
    results = {}

    print("\n========== 全系列词嵌入对比实验 ==========\n")
    for emb in EMB_TYPES:
        print(f"\n===== 当前：{emb} =====")
        res = run_nn(filename, emb, models, 'FCN')
        results[emb] = res

    # ===================== ✅ 最终保存：所有模型嵌入时间 =====================
    # emb_time_df = pd.DataFrame(embedding_time_list)
    # emb_time_df.to_csv("./Results/embedding_time_total.csv", index=False, encoding="utf-8-sig")
    # print("\n✅ 所有文本嵌入时间已保存至 embedding_time_total.csv")

    # 输出论文表格
    print("\n\n==================== 最终实验结果（论文直接复制）====================")
    print(f"{'词嵌入类型':<12} {'准确率(%)':<10} {'精确率':<10} {'召回率':<10} {'F1':<10}")
    print("-"*60)
    for k, v in results.items():
        print(f"{k:<12} {v['acc']:<10.2f} {v['pre']:<10.4f} {v['rec']:<10.4f} {v['f1']:<10.4f}")