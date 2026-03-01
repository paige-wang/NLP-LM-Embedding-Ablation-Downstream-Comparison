# PROJECT_BLUEPRINT.md
# ST5230 Assignment 1 — LM Embedding Ablation & Downstream Comparison
# 完整架构蓝图 · 工程约束 · 验收标准

**课程：** ST5230 Applied Natural Language Processing
**目标：** High Credit（A 档）
**产出：** PDF 报告（≤6 页）+ 代码（GitHub 链接或 zip）
**最后更新：** 2026-03-01

---

## 目录

1. [项目叙事定位](#1-项目叙事定位)
2. [技术栈与环境](#2-技术栈与环境)
3. [仓库目录结构](#3-仓库目录结构)
4. [数据集决策](#4-数据集决策)
5. [所有已确认的实验决策](#5-所有已确认的实验决策)
6. [超参数汇总表](#6-超参数汇总表)
7. [阶段与文件清单](#7-阶段与文件清单)
8. [模块架构详解](#8-模块架构详解)
9. [工程约束与验收标准](#9-工程约束与验收标准)
10. [验收清单与收敛基线](#10-验收清单与收敛基线)
11. [报告结构规划](#11-报告结构规划)
12. [Intent → Agent 映射表](#12-intent--agent-映射表)

---

## 1. 项目叙事定位

本作业不只是"跑通四个模型"，而是构建一个**从语言建模到 NLU 探针再到路由网关**的完整叙事链：

```
WikiText-2 语料
    ↓ 训练
四种 LM（NGram / RNN / LSTM / Transformer）
    ↓ 消融实验（3 种 Embedding）
最优 Transformer LM（表征提取器）
    ↓ Mean Pooling → Linear(300→7)
SNIPS 意图分类（7 类）
    ↓ 对应
7 个专用 Agent（Traffic Routing Gateway）
```

报告结论落点：Transformer 表征最可分 → 可扩展为 Multi-Agent 调度 → LLM4Rec 铺垫。

---

## 2. 技术栈与环境

| 包 | 最低版本 | 用途 |
|----|---------|------|
| `torch` | 2.2.0 | 训练框架 |
| `transformers` | 4.40.0 | 预训练模型/分词器 |
| `datasets` | 2.19.0 | 数据集加载（WikiText-2 / SNIPS） |
| `tokenizers` | 0.19.0 | 快速分词器 |
| `gensim` | 4.3.0 | Word2Vec 训练（Embedding-2） |
| `numpy` | 1.26.0 | 数值计算 |
| `scikit-learn` | 1.4.0 | 分类指标 / 混淆矩阵 |
| `matplotlib` | 3.8.0 | 绘图 |
| `tqdm` | 4.66.0 | 进度条 |

**运行环境：** CUDA GPU 优先；CPU fallback 必须可运行。
**包管理：** pip + `requirements.txt`，版本锁定。
**平台注意：** Windows 下 DataLoader `num_workers` 须置 0 或加 `if __name__ == '__main__'` 守卫。

---

## 3. 仓库目录结构

```
.
├── PROJECT_BLUEPRINT.md   ← 本文件（架构权威参考）
├── CLAUDE.md              ← 工作约定（代码风格 / 流程）
├── ARCHITECTURE.md        ← 类设计与数据流
├── STATE.md               ← 任务看板（每个里程碑后更新）
├── EXPERIMENTS.md         ← 结果表 + 超参注册（每次 run 后更新）
├── requirements.md        ← 作业规范（只读）
├── requirements.txt       ← pip 依赖（版本锁定）
│
├── data/
│   ├── README.md          ← 数据下载说明
│   ├── raw/               ← GloVe.6B.300d.txt（不入 git）
│   └── processed/         ← vocab.json / *_ids.npy / word2vec.bin
│
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── config.py      ← LMConfig / EmbeddingConfig / DownstreamConfig
│   │   ├── seed.py        ← set_seed(42)
│   │   ├── logging_utils.py ← get_logger(run_name)
│   │   └── metrics.py     ← perplexity / accuracy / f1 / confusion_matrix
│   ├── data/
│   │   ├── vocabulary.py  ← Vocabulary（build / encode / decode / save / load）
│   │   ├── tokenizer.py   ← SimpleTokenizer（空格 + 小写化）
│   │   ├── lm_dataset.py  ← LMDataset（滑动窗口，seq_len=64）
│   │   └── downstream_dataset.py ← SNIPSDataset（7 类意图）
│   ├── models/
│   │   ├── base_lm.py     ← BaseLanguageModel（ABC）
│   │   ├── ngram_lm.py    ← NGramLanguageModel（trigram + Laplace）
│   │   ├── rnn_lm.py      ← RNNLanguageModel（2层，hidden=512）
│   │   ├── lstm_lm.py     ← LSTMLanguageModel（2层，hidden=512）
│   │   └── transformer_lm.py ← TransformerLanguageModel（自定义，4层）
│   ├── embeddings/
│   │   ├── base.py        ← BaseEmbedding（ABC）
│   │   ├── trainable.py   ← TrainableEmbedding
│   │   ├── fixed_self.py  ← FixedSelfTrainedEmbedding（Word2Vec 冻结）
│   │   ├── fixed_pretrained.py ← FixedPretrainedEmbedding（GloVe 冻结）
│   │   └── factory.py     ← build_embedding(cfg, vocab)
│   └── downstream/
│       ├── classifier.py  ← IntentClassifier（MeanPool → Linear(300→7)）
│       └── trainer.py     ← DownstreamTrainer（frozen / finetune 两种模式）
│
├── scripts/
│   ├── train_lm.py        ← Part I 入口（--model {ngram,rnn,lstm,transformer}）
│   ├── ablation.py        ← Part II 入口（--model × --embed 网格）
│   └── downstream.py      ← Part III 入口（--mode {frozen,finetune}）
│
├── notebooks/             ← EDA 与结果可视化
└── outputs/
    ├── checkpoints/       ← <model>_<run_id>/
    ├── logs/              ← <run_name>.log
    └── figures/           ← *.png
```

---

## 4. 数据集决策

| 用途 | 数据集 | 加载方式 | 备注 |
|------|--------|---------|------|
| LM 训练 | WikiText-2 (`wikitext/wikitext-2-raw-v1`) | `datasets.load_dataset` | 自动缓存；空行需过滤 |
| 下游分类 | SNIPS (`DeepPavlov/snips`) | `datasets.load_dataset` | 7 类意图；无需手动下载 |
| Embedding-3 | GloVe.6B.300d | 手动下载至 `data/raw/` | Stanford NLP 官网；300d 对齐 |

**WikiText-2 统计（预估）：**

| 分割 | 约 Token 数 |
|------|------------|
| train | ~2.1M |
| validation | ~217k |
| test | ~245k |

---

## 5. 所有已确认的实验决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| LM 训练数据 | WikiText-2 | 轻量（~2M tokens），适合有限算力 |
| 下游任务数据 | SNIPS 7类意图 | 多分类更能体现表征可分性；对应 Agent 调度叙事 |
| 词表大小 | 20,000 | 覆盖率与内存的平衡点 |
| Embedding 维度 | 300 | 与 GloVe.6B.300d 对齐，无需投影层 |
| NGram | trigram，Laplace 平滑 | 作业要求；k=1 |
| RNN | 2层，hidden=512，dropout=0.5 | 参数量适中 |
| LSTM | 2层，hidden=512，dropout=0.5 | 与 RNN 对比配置对齐 |
| Transformer | 4层，d_model=300，heads=6，FFN=1200 | 自定义实现；heads=6 整除 300 |
| Embedding-1 | Trainable（随机初始化，联合训练） | 基线 |
| Embedding-2 | Fixed-self（gensim Word2Vec，冻结） | 同语料，排除迁移优势 |
| Embedding-3 | Fixed-pretrained（GloVe.6B.300d，冻结） | 语义先验最强 |
| 下游基础 LM | Transformer（最优模型） | 对比价值最高 |
| 表征提取 | Mean Pooling（最后一层所有 token 平均） | 简单、可解释 |
| 分类器 | Linear(300 → 7) | 满足"simple downstream model"要求 |
| 对比条件 | Frozen（只训 Linear）vs Fine-tuned（全参数） | 直接体现 LM 表征质量 |

---

## 6. 超参数汇总表

```python
@dataclass
class LMConfig:
    # 词表 / 分词
    vocab_size: int = 20000
    embed_dim: int = 300
    seq_len: int = 64

    # 训练
    batch_size: int = 64
    lr: float = 3e-4        # Adam
    epochs: int = 30
    dropout: float = 0.5
    grad_clip: float = 1.0
    warmup_steps: int = 1000

    # RNN / LSTM
    hidden_size: int = 512
    num_layers: int = 2

    # Transformer（自定义）
    d_model: int = 300
    nhead: int = 6
    num_transformer_layers: int = 4
    dim_feedforward: int = 1200

    # NGram
    ngram_n: int = 3
    ngram_smoothing: float = 1.0   # Laplace add-1

    seed: int = 42
    device: str = "auto"           # CUDA if available else CPU


@dataclass
class DownstreamConfig:
    num_classes: int = 7
    embed_dim: int = 300

    frozen_lr: float = 1e-3        # 只训练 linear head
    finetune_lr: float = 5e-5      # 全参数微调

    batch_size: int = 32
    epochs: int = 10
    grad_clip: float = 1.0
    dropout: float = 0.1

    seed: int = 42
    device: str = "auto"
```

---

## 7. 阶段与文件清单

### Phase 1 — 框架搭建

| 文件 | 状态 | 关键职责 |
|------|------|---------|
| `requirements.txt` | ✅ 已创建 | 锁定依赖版本 |
| `data/README.md` | ✅ 已创建 | 数据下载说明 |
| `src/utils/config.py` | ✅ 已创建 | LMConfig / EmbeddingConfig / DownstreamConfig |
| `src/utils/seed.py` | ✅ 已创建 | `set_seed(42)` |
| `src/utils/logging_utils.py` | ✅ 已创建 | `get_logger(run_name)` |
| `src/utils/metrics.py` | ✅ 已创建 | PPL / accuracy / F1 / confusion matrix |
| `src/data/vocabulary.py` | 🔲 待实现 | Vocabulary（20k + 4 特殊 token） |
| `src/data/tokenizer.py` | 🔲 待实现 | SimpleTokenizer（空格 + 小写化） |
| `src/data/lm_dataset.py` | 🔲 待实现 | LMDataset（滑动窗口） |
| `src/data/downstream_dataset.py` | 🔲 待实现 | SNIPSDataset（7 类意图） |

### Phase 2 — Part I：四个 LM

| 文件 | 状态 | 关键职责 |
|------|------|---------|
| `src/models/base_lm.py` | 🔲 待实现 | BaseLanguageModel（ABC），`perplexity()` 共享实现 |
| `src/models/ngram_lm.py` | 🔲 待实现 | trigram count-based，无 `nn.Module` |
| `src/models/rnn_lm.py` | 🔲 待实现 | 2层 RNN，接受 `embedding` 参数 |
| `src/models/lstm_lm.py` | 🔲 待实现 | 2层 LSTM，接受 `embedding` 参数 |
| `src/models/transformer_lm.py` | 🔲 待实现 | 自定义：MHSA + PE + 4层 Block |
| `scripts/train_lm.py` | 🔲 待实现 | 统一训练入口，含梯度累加 + OOM 容错 |

### Phase 3 — Part II：Embedding 消融

| 文件 | 状态 | 关键职责 |
|------|------|---------|
| `src/embeddings/base.py` | 🔲 待实现 | BaseEmbedding（ABC） |
| `src/embeddings/trainable.py` | 🔲 待实现 | `nn.Embedding`，联合训练 |
| `src/embeddings/fixed_self.py` | 🔲 待实现 | gensim Word2Vec，`requires_grad=False` |
| `src/embeddings/fixed_pretrained.py` | 🔲 待实现 | GloVe.6B.300d 加载，OOV 随机初始化 |
| `src/embeddings/factory.py` | 🔲 待实现 | `build_embedding(cfg, vocab)` 工厂函数 |
| `scripts/ablation.py` | 🔲 待实现 | 3 embedding × 3 model 网格运行 |

### Phase 4 — Part III：下游任务

| 文件 | 状态 | 关键职责 |
|------|------|---------|
| `src/downstream/classifier.py` | 🔲 待实现 | IntentClassifier（MeanPool + Linear(300→7)） |
| `src/downstream/trainer.py` | 🔲 待实现 | DownstreamTrainer（frozen / finetune 模式） |
| `scripts/downstream.py` | 🔲 待实现 | 训练入口，--mode {frozen,finetune} |

---

## 8. 模块架构详解

### 8.1 数据流

```
HuggingFace datasets (WikiText-2)
    │  SimpleTokenizer.tokenize()   [空格分词 + 小写化]
    ▼
List[List[str]]                     [逐句 token 列表]
    │  Vocabulary.build() + encode()
    ▼
List[int]                           [拼接后的全局 token ID 序列]
    │  numpy save → data/processed/*_ids.npy
    │  LMDataset(seq_len=64, stride=1)
    ▼
(input_ids [B,64], target_ids [B,64])  [target = input 左移 1 位]
    │  DataLoader(batch_size=64)
    ▼
模型训练
```

### 8.2 特殊 Token 约定

| Token | ID | 用途 |
|-------|----|------|
| `<pad>` | 0 | 批次填充（LMDataset 若需要） |
| `<unk>` | 1 | 词表外词 |
| `<bos>` | 2 | 句子开始 |
| `<eos>` | 3 | 句子结束 |
| 实际词汇 | 4 ~ 20003 | 高频 top-20000 |

### 8.3 神经 LM 通用接口

```python
class BaseLanguageModel(ABC):
    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids:  [batch, seq_len]
        # return:     [batch, seq_len, vocab_size]  ← 必须显式注释此形状
        ...

    def encode(self, input_ids: Tensor) -> Tensor:
        # 提取用于下游任务的表征
        # return:     [batch, seq_len, d_model]
        ...

    def perplexity(self, dataloader: DataLoader) -> float:
        # 所有神经 LM 的共享实现，mask <pad> token
        ...

    def generate(self, prompt: str, max_new_tokens: int, vocab: Vocabulary) -> str:
        ...
```

### 8.4 Transformer 自定义实现要点

```
TransformerLanguageModel
├── embedding: BaseEmbedding        → [B, T, 300]
├── positional_encoding: PE         → sinusoidal，[1, T, 300]
├── blocks: nn.ModuleList
│   └── TransformerBlock × 4
│       ├── pre-norm: LayerNorm(300)
│       ├── MultiHeadSelfAttention  (d_model=300, heads=6, head_dim=50)
│       │   └── causal mask: upper-triangular bool mask，防止看未来 token
│       ├── pre-norm: LayerNorm(300)
│       └── FFN: Linear(300→1200) → GELU → Linear(1200→300)
├── layer_norm: LayerNorm(300)      → 最终归一化
└── lm_head: Linear(300→20000, bias=False)  → [B, T, vocab_size]
```

**权重共享（可选）：** `lm_head.weight = embedding.weight`（tied embeddings），可减少参数量 ~6M。

### 8.5 Embedding 消融矩阵

| | Trainable | Fixed-Self (Word2Vec) | Fixed-Pretrained (GloVe) |
|-|-----------|----------------------|--------------------------|
| RNN | ✓ | ✓ | ✓ |
| LSTM | ✓ | ✓ | ✓ |
| Transformer | ✓ | ✓ | ✓ |
| NGram | ✗（无 embedding 层） | — | — |

共 **9 个实验**（不含 NGram）。

### 8.6 下游任务架构

```python
class IntentClassifier(nn.Module):
    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids:  [batch, seq_len]
        hidden = self.lm.encode(input_ids)   # [batch, seq_len, 300]
        pooled = hidden.mean(dim=1)          # [batch, 300]  ← 在 seq 维度 mean pool
        logits = self.classifier(pooled)     # [batch, 7]
        return logits
```

---

## 9. 工程约束与验收标准

> 本章节集合了**用户指定的 4 条强制要求**与**深度工程 pitfall 分析**，
> 所有实现必须满足这些约束，否则不允许标记为完成。

---

### 9.1 【强制】维度严格断言

**要求：** 所有 `forward` 方法必须在关键操作前后添加显式张量形状注释；下游
Mean Pooling 必须在 `seq` 维度操作，送入线性层前严格保证为 `[batch, 300]`，
严防广播错误导致的静默错误。

**实施细节：**

```python
# transformer_lm.py forward 示例
def forward(self, input_ids: Tensor) -> Tensor:
    # input_ids:  [B, T]
    x = self.embedding(input_ids)         # [B, T, 300]
    x = x + self.positional_encoding(x)  # [B, T, 300]
    for block in self.blocks:
        x = block(x, causal_mask)         # [B, T, 300]
    x = self.layer_norm(x)               # [B, T, 300]
    logits = self.lm_head(x)             # [B, T, 20000]
    return logits

# downstream/classifier.py 中的断言
def forward(self, input_ids: Tensor) -> Tensor:
    hidden = self.lm.encode(input_ids)   # [B, T, 300]
    pooled = hidden.mean(dim=1)          # [B, 300]
    assert pooled.shape[-1] == 300, f"Expected 300, got {pooled.shape}"
    assert pooled.ndim == 2, f"Expected 2D tensor before linear, got {pooled.ndim}D"
    return self.classifier(pooled)       # [B, 7]
```

**额外检查点：**
- `TransformerBlock` 中的 `causal_mask` 必须是 `[T, T]` 上三角 bool 矩阵，非 `[B, T, T]`
- `positional_encoding` 输出必须可广播到 `[B, T, 300]`（即形状为 `[1, T, 300]`）
- `MultiHeadSelfAttention` 中 `head_dim = d_model // nhead = 50`，确保 `300 % 6 == 0`

---

### 9.2 【强制】显存 OOM 容错

**要求：** 训练脚本必须内置梯度累加；默认 `batch=64`，若捕获 `CUDA OOM`，
自动 fallback 至 `batch=16` + 累加 4 步（等效 batch=64），训练继续而非崩溃。

**实施模板（`scripts/train_lm.py`）：**

```python
GRAD_ACCUM_STEPS = 1
EFFECTIVE_BATCH = cfg.batch_size  # 初始 64

for epoch in range(cfg.epochs):
    optimizer.zero_grad()
    for step, (x, y) in enumerate(train_loader):
        try:
            loss = model(x, y) / GRAD_ACCUM_STEPS
            loss.backward()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.warning("CUDA OOM detected — switching to batch=16, accum=4")
                # 重建 DataLoader，batch=16，GRAD_ACCUM_STEPS=4
                train_loader = rebuild_loader(batch_size=16)
                GRAD_ACCUM_STEPS = 4
                EFFECTIVE_BATCH = 16 * 4  # = 64，等效不变
                optimizer.zero_grad()
                continue
            raise
```

**额外注意：**
- `torch.cuda.empty_cache()` 只释放缓存，无法解决模型本身过大的问题；
  若 fallback 后仍 OOM，应再次 fallback 至 `batch=8, accum=8`，记录日志并继续
- 梯度累加时 `loss` 必须除以 `GRAD_ACCUM_STEPS`，否则等效学习率翻倍

---

### 9.3 【强制】自动化日志落盘

**要求：** 严禁仅在控制台打印结果。PPL、准确率、耗时等核心指标，
每 epoch 后必须自动追加写入 `outputs/EXPERIMENTS.md`。

**实施规范：**

```python
# src/utils/logging_utils.py 中新增
def log_experiment_result(
    run_name: str,
    epoch: int,
    metrics: dict,
    filepath: str = "outputs/EXPERIMENTS.md"
) -> None:
    """Append one epoch result row to outputs/EXPERIMENTS.md."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    # 表格行格式：| run_name | epoch | ppl | acc | time |
    row = (
        f"| {run_name} | {epoch} "
        + " | ".join(str(round(v, 4)) for v in metrics.values())
        + f" | {timestamp} |\n"
    )
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(row)
```

**每次 epoch 后必须落盘的指标：**

| 阶段 | 必须记录 |
|------|---------|
| Part I（LM 训练） | train_loss, val_ppl, epoch_time_sec, total_params |
| Part II（消融） | embed_type, model, val_ppl, convergence_epoch |
| Part III（下游） | mode（frozen/finetune）, val_acc, val_macro_f1, epoch |

**混淆矩阵必须用 `plt.savefig()`，严禁 `plt.show()`。**

---

### 9.4 【强制】明确的验收清单与收敛基线

见第 10 章。

---

### 9.5 【工程 Pitfall】NGram 内存炸裂

**问题：** Trigram 若用稠密矩阵存储，20k³ ≈ 8 万亿 entries，直接 OOM。
**解决：** 必须用稀疏字典 `dict[tuple[int,int], Counter[int]]` 存储，
按需查询，不预分配全表。词表修剪至 10k-20k 可控。

---

### 9.6 【工程 Pitfall】WikiText-2 空行与 Section Header

**问题：** WikiText-2 raw 格式包含大量空行（`""`）和 `= Section Title =` 行，
直接加入语料会污染 n-gram 统计和 LM 困惑度。
**解决：** 预处理时过滤 `len(line.strip()) < 10` 的行；
过滤以 `=` 开头和结尾的行（section headers）。

---

### 9.7 【工程 Pitfall】Causal Mask 缺失

**问题：** Transformer LM 若未加上三角因果 mask，训练时可以看到未来 token，
loss 会极低但 PPL 虚假，生成时失效（因为推理无法看未来）。
**解决：** 在每个 `TransformerBlock` 的 attention 中强制传入：
```python
causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
# attn_weights.masked_fill_(causal_mask, float('-inf'))
```

---

### 9.8 【工程 Pitfall】Frozen Embedding 被意外解冻

**问题：** 对整个模型调用 `model.train()` 后，所有参数的 `requires_grad`
不会被重置，但 Dropout/BN 的行为会切换。然而如果 optimizer 在创建时包含了
`embedding.parameters()`（而非仅 non-frozen 参数），反向传播仍会计算梯度。
**解决：**
```python
# 在构建 optimizer 时显式排除冻结参数
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=cfg.lr)
# 并在 forward 后断言
assert not embedding.weight.requires_grad, "Embedding should be frozen!"
```

---

### 9.9 【工程 Pitfall】GloVe OOV 词随机初始化

**问题：** GloVe 词表与 WikiText-2 top-20k 不完全重合。若 OOV 词赋 0 向量，
模型无法区分它们；若赋 `<unk>` 向量，所有 OOV 语义相同。
**解决：** OOV 词用 `torch.randn(300) * 0.01` 随机初始化（小标准差），
并记录覆盖率（命中词数 / 20000）写入日志，预期 WikiText-2 覆盖率 > 85%。

---

### 9.10 【工程 Pitfall】PPL 计算包含 Padding Token

**问题：** 若 `<pad>` token 的 NLL 被计入平均，会低估真实 PPL。
**解决：**
```python
# CrossEntropyLoss 中设置 ignore_index
criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad_id=0
```

---

### 9.11 【工程 Pitfall】Windows DataLoader num_workers

**问题：** Windows 下 PyTorch DataLoader 使用 `spawn` 多进程，
若在 `if __name__ == '__main__':` 外调用 `num_workers > 0`，会产生
无限进程递归（进程风暴）。
**解决：** 所有训练脚本强制以 `if __name__ == '__main__':` 包裹主逻辑，
或在配置中默认 `num_workers=0`，并在注释中说明原因。

---

### 9.12 【工程 Pitfall】Warmup Scheduler Step 位置

**问题：** LR scheduler 若每 epoch 而非每 step 调用 `step()`，warmup 期
学习率按 epoch 而非 step 线性增长，warmup_steps=1000 实际变成 1000 epochs。
**解决：** 在每个 batch 的 `optimizer.step()` 之后立即调用 `scheduler.step()`，
并在日志中打印当前 `lr` 以便验证。

---

### 9.13 【工程 Pitfall】Frozen 模式下 LM 的 Dropout

**问题：** 下游任务 frozen 模式下，若 LM 仍在 `train()` 模式，
Dropout 随机 mask 会导致 mean pooling 结果不确定，分类器训练不稳定。
**解决：** frozen 模式下，LM 的 eval/train 状态：
```python
if mode == "frozen":
    lm.eval()  # 关闭 Dropout + BN 的 train 行为
    for p in lm.parameters():
        p.requires_grad = False
```

---

### 9.14 【工程 Pitfall】Checkpoint 命名冲突

**问题：** 若两次实验使用相同 `run_id`，新 checkpoint 会覆盖旧的，
导致 ablation 结果无法追溯。
**解决：** `run_id` 格式为 `{model}_{embed}_{timestamp}`，例如
`transformer_trainable_20260301_143022`，由训练脚本自动生成。

---

### 9.15 【工程 Pitfall】Word2Vec 词表与 LM 词表对齐

**问题：** gensim Word2Vec 有自己的词表（基于训练语料频率），
与 `Vocabulary` 的 top-20k 词表不完全一致。对齐方式错误会导致
embedding 矩阵行与 token ID 错位，静默传入错误语义。
**解决：**
```python
# 构建 fixed_self embedding 矩阵时，以 LM Vocabulary 为准
weight = torch.zeros(vocab_size, embed_dim)
for token, idx in vocab.token2id.items():
    if token in w2v_model.wv:
        weight[idx] = torch.tensor(w2v_model.wv[token])
    else:
        weight[idx] = torch.randn(embed_dim) * 0.01
```

---

## 10. 验收清单与收敛基线

### 10.1 功能验收（必须全部通过）

- [ ] `python scripts/train_lm.py --model ngram` 无报错完成，PPL 写入日志
- [ ] `python scripts/train_lm.py --model rnn` 无报错完成，`outputs/checkpoints/` 有存档
- [ ] `python scripts/train_lm.py --model lstm` 无报错完成
- [ ] `python scripts/train_lm.py --model transformer` 无报错完成
- [ ] `python scripts/ablation.py --model rnn --embed trainable` 无报错
- [ ] `python scripts/ablation.py --model transformer --embed fixed_pretrained` 无报错
- [ ] `python scripts/downstream.py --mode frozen` 无报错完成
- [ ] `python scripts/downstream.py --mode finetune` 无报错完成
- [ ] `outputs/logs/` 下存在对应 `.log` 文件
- [ ] `outputs/figures/` 下存在混淆矩阵图
- [ ] `outputs/EXPERIMENTS.md` 有每次 epoch 的自动追加记录
- [ ] 所有模型在 CPU 上可运行（GPU 非必须）

### 10.2 收敛基线（高信心预估）

> 注：WikiText-2 规模较小，以下基线为合理训练配置下的预期区间。
> 若结果明显偏离，优先检查 causal mask、PPL 计算（是否包含 pad）、
> 以及学习率 scheduler 位置。

#### Part I — LM Perplexity（WikiText-2 test set）

| 模型 | 预期 Test PPL | 说明 |
|------|-------------|------|
| NGram (trigram, Laplace) | 300 ~ 800 | 受 Laplace 平滑影响较大；基线 |
| RNN (2层, hidden=512) | 120 ~ 180 | 收敛较慢，梯度消失风险 |
| LSTM (2层, hidden=512) | 90 ~ 130 | 比 RNN 更稳定 |
| Transformer (4层, custom) | 60 ~ 110 | 期望最优；依赖 causal mask 正确 |

**最低通过基线：** Transformer PPL < 150（若 > 150 说明有实现错误）

#### Part II — Embedding 消融（相对 Trainable 的 PPL 变化）

| Embedding 类型 | 预期效果 | 分析 |
|----------------|---------|------|
| Trainable（基线） | PPL baseline | 联合训练，语料适配最好 |
| Fixed-self (Word2Vec) | PPL +5% ~ +20% | 同语料但未联合优化 |
| Fixed-pretrained (GloVe) | PPL +0% ~ +15% | 可能优于或接近 Word2Vec |

**Transformer + Trainable 应为 9 个格子中 PPL 最低。**

#### Part III — 下游意图分类（SNIPS test set）

| 模式 | 预期 Accuracy | 预期 Macro F1 |
|------|-------------|--------------|
| Frozen LM + Linear | ≥ 70% | ≥ 0.68 |
| Fine-tuned LM + Linear | ≥ 85% | ≥ 0.84 |

**Fine-tuned 必须显著优于 Frozen，否则报告讨论其原因（如过拟合、LR 过大）。**

**最低通过基线：**
- Frozen accuracy ≥ 65%（若更低说明 mean pooling 或 encoder 实现有误）
- Finetune accuracy ≥ 80%（若更低说明 LR 过大或 frozen 模式 eval 未正确设置）

### 10.3 代码质量检查

- [ ] 所有公开函数有 Google-style docstring
- [ ] 所有公开函数有 type hints
- [ ] 无 magic number（超参均在 `config.py`）
- [ ] `set_seed(42)` 在每个训练脚本的第一行之后立即调用
- [ ] 无 `plt.show()`（只有 `plt.savefig()`）
- [ ] 所有文件操作显式指定 `encoding="utf-8"`

---

## 11. 报告结构规划

```
第 0.5 页  Setup
           ├── 数据集：WikiText-2（LM）+ SNIPS（下游）
           ├── 超参表（vocab=20k, dim=300, seq=64, ...）
           └── 硬件：GPU 型号 / CPU fallback

第 1.5 页  Part I — 四模型对比
           ├── Table 1：train_time | inference_time | val_ppl | test_ppl
           ├── Figure 1：4 条 val loss 曲线（同一图）
           └── 文本生成样例（每模型 1-2 句）

第 1.5 页  Part II — Embedding 消融
           ├── Table 2：3×3 PPL 矩阵（模型 × embedding）
           ├── Figure 2：收敛曲线（RNN/LSTM/Transformer × 3 embedding）
           └── 分析：语义先验 vs. 语料适配的权衡

第 1.5 页  Part III — NLU 探针 + 路由网关
           ├── Table 3：Frozen vs Fine-tuned（Acc / F1 / Precision / Recall）
           ├── Figure 3：7×7 混淆矩阵（Frozen + Fine-tuned 各一张）
           └── Agent 映射表（7 intent → 7 agent）

第 0.5 页  Discussion & Conclusion
           ├── 为什么 Transformer 表征最可分
           ├── Frozen 与 Fine-tuned 差距的来源
           └── 延伸：LM 作为 NLU 探针 → Multi-Agent 调度 → LLM4Rec
```

---

## 12. Intent → Agent 映射表

| SNIPS Intent | Agent | 描述 |
|-------------|-------|------|
| GetWeather | WeatherAgent | 查询天气预报 |
| BookRestaurant | RestaurantAgent | 餐厅预订 |
| PlayMusic | MusicAgent | 音乐播放控制 |
| AddToPlaylist | PlaylistAgent | 播放列表管理 |
| RateBook | ReviewAgent | 图书评分与评论 |
| SearchCreativeWork | ContentAgent | 创意内容检索 |
| SearchScreeningEvent | TicketAgent | 电影场次与票务查询 |

**核心叙事：** IntentClassifier 的输出可直接作为路由决策，
将用户 utterance 分发到对应的专用 Agent——这是从"语言建模"到
"智能系统"的自然过渡，也是 LM 表征能力在工业场景中的直接价值体现。

---

*文件生成时间：2026-03-01 | 版本：1.0*
*与 CLAUDE.md / ARCHITECTURE.md / STATE.md / EXPERIMENTS.md 共同构成项目文档体系。*
