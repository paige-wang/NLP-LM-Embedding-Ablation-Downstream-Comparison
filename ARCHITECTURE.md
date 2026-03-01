# ARCHITECTURE.md — Architecture Blueprint

Cross-references:
- Conventions & tech stack → [CLAUDE.md](CLAUDE.md)
- Experiment configs       → [EXPERIMENTS.md](EXPERIMENTS.md)
- Task backlog             → [STATE.md](STATE.md)

---

## 1. Directory Structure

See `CLAUDE.md § 3` for the canonical layout.

---

## 2. Data Pipeline

### 2.1 Classes

```
src/data/
├── dataset.py       ← LMDataset, DownstreamDataset
├── tokenizer.py     ← Tokenizer (wraps HuggingFace tokenizers)
└── vocab.py         ← Vocabulary (token ↔ index, special tokens)
```

**`Vocabulary`**
```python
class Vocabulary:
    def __init__(self, min_freq: int = 2, max_size: int | None = None): ...
    def build(self, texts: list[str]) -> None: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def __len__(self) -> int: ...
    # Persists to / loads from JSON
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "Vocabulary": ...
```

**`LMDataset`** (torch.utils.data.Dataset)
```python
class LMDataset(Dataset):
    def __init__(self, token_ids: list[int], seq_len: int): ...
    def __getitem__(self, idx) -> tuple[Tensor, Tensor]: ...  # (input, target) shifted by 1
```

### 2.2 Data Flow

```
Raw text files
     │  Tokenizer.tokenize()
     ▼
Token sequences
     │  Vocabulary.build() + encode()
     ▼
Integer ID sequences  ─── saved to data/processed/
     │  LMDataset(seq_len=N)
     ▼
(input_ids, target_ids) batches  ──► model training
```

---

## 3. Language Models (Part I)

### 3.1 Base Interface

```python
# src/models/base.py
class BaseLanguageModel(ABC):
    @abstractmethod
    def forward(self, input_ids: Tensor) -> Tensor: ...          # logits
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int) -> str: ...
    def perplexity(self, dataloader: DataLoader) -> float: ...   # shared impl
```

### 3.2 Model Implementations

| Class | File | Key params |
|-------|------|-----------|
| `NGramLanguageModel` | `src/models/ngram.py` | `n`, `smoothing` |
| `RNNLanguageModel` | `src/models/rnn.py` | `hidden_size`, `num_layers`, `embedding` |
| `LSTMLanguageModel` | `src/models/lstm.py` | `hidden_size`, `num_layers`, `dropout`, `embedding` |
| `TransformerLanguageModel` | `src/models/transformer.py` | `d_model`, `nhead`, `num_layers`, `dim_feedforward`, `embedding` |

All neural LMs accept an `embedding: BaseEmbedding` argument (see §4), enabling
plug-and-play ablation without changing model internals.

---

## 4. Embedding Layer Abstraction (Part II)

```python
# src/embeddings/base.py
class BaseEmbedding(nn.Module, ABC):
    embedding_dim: int
    @abstractmethod
    def forward(self, token_ids: Tensor) -> Tensor: ...
```

| Class | File | Description |
|-------|------|-------------|
| `TrainableEmbedding` | `src/embeddings/trainable.py` | `nn.Embedding`, trained end-to-end |
| `FixedSelfTrainedEmbedding` | `src/embeddings/fixed_self.py` | Word2Vec/GloVe trained on same corpus; weights frozen |
| `FixedPretrainedEmbedding` | `src/embeddings/fixed_pretrained.py` | Public pretrained vectors (e.g., GloVe-840B); weights frozen |

**Factory:**
```python
# src/embeddings/factory.py
def build_embedding(cfg: EmbeddingConfig, vocab: Vocabulary) -> BaseEmbedding: ...
```

---

## 5. Downstream Task (Part III)

```python
# src/downstream/classifier.py
class SentimentClassifier(nn.Module):
    def __init__(self, lm: BaseLanguageModel, num_classes: int, freeze_lm: bool): ...
    def forward(self, input_ids: Tensor) -> Tensor: ...   # class logits

# src/downstream/trainer.py
class DownstreamTrainer:
    def train(self, model, train_loader, val_loader, cfg) -> dict: ...
    def evaluate(self, model, loader) -> dict: ...        # accuracy, F1, etc.
```

---

## 6. Utility Modules

```
src/utils/
├── config.py    ← dataclass configs for each model/experiment
├── metrics.py   ← perplexity(), accuracy(), f1()
├── seed.py      ← set_seed(seed: int)
└── logging.py   ← get_logger(name: str) → Logger
```

---

## 7. Training Scripts

| Script | Part | Responsibilities |
|--------|------|-----------------|
| `scripts/train_lm.py` | I | Instantiate model + embedding; run train loop; log metrics; save checkpoint |
| `scripts/ablation.py` | II | Grid over 3 embedding configs × 2 neural LMs; record results to EXPERIMENTS.md |
| `scripts/downstream.py` | III | Load LM checkpoint; attach classifier head; run downstream training |
