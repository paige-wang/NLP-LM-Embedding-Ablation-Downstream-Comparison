"""Dataclass configs for language modeling, embeddings, and downstream runs."""

from dataclasses import dataclass


@dataclass
class LMConfig:
    """Hyperparameters shared by all language model experiments."""

    vocab_size: int = 20000
    embed_dim: int = 300
    seq_len: int = 64

    batch_size: int = 64
    lr: float = 3e-4
    epochs: int = 30
    dropout: float = 0.5
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    num_workers: int = 0  # Windows-safe default
    max_steps_per_epoch: int = 0  # 0 means full epoch

    hidden_size: int = 512
    num_layers: int = 2

    d_model: int = 300
    nhead: int = 6
    num_transformer_layers: int = 4
    dim_feedforward: int = 1200

    ngram_n: int = 3
    ngram_smoothing: float = 1.0

    data_dir: str = "data/processed"
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    figures_dir: str = "outputs/figures"

    seed: int = 42
    device: str = "auto"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding ablation variants."""

    embed_type: str = "trainable"  # trainable | fixed_self | fixed_pretrained
    embed_dim: int = 300
    vocab_size: int = 20000
    freeze: bool = False

    w2v_window: int = 5
    w2v_min_count: int = 2
    w2v_epochs: int = 10
    w2v_workers: int = 1
    w2v_save_path: str = "data/processed/word2vec.bin"

    glove_path: str = "data/raw/glove.6B.300d.txt"


@dataclass
class DownstreamConfig:
    """Hyperparameters for SNIPS downstream intent classification."""

    num_classes: int = 7
    embed_dim: int = 300
    seq_len: int = 32

    frozen_lr: float = 1e-3
    finetune_lr: float = 5e-5
    head_lr_multiplier: float = 10.0   # classifier head LR = finetune_lr * multiplier
    batch_size: int = 32
    epochs: int = 10
    grad_clip: float = 1.0
    dropout: float = 0.1
    num_workers: int = 0
    max_steps_per_epoch: int = 0
    warmup_steps: int = 200
    early_stopping_patience: int = 5

    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    figures_dir: str = "outputs/figures"

    seed: int = 42
    device: str = "auto"
