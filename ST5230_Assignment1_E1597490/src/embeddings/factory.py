"""Factory for embedding ablation variants."""

from collections.abc import Iterable

from src.data.vocabulary import Vocabulary
from src.embeddings.base import BaseEmbedding
from src.embeddings.fixed_pretrained import FixedPretrainedEmbedding
from src.embeddings.fixed_self import FixedSelfTrainedEmbedding
from src.embeddings.trainable import TrainableEmbedding
from src.utils.config import EmbeddingConfig


def build_embedding(
    cfg: EmbeddingConfig,
    vocab: Vocabulary,
    token_sequences: Iterable[list[str]] | None = None,
) -> tuple[BaseEmbedding, dict[str, float]]:
    """Build embedding module and return metadata."""
    meta: dict[str, float] = {}
    if cfg.embed_type == "trainable":
        return TrainableEmbedding(len(vocab), cfg.embed_dim, pad_id=vocab.pad_id), meta

    if cfg.embed_type == "fixed_self":
        embedding = FixedSelfTrainedEmbedding.from_corpus(
            vocab=vocab,
            token_sequences=token_sequences or [],
            embed_dim=cfg.embed_dim,
            save_path=cfg.w2v_save_path,
            window=cfg.w2v_window,
            min_count=cfg.w2v_min_count,
            epochs=cfg.w2v_epochs,
            workers=cfg.w2v_workers,
            pad_id=vocab.pad_id,
        )
        return embedding, meta

    if cfg.embed_type == "fixed_pretrained":
        embedding, coverage = FixedPretrainedEmbedding.from_glove(
            vocab=vocab,
            glove_path=cfg.glove_path,
            embed_dim=cfg.embed_dim,
            pad_id=vocab.pad_id,
        )
        meta["glove_coverage"] = coverage
        return embedding, meta

    raise ValueError(f"Unsupported embed_type: {cfg.embed_type}")
