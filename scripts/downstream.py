"""Part III entry: downstream SNIPS intent classification."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.downstream_dataset import SNIPSDataset
from src.data.pipeline import build_or_load_vocab_and_ids, load_snips_splits
from src.data.tokenizer import SimpleTokenizer
from src.embeddings.trainable import TrainableEmbedding
from src.downstream.classifier import IntentClassifier
from src.downstream.trainer import DownstreamTrainer
from src.models.transformer_lm import TransformerLanguageModel
from src.utils.config import DownstreamConfig, LMConfig
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["frozen", "finetune"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=0)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_backbone(vocab_size: int, pad_id: int, lm_cfg: LMConfig) -> TransformerLanguageModel:
    embedding = TrainableEmbedding(vocab_size=vocab_size, embedding_dim=lm_cfg.embed_dim, pad_id=pad_id)
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        embedding=embedding,
        d_model=lm_cfg.d_model,
        nhead=lm_cfg.nhead,
        num_layers=lm_cfg.num_transformer_layers,
        dim_feedforward=lm_cfg.dim_feedforward,
        dropout=lm_cfg.dropout,
        pad_id=pad_id,
    )
    return model


def main() -> None:
    args = parse_args()
    cfg = DownstreamConfig()
    lm_cfg = LMConfig()
    set_seed(cfg.seed)
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.max_steps_per_epoch = args.max_steps

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"downstream_{args.mode}_{timestamp}"
    logger = get_logger(run_name, cfg.log_dir)

    tokenizer = SimpleTokenizer()
    vocab, _, _ = build_or_load_vocab_and_ids(lm_cfg.data_dir, lm_cfg.vocab_size, tokenizer)
    snips = load_snips_splits()
    class_names = snips["labels"]  # type: ignore[assignment]
    train_ds = SNIPSDataset(snips["train"], vocab=vocab, seq_len=cfg.seq_len)  # type: ignore[arg-type]
    val_ds = SNIPSDataset(snips["validation"], vocab=vocab, seq_len=cfg.seq_len)  # type: ignore[arg-type]
    test_ds = SNIPSDataset(snips["test"], vocab=vocab, seq_len=cfg.seq_len)  # type: ignore[arg-type]

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    lm = build_backbone(vocab_size=len(vocab), pad_id=vocab.pad_id, lm_cfg=lm_cfg)
    model = IntentClassifier(lm=lm, embed_dim=cfg.embed_dim, num_classes=cfg.num_classes)
    trainer = DownstreamTrainer(
        model=model,
        cfg=cfg,
        run_name=run_name,
        mode=args.mode,
        logger=logger,
        class_names=class_names,  # type: ignore[arg-type]
    )

    trainer.train(train_loader, val_loader)
    test_metrics = trainer.evaluate(test_loader)
    logger.info(
        "test mode=%s acc=%.4f macro_f1=%.4f device=%s",
        args.mode,
        test_metrics["accuracy"],
        test_metrics["macro_f1"],
        resolve_device(cfg.device),
    )


if __name__ == "__main__":
    main()
