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
from src.embeddings.fixed_pretrained import FixedPretrainedEmbedding
from src.downstream.classifier import IntentClassifier
from src.downstream.trainer import DownstreamTrainer
from src.models.transformer_lm import TransformerLanguageModel
from src.utils.config import DownstreamConfig, LMConfig
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["frozen", "finetune"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override backbone LR (finetune: default 5e-5; frozen: default 1e-3)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to a pretrained LM .pt checkpoint to warm-start the backbone.",
    )
    parser.add_argument(
        "--rebuild_vocab",
        action="store_true",
        help=(
            "Force delete cached vocab.json and token-id arrays, then rebuild "
            "from WikiText-2 + SNIPS training data.  Use this when the cached "
            "vocab is tiny (e.g. built from the fallback corpus)."
        ),
    )
    parser.add_argument(
        "--embed_init",
        choices=["random", "glove", "pretrained"],
        default="random",
        help=(
            "Embedding initialization strategy: "
            "'random' = default random init, "
            "'glove' = initialize embedding layer from GloVe vectors, "
            "'pretrained' = load full checkpoint via --ckpt_path (default behavior)."
        ),
    )
    parser.add_argument(
        "--glove_path",
        type=str,
        default="data/raw/glove.6B.300d.txt",
        help="Path to GloVe embeddings file (used when --embed_init=glove).",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_backbone(vocab_size: int, pad_id: int, lm_cfg: LMConfig) -> TransformerLanguageModel:
    embedding = TrainableEmbedding(
        vocab_size=vocab_size, embedding_dim=lm_cfg.embed_dim, pad_id=pad_id
    )
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


def load_checkpoint(model: TransformerLanguageModel, ckpt_path: str, logger) -> None:
    """Load a pretrained LM checkpoint into the backbone.

    Uses strict=False so that size-mismatched keys (embedding / lm_head when
    vocab changed) are skipped gracefully.
    """
    path = Path(ckpt_path)
    if not path.exists():
        logger.warning("Checkpoint not found at %s — starting from random init.", ckpt_path)
        return
    state = torch.load(path, map_location="cpu")
    # Unwrap nested dicts produced by some save conventions
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = set(state.keys()) - set(missing)
    logger.info(
        "Loaded checkpoint %s  matched=%d  missing=%d  unexpected=%d",
        ckpt_path,
        len(loaded),
        len(missing),
        len(unexpected),
    )


def main() -> None:
    args = parse_args()
    cfg = DownstreamConfig()
    lm_cfg = LMConfig()
    set_seed(cfg.seed)

    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.max_steps_per_epoch = args.max_steps

    # Apply LR overrides (never silently override to 1e-3)
    if args.mode == "finetune":
        if args.lr is not None:
            cfg.finetune_lr = args.lr
        # else keep config default: 5e-5
    else:
        if args.lr is not None:
            cfg.frozen_lr = args.lr

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    init_tag = args.embed_init if args.embed_init != "random" else ""
    run_name = f"downstream_{args.mode}{'_' + init_tag if init_tag else ''}_{timestamp}"
    logger = get_logger(run_name, cfg.log_dir)

    logger.info(
        "mode=%s  epochs=%d  batch_size=%d  finetune_lr=%.2e  frozen_lr=%.2e",
        args.mode,
        cfg.epochs,
        cfg.batch_size,
        cfg.finetune_lr,
        cfg.frozen_lr,
    )

    tokenizer = SimpleTokenizer()

    # ── Load SNIPS splits first so we can augment the vocab ──────────────────
    snips = load_snips_splits()
    class_names: list[str] = snips["labels"]  # type: ignore[assignment]

    # Tokenize SNIPS training texts → used to extend vocab coverage
    snips_train_tokens: list[list[str]] = [
        tokenizer.tokenize(ex.text)  # type: ignore[union-attr]
        for ex in snips["train"]     # type: ignore[union-attr]
        if ex.text.strip()           # type: ignore[union-attr]
    ]

    # ── Build or rebuild vocabulary ───────────────────────────────────────────
    vocab, _, _ = build_or_load_vocab_and_ids(
        lm_cfg.data_dir,
        lm_cfg.vocab_size,
        tokenizer,
        supplement_texts=snips_train_tokens,
        force=args.rebuild_vocab,
    )
    logger.info("Vocabulary size: %d", len(vocab))

    # ── Datasets & data-loaders ───────────────────────────────────────────────
    train_ds = SNIPSDataset(snips["train"], vocab=vocab, seq_len=cfg.seq_len)       # type: ignore
    val_ds   = SNIPSDataset(snips["validation"], vocab=vocab, seq_len=cfg.seq_len)  # type: ignore
    test_ds  = SNIPSDataset(snips["test"], vocab=vocab, seq_len=cfg.seq_len)        # type: ignore

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    # ── Build backbone and apply initialization strategy ────────────────────
    lm = build_backbone(vocab_size=len(vocab), pad_id=vocab.pad_id, lm_cfg=lm_cfg)

    if args.embed_init == "glove":
        # Initialize only the embedding layer from GloVe; other layers stay random
        glove_emb, coverage = FixedPretrainedEmbedding.from_glove(
            vocab=vocab, glove_path=args.glove_path, embed_dim=lm_cfg.embed_dim, pad_id=vocab.pad_id
        )
        lm.embedding.embedding.weight.data.copy_(glove_emb.weight.data)
        logger.info("embed_init=glove  glove_coverage=%.4f", coverage)
    elif args.embed_init == "pretrained" or args.ckpt_path:
        if args.ckpt_path:
            load_checkpoint(lm, args.ckpt_path, logger)
        else:
            logger.warning("embed_init=pretrained but no --ckpt_path provided; using random init.")
    else:
        logger.info("embed_init=random")

    model = IntentClassifier(lm=lm, embed_dim=cfg.embed_dim, num_classes=cfg.num_classes)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = DownstreamTrainer(
        model=model,
        cfg=cfg,
        run_name=run_name,
        mode=args.mode,
        logger=logger,
        class_names=class_names,
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
