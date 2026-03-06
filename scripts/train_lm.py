"""Part I entry: train ngram/rnn/lstm/transformer language models."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.lm_dataset import LMDataset
from src.data.pipeline import build_or_load_vocab_and_ids
from src.data.tokenizer import SimpleTokenizer
from src.embeddings.factory import build_embedding
from src.models.lstm_lm import LSTMLanguageModel
from src.models.ngram_lm import NGramLanguageModel
from src.models.rnn_lm import RNNLanguageModel
from src.models.transformer_lm import TransformerLanguageModel
from src.utils.config import EmbeddingConfig, LMConfig
from src.utils.logging_utils import get_logger, log_experiment_result
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ngram", "rnn", "lstm", "transformer"], required=True)
    parser.add_argument("--embed", dest="embed", choices=["trainable", "fixed_self", "fixed_pretrained"], default="trainable")
    parser.add_argument(
        "--embed_type",
        dest="embed",
        choices=["trainable", "fixed_self", "fixed_pretrained"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default=None, help="Processed data directory override.")
    parser.add_argument("--lr", type=float, default=None, help="Learning-rate override.")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda or cpu.")
    parser.add_argument(
        "--glove_path",
        type=str,
        default="data/raw/glove.6B.300d.txt",
        help="Path to GloVe embeddings file (used when --embed=fixed_pretrained).",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def make_model(
    model_name: str,
    lm_cfg: LMConfig,
    emb_cfg: EmbeddingConfig,
    vocab,
    token_sequences: list[list[str]],
):
    embedding, embedding_meta = build_embedding(emb_cfg, vocab, token_sequences=token_sequences)
    if model_name == "rnn":
        model = RNNLanguageModel(
            vocab_size=len(vocab),
            embedding=embedding,
            hidden_size=lm_cfg.hidden_size,
            num_layers=lm_cfg.num_layers,
            dropout=lm_cfg.dropout,
            pad_id=vocab.pad_id,
        )
    elif model_name == "lstm":
        model = LSTMLanguageModel(
            vocab_size=len(vocab),
            embedding=embedding,
            hidden_size=lm_cfg.hidden_size,
            num_layers=lm_cfg.num_layers,
            dropout=lm_cfg.dropout,
            pad_id=vocab.pad_id,
        )
    elif model_name == "transformer":
        model = TransformerLanguageModel(
            vocab_size=len(vocab),
            embedding=embedding,
            d_model=lm_cfg.d_model,
            nhead=lm_cfg.nhead,
            num_layers=lm_cfg.num_transformer_layers,
            dim_feedforward=lm_cfg.dim_feedforward,
            dropout=lm_cfg.dropout,
            pad_id=vocab.pad_id,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model, embedding_meta


def rebuild_loader(dataset: LMDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def train_neural_model(args: argparse.Namespace, cfg: LMConfig, run_name: str, logger) -> None:
    tokenizer = SimpleTokenizer()
    vocab, split_ids, train_token_sequences = build_or_load_vocab_and_ids(
        data_dir=cfg.data_dir,
        vocab_size=cfg.vocab_size,
        tokenizer=tokenizer,
    )
    adaptive_seq_len = cfg.seq_len
    train_len = int(len(split_ids["train"]))
    val_len = int(len(split_ids["validation"]))
    min_len = min(train_len, val_len)
    if min_len <= 2:
        raise ValueError("Not enough token ids to build LM dataset; need at least 3 tokens per split.")
    max_allowed = min_len - 2
    adaptive_seq_len = max(1, min(cfg.seq_len, max_allowed))
    logger.info("using_seq_len=%d", adaptive_seq_len)
    train_ds = LMDataset(split_ids["train"], seq_len=adaptive_seq_len, stride=1)
    val_ds = LMDataset(split_ids["validation"], seq_len=adaptive_seq_len, stride=1)
    batch_size = args.batch_size or cfg.batch_size
    train_loader = rebuild_loader(train_ds, batch_size=batch_size, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers)

    emb_cfg = EmbeddingConfig(
        embed_type=args.embed,
        embed_dim=cfg.embed_dim,
        vocab_size=cfg.vocab_size,
        glove_path=args.glove_path,
    )
    model, embedding_meta = make_model(args.model, cfg, emb_cfg, vocab, train_token_sequences)
    device = resolve_device(cfg.device)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    warmup_steps = max(cfg.warmup_steps, 1)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps))

    grad_accum_steps = 1
    effective_batch = batch_size
    oom_levels = [(16, 4), (8, 8)]
    oom_idx = 0

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("run=%s model=%s embed=%s params=%d device=%s", run_name, args.model, args.embed, total_params, device)
    if "glove_coverage" in embedding_meta:
        logger.info("glove_coverage=%.4f", embedding_meta["glove_coverage"])

    epochs = args.epochs if args.epochs is not None else cfg.epochs
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        seen_steps = 0
        for step, (input_ids, target_ids) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            try:
                loss = model.compute_loss(input_ids, target_ids) / grad_accum_steps
                loss.backward()
            except RuntimeError as exc:
                is_oom = "out of memory" in str(exc).lower() and torch.cuda.is_available()
                if not is_oom:
                    raise
                torch.cuda.empty_cache()
                if oom_idx >= len(oom_levels):
                    raise
                new_batch, new_accum = oom_levels[oom_idx]
                oom_idx += 1
                train_loader = rebuild_loader(train_ds, batch_size=new_batch, num_workers=cfg.num_workers)
                val_loader = DataLoader(val_ds, batch_size=new_batch, shuffle=False, num_workers=cfg.num_workers)
                grad_accum_steps = new_accum
                effective_batch = new_batch * new_accum
                logger.warning(
                    "CUDA OOM detected, switching to batch=%d accum=%d effective_batch=%d",
                    new_batch,
                    new_accum,
                    effective_batch,
                )
                optimizer.zero_grad()
                continue

            running_loss += float(loss.item()) * grad_accum_steps
            seen_steps += 1

            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if args.max_steps and step >= args.max_steps:
                break

        epoch_time = time.time() - epoch_start
        train_loss = running_loss / max(seen_steps, 1)
        val_ppl = model.perplexity(val_loader, device=device)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            "epoch=%d train_loss=%.4f val_ppl=%.4f lr=%.6f time=%.2fs effective_batch=%d",
            epoch,
            train_loss,
            val_ppl,
            current_lr,
            epoch_time,
            effective_batch,
        )
        log_experiment_result(
            run_name=run_name,
            epoch=epoch,
            metrics={
                "train_loss": train_loss,
                "val_ppl": val_ppl,
                "epoch_time_sec": epoch_time,
                "total_params": float(total_params),
            },
        )

    checkpoint_dir = Path(cfg.checkpoint_dir) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model,
            "state_dict": model.state_dict(),
            "vocab_path": str(Path(cfg.data_dir) / "vocab.json"),
            "config": cfg.__dict__,
        },
        checkpoint_dir / "model.pt",
    )
    logger.info("checkpoint_saved=%s", checkpoint_dir / "model.pt")


def train_ngram(args: argparse.Namespace, cfg: LMConfig, run_name: str, logger) -> None:
    tokenizer = SimpleTokenizer()
    vocab, split_ids, _ = build_or_load_vocab_and_ids(
        data_dir=cfg.data_dir,
        vocab_size=cfg.vocab_size,
        tokenizer=tokenizer,
    )
    model = NGramLanguageModel(
        vocab_size=len(vocab),
        n=cfg.ngram_n,
        smoothing=cfg.ngram_smoothing,
    )
    model.fit(split_ids["train"].tolist())
    val_ppl = model.perplexity(split_ids["validation"].tolist())
    test_ppl = model.perplexity(split_ids["test"].tolist())
    logger.info("ngram_val_ppl=%.4f ngram_test_ppl=%.4f", val_ppl, test_ppl)
    log_experiment_result(
        run_name=run_name,
        epoch=1,
        metrics={
            "train_loss": 0.0,
            "val_ppl": val_ppl,
            "epoch_time_sec": 0.0,
            "total_params": 0,
        },
    )


def main() -> None:
    args = parse_args()
    cfg = LMConfig()
    set_seed(cfg.seed)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_steps:
        cfg.max_steps_per_epoch = args.max_steps
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.lr is not None:
        cfg.lr = args.lr
    if args.device is not None:
        cfg.device = args.device

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_{args.embed}_{timestamp}"
    logger = get_logger(run_name, log_dir=cfg.log_dir)

    if args.model == "ngram":
        train_ngram(args, cfg, run_name, logger)
    else:
        train_neural_model(args, cfg, run_name, logger)


if __name__ == "__main__":
    main()
