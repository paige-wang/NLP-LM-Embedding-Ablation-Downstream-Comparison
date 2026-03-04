"""Part II entry: embedding ablation over neural language models."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.lm_dataset import LMDataset
from src.data.pipeline import build_or_load_vocab_and_ids
from src.data.tokenizer import SimpleTokenizer
from src.embeddings.factory import build_embedding
from src.models.lstm_lm import LSTMLanguageModel
from src.models.rnn_lm import RNNLanguageModel
from src.models.transformer_lm import TransformerLanguageModel
from src.utils.config import EmbeddingConfig, LMConfig
from src.utils.logging_utils import get_logger, log_experiment_result
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rnn", "lstm", "transformer"], default=None)
    parser.add_argument("--embed", choices=["trainable", "fixed_self", "fixed_pretrained"], default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=0)
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


def make_model(model_name: str, cfg: LMConfig, vocab_size: int, embedding, pad_id: int):
    if model_name == "rnn":
        return RNNLanguageModel(vocab_size, embedding, cfg.hidden_size, cfg.num_layers, cfg.dropout, pad_id=pad_id)
    if model_name == "lstm":
        return LSTMLanguageModel(vocab_size, embedding, cfg.hidden_size, cfg.num_layers, cfg.dropout, pad_id=pad_id)
    if model_name == "transformer":
        return TransformerLanguageModel(
            vocab_size=vocab_size,
            embedding=embedding,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_transformer_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            pad_id=pad_id,
        )
    raise ValueError(f"Unsupported model {model_name}")


def run_single(
    model_name: str,
    embed_name: str,
    cfg: LMConfig,
    args: argparse.Namespace,
    split_ids,
    vocab,
    token_sequences,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{embed_name}_{timestamp}"
    logger = get_logger(run_name, cfg.log_dir)
    logger.info("ablation_run model=%s embed=%s", model_name, embed_name)

    min_len = min(len(split_ids["train"]), len(split_ids["validation"]))
    if min_len <= 2:
        raise ValueError("Not enough token ids to build LM dataset; need at least 3 tokens per split.")
    seq_len = max(1, min(cfg.seq_len, min_len - 2))
    train_ds = LMDataset(split_ids["train"], seq_len=seq_len, stride=1)
    val_ds = LMDataset(split_ids["validation"], seq_len=seq_len, stride=1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers)

    emb_cfg = EmbeddingConfig(
        embed_type=embed_name,
        embed_dim=cfg.embed_dim,
        vocab_size=cfg.vocab_size,
        glove_path=args.glove_path,
    )
    embedding, meta = build_embedding(emb_cfg, vocab, token_sequences=token_sequences)
    model = make_model(model_name, cfg, len(vocab), embedding, pad_id=vocab.pad_id)
    device = resolve_device(cfg.device)
    model.to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    best_ppl = float("inf")
    best_epoch = 1
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total_loss = 0.0
        steps = 0
        for step, (input_ids, target_ids) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            loss = model.compute_loss(input_ids, target_ids)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
            if args.max_steps and step >= args.max_steps:
                break
        train_loss = total_loss / max(steps, 1)
        val_ppl = model.perplexity(val_loader, device)
        epoch_time = time.time() - start
        logger.info(
            "epoch=%d train_loss=%.4f val_ppl=%.4f time=%.2f",
            epoch,
            train_loss,
            val_ppl,
            epoch_time,
        )
        log_experiment_result(
            run_name=run_name,
            epoch=epoch,
            metrics={
                "embed_type": embed_name,
                "model": model_name,
                "val_ppl": val_ppl,
                "convergence_epoch": epoch,
            },
        )
        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_epoch = epoch

    if "glove_coverage" in meta:
        logger.info("glove_coverage=%.4f", meta["glove_coverage"])
    logger.info("best_val_ppl=%.4f convergence_epoch=%d", best_ppl, best_epoch)

    ckpt_dir = Path(cfg.checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_name": model_name, "embed_name": embed_name, "state_dict": model.state_dict()},
        ckpt_dir / "model.pt",
    )


def main() -> None:
    args = parse_args()
    cfg = LMConfig()
    set_seed(cfg.seed)
    tokenizer = SimpleTokenizer()
    vocab, split_ids, token_sequences = build_or_load_vocab_and_ids(cfg.data_dir, cfg.vocab_size, tokenizer)

    models = [args.model] if args.model else ["rnn", "lstm", "transformer"]
    embeds = [args.embed] if args.embed else ["trainable", "fixed_self", "fixed_pretrained"]
    for model_name in models:
        for embed_name in embeds:
            run_single(model_name, embed_name, cfg, args, split_ids, vocab, token_sequences)


if __name__ == "__main__":
    main()
