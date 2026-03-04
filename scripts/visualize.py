"""Unified visualization script for all report figures.

Subcommands:
  convergence  — Training convergence curves (Part I & Part II)
  coverage     — Vocabulary coverage stacked bar chart
  tsne         — t-SNE embedding space visualization
  cosine       — Pairwise cosine similarity heatmaps
  transfer     — Initialization strategy comparison bar chart
  all          — Generate all available plots
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Shared constants ─────────────────────────────────────────────────────────
COLORS = {
    "rnn": "#4472C4",
    "lstm": "#ED7D31",
    "transformer": "#70AD47",
    "trigram": "#808080",
    "trainable": "#4472C4",
    "fixed_self": "#ED7D31",
    "fixed_pretrained": "#70AD47",
    "random": "#4472C4",
    "glove": "#ED7D31",
    "pretrained": "#70AD47",
    "finetune": "#C00000",
    "frozen": "#4472C4",
}

DEFAULT_WORDS = [
    "restaurant", "book", "weather", "music", "playlist", "song",
    "movie", "hotel", "food", "rain", "play", "search", "rate",
    "screen", "add", "get", "find", "show",
]


# ── Log parsing ──────────────────────────────────────────────────────────────

def parse_lm_logs(log_dir: str) -> dict[str, dict[str, list]]:
    """Parse ablation/train_lm logs → {run_key: {epochs, train_loss, val_ppl}}.

    run_key is formatted as 'model_embed' (e.g. 'rnn_trainable').
    When multiple runs share the same key, the longest run is kept.
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return {}

    epoch_re = re.compile(
        r"epoch=(\d+)\s+train_loss=([\d.]+)\s+val_ppl=([\d.]+)"
    )
    run_re = re.compile(r"(?:ablation_run|lm_run)\s+model=(\w+)\s+embed=(\w+)")
    run_re2 = re.compile(r"run=\w+\s+model=(\w+)\s+embed=(\w+)")

    all_runs: dict[str, list[dict[str, list]]] = defaultdict(list)

    for log_file in sorted(log_path.glob("*.log")):
        text = log_file.read_text(encoding="utf-8", errors="replace")
        # Determine model/embed from header line
        m = run_re.search(text) or run_re2.search(text)
        if not m:
            continue
        model, embed = m.group(1), m.group(2)
        key = f"{model}_{embed}"

        epochs, losses, ppls = [], [], []
        for em in epoch_re.finditer(text):
            epochs.append(int(em.group(1)))
            losses.append(float(em.group(2)))
            ppls.append(float(em.group(3)))

        if epochs:
            all_runs[key].append({
                "epochs": epochs, "train_loss": losses, "val_ppl": ppls,
            })

    # Keep longest run per key
    result: dict[str, dict[str, list]] = {}
    for key, runs in all_runs.items():
        result[key] = max(runs, key=lambda r: len(r["epochs"]))
    return result


def parse_downstream_logs(log_dir: str) -> dict[str, dict[str, list]]:
    """Parse downstream logs → {run_key: {epochs, train_loss, val_acc, val_f1}}.

    run_key examples: 'frozen', 'finetune', 'frozen_glove', 'frozen_pretrained'.
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return {}

    epoch_re = re.compile(
        r"epoch=(\d+)\s+mode=(\w+)\s+train_loss=([\d.]+)\s+"
        r"val_acc=([\d.]+)\s+val_macro_f1=([\d.]+)"
    )
    init_re = re.compile(r"embed_init=(\w+)")

    all_runs: dict[str, list[dict[str, list]]] = defaultdict(list)

    for log_file in sorted(log_path.glob("downstream_*.log")):
        text = log_file.read_text(encoding="utf-8", errors="replace")
        im = init_re.search(text)
        init_tag = im.group(1) if im else ""

        epochs, losses, accs, f1s = [], [], [], []
        mode = None
        for em in epoch_re.finditer(text):
            mode = em.group(2)
            epochs.append(int(em.group(1)))
            losses.append(float(em.group(3)))
            accs.append(float(em.group(4)))
            f1s.append(float(em.group(5)))

        if epochs and mode:
            key = f"{mode}{'_' + init_tag if init_tag else ''}"
            all_runs[key].append({
                "epochs": epochs, "train_loss": losses,
                "val_acc": accs, "val_f1": f1s,
            })

    result: dict[str, dict[str, list]] = {}
    for key, runs in all_runs.items():
        result[key] = max(runs, key=lambda r: len(r["epochs"]))
    return result


# ── Subcommand: convergence ──────────────────────────────────────────────────

def cmd_convergence(args: argparse.Namespace) -> None:
    """Generate Part I and Part II convergence curves."""
    runs = parse_lm_logs(args.log_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Part I: model comparison ---
    part1_keys = ["rnn_trainable", "lstm_trainable", "transformer_trainable"]
    part1_labels = {"rnn_trainable": "RNN", "lstm_trainable": "LSTM",
                    "transformer_trainable": "Transformer"}
    part1_colors = {"rnn_trainable": COLORS["rnn"], "lstm_trainable": COLORS["lstm"],
                    "transformer_trainable": COLORS["transformer"]}

    available = [k for k in part1_keys if k in runs]
    if available:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for key in available:
            r = runs[key]
            ax.plot(r["epochs"], r["val_ppl"], marker="o", markersize=3,
                    color=part1_colors[key], label=part1_labels[key], linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Perplexity")
        ax.set_title("Part I: Language Model Convergence")
        ax.set_yscale("log")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(out / "part1_convergence.png", dpi=200)
        plt.close(fig)
        print(f"  Saved: {out / 'part1_convergence.png'}")

    # --- Part II: ablation convergence (1×3 subplots) ---
    models = ["rnn", "lstm", "transformer"]
    embeds = ["trainable", "fixed_self", "fixed_pretrained"]
    embed_labels = {"trainable": "Trainable", "fixed_self": "Word2Vec", "fixed_pretrained": "GloVe"}

    has_ablation = any(f"{m}_{e}" in runs for m in models for e in embeds if e != "trainable")
    if has_ablation:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for i, model in enumerate(models):
            ax = axes[i]
            for embed in embeds:
                key = f"{model}_{embed}"
                if key not in runs:
                    continue
                r = runs[key]
                ax.plot(r["epochs"], r["val_ppl"], marker="o", markersize=2,
                        color=COLORS[embed], label=embed_labels[embed], linewidth=1.3)
            ax.set_xlabel("Epoch")
            if i == 0:
                ax.set_ylabel("Validation Perplexity")
            ax.set_title(model.upper())
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        fig.suptitle("Part II: Embedding Ablation Convergence", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(out / "part2_ablation_convergence.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out / 'part2_ablation_convergence.png'}")

    # --- Part II: grouped bar chart (best PPL per model×embed) ---
    if has_ablation:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(models))
        width = 0.25
        for j, embed in enumerate(embeds):
            ppls = []
            for model in models:
                key = f"{model}_{embed}"
                if key in runs:
                    ppls.append(min(runs[key]["val_ppl"]))
                else:
                    ppls.append(0)
            bars = ax.bar(x + j * width, ppls, width, label=embed_labels[embed],
                          color=COLORS[embed], alpha=0.85)
            for bar, ppl in zip(bars, ppls):
                if ppl > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f"{ppl:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.upper() for m in models])
        ax.set_ylabel("Best Validation Perplexity")
        ax.set_title("Part II: Embedding Ablation — Best PPL")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.9)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(out / "part2_ablation_bars.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out / 'part2_ablation_bars.png'}")

    if not available and not has_ablation:
        print("  [convergence] No LM log data found.")


# ── Subcommand: coverage ─────────────────────────────────────────────────────

def cmd_coverage(args: argparse.Namespace) -> None:
    """Generate vocabulary coverage stacked bar chart."""
    from src.data.pipeline import build_or_load_vocab_and_ids
    from src.data.tokenizer import SimpleTokenizer

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = SimpleTokenizer()
    vocab, _, _ = build_or_load_vocab_and_ids("data/processed", 20000, tokenizer)
    vocab_size = len(vocab)

    coverages = {"Trainable": 1.0}  # trainable always 100%

    # Word2Vec coverage
    w2v_path = Path("data/processed/word2vec.bin")
    if w2v_path.exists():
        try:
            from gensim.models import Word2Vec
            w2v = Word2Vec.load(str(w2v_path))
            hits = sum(1 for t in vocab.token2id if t in w2v.wv)
            coverages["Word2Vec"] = hits / vocab_size
        except Exception:
            coverages["Word2Vec"] = 0.0
    else:
        coverages["Word2Vec"] = 0.0

    # GloVe coverage
    glove_path = Path(args.glove_path) if args.glove_path else None
    if glove_path and glove_path.exists():
        glove_tokens: set[str] = set()
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                token = line.split(" ", 1)[0]
                glove_tokens.add(token)
        hits = sum(1 for t in vocab.token2id if t in glove_tokens)
        coverages["GloVe"] = hits / vocab_size
    else:
        coverages["GloVe"] = 0.0

    labels = list(coverages.keys())
    covered = [coverages[l] for l in labels]
    uncovered = [1.0 - c for c in covered]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax.bar(x, covered, color=COLORS["trainable"], label="Covered", alpha=0.85)
    ax.bar(x, uncovered, bottom=covered, color="#D3D3D3", label="OOV (random init)", alpha=0.7)
    for i, c in enumerate(covered):
        ax.text(i, c / 2, f"{c * 100:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Coverage Fraction")
    ax.set_title("Vocabulary Coverage by Embedding Source")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / "vocab_coverage.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out / 'vocab_coverage.png'}")


# ── Subcommand: tsne ─────────────────────────────────────────────────────────

def cmd_tsne(args: argparse.Namespace) -> None:
    """Generate t-SNE embedding space visualization."""
    import torch
    from sklearn.manifold import TSNE

    from src.data.pipeline import build_or_load_vocab_and_ids
    from src.data.tokenizer import SimpleTokenizer

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    words = args.words.split(",") if args.words else DEFAULT_WORDS
    tokenizer = SimpleTokenizer()
    vocab, _, _ = build_or_load_vocab_and_ids("data/processed", 20000, tokenizer)

    # Filter to words in vocab
    word_ids = [(w, vocab.token2id[w]) for w in words if w in vocab.token2id]
    if len(word_ids) < 3:
        print("  [tsne] Not enough words found in vocabulary.")
        return

    embeddings: dict[str, np.ndarray] = {}

    # GloVe embeddings
    if args.glove_path and Path(args.glove_path).exists():
        from src.embeddings.fixed_pretrained import FixedPretrainedEmbedding
        glove_emb, _ = FixedPretrainedEmbedding.from_glove(vocab, args.glove_path, 300)
        vecs = glove_emb.weight.detach().numpy()
        embeddings["GloVe"] = vecs[[idx for _, idx in word_ids]]

    # Trainable checkpoint embeddings
    if args.ckpt_path and Path(args.ckpt_path).exists():
        state = torch.load(args.ckpt_path, map_location="cpu")
        sd = state.get("model_state_dict", state.get("state_dict", state))
        for key_name in ["embedding.embedding.weight", "embedding.weight"]:
            if key_name in sd:
                vecs = sd[key_name].numpy()
                embeddings["Trained"] = vecs[[idx for _, idx in word_ids]]
                break

    if not embeddings:
        print("  [tsne] No embedding sources available.")
        return

    n_panels = len(embeddings)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    word_labels = [w for w, _ in word_ids]
    perplexity = min(5, len(word_ids) - 1)

    for ax, (name, vecs) in zip(axes, embeddings.items()):
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        coords = tsne.fit_transform(vecs)
        ax.scatter(coords[:, 0], coords[:, 1], c=COLORS["transformer"], s=40, alpha=0.8)
        for j, label in enumerate(word_labels):
            ax.annotate(label, (coords[j, 0], coords[j, 1]),
                        fontsize=8, ha="center", va="bottom",
                        textcoords="offset points", xytext=(0, 4))
        ax.set_title(f"{name} Embedding Space")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("t-SNE Visualization of Selected Word Embeddings", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "tsne_embeddings.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / 'tsne_embeddings.png'}")


# ── Subcommand: cosine ───────────────────────────────────────────────────────

def cmd_cosine(args: argparse.Namespace) -> None:
    """Generate pairwise cosine similarity heatmaps."""
    import torch
    from sklearn.metrics.pairwise import cosine_similarity

    from src.data.pipeline import build_or_load_vocab_and_ids
    from src.data.tokenizer import SimpleTokenizer

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    words = args.words.split(",") if args.words else DEFAULT_WORDS[:12]
    tokenizer = SimpleTokenizer()
    vocab, _, _ = build_or_load_vocab_and_ids("data/processed", 20000, tokenizer)

    word_ids = [(w, vocab.token2id[w]) for w in words if w in vocab.token2id]
    if len(word_ids) < 3:
        print("  [cosine] Not enough words found in vocabulary.")
        return

    embeddings: dict[str, np.ndarray] = {}

    if args.glove_path and Path(args.glove_path).exists():
        from src.embeddings.fixed_pretrained import FixedPretrainedEmbedding
        glove_emb, _ = FixedPretrainedEmbedding.from_glove(vocab, args.glove_path, 300)
        vecs = glove_emb.weight.detach().numpy()
        embeddings["GloVe"] = vecs[[idx for _, idx in word_ids]]

    if args.ckpt_path and Path(args.ckpt_path).exists():
        state = torch.load(args.ckpt_path, map_location="cpu")
        sd = state.get("model_state_dict", state.get("state_dict", state))
        for key_name in ["embedding.embedding.weight", "embedding.weight"]:
            if key_name in sd:
                vecs = sd[key_name].numpy()
                embeddings["Trained"] = vecs[[idx for _, idx in word_ids]]
                break

    if not embeddings:
        print("  [cosine] No embedding sources available.")
        return

    word_labels = [w for w, _ in word_ids]
    n_panels = len(embeddings)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (name, vecs) in zip(axes, embeddings.items()):
        sim = cosine_similarity(vecs)
        im = ax.imshow(sim, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(len(word_labels)))
        ax.set_yticks(range(len(word_labels)))
        ax.set_xticklabels(word_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(word_labels, fontsize=7)
        ax.set_title(f"{name} Cosine Similarity")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Pairwise Cosine Similarity of Selected Words", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "cosine_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / 'cosine_heatmap.png'}")


# ── Subcommand: transfer ─────────────────────────────────────────────────────

def cmd_transfer(args: argparse.Namespace) -> None:
    """Generate initialization strategy comparison bar chart."""
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Parse results from CLI or downstream logs
    if args.results:
        # Format: "random:85.2,glove:88.1,pretrained:91.5,finetune:97.0"
        pairs = [p.split(":") for p in args.results.split(",")]
        labels = [p[0].strip() for p in pairs]
        accs = [float(p[1].strip()) for p in pairs]
    else:
        # Try parsing from logs
        ds_runs = parse_downstream_logs(args.log_dir)
        if not ds_runs:
            print("  [transfer] No downstream results available. Use --results to provide them.")
            return
        labels, accs = [], []
        for key in ["frozen", "frozen_glove", "frozen_pretrained", "finetune"]:
            if key in ds_runs:
                best_acc = max(ds_runs[key]["val_acc"]) * 100
                display = key.replace("frozen_", "").replace("frozen", "Random Init")
                display = display.replace("glove", "GloVe Init").replace("pretrained", "Pretrained")
                display = display.replace("finetune", "Fine-tuned")
                labels.append(display)
                accs.append(best_acc)

    if not labels:
        print("  [transfer] No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    colors_list = [COLORS.get(l.lower().split()[0], COLORS["random"]) for l in labels]
    bars = ax.bar(x, accs, color=colors_list, alpha=0.85, width=0.55)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Part III: Initialization Strategy Comparison")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(accs) * 1.15 if accs else 100)
    fig.tight_layout()
    fig.savefig(out / "transfer_comparison.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: {out / 'transfer_comparison.png'}")


# ── Subcommand: downstream convergence ───────────────────────────────────────

def cmd_downstream_convergence(args: argparse.Namespace) -> None:
    """Generate downstream training convergence curves."""
    ds_runs = parse_downstream_logs(args.log_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not ds_runs:
        print("  [downstream] No downstream log data found.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for key, data in sorted(ds_runs.items()):
        label = key.replace("_", " ").title()
        color = COLORS.get(key.split("_")[0], COLORS["random"])
        ax1.plot(data["epochs"], data["val_acc"], marker="o", markersize=3,
                 label=label, color=color, linewidth=1.3)
        ax2.plot(data["epochs"], data["train_loss"], marker="s", markersize=3,
                 label=label, color=color, linewidth=1.3)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Accuracy")
    ax1.set_title("Downstream — Val Accuracy")
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle="--")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Loss")
    ax2.set_title("Downstream — Train Loss")
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("Part III: Downstream Training Dynamics", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "downstream_convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out / 'downstream_convergence.png'}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate report figures from experiment logs and checkpoints."
    )
    parser.add_argument("--log_dir", default="outputs/logs", help="Directory with .log files")
    parser.add_argument("--output_dir", default="outputs/figures", help="Output directory for PNGs")
    parser.add_argument("--glove_path", default="data/raw/glove.6B.300d.txt", help="GloVe file path")
    parser.add_argument("--ckpt_path", default=None, help="Trained model checkpoint for embedding viz")
    parser.add_argument("--words", default=None, help="Comma-separated word list for tsne/cosine")
    parser.add_argument("--results", default=None,
                        help="Manual results for transfer chart (e.g. 'random:85,glove:88')")
    parser.add_argument("--all", action="store_true", help="Generate all available plots")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("convergence", help="Training convergence curves")
    sub.add_parser("coverage", help="Vocabulary coverage bar chart")
    sub.add_parser("tsne", help="t-SNE embedding visualization")
    sub.add_parser("cosine", help="Cosine similarity heatmaps")
    sub.add_parser("transfer", help="Init strategy comparison bars")
    sub.add_parser("downstream", help="Downstream training curves")
    sub.add_parser("all", help="Generate all available plots")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.all or args.command == "all":
        print("Generating all available plots...")
        for cmd_fn in [cmd_convergence, cmd_coverage, cmd_tsne, cmd_cosine,
                       cmd_transfer, cmd_downstream_convergence]:
            try:
                cmd_fn(args)
            except Exception as e:
                print(f"  Warning: {cmd_fn.__name__} failed: {e}")
        return

    dispatch = {
        "convergence": cmd_convergence,
        "coverage": cmd_coverage,
        "tsne": cmd_tsne,
        "cosine": cmd_cosine,
        "transfer": cmd_transfer,
        "downstream": cmd_downstream_convergence,
    }

    if args.command and args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
