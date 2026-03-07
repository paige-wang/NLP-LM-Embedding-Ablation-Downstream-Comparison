"""Dataset loading and preprocessing helpers."""

import csv
from pathlib import Path

from datasets import load_dataset
import numpy as np

from src.data.downstream_dataset import SNIPSExample, parse_snips_split
from src.data.tokenizer import SimpleTokenizer
from src.data.vocabulary import Vocabulary

SNIPS_DATA_DIR = Path("data/snips")


def _reconstruct_sequences_from_ids(ids: np.ndarray, vocab: Vocabulary) -> list[list[str]]:
    """Reconstruct token sequences from cached LM ids split by BOS/EOS markers."""
    sequences: list[list[str]] = []
    current: list[int] = []
    for idx in ids.tolist():
        if idx == vocab.bos_id:
            current = []
            continue
        if idx == vocab.eos_id:
            if current:
                sequences.append(vocab.decode_ids(current, skip_special=True))
            current = []
            continue
        current.append(idx)
    if current:
        sequences.append(vocab.decode_ids(current, skip_special=True))
    return [seq for seq in sequences if seq]


def _fallback_wikitext() -> dict[str, list[str]]:
    """Provide tiny local fallback corpus when HF download is unavailable."""
    return {
        "train": [
            "Language models learn token distributions from text corpora.",
            "Transformer models use causal attention masks during training.",
            "Recurrent models include RNN and LSTM architectures.",
        ],
        "validation": [
            "Embeddings can be trainable or frozen.",
            "Perplexity measures language model quality.",
        ],
        "test": [
            "Intent classification can use frozen or fine tuned language models.",
            "SNIPS contains seven intent labels.",
        ],
    }


def load_wikitext_splits() -> dict[str, list[str]]:
    """Load WikiText-2 raw splits, with fallback if network is unavailable."""
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        return {
            "train": list(dataset["train"]["text"]),
            "validation": list(dataset["validation"]["text"]),
            "test": list(dataset["test"]["text"]),
        }
    except Exception:
        return _fallback_wikitext()


def build_or_load_vocab_and_ids(
    data_dir: str,
    vocab_size: int,
    tokenizer: SimpleTokenizer,
    supplement_texts: list[list[str]] | None = None,
    force: bool = False,
) -> tuple[Vocabulary, dict[str, np.ndarray], list[list[str]]]:
    """Build/load vocabulary and token-id arrays for LM training.

    Args:
        data_dir: Directory for processed data cache.
        vocab_size: Maximum vocabulary size.
        tokenizer: Tokenizer instance.
        supplement_texts: Extra tokenized sequences (e.g. from SNIPS) added to
            the vocab AFTER the main WikiText-2 build.  They do NOT affect the
            token-id .npy arrays, only the vocabulary coverage.
        force: If True, delete cached vocab/ids and rebuild from scratch.
    """
    processed_dir = Path(data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = processed_dir / "vocab.json"
    split_paths = {
        "train": processed_dir / "train_ids.npy",
        "validation": processed_dir / "val_ids.npy",
        "test": processed_dir / "test_ids.npy",
    }

    if force:
        for p in [vocab_path, *split_paths.values()]:
            p.unlink(missing_ok=True)

    if vocab_path.exists() and all(path.exists() for path in split_paths.values()):
        vocab = Vocabulary.load(str(vocab_path))
        # If supplement tokens were requested, augment in-place (no rebuild needed)
        if supplement_texts:
            vocab.build(supplement_texts)
            vocab.save(str(vocab_path))
        split_ids = {name: np.load(path) for name, path in split_paths.items()}
        train_token_sequences = _reconstruct_sequences_from_ids(split_ids["train"], vocab)
        return vocab, split_ids, train_token_sequences

    raw_splits = load_wikitext_splits()
    token_splits: dict[str, list[list[str]]] = {}
    for split_name, lines in raw_splits.items():
        tokenized: list[list[str]] = []
        for line in lines:
            if not tokenizer.is_valid_wikitext_line(line):
                continue
            tokens = tokenizer.tokenize(line)
            if tokens:
                tokenized.append(tokens)
        token_splits[split_name] = tokenized

    vocab = Vocabulary(max_size=vocab_size)
    vocab.build(token_splits["train"])
    if supplement_texts:
        vocab.build(supplement_texts)
    vocab.save(str(vocab_path))

    split_ids: dict[str, np.ndarray] = {}
    for split_name, seqs in token_splits.items():
        merged_ids: list[int] = []
        for seq in seqs:
            merged_ids.extend(vocab.encode_tokens(seq, add_bos_eos=True))
        array = np.asarray(merged_ids, dtype=np.int64)
        split_ids[split_name] = array
        np.save(split_paths[split_name], array)
    return vocab, split_ids, token_splits["train"]


def _fallback_snips() -> dict[str, list[SNIPSExample]]:
    label_names = [
        "GetWeather",
        "BookRestaurant",
        "PlayMusic",
        "AddToPlaylist",
        "RateBook",
        "SearchCreativeWork",
        "SearchScreeningEvent",
    ]
    examples = [
        SNIPSExample(text="what is the weather today", label=0),
        SNIPSExample(text="book a table for two tonight", label=1),
        SNIPSExample(text="play some jazz music", label=2),
        SNIPSExample(text="add this song to my playlist", label=3),
        SNIPSExample(text="rate this book five stars", label=4),
        SNIPSExample(text="search creative work by pixar", label=5),
        SNIPSExample(text="find movie screenings nearby", label=6),
    ]
    return {"train": examples * 4, "validation": examples, "test": examples, "labels": label_names}


def _load_snips_label_names() -> list[str]:
    """Read label names from data/snips/labels.txt, falling back to hardcoded list."""
    labels_path = SNIPS_DATA_DIR / "labels.txt"
    if labels_path.exists():
        names = [ln.strip() for ln in labels_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if names:
            return names
    return [
        "AddToPlaylist", "BookRestaurant", "GetWeather",
        "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent",
    ]


def _load_snips_csv(split_file: Path) -> list[SNIPSExample]:
    """Load a SNIPS split from a local CSV file (columns: text, label)."""
    examples: list[SNIPSExample] = []
    with open(split_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            label = int(row["label"])
            examples.append(SNIPSExample(text=text, label=label))
    return examples


def _load_snips_from_local_csv() -> dict[str, list[SNIPSExample] | list[str]] | None:
    """Try loading all three SNIPS splits from local CSV files in data/snips/.

    Returns None if the directory or any required file is missing.
    """
    required = {
        "train": SNIPS_DATA_DIR / "train.csv",
        "validation": SNIPS_DATA_DIR / "val.csv",
        "test": SNIPS_DATA_DIR / "test.csv",
    }
    if not all(p.exists() for p in required.values()):
        return None
    return {
        "train": _load_snips_csv(required["train"]),
        "validation": _load_snips_csv(required["validation"]),
        "test": _load_snips_csv(required["test"]),
        "labels": _load_snips_label_names(),
    }


def load_snips_splits() -> dict[str, list[SNIPSExample] | list[str]]:
    """Load SNIPS splits: local CSV → HuggingFace → synthetic fallback."""
    # 1. Try local CSV files (fastest, no network)
    local = _load_snips_from_local_csv()
    if local is not None:
        return local

    # 2. Try HuggingFace (DeepPavlov/snips has 'utterance'+'label' columns)
    try:
        dataset = load_dataset("DeepPavlov/snips")
        label_names = _load_snips_label_names()

        def _parse_hf(split):
            examples = []
            for row in split:
                text = row.get("utterance", row.get("text", "")).strip()
                label_val = row.get("label", row.get("intent", 0))
                label = int(label_val) if isinstance(label_val, int) else 0
                examples.append(SNIPSExample(text=text, label=label))
            return examples

        train_split = dataset["train"]
        test_split = dataset["test"]
        # HF version has no validation split — use last 10 % of train
        cut = max(1, int(len(train_split) * 0.9))
        return {
            "train": _parse_hf(train_split.select(range(cut))),
            "validation": _parse_hf(train_split.select(range(cut, len(train_split)))),
            "test": _parse_hf(test_split),
            "labels": label_names,
        }
    except Exception:
        pass

    # 3. Tiny synthetic fallback
    return _fallback_snips()
