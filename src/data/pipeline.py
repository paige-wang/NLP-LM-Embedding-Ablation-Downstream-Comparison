"""Dataset loading and preprocessing helpers."""

from pathlib import Path

from datasets import load_dataset
import numpy as np

from src.data.downstream_dataset import SNIPSExample, parse_snips_split
from src.data.tokenizer import SimpleTokenizer
from src.data.vocabulary import Vocabulary


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
) -> tuple[Vocabulary, dict[str, np.ndarray], list[list[str]]]:
    """Build/load vocabulary and token-id arrays for LM training."""
    processed_dir = Path(data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = processed_dir / "vocab.json"
    split_paths = {
        "train": processed_dir / "train_ids.npy",
        "validation": processed_dir / "val_ids.npy",
        "test": processed_dir / "test_ids.npy",
    }

    if vocab_path.exists() and all(path.exists() for path in split_paths.values()):
        vocab = Vocabulary.load(str(vocab_path))
        split_ids = {name: np.load(path) for name, path in split_paths.items()}
        return vocab, split_ids, []

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


def load_snips_splits() -> dict[str, list[SNIPSExample] | list[str]]:
    """Load SNIPS splits with fallback."""
    try:
        dataset = load_dataset("DeepPavlov/snips")
        return {
            "train": parse_snips_split(dataset["train"]),
            "validation": parse_snips_split(dataset["validation"]),
            "test": parse_snips_split(dataset["test"]),
            "labels": list(dataset["train"].features["intent"].names),
        }
    except Exception:
        return _fallback_snips()
