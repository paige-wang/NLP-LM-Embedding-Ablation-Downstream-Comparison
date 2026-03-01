# Data Download Instructions

## Language Model Training: WikiText-2

WikiText-2 is loaded automatically via HuggingFace `datasets`:

```python
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
```

This will cache to `~/.cache/huggingface/datasets/` on first run.

## Downstream Task: SNIPS Intent Classification

SNIPS (7-class intent recognition) is loaded automatically via HuggingFace `datasets`:

```python
from datasets import load_dataset
ds = load_dataset("DeepPavlov/snips")
```

The 7 intent classes are:
1. GetWeather
2. BookRestaurant
3. PlayMusic
4. AddToPlaylist
5. RateBook
6. SearchCreativeWork
7. SearchScreeningEvent

## Pretrained Embeddings: GloVe.6B.300d

Download from: https://nlp.stanford.edu/projects/glove/

```bash
# Download and extract
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
cp glove.6B.300d.txt data/raw/glove.6B.300d.txt
```

Or manually download `glove.6B.zip` from the Stanford NLP page and extract
`glove.6B.300d.txt` to `data/raw/`.

## Processed Data

After running `scripts/train_lm.py` for the first time, the following files
will be created automatically in `data/processed/`:

- `vocab.json` — vocabulary (token → index mapping)
- `train_ids.npy` — token IDs for training split
- `val_ids.npy` — token IDs for validation split
- `test_ids.npy` — token IDs for test split
- `word2vec.bin` — Word2Vec vectors trained on WikiText-2

**Do not commit any files in `data/raw/` or `data/processed/`.**
