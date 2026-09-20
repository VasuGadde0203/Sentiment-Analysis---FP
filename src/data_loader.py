import re

import pandas as pd

from src.config import DATA_CSV_PATH, SEED, TRAIN_SIZE, TEST_SIZE, VAL_SIZE, FULL_DATASET

_BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Strip HTML line-break tags that pollute the vocabulary/tokenizer."""
    return _BR_TAG_RE.sub(" ", text)


def _load_raw_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV_PATH)
    df["text"] = df["review"].astype(str).apply(clean_text)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    return df[["text", "label"]]


def get_splits(
    full: bool = FULL_DATASET,
    train_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE,
    test_size: int = TEST_SIZE,
    seed: int = SEED,
):
    """
    Single source of truth for the train/val/test split so every model
    (custom LSTM, ULMFiT, BERT) is trained and evaluated on identical data.

    - train: used for gradient updates
    - val:   used for per-epoch validation loss / convergence tracking
    - test:  held out, used only for the final comparison metrics
    """
    df = _load_raw_dataframe()
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if full:
        n = len(df)
        test_size = int(n * 0.2)
        val_size = int(n * 0.1)
        train_size = n - test_size - val_size

    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:train_size + val_size].reset_index(drop=True)
    test_df = df.iloc[
        train_size + val_size: train_size + val_size + test_size
    ].reset_index(drop=True)

    return train_df, val_df, test_df


def load_imdb(
    full: bool = FULL_DATASET,
    train_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE,
    test_size: int = TEST_SIZE,
    seed: int = SEED,
):
    """Returns a HuggingFace-style DatasetDict with 'train'/'val'/'test' splits
    (columns: text, label) so existing preprocess/collate code keeps working."""
    from datasets import Dataset, DatasetDict

    train_df, val_df, test_df = get_splits(full, train_size, val_size, test_size, seed)
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "val": Dataset.from_pandas(val_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )
