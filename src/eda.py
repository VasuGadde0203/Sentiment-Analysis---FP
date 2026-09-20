"""Exploratory data analysis on the IMDB dataset used by every model.

Not required by the GUVI brief, but useful groundwork before training: it
confirms the dataset is what the code assumes it is (balanced labels, review
length distribution, most-common words per class) and produces plots for the
project report. Run with `python -m src.eda`.
"""
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from src.config import DATA_CSV_PATH, REPORTS_DIR
from src.data_loader import clean_text


def load_clean_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV_PATH)
    df["text"] = df["review"].astype(str).apply(clean_text)
    n_dupes = df["text"].duplicated().sum()
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Rows: {len(df)} (dropped {n_dupes} exact-duplicate reviews)")
    print(f"Missing values:\n{df[['review', 'sentiment']].isnull().sum()}")
    return df


def run():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df = load_clean_df()
    df["word_count"] = df["text"].str.split().apply(len)

    print("\nLabel distribution:")
    print(df["sentiment"].value_counts())
    print("\nReview length (words) by class:")
    print(df.groupby("sentiment")["word_count"].describe())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    df["sentiment"].value_counts().plot(
        kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"]
    )
    axes[0].set_title("Sentiment label distribution")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    for label, color in [("positive", "#4C72B0"), ("negative", "#DD8452")]:
        subset = df.loc[df["sentiment"] == label, "word_count"]
        axes[1].hist(subset.clip(upper=800), bins=40, alpha=0.6, label=label, color=color)
    axes[1].set_title("Review length (words, clipped at 800)")
    axes[1].set_xlabel("Word count")
    axes[1].set_ylabel("Number of reviews")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(REPORTS_DIR, "eda_overview.png")
    plt.savefig(out_path)
    plt.close()
    print(f"\nSaved: {out_path}")

    for label in ["positive", "negative"]:
        counter = Counter()
        for text in df.loc[df["sentiment"] == label, "text"]:
            counter.update(text.lower().split())
        top20 = counter.most_common(20)
        print(f"\nTop 20 words in {label} reviews (no stopword removal -- "
              f"dominated by common function words, which is expected):")
        print(top20)


if __name__ == "__main__":
    run()
