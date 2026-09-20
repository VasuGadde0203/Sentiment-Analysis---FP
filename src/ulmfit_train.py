import argparse
import json
import os
import time

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from fastai.text.all import AWD_LSTM, TextDataLoaders, accuracy, text_classifier_learner

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    TEST_SIZE,
    TRAIN_SIZE,
    ULMFIT_DROP_MULT,
    ULMFIT_EPOCHS,
    VAL_SIZE,
    set_seed,
)
from src.data_loader import get_splits


def _sklearn_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def train_ulmfit(train_size=TRAIN_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE, epochs=ULMFIT_EPOCHS):
    set_seed()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Same train/val/test split used by the custom LSTM, for a fair comparison.
    train_df, val_df, test_df = get_splits(train_size=train_size, val_size=val_size, test_size=test_size)
    combined = pd.concat(
        [train_df.assign(is_valid=False), val_df.assign(is_valid=True)],
        ignore_index=True,
    )

    dls = TextDataLoaders.from_df(
        combined,
        text_col="text",
        label_col="label",
        valid_col="is_valid",
        seed=SEED,
        bs=32,  # lower than fastai's default 64 to reduce peak memory once unfrozen
        num_workers=0,  # avoid spawning many DataLoader worker subprocesses on Windows,
                        # which caused severe CPU oversubscription and an OOM-related crash
    )

    learn = text_classifier_learner(
        dls,
        AWD_LSTM,
        drop_mult=ULMFIT_DROP_MULT,
        metrics=accuracy,
    )

    # Replicate Learner.fine_tune's default schedule manually (1 frozen warm-up
    # epoch + N unfrozen epochs) so we can capture per-epoch history from both
    # stages -- learn.recorder.values is reset internally on every fit() call,
    # so a single fine_tune() call would silently drop the frozen epoch's row.
    start_time = time.time()
    learn.freeze()
    learn.fit_one_cycle(1, slice(2e-3), pct_start=0.99)
    recorded_values = list(learn.recorder.values)

    learn.unfreeze()
    learn.fit_one_cycle(epochs, slice(1e-3 / 100, 1e-3), pct_start=0.3, div=5.0)
    recorded_values += list(learn.recorder.values)
    training_time = time.time() - start_time

    # Each row is [train_loss, valid_loss, accuracy] for that epoch.
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    for row in recorded_values:
        history["train_loss"].append(float(row[0]))
        history["val_loss"].append(float(row[1]))
        history["val_accuracy"].append(float(row[2]))

    best_epoch = int(min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i])) + 1

    # Final held-out test evaluation using the same sklearn metrics as the other models.
    test_dl = learn.dls.test_dl(test_df["text"].tolist())
    preds, _ = learn.get_preds(dl=test_dl)
    pred_labels = preds.argmax(dim=1).numpy()
    test_metrics = _sklearn_metrics(test_df["label"].values, pred_labels)

    learn.export(os.path.join(MODELS_DIR, "ulmfit_model.pkl"))

    results = {
        "model": "ulmfit_awd_lstm",
        "epochs_trained": epochs + 1,  # +1 for the frozen warm-up epoch
        "best_epoch": best_epoch,
        "training_time_seconds": training_time,
        "history": history,
        "test_metrics": test_metrics,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }
    with open(os.path.join(REPORTS_DIR, "ulmfit_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest epoch: {best_epoch}")
    print("Test metrics:", test_metrics)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--val_size", type=int, default=VAL_SIZE)
    parser.add_argument("--test_size", type=int, default=TEST_SIZE)
    parser.add_argument("--epochs", type=int, default=ULMFIT_EPOCHS)
    args = parser.parse_args()

    train_ulmfit(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        epochs=args.epochs,
    )
