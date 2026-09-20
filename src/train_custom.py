import argparse
import json
import os
import pickle
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    LR,
    MODELS_DIR,
    REPORTS_DIR,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
    set_seed,
)
from src.custom_lstm import SentimentLSTM
from src.data_loader import load_imdb
from src.evaluate import evaluate
from src.preprocess import build_vocab, collate_fn


def train(train_size=TRAIN_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE, epochs=EPOCHS):
    set_seed()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    dataset = load_imdb(train_size=train_size, val_size=val_size, test_size=test_size)
    vocab = build_vocab(dataset)

    model = SentimentLSTM(len(vocab)).to(DEVICE)

    train_loader = DataLoader(
        dataset["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, vocab),
    )
    val_loader = DataLoader(
        dataset["val"],
        batch_size=BATCH_SIZE,
        collate_fn=lambda x: collate_fn(x, vocab),
    )
    test_loader = DataLoader(
        dataset["test"],
        batch_size=BATCH_SIZE,
        collate_fn=lambda x: collate_fn(x, vocab),
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for texts, labels in train_loader:
            texts, labels = texts.to(DEVICE), labels.float().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, DEVICE, criterion=criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "custom_lstm.pt"))

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
        )

    training_time = time.time() - start_time

    # Reload the best checkpoint (lowest val loss) before final test evaluation.
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "custom_lstm.pt")))
    test_metrics = evaluate(model, test_loader, DEVICE, criterion=criterion)

    with open(os.path.join(MODELS_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    results = {
        "model": "custom_lstm",
        "epochs_trained": epochs,
        "best_epoch": best_epoch,
        "training_time_seconds": training_time,
        "history": history,
        "test_metrics": test_metrics,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }
    with open(os.path.join(REPORTS_DIR, "custom_lstm_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest epoch: {best_epoch} (val loss {best_val_loss:.4f})")
    print("Test metrics:", test_metrics)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--val_size", type=int, default=VAL_SIZE)
    parser.add_argument("--test_size", type=int, default=TEST_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    train(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        epochs=args.epochs,
    )
