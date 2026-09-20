import argparse
import json
import os
import time

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    BERT_BATCH_SIZE,
    BERT_EPOCHS,
    BERT_LR,
    BERT_MAX_LEN,
    BERT_MODEL_NAME,
    DEVICE,
    MODELS_DIR,
    REPORTS_DIR,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
    set_seed,
)
from src.data_loader import get_splits
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class IMDBBertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _run_epoch(model, dataloader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_preds.extend(outputs.logits.argmax(dim=1).detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    return metrics


def train_bert(
    train_size=TRAIN_SIZE,
    val_size=VAL_SIZE,
    test_size=TEST_SIZE,
    epochs=BERT_EPOCHS,
    max_len=BERT_MAX_LEN,
    batch_size=BERT_BATCH_SIZE,
):
    set_seed()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train_df, val_df, test_df = get_splits(train_size=train_size, val_size=val_size, test_size=test_size)

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=2).to(DEVICE)

    train_ds = IMDBBertDataset(train_df["text"], train_df["label"], tokenizer, max_len)
    val_ds = IMDBBertDataset(val_df["text"], val_df["label"], tokenizer, max_len)
    test_ds = IMDBBertDataset(test_df["text"], test_df["label"], tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=BERT_LR)

    bert_model_dir = os.path.join(MODELS_DIR, "bert")
    os.makedirs(bert_model_dir, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

    for epoch in range(epochs):
        train_metrics = _run_epoch(model, train_loader, DEVICE, optimizer=optimizer)
        val_metrics = _run_epoch(model, val_loader, DEVICE, optimizer=None)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            model.save_pretrained(bert_model_dir)
            tokenizer.save_pretrained(bert_model_dir)

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_metrics['loss']:.4f} "
            f"| Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
        )

    training_time = time.time() - start_time

    # Reload best checkpoint before final test evaluation.
    model = AutoModelForSequenceClassification.from_pretrained(bert_model_dir).to(DEVICE)
    test_metrics = _run_epoch(model, test_loader, DEVICE, optimizer=None)
    test_metrics.pop("loss")

    results = {
        "model": "bert_base_uncased",
        "epochs_trained": epochs,
        "best_epoch": best_epoch,
        "training_time_seconds": training_time,
        "history": history,
        "test_metrics": test_metrics,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }
    with open(os.path.join(REPORTS_DIR, "bert_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest epoch: {best_epoch} (val loss {best_val_loss:.4f})")
    print("Test metrics:", test_metrics)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--val_size", type=int, default=VAL_SIZE)
    parser.add_argument("--test_size", type=int, default=TEST_SIZE)
    parser.add_argument("--epochs", type=int, default=BERT_EPOCHS)
    parser.add_argument("--max_len", type=int, default=BERT_MAX_LEN)
    parser.add_argument("--batch_size", type=int, default=BERT_BATCH_SIZE)
    args = parser.parse_args()

    train_bert(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        epochs=args.epochs,
        max_len=args.max_len,
        batch_size=args.batch_size,
    )
