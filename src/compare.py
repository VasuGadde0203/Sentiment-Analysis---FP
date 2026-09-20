import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from src.config import REPORTS_DIR

RESULT_FILES = {
    "Custom LSTM": "custom_lstm_results.json",
    "AWD-LSTM (ULMFiT)": "ulmfit_results.json",
    "BERT (base-uncased)": "bert_results.json",
}


def _load_results():
    results = {}
    for name, filename in RESULT_FILES.items():
        path = os.path.join(REPORTS_DIR, filename)
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
        else:
            print(f"Skipping '{name}': {filename} not found (run its training script first).")
    return results


def build_comparison_table(results):
    rows = []
    for name, r in results.items():
        m = r["test_metrics"]
        rows.append(
            {
                "Model": name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
                "Best Epoch": r["best_epoch"],
                "Epochs Trained": r["epochs_trained"],
                "Training Time (s)": round(r["training_time_seconds"], 1),
            }
        )
    return pd.DataFrame(rows)


def plot_loss_curves(results, out_path):
    plt.figure(figsize=(8, 5))
    for name, r in results.items():
        epochs = range(1, len(r["history"]["train_loss"]) + 1)
        plt.plot(epochs, r["history"]["train_loss"], marker="o", label=f"{name} - train")
        plt.plot(epochs, r["history"]["val_loss"], marker="x", linestyle="--", label=f"{name} - val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss per Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metrics_bar(table, out_path):
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x = range(len(table))
    width = 0.2

    plt.figure(figsize=(8, 5))
    for i, metric in enumerate(metrics):
        offsets = [pos + i * width for pos in x]
        plt.bar(offsets, table[metric], width=width, label=metric)

    plt.xticks([pos + width * 1.5 for pos in x], table["Model"], rotation=15)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Test Set Metrics Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    return "\n".join([header, separator, *rows])


def compare():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    results = _load_results()

    if not results:
        print("No results found. Run train_custom.py, ulmfit_train.py and/or bert_train.py first.")
        return None

    table = build_comparison_table(results)

    csv_path = os.path.join(REPORTS_DIR, "comparison_table.csv")
    md_path = os.path.join(REPORTS_DIR, "comparison_table.md")
    table.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(_to_markdown(table))

    plot_loss_curves(results, os.path.join(REPORTS_DIR, "loss_curves.png"))
    plot_metrics_bar(table, os.path.join(REPORTS_DIR, "metrics_comparison.png"))

    print(table.to_string(index=False))
    print(f"\nSaved: {csv_path}, {md_path}")
    print(f"Saved plots: loss_curves.png, metrics_comparison.png (in {REPORTS_DIR})")
    return table


if __name__ == "__main__":
    compare()
