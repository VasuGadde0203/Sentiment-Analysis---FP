# Sentiment Analysis: Custom LSTM vs. AWD-LSTM (ULMFiT) vs. BERT

GUVI Final Project + Extension. Trains and compares three architectures on the
IMDb movie review sentiment classification task (binary: positive/negative):

1. **Custom LSTM** — trained from scratch in PyTorch.
2. **AWD-LSTM (ULMFiT)** — pretrained language model fine-tuned with fastai.
3. **BERT (bert-base-uncased)** — pretrained transformer fine-tuned with HuggingFace `transformers`.

All three are trained and evaluated on **identical data splits** and **identical
metrics** (Accuracy, Precision, Recall, F1, train/val loss per epoch) so their
results are directly comparable.

## Project layout

```
data/IMDB Dataset.csv     # Kaggle IMDB dataset (50k reviews, review/sentiment columns)
src/
  config.py               # hyperparameters, paths, reproducibility (seed)
  data_loader.py           # single source of truth for the train/val/test split
  preprocess.py            # tokenizer, vocab, padding/collate for the custom LSTM
  custom_lstm.py            # SentimentLSTM model definition
  train_custom.py           # trains + evaluates the custom LSTM, saves model + results
  ulmfit_train.py            # fine-tunes AWD-LSTM via fastai, saves model + results
  bert_train.py               # fine-tunes bert-base-uncased, saves model + results
  evaluate.py                  # shared sklearn-metrics evaluation helper
  compare.py                    # builds the comparison table + plots from all results
  eda.py                         # dataset EDA: label balance, review-length distribution, top words
  predict.py                      # load a saved model + vocab and predict sentiment for new text
models/                    # saved model checkpoints (created on first run)
reports/                   # per-model results JSON + comparison table/plots (created on first run)
main.py                    # CLI entry point
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

**Environment note:** `requirements.txt` pins `fastai==2.7.19` with
`fastcore<1.8,>=1.5.29` and `torch==2.6.0`/`torchvision==0.21.0`/`torchaudio==2.6.0`.
The latest `fastai` (2.8.8) pulls in `fastcore>=2.x`, which switched to
`plum-dispatch` internally and breaks `fastai`'s doc-generation code at
import time (`AttributeError: 'Function' object attribute '__doc__' is
read-only`) — `from fastai.text.all import *` crashes immediately. These
pinned versions are a verified-working combination; do not upgrade `fastai`/
`fastcore` independently.

## Running

```bash
# Individually
python -m src.train_custom              # custom LSTM
python -m src.ulmfit_train               # AWD-LSTM / ULMFiT
python -m src.bert_train                  # BERT
python -m src.compare                      # comparison table + plots (reports/)

# Or via the CLI wrapper
python main.py lstm   # or: ulmfit | bert | compare | all
```

Each training script accepts `--train_size`, `--val_size`, `--test_size`
(and `--epochs`) to override the split sizes in `src/config.py` — useful for
a quick correctness smoke-test before committing to a full run, e.g.:

```bash
python -m src.train_custom --train_size 500 --val_size 100 --test_size 100 --epochs 2
```

## Data split

`src/data_loader.get_splits()` is the single source of truth for the train/
val/test split (fixed seed = 42), used identically by all three model
scripts. By default (`FULL_DATASET = False` in `config.py`, `MAX_SAMPLES =
5000`) it samples **3,500 train / 750 val / 750 test** reviews (a 70/15/15
split of 5,000) out of the ~49,582 available after deduplication — sized
for iterating on a CPU-only machine. Set `FULL_DATASET = True` to use a
70/10/20 split over the full deduplicated dataset instead (recommended on a
GPU — BERT fine-tuning in particular is expensive on CPU).

- **train**: gradient updates
- **val**: per-epoch loss/accuracy tracking (checkpoint selection is based on
  lowest val loss)
- **test**: held out entirely until the final evaluation reported in the
  comparison table

## Reproducibility

`src/config.py: set_seed()` seeds `random`, `numpy`, and `torch` (called at
the start of every training script) so results are reproducible given the
same data split and hyperparameters.

## Results

Run `python -m src.compare` after training the models you want compared. It
reads `reports/{custom_lstm,ulmfit,bert}_results.json` (skipping any that
don't exist yet) and writes:

- `reports/comparison_table.csv` / `.md` — Accuracy/Precision/Recall/F1,
  best epoch, training time
- `reports/loss_curves.png` — train/val loss per epoch, all models overlaid
- `reports/metrics_comparison.png` — bar chart of test-set metrics

### Verified results (3,500 train / 750 val / 750 test, seed 42)

Measured by actually running `train_custom.py` and `ulmfit_train.py` end to
end (see `AUDIT_REPORT.md` for full methodology notes):

| Model | Accuracy | Precision | Recall | F1 | Best epoch | Training time |
|---|---|---|---|---|---|---|
| Custom LSTM | 0.496 | 0.494 | 0.986 | 0.658 | 1 / 5 | ~27 min |
| AWD-LSTM (ULMFiT) | 0.855 | 0.844 | 0.864 | 0.854 | 5 / 5 | ~26 min |
| BERT (base-uncased) | not run | — | — | — | — | — |

The custom LSTM barely beats chance and overfits almost immediately (val
loss rises every epoch after epoch 1, while train loss keeps falling) —
3,500 examples is not enough to learn useful word representations from
scratch. ULMFiT, starting from AWD-LSTM weights pretrained on WikiText-103,
reaches 85.5% test accuracy with smoothly decreasing train *and* val loss.
This is the effect the project is designed to demonstrate: transfer
learning's advantage is largest precisely when labeled data is scarce.

**BERT was not trained in the sandboxed environment this audit ran in** —
its network policy blocks `huggingface.co`, which `bert_train.py` needs to
download `bert-base-uncased`. The code path was verified structurally
(swapped in a tiny offline random-weight BERT and confirmed the full
train/eval/checkpoint/save loop runs without error) but not with real
weights. Run `python -m src.bert_train --train_size 3500 --val_size 750
--test_size 750` on a machine with normal internet access to get real BERT
numbers, then re-run `python -m src.compare`.
