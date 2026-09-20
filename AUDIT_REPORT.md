# Project Audit Report

**Scope:** Requirement-by-requirement audit of this repository against the
actual GUVI project brief (`Final Project document.docx`), performed by
inspecting every file in `src/`, `notebooks/`, `main.py`, and by **actually
running** the training pipeline in a clean environment (not just reading the
code) to distinguish verified behavior from unverified claims.

**Important correction to the original task framing:** the task instructions
this audit was requested under described a TF-IDF + classical-ML pipeline
(Naive Bayes / Logistic Regression / Linear SVC, 4,000/5,000/5,000 split).
That does not match this project. GUVI's actual document requires **three
deep-learning architectures** (a from-scratch custom LSTM, a pretrained
AWD-LSTM fine-tuned via ULMFiT, and fine-tuned BERT-base-uncased) compared
against each other on the IMDb 50k dataset — no TF-IDF or classical ML
anywhere in the brief. This audit is against the real GUVI requirements
(confirmed with the user before proceeding).

## Classification key

1. **Completed and correct** — inspected, and (where feasible in this
   environment) executed successfully.
2. **Implemented but requires correction** — present, but had a bug or
   methodological problem, now fixed (noted).
3. **Partially implemented** — covers part of the requirement.
4. **Missing** — not present.
5. **Unable to verify** — code looks correct on inspection but could not be
   executed here; distinguished explicitly from "verified."

---

## Part 1: Custom LSTM vs. AWD-LSTM (ULMFiT)

### 1. Load the IMDb movie reviews dataset (50,000 labeled samples)

**Status: Completed and correct.**

`src/data_loader.py` reads a Kaggle-format `data/IMDB Dataset.csv`
(`review`, `sentiment` columns). Verified independently by loading the CSV
directly: **50,000 rows, 2 columns (`review` str, `sentiment` str), zero
missing values, labels perfectly balanced at 25,000/25,000.** This matches
GUVI's stated dataset size exactly.

`data/` is (correctly) gitignored — the dataset is a large third-party
file that doesn't belong in the git history. For this audit, since
`huggingface.co` is not reachable from this sandboxed session's network
policy (see the BERT section below), I obtained the same 50k-row IMDb
dataset from a public GitHub mirror instead, and verified its shape/content
matched what the code expects.

**Found and fixed:** the CSV contains **418 exact-duplicate reviews** (a
known property of this dataset — some reviews were scraped twice). The
original `_load_raw_dataframe()` did not deduplicate before splitting,
which means a duplicate could land in *both* the train and test sets,
silently leaking a memorized example into "held-out" evaluation. Fixed by
adding `.drop_duplicates(subset=["text"])` before the shuffle/split in
`src/data_loader.py`. Post-dedup: 49,582 unique rows.

### 2. Preprocess text using tokenization and padding

**Status: Completed and correct.**

`src/preprocess.py`: whitespace + lowercase tokenization, a vocabulary
built **only from the training split** (`build_vocab(dataset['train'])`) —
correctly avoids fitting on validation/test data — capped at 20,000 words
by frequency, with `<pad>`/`<unk>` reserved. `encode()` truncates to
`MAX_SEQ_LEN=300`. `collate_fn` right-pads each batch to the longest
sequence in it via `pad_sequence`. `src/custom_lstm.py`'s embedding layer
uses `padding_idx=0` so the pad token never receives a gradient.

Deliberately **not** stripped: punctuation, casing, stopwords, negations.
This is the right call for a sequence model — "not good" vs. "good" differ
only by a word a stopword filter would often remove, and casing/punctuation
carry real sentiment signal (e.g. "AMAZING!!!" vs. "amazing"). The only
cleaning applied is stripping literal `<br />` HTML tags, which are scraper
artifacts, not content.

### 3. Build and train a custom LSTM model in PyTorch

**Status: Completed and correct — verified by actual execution.**

`src/custom_lstm.py`: embedding → 2-layer LSTM (hidden size 256, dropout
0.3) → linear → sigmoid, for binary classification. `src/train_custom.py`
runs the full train/val loop, checkpoints on **lowest validation loss**
(not last epoch, not training loss — avoids saving an overfit model), then
reloads that checkpoint for final test evaluation.

**Ran the real training job** on the agreed 3,500/750/750 split (see
Results below): 5 epochs, ~27 minutes wall-clock on this 4-core CPU
sandbox. Produced `models/custom_lstm.pt`, `models/vocab.pkl`,
`reports/custom_lstm_results.json`.

### 4. Fine-tune a pretrained AWD-LSTM model using ULMFiT methodology

**Status: Completed and correct — verified by actual execution.**

`src/ulmfit_train.py` uses fastai's `AWD_LSTM` + `text_classifier_learner`.
It deliberately does **not** call `learn.fine_tune(epochs)` in one line;
instead it replicates that method's own two-stage schedule manually
(`freeze()` → `fit_one_cycle(1, ...)` → `unfreeze()` →
`fit_one_cycle(epochs, ...)`), because fastai's `Recorder` resets its
per-epoch history on every internal `.fit()` call, and `fine_tune()` makes
two such calls — a single call would silently drop the frozen warm-up
epoch's loss/accuracy row from the history used to report "epochs to
convergence." I checked this claim against fastai's actual source
(`Learner.fine_tune`, `Recorder.before_fit`) — it's correct, not a
plausible-sounding justification.

**Ran the real fine-tuning job** on the same 3,500/750/750 split: this
downloaded the genuine pretrained AWD-LSTM (WikiText-103, `wt103-fwd`, ~105
MB) from fastai's S3 model zoo and fine-tuned it — 1 frozen + 4 unfrozen
epochs, ~13 minutes wall-clock. Produced `models/ulmfit_model.pkl` and
`reports/ulmfit_results.json`. (`s3.amazonaws.com` is reachable from this
sandbox's network policy, unlike `huggingface.co` — see the BERT section.)

### 5. Evaluate both models using identical metrics

**Status: Completed and correct.**

Both models are evaluated with the same `sklearn` functions
(`accuracy_score`, `precision_score`, `recall_score`, `f1_score`) on the
same test split, produced by the same seeded `get_splits()` function
(`src/data_loader.py`) — this is the single source of truth every script
imports, so a difference in reported accuracy can't be an artifact of the
two models secretly seeing different data.

### 6. Comparative analysis: accuracy, convergence, generalization

**Status: Completed and correct** for the two models actually run in this
environment.

`src/compare.py` builds a comparison table (Accuracy/Precision/Recall/F1/
Best Epoch/Epochs Trained/Training Time) and two plots (overlaid train/val
loss curves for convergence comparison, grouped bar chart of test metrics).
It gracefully skips any model whose `reports/*_results.json` doesn't exist
yet rather than crashing — verified this behavior directly, since BERT's
results file is absent in this environment (see below).

### 7. Evaluation metrics required: Accuracy, Precision, Recall, F1, train/val loss, epochs to convergence

**Status: Completed and correct.** All six are present in every results
JSON and the comparison table/plots. "Epochs to convergence" is represented
via `best_epoch` (lowest val loss) plus the full per-epoch loss history
rather than a single scalar — this is the more informative version of the
same requirement.

### 8. Deliverables: LSTM source, ULMFiT fine-tuning code, trained model files, evaluation/comparison report, documentation

**Status: Completed and correct** (BERT trained-model-file deliverable
excepted — see Part 2).

### 9. Guidelines: modular coding, reproducibility via fixed seeds, documented hyperparameters

**Status: Completed and correct.** Clean `src/` package (one concern per
file), `set_seed()` (seeds `random`, `numpy`, `torch`, `torch.cuda`) called
at the top of every training script, all hyperparameters centralized in
`src/config.py`, documented in `README.md`.

---

## Part 2: BERT extension

### 10–11. Same IMDb split; BERT WordPiece tokenization

**Status: Completed and correct in code; unable to verify by execution in
this environment (see limitation below).**

`src/bert_train.py` calls the same `get_splits()` used by the other two
models. `IMDBBertDataset` tokenizes with `AutoTokenizer` (WordPiece for
`bert-base-uncased`), truncating/padding to `BERT_MAX_LEN=256`, and builds
the attention mask so padding is ignored by self-attention.

### 12–13. Fine-tune BERT-base-uncased with a classification head, supervised training

**Status: Completed and correct in code; unable to verify by execution.**

`AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME,
num_labels=2)`, manual PyTorch training loop (not HF `Trainer`, deliberately
— documented reasoning: keeps the "training time"/"loss per epoch"
measurements structurally comparable to the custom LSTM's own manual loop,
and avoids an `accelerate` dependency). Checkpoints on best validation
loss, same as the custom LSTM.

### 14. Compare BERT with LSTM-based models

**Status: Partially implemented, by necessity.** `compare.py`'s logic is
ready and correct (see #6), but a fourth column for BERT can't be
populated in this environment. The comparison table produced here has 2 of
the 3 required models.

### 15. Analyze performance, computational cost, robustness

**Status: Partially implemented.** Training time is tracked and logged for
all three scripts by design; for BERT specifically, only *code-level* cost
characteristics can be reported here (parameter count, `BERT_BATCH_SIZE=8`
chosen deliberately small for CPU memory headroom, `BERT_MAX_LEN=256`), not
a measured wall-clock number, since the job never ran.

### 16. Same evaluation metrics

**Status: Completed and correct in code** (same `sklearn` functions,
same schema as the other two models' results JSON).

### 17. Deliverables: BERT fine-tuning code, trained BERT model, comparative analysis, performance plots, documentation

**Status: Implemented but requires correction; documented limitation, not
a code defect.**

- BERT fine-tuning code: **done, and structurally verified** (see below).
- Trained BERT model file: **missing** — cannot be produced in this
  environment.
- Comparative analysis / performance plots: **partial** — 2 of 3 models.
- Documentation: this report + README updates.

### 18. Guidelines: standard fine-tuning LR, experiment logs, ethical/reproducible practices

**Status: Completed and correct in code.** `BERT_LR=2e-5` is the standard
BERT fine-tuning learning rate from the original paper; `set_seed()` is
called; every epoch's loss/accuracy is logged to the results JSON.

---

## Hard environment limitation: BERT could not be executed here

This sandboxed session's network egress policy allows `pypi.org`,
`files.pythonhosted.org`, and GitHub, but **blocks `huggingface.co` and
`cdn-lfs.huggingface.co`** (confirmed via the proxy's own diagnostics —
`connect_rejected`, policy denial, not a transient failure). `transformers`
downloads both the `bert-base-uncased` tokenizer and weights from
`huggingface.co`, so `bert_train.py` cannot fetch them here. By contrast,
`s3.amazonaws.com` (where fastai's pretrained AWD-LSTM lives) **is**
reachable, which is why ULMFiT could be run for real but BERT could not.

**What I did instead, to avoid the dishonest options of either skipping
verification silently or fabricating results:** I built a tiny,
locally-authored BERT-architecture model (2 layers, hidden size 16, random
weights, a hand-written 56-token vocab) entirely offline — no network calls
— and pointed `bert_train.py` at it via `config.BERT_MODEL_NAME` (a runtime
override, not a code change). This ran the *actual, unmodified*
`train_bert()` function end-to-end: dataset construction, attention masks,
the training/eval loop, best-checkpoint selection, save/reload, and the
results JSON schema all executed without error. The accuracy/loss numbers
from that run are meaningless (random tiny model, random tiny vocab) and
were **deleted**, not kept — they must never be mistaken for real BERT
results. This confirms `bert_train.py` has no runtime bugs; it genuinely
just needs network access to `huggingface.co`, which your own machine (or
any environment with more permissive egress) will have.

**To get real BERT results:** run `python -m src.bert_train` (optionally
with `--train_size 3500 --val_size 750 --test_size 750` to match the other
two models) on a machine that can reach `huggingface.co` — your local
setup, Google Colab, or an environment here configured with a more
permissive network policy. It will download `bert-base-uncased` (~440 MB)
once, then fine-tune normally with no other changes needed. Then re-run
`python -m src.compare` to fold it into the comparison table/plots.

---

## Other findings

| Item | Classification | Detail |
|---|---|---|
| `notebooks/sentiment_analysis.ipynb` | Superseded, not deleted | Only implements the custom-LSTM piece, no val split, no ULMFiT/BERT, no model saving, uses HF's `imdb` dataset directly rather than the Kaggle CSV. Flagged with a note at the top pointing to `src/` as the real pipeline, rather than deleted — it's a legitimate record of the original exploration. |
| `requirements.txt: nltk` | Fixed | Listed but never imported anywhere in the codebase; removed. |
| EDA | Added (not required by GUVI, but useful) | `src/eda.py` — label balance, review-length distribution by class, top words per class. GUVI's document doesn't mandate EDA, but it's cheap groundwork and useful for the project report/viva. |
| Inference / prediction | Added (not required by GUVI, but useful) | `src/predict.py` — loads the saved custom-LSTM checkpoint + vocab independently of training, predicts on new text using identical preprocessing. GUVI's deliverables list doesn't include a serving/inference component, so this is a bonus, not a gap fix. |
| `EVALUATION_PREP.md` "TBD" results / "run and verified end-to-end" claim | Corrected | The results were placeholders and the "verified" claim could not be reproduced from the repo alone (no `models/`/`reports/` present, correctly gitignored). This audit re-ran everything and replaced placeholders with real measured numbers (see below), or explicitly marked BERT as not executed here. |
