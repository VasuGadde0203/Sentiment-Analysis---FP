# GUVI Evaluation Prep — Sentiment Analysis (LSTM vs. ULMFiT vs. BERT)

This document is your speaking notes for the evaluation. It explains the
project in plain language, walks through the code an evaluator is likely to
ask about, and gives you rehearsed answers to likely questions.

> **Note on numbers:** result values referenced below as "TBD" will be filled
> in once the full-scale (10k train / 1k val / 2k test) training run
> finishes — see `reports/comparison_table.md` for the final numbers once it
> completes. The pipeline itself (data loading → training → evaluation →
> saving → comparison) has been run and verified end-to-end at small scale
> for all three models, so the code path is proven correct; only the
> absolute metric values depend on the full run.

## 1. Project Overview

**Problem:** classify a movie review as positive or negative sentiment.

**Why it matters (business use cases):** the same binary text-classification
pattern applies to product review analysis, social media opinion mining,
brand monitoring, e-commerce feedback, and call-center transcript analysis —
IMDb reviews are just a large, clean, publicly available stand-in dataset.

**The actual point of the project** isn't just "classify sentiment" — it's a
**controlled comparison of three ways to get there**:

1. Train a neural network (LSTM) completely from scratch, with no prior
   knowledge of English.
2. Take a language model that already learned English from a huge unlabeled
   corpus (AWD-LSTM, via ULMFiT) and fine-tune it for this one task.
3. Take a transformer that already learned English a different way (BERT)
   and fine-tune that instead.

The interesting result isn't "which model wins" — it's *why*: transfer
learning (2 and 3) starts with useful language knowledge already baked in,
so it should reach good accuracy in far fewer epochs than training from
scratch (1), and the transformer's self-attention should handle long-range
context in a review better than an LSTM's sequential processing.

**Users:** nobody-facing directly — this is a research/benchmarking
deliverable (source code + trained models + a comparison report), not a
deployed app.

## 2. Architecture and Data Flow

There's no web server, database, or API here — it's an ML training
pipeline. The "architecture" is really a data pipeline with three parallel
branches that reconverge for comparison:

```
data/IMDB Dataset.csv (50k reviews: review, sentiment)
        |
src/data_loader.py: get_splits()
  - cleans HTML (<br />) out of the text
  - shuffles with a fixed seed (42) for reproducibility
  - splits into train / val / test (10k / 1k / 2k by default)
        |
        +--------------------+--------------------+
        |                    |                    |
   Custom LSTM           AWD-LSTM (ULMFiT)       BERT
 (src/train_custom.py)  (src/ulmfit_train.py)  (src/bert_train.py)
        |                    |                    |
   models/custom_lstm.pt  models/ulmfit_model.pkl  models/bert/
   reports/custom_lstm_    reports/ulmfit_          reports/bert_
     results.json            results.json            results.json
        |                    |                    |
        +--------------------+--------------------+
                             |
                    src/compare.py
                             |
        reports/comparison_table.csv, loss_curves.png,
                metrics_comparison.png
```

Every branch is evaluated the same way: same test set, same sklearn metric
functions (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`),
same JSON result schema — that's what makes the final comparison fair rather
than apples-to-oranges.

## 3. Feature-by-Feature Explanation

### 3.1 Data loading & splitting (`src/data_loader.py`)

- **What it does:** reads the Kaggle IMDB CSV, strips `<br />` HTML tags
  (the raw reviews are full of them — left in, they'd pollute the
  vocabulary with a meaningless high-frequency "token"), maps
  `positive`/`negative` to `1`/`0`, shuffles with a fixed seed, and slices
  out train/val/test.
- **Why it's centralized:** all three models call `get_splits()` (or
  `load_imdb()`, a thin wrapper that returns a HuggingFace-style
  `DatasetDict` for the parts of the code that expect that interface). One
  function owning the split guarantees all three models are trained and
  tested on **exactly the same rows** — otherwise a difference in accuracy
  could just be a difference in what data each model happened to see.

### 3.2 Custom LSTM (`src/custom_lstm.py`, `src/preprocess.py`, `src/train_custom.py`)

- **What it does:** a from-scratch vocabulary (top 20,000 words by
  frequency, whitespace tokenized) feeds a trainable embedding layer, a
  2-layer LSTM, and a linear + sigmoid output for binary classification.
- **Why this architecture:** it's the deliberate "no prior knowledge"
  baseline the whole comparison is built around — every parameter,
  including the word embeddings, starts randomly initialized and is
  learned purely from the labeled training set.
- **What happens internally during training:** each epoch runs a full pass
  over the training batches (forward pass → BCE loss → backward pass →
  Adam step), then evaluates on the held-out validation set. The checkpoint
  with the **lowest validation loss** (not just the last epoch) is the one
  saved to `models/custom_lstm.pt` and the one used for final test
  evaluation — this avoids reporting an overfit late epoch.

### 3.3 AWD-LSTM / ULMFiT (`src/ulmfit_train.py`)

- **What it does:** loads a language model (AWD-LSTM) that fastai
  pretrained on a large general-English corpus (WikiText-103), replaces its
  output head with a sentiment classifier head, and fine-tunes it on the
  IMDb training split using the ULMFiT recipe: **1 epoch with the
  pretrained backbone frozen** (so only the new classifier head adapts
  first, without corrupting the pretrained weights with large early
  gradients), then **unfrozen fine-tuning** for the remaining epochs with a
  lower, discriminative learning rate.
- **Why manually replaying `fine_tune()`'s two stages** instead of calling
  fastai's `learn.fine_tune(epochs)` directly: fastai's `Recorder` callback
  resets its per-epoch history on every internal `fit()` call, and
  `fine_tune()` makes two such calls internally (frozen, then unfrozen). A
  single `fine_tune()` call would silently **lose the frozen warm-up
  epoch's loss/accuracy row** from the recorded history. Calling
  `learn.freeze()` + `fit_one_cycle(...)` and then `learn.unfreeze()` +
  `fit_one_cycle(...)` separately — with fastai's own default arguments —
  produces an identical training run while letting us capture both stages'
  history for the comparison report.

### 3.4 BERT (`src/bert_train.py`)

- **What it does:** tokenizes reviews with BERT's WordPiece tokenizer
  (adding `[CLS]`/`[SEP]`, truncating/padding to a fixed length, building
  an attention mask so padding is ignored), then fine-tunes
  `bert-base-uncased` with a classification head on top using a manual
  PyTorch training loop (not HuggingFace's `Trainer`, to avoid an extra
  `accelerate` dependency and to keep the training loop's structure — and
  its per-epoch loss tracking — consistent with the custom LSTM's loop).
- **Why a manual loop mirrors the custom LSTM's structure:** so the
  "training time" and "loss per epoch" numbers in the comparison are
  measuring the same thing across models, not an artifact of a
  higher-level training framework doing something different under the
  hood.

### 3.5 Comparison report (`src/compare.py`)

- Loads whichever of the three `reports/*_results.json` files exist,
  builds a table (Accuracy/Precision/Recall/F1/best epoch/training time),
  and renders two plots: overlaid train/val loss curves (to visually
  compare convergence speed) and a grouped bar chart of test metrics.

## 4. Important Code an Evaluator Might Question

| File | What to explain if asked |
|---|---|
| [src/data_loader.py](src/data_loader.py) `get_splits()` | Why one function is the single source of truth for the data split across all three models (fair comparison), and why it's seeded. |
| [src/custom_lstm.py](src/custom_lstm.py) `SentimentLSTM` | Why `padding_idx=0` on the embedding (so the `<pad>` token doesn't get a trainable, meaningless embedding that could leak into the hidden state). |
| [src/train_custom.py](src/train_custom.py) `train()` | Why checkpointing is based on *validation* loss, not training loss or the last epoch (avoids reporting an overfit model). |
| [src/ulmfit_train.py](src/ulmfit_train.py) `train_ulmfit()` | The freeze → fine-tune-head → unfreeze → fine-tune-everything sequence, and why it's replicated manually instead of calling `fine_tune()` in one line. |
| [src/bert_train.py](src/bert_train.py) `IMDBBertDataset` / `_run_epoch()` | How attention masks handle variable-length reviews within a fixed `max_len`, and why the same `_run_epoch()` function (with `optimizer=None`) is reused for both training and evaluation instead of duplicating the loop. |
| [src/evaluate.py](src/evaluate.py) `evaluate()` | Why one shared function computes metrics for the custom LSTM, used identically at both validation time (with loss) and test time. |
| [src/compare.py](src/compare.py) | How it stays correct even if you've only trained 1 or 2 of the 3 models (skips missing ones rather than crashing). |

## 5. Database

**Not applicable.** This project has no database — the dataset is a static
CSV file, and results are persisted as JSON files (`reports/`) and model
checkpoints (`models/`). If asked "why no database," the honest answer is
that the project's inputs and outputs are inherently file-based (a fixed
research dataset in, trained model files + a metrics report out) — a
database would add complexity with no corresponding requirement.

## 6. API

**Not applicable.** This is an offline training/evaluation pipeline, not a
served application — there's no inference API. (If asked "how would you
serve this," a reasonable answer: wrap the saved model + tokenizer/vocab in
a small FastAPI/Flask endpoint that loads the checkpoint once at startup and
exposes a `/predict` route — straightforward because all three training
scripts already save everything needed to reload a model for inference.)

## 7. Security

**Not applicable in the traditional sense** — there's no authentication,
user input, or network-facing surface. The only "security-adjacent"
practice actually present: fixed random seeds and pinned dependency
versions for reproducibility, and no secrets/credentials anywhere in the
codebase (the HuggingFace Hub download for BERT weights is anonymous/public).

## 8. Challenges and Decisions

**1. `fastai`'s latest version couldn't even be imported.**
Problem: `pip install fastai` (unpinned) resolved to fastai 2.8.8, which
requires `fastcore>=1.14.6`. The newest available fastcore is 2.x, which
internally switched to the `plum-dispatch` library for multiple dispatch.
`fastai`'s doc-generation decorator (`@docs`) tries to set `.__doc__` on
dispatched methods, but `plum`'s `Function` objects don't allow that —
`from fastai.text.all import *` crashed on import with
`AttributeError: 'Function' object attribute '__doc__' is read-only`,
before a single line of my code even ran.
Why it happened: a genuine upstream version-compatibility gap between two
libraries released asynchronously.
How it was solved: downgraded to `fastai==2.7.19`, which pins
`fastcore<1.8,>=1.5.29` — a combination confirmed to import and run
correctly. Also had to align `torch`/`torchvision`/`torchaudio` to versions
that are mutually ABI-compatible with that fastai version.
Why this is the right fix: pinning to a known-compatible release triple is
the standard, low-risk fix for this exact class of problem, versus e.g.
monkeypatching fastcore's internals (fragile) or downgrading only fastcore
without also satisfying fastai's minimum-version floor (impossible — no
fastcore 1.x release satisfies `>=1.14.6`, so the "floor" itself only
resolves to a broken 2.x release; pinning fastai down is what actually
fixes it).

**2. `ulmfit_train.py`'s original `fine_tune()` call would have silently
under-reported metrics.**
Problem: fastai's `Recorder.before_fit()` resets `self.values = []` on
*every* `.fit()` call. `Learner.fine_tune()` calls `fit_one_cycle()` twice
internally (frozen warm-up, then unfrozen). A single `fine_tune(N)` call
would leave `learn.recorder.values` holding only the last `fit_one_cycle`'s
rows — the frozen epoch's train/val loss would be silently dropped from any
history/plot built from it.
How it was solved: replicated `fine_tune()`'s two `fit_one_cycle` calls
manually (with its own default hyperparameters), capturing
`learn.recorder.values` after each call and concatenating them.
Why this matters for the evaluation deliverable: "epochs to convergence" is
one of the project's required evaluation metrics — silently losing an
epoch's data would misreport it.

**3. Custom LSTM's `evaluate()` had a runtime bug (missing `import torch`)
that would crash as soon as it was called.**
Found while reading the original code as part of the audit — `evaluate()`
used `torch.no_grad()` without importing `torch`. Fixed by adding the
import and, while touching the function, extending it to optionally compute
loss (needed for per-epoch validation-loss tracking) and return a metrics
dict instead of only printing.

**4. Training throughput on this machine was wildly inconsistent — same
code, sometimes 20x slower.**
Problem: identical batches of 64 examples took ~1 second in one run and
~30 seconds in another. This is a shared, multi-tenant machine — other
concurrent, unrelated Python/ML sessions were competing for CPU (confirmed
via `Get-Process`/`tasklist` showing other project processes and elevated
overall CPU utilization when it was slow, and zero contention when it was
fast). Not a bug in this codebase.
How it was handled: (a) added `--train_size`/`--val_size`/`--test_size`/
`--epochs` CLI overrides to every training script so correctness can be
smoke-tested on a tiny slice in seconds regardless of machine load, and
(b) ran the agreed 10k/2k-scale training as a true background process with
no artificial timeout, rather than assuming a fixed wall-clock budget.

## 9. Likely Evaluator Questions

**Q1: Why compare an LSTM trained from scratch against two pretrained
models — isn't that an unfair fight?**
A: That's the point of the project. It's not meant to be a fair fight — it's
meant to *measure* the size of the transfer-learning advantage. The
custom LSTM is the "no prior knowledge" control group.

**Q2: Why IMDb specifically?**
A: It's a well-established, clean, binary-labeled sentiment benchmark
(50k reviews) that's large enough to train a model from scratch on, small
enough to fine-tune pretrained models on quickly, and standard enough that
results are comparable to published baselines.

**Q3: Why PyTorch for the custom LSTM but fastai for AWD-LSTM?**
A: fastai *is* PyTorch underneath — it's the library that ships the
pretrained AWD-LSTM weights and implements the ULMFiT fine-tuning recipe
(discriminative learning rates, gradual unfreezing) as documented, tested
building blocks. Reimplementing that from scratch would be reinventing a
well-established recipe, not adding rigor.

**Q4: Explain your architecture end-to-end.**
A: (walk through the data-flow diagram in section 2 — one CSV, one shared
split function, three independent training scripts producing a model
checkpoint and a results JSON each, then one comparison script that reads
all of them.)

**Q5: How do you make sure the three models are compared fairly?**
A: Same `get_splits()` function (same seed, same row indices) for train/
val/test across all three; same sklearn metric functions computed the same
way on the same held-out test set; results serialized to the same JSON
schema so `compare.py` treats them uniformly.

**Q6: What's the difference between your validation set and test set?**
A: Validation is used *during* training — every epoch, to track
convergence and pick the best checkpoint. Test is touched exactly once, at
the very end, only by the best checkpoint, purely for the final reported
metrics. This prevents the reported numbers from being inflated by
(indirectly) tuning on the test set.

**Q7: Why track validation loss instead of validation accuracy for
checkpoint selection?**
A: Loss is a continuous, more sensitive signal than accuracy (which can
plateau in coarse steps as predictions cross the 0.5 threshold), so it
gives a more reliable "best epoch" signal, especially early in training.

**Q8: What is ULMFiT, in plain terms?**
A: A recipe for fine-tuning a language model for a new text task: first
train the model's own next-word-prediction ability further on the target
domain if needed, replace the output layer with a classifier, freeze
everything except the new layer and train it briefly so it doesn't get
knocked around by large early gradients, then unfreeze the whole model and
keep fine-tuning at a lower learning rate.

**Q9: What is AWD-LSTM?**
A: An LSTM language model architecture with several specific regularization
techniques (weight-dropped LSTM, embedding dropout, weight tying) designed
to train language models well without overfitting — it's the backbone
architecture ULMFiT was originally built and demonstrated on.

**Q10: Why BERT instead of, say, GPT for the transformer extension?**
A: The project brief specifically asks for BERT, and BERT (bidirectional,
encoder-only) is the natural fit for a classification task — it looks at
the whole sequence at once, unlike an autoregressive decoder-only model
built for text generation.

**Q11: How does BERT's tokenizer differ from your custom LSTM's tokenizer?**
A: The custom LSTM uses a simple whitespace/lowercase tokenizer with a
20,000-word vocabulary built only from the training set — anything unseen
becomes `<unk>`. BERT uses WordPiece subword tokenization with a large,
fixed, pretrained vocabulary — it can represent virtually any word by
breaking it into known sub-word pieces, so it never has an true
out-of-vocabulary problem the same way.

**Q12: What happens if the input review is longer than your max sequence
length?**
A: It's truncated. The custom LSTM truncates (and pads) to `MAX_SEQ_LEN`
(300 tokens); BERT truncates (and pads) to `BERT_MAX_LEN` (256 tokens,
within BERT's 512-token architectural limit).

**Q13: What happens with an empty or missing review?**
A: The tokenizer/CSV loading doesn't special-case empty strings — an empty
review tokenizes to an empty list, which the model still processes as a
zero-length (fully padded) sequence. This dataset doesn't contain missing
reviews (the Kaggle CSV is complete), so this wasn't hardened further; for
a production inference API this is exactly the kind of input validation
you'd add at the API boundary rather than inside the model.

**Q14: How do you know the model isn't just memorizing the training set?**
A: The test set is never touched during training or checkpoint selection,
and validation loss (not training loss) drives which checkpoint is kept —
if the model were purely memorizing, validation loss would rise even as
training loss kept falling, which is exactly the checkpointing logic guards
against.

**Q15: What's precision/recall/F1 measuring here, concretely?**
A: Treating "positive" as the positive class: precision is "of the reviews
I called positive, how many actually were," recall is "of the actually
positive reviews, how many did I catch," and F1 is their harmonic mean —
useful because accuracy alone can hide an imbalance between the two error
types.

**Q16: How would you scale this to a much larger dataset or to production?**
A: Move training to GPU (all three scripts already use `DEVICE =
torch.device("cuda" if available else "cpu")`, so no code change is needed,
just hardware); for BERT specifically, use mixed-precision training and a
larger batch size to use a GPU efficiently; for serving, export the best
checkpoint and wrap it in a lightweight inference API.

**Q17: What would you improve with more time/compute?**
A: Run the full 50k-review dataset (currently trained on a 10k/1k/2k subset
for iteration speed on a CPU-only machine — `FULL_DATASET = True` in
`config.py` switches to it), a small hyperparameter sweep (learning rate,
dropout) per model, and possibly a subword tokenizer for the custom LSTM
too, to reduce its out-of-vocabulary rate.

**Q18: Why did you choose a 2-layer LSTM with 256 hidden units for the
custom model specifically?**
A: A modest, commonly-used starting configuration for binary text
classification on a dataset this size — large enough to have real
capacity, small enough to train in reasonable time on CPU without
immediately overfitting a 10k-example training set.

**Q19: What does `padding_idx=0` actually do and why does it matter?**
A: It tells the embedding layer to keep index 0's (the `<pad>` token's)
embedding vector fixed at zero and to never receive a gradient — without
it, the padding token would get its own trainable embedding, which is
meaningless (padding carries no information) and could add noise to the
LSTM's hidden state, especially for shorter reviews with more padding.

**Q20: How is reproducibility ensured?**
A: `src/config.py: set_seed()` seeds Python's `random`, NumPy, and PyTorch
(including CUDA) and is called at the start of every training script,
before the data split, model initialization, or training loop — so the
same run with the same config produces the same split and the same
initial weights.

**Q21: Why is the comparison table/plot generation a separate script
instead of built into each training script?**
A: Separation of concerns — `compare.py` doesn't need to know *how* a model
was trained, only that a `reports/<model>_results.json` file with a known
schema exists. It also means the comparison can be regenerated any time
(e.g., after only 1 of the 3 models has been trained) without retraining
anything.

**Q22: What's the biggest limitation of this project as it stands?**
A: Compute — the "full-scale" 10k/2k run (and especially the true 50k
full-dataset run) is expensive to run repeatedly on CPU, particularly for
BERT. That's a hardware constraint, not a methodology gap: the code,
metrics, and comparison logic are already correct and complete; running
them longer just produces more statistically reliable numbers on top of
the same pipeline.
