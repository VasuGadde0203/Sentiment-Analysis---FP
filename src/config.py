import os
import random

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "IMDB Dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Single source of truth for the development/testing dataset size. The
# official project requirement is the full 50,000-sample IMDb dataset
# (FULL_DATASET = True below); MAX_SAMPLES is a development-time knob to
# iterate faster on this CPU-only machine. Raise it back to 50000 (and set
# FULL_DATASET = True) to run the official full-scale experiment -- no other
# code changes are needed, every training script reads its split sizes from
# this one place.
MAX_SAMPLES = 5000
FULL_DATASET = False

# 70/15/15 split of MAX_SAMPLES, shared identically by all three models via
# src.data_loader.get_splits() so the comparison stays fair.
TRAIN_SIZE = int(MAX_SAMPLES * 0.70)
VAL_SIZE = int(MAX_SAMPLES * 0.15)
TEST_SIZE = MAX_SAMPLES - TRAIN_SIZE - VAL_SIZE  # remainder, avoids rounding drift

# Custom LSTM hyperparameters
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LEN = 300
LR = 1e-3
EPOCHS = 5

# ULMFiT (AWD-LSTM) hyperparameters
ULMFIT_EPOCHS = 4
ULMFIT_DROP_MULT = 0.5

# BERT hyperparameters
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LEN = 256
BERT_BATCH_SIZE = 8
BERT_LR = 2e-5
BERT_EPOCHS = 2
