"""Load a trained model independently of its training script and predict
sentiment for new, unseen text.

Uses exactly the same tokenizer/vocab and preprocessing as training, since
it's built from the same `src.preprocess` functions and the vocab pickled
alongside the model checkpoint.

    python -m src.predict "This movie was absolutely wonderful."
"""
import argparse
import os
import pickle

import torch

from src.config import DEVICE, MODELS_DIR
from src.custom_lstm import SentimentLSTM
from src.preprocess import encode


def load_custom_lstm():
    with open(os.path.join(MODELS_DIR, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    model = SentimentLSTM(len(vocab)).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "custom_lstm.pt"), map_location=DEVICE)
    )
    model.eval()
    return model, vocab


def predict_custom_lstm(text: str, model=None, vocab=None):
    """Returns (label, positive_probability). Loads the saved model/vocab if
    not passed in, so this also works as a one-off call."""
    if model is None or vocab is None:
        model, vocab = load_custom_lstm()

    ids = encode(text, vocab)
    if not ids:  # empty/unknown-only text -> single <unk> token, avoids a 0-length tensor
        ids = [vocab["<unk>"]]
    tensor = torch.tensor(ids).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob_positive = model(tensor).item()

    label = "positive" if prob_positive > 0.5 else "negative"
    return label, prob_positive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment for new text using the trained custom LSTM.")
    parser.add_argument("text", type=str, help="Review text to classify.")
    args = parser.parse_args()

    model, vocab = load_custom_lstm()
    label, prob = predict_custom_lstm(args.text, model, vocab)
    print(f"Text: {args.text!r}")
    print(f"Prediction: {label}  (P(positive) = {prob:.4f})")
