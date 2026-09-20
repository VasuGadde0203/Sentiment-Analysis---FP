from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from src.config import MAX_VOCAB_SIZE, MAX_SEQ_LEN


def tokenize(text):
    # Fast tokenizer
    return text.lower().split()


def build_vocab(dataset):
    counter = Counter()

    for text in dataset['train']['text']:
        tokens = text.lower().split()
        counter.update(tokens)

    vocab = {'<pad>': 0, '<unk>': 1}

    for word, _ in counter.most_common(MAX_VOCAB_SIZE - 2):
        vocab[word] = len(vocab)

    return vocab


def encode(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(token, vocab['<unk>']) for token in tokens[:MAX_SEQ_LEN]]


def collate_fn(batch, vocab):
    texts = [torch.tensor(encode(x['text'], vocab)) for x in batch]
    labels = torch.tensor([x['label'] for x in batch])

    texts = pad_sequence(texts, batch_first=True)
    return texts, labels