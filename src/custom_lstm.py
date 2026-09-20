import torch
import torch.nn as nn
from src.config import EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        
        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_DIM,
            NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT
        )
        
        self.fc = nn.Linear(HIDDEN_DIM, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return self.sigmoid(out).squeeze()