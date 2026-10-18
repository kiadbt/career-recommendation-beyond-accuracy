"""STEP baseline classes; extracted unchanged. See THIRD_PARTY_STEP_LICENSE.txt.
Source revision: 158e2e81def9aa5874e3fc76a9e660ed3d2a90e9.
"""
import torch
from torch import nn
from torch.nn import functional as F

class MLPTransformation(nn.Module):
    """MLP with residual connection for career path prediction."""
    def __init__(self, input_size=768, hidden_size=512, output_size=768, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.temperature = nn.Parameter(torch.tensor(5.0))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        identity = x
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = out + identity
        out = F.normalize(out, p=2, dim=-1)
        return out

    def compute_similarity(self, embeddings, esco_embeddings):
        return embeddings @ esco_embeddings.T * self.temperature

class CareerBiLSTM(nn.Module):
    """Bidirectional LSTM for career path prediction."""
    def __init__(self, input_size=768, hidden_size=16, num_classes=100, dropout=0.1):
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        x = self.input_dropout(x)
        lengths_clamped = lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_clamped, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        combined = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.classifier(combined)
