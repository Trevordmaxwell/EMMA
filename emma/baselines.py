from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUClassifier(nn.Module):
    """Simple GRU baseline for Needle: classify the value at the query step."""
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, num_values: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.head = nn.Linear(hid_dim, num_values)

    def forward(self, tokens: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        # tokens: (B, L)
        x = self.embed(tokens)
        out, h = self.gru(x)  # out: (B, L, H)
        B = tokens.size(0)
        idx = query_pos.view(B, 1, 1).expand(-1, 1, out.size(-1))  # (B,1,H)
        gathered = out.gather(1, idx).squeeze(1)  # (B, H)
        logits = self.head(gathered)
        return logits
