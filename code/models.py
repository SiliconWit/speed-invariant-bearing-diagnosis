"""Encoder, projection head and classifier.

The encoder ends in global average pooling, which is the part that matters.
Pooling over the frequency axis makes the embedding insensitive to where a
pattern sits along that axis, and on a log axis a speed change moves patterns
along that axis without otherwise changing them.  The architecture is small on
purpose: the claim is about the representation, and a large encoder would let
capacity stand in for the argument.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, nbins=256, k=64, width=32):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 2
        self.body = nn.Sequential(
            nn.Conv1d(1, c1, 9, padding=4), nn.BatchNorm1d(c1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, 9, padding=4), nn.BatchNorm1d(c2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, 9, padding=4), nn.BatchNorm1d(c3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),            # translation insensitivity lives here
        )
        self.proj = nn.Linear(c3, k)

    def forward(self, x):
        h = self.body(x.unsqueeze(1)).flatten(1)
        z = self.proj(h)
        return F.normalize(z, dim=1)            # onto the unit sphere


class Net(nn.Module):
    """Encoder plus a linear head reading the same normalised embedding.

    Keeping the head on z rather than on an unnormalised feature means the
    margin geometry of Proposition 1 describes the space the classifier reads from.
    """
    def __init__(self, nbins=256, k=64, n_class=4, width=32):
        super().__init__()
        self.enc = Encoder(nbins, k, width)
        self.head = nn.Linear(k, n_class)

    def forward(self, x):
        z = self.enc(x)
        return self.head(z), z
