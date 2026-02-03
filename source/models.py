import torch
from torch import nn

class SimpleFlow(nn.Module):
    def __init__(self, dim: int = 2, hidden: int = 64):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(dim + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, dim))

    def forward(self, x_t, t):
        return self.model(torch.cat([x_t, t], dim=1))