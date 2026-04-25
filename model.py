"""Model definitions for self-pruning neural network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """Linear layer with a learnable gate per weight."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.gate_scores, mean=0.0, std=1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


class PrunableMLP(nn.Module):
    """Simple MLP for CIFAR-10 with only PrunableLinear layers."""

    def __init__(self, input_dim: int = 3 * 32 * 32, hidden_dims: tuple = (1024, 512), num_classes: int = 10) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(input_dim, h1)
        self.fc2 = PrunableLinear(h1, h2)
        self.fc3 = PrunableLinear(h2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
