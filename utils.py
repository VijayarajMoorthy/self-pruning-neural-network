"""Utility helpers for training, sparsity, evaluation, and reporting."""

import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

from model import PrunableLinear


def setup_logging(log_file: str) -> None:
    """Set up logger to print to console and file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="w")],
        force=True,
    )


def compute_sparsity_loss(model: torch.nn.Module) -> torch.Tensor:
    """Sum all gate values across all prunable layers."""
    device = next(model.parameters()).device
    loss = torch.tensor(0.0, device=device)
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            loss = loss + torch.sigmoid(module.gate_scores).sum()
    return loss


@torch.no_grad()
def compute_sparsity(model: torch.nn.Module, threshold: float = 1e-2) -> float:
    """Return percentage of gates below the provided threshold."""
    total = 0
    below = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            below += (gates < threshold).sum().item()
    return 100.0 * below / max(total, 1)


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str) -> float:
    """Evaluate model accuracy in percent."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def get_all_gate_values(model: torch.nn.Module) -> torch.Tensor:
    """Collect all gate values from all prunable layers into one tensor."""
    values = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            values.append(torch.sigmoid(module.gate_scores).flatten().cpu())
    return torch.cat(values)


def plot_gate_distribution(gate_values: torch.Tensor, out_path: str) -> None:
    """Save histogram of gate values."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(gate_values.numpy(), bins=50, edgecolor="black", alpha=0.8)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate_report(report_path: str, results: List[Dict], best_lambda: float) -> None:
    """Generate markdown report automatically from experiment results."""
    table_rows = "\n".join(
        f"| {result['lambda']:.4g} | {result['accuracy']:.2f} | {result['sparsity']:.2f} |" for result in results
    )

    observations = [
        "- As lambda increases, sparsity pressure becomes stronger and more gates move toward zero.",
        "- Lower lambda usually preserves model capacity and accuracy.",
        "- Higher lambda generally improves pruning but can reduce test accuracy.",
        f"- Best model in this run uses lambda={best_lambda:.4g}.",
    ]

    content = f"""# Self-Pruning Neural Network Report

## Why L1 on sigmoid gates induces sparsity

Each trainable weight is multiplied by a gate value: `gate = sigmoid(gate_scores)`, where each gate lies in `(0, 1)`.
The sparsity objective penalizes the sum of gate values:

`SparsityLoss = sum(sigmoid(gate_scores))`

Adding this term to classification loss encourages many gates to shrink toward zero.
When a gate becomes very small, the corresponding effective weight is nearly removed.
This makes pruning differentiable and trainable end-to-end with backpropagation.

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
|---:|---:|---:|
{table_rows}

## Observations

{chr(10).join(observations)}

## Histogram

The gate value histogram for the best model is saved at:
`outputs/plots/gate_distribution.png`
"""

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(content)
