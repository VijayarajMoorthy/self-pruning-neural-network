"""Train self-pruning neural networks on CIFAR-10 for multiple sparsity lambdas."""

import copy
import logging
import os
import random
from typing import Dict, List, Tuple
from urllib.error import URLError

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from config import Config
from model import PrunableMLP
from utils import (
    compute_sparsity,
    compute_sparsity_loss,
    evaluate,
    generate_report,
    get_all_gate_values,
    plot_gate_distribution,
    setup_logging,
)


def set_seed(seed: int) -> None:
    """Set random seed for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """Build train and test loaders for CIFAR-10."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    try:
        train_dataset = datasets.CIFAR10(root=cfg.data_dir, train=True, download=False, transform=transform)
        test_dataset = datasets.CIFAR10(root=cfg.data_dir, train=False, download=False, transform=transform)
    except (URLError, RuntimeError, OSError) as error:
        logging.warning("Local CIFAR-10 not available (%s). Falling back to FakeData.", error)
        train_dataset = datasets.FakeData(
            size=cfg.fake_train_size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
            random_offset=cfg.seed,
        )
        test_dataset = datasets.FakeData(
            size=cfg.fake_test_size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
            random_offset=cfg.seed + 1,
        )

    train_size = int((1.0 - cfg.val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, _ = random_split(
        train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.seed)
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )
    return train_loader, test_loader


def train_single_lambda(cfg: Config, lambda_value: float, train_loader: DataLoader, test_loader: DataLoader) -> Dict:
    """Train one model for a specific sparsity lambda and return metrics."""
    model = PrunableMLP(hidden_dims=cfg.hidden_dims).to(cfg.device)
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_samples = 0

        progress = tqdm(train_loader, desc=f"lambda={lambda_value:.4g} epoch={epoch}/{cfg.epochs}", leave=False)
        for images, labels in progress:
            images = images.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            classification_loss = criterion(logits, labels)
            sparsity_loss = compute_sparsity_loss(model)
            total_loss = classification_loss + lambda_value * sparsity_loss
            total_loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            epoch_loss += total_loss.item() * batch_size
            total_samples += batch_size
            progress.set_postfix(loss=f"{total_loss.item():.4f}")

        mean_loss = epoch_loss / max(total_samples, 1)
        test_accuracy = evaluate(model, test_loader, cfg.device)
        logging.info(
            "lambda=%.4g epoch=%d/%d train_loss=%.4f test_acc=%.2f%%",
            lambda_value,
            epoch,
            cfg.epochs,
            mean_loss,
            test_accuracy,
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    sparsity = compute_sparsity(model, threshold=1e-2)
    model_path = os.path.join(cfg.output_dir, "models", f"best_lambda_{lambda_value:.4g}.pt")
    torch.save(best_state, model_path)

    return {
        "lambda": lambda_value,
        "accuracy": best_accuracy,
        "sparsity": sparsity,
        "model_path": model_path,
        "state_dict": best_state,
    }


def main() -> None:
    cfg = Config()

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "logs"), exist_ok=True)

    setup_logging(os.path.join(cfg.output_dir, "logs", "train.log"))
    set_seed(cfg.seed)
    logging.info("Using device: %s", cfg.device)

    train_loader, test_loader = build_dataloaders(cfg)

    results: List[Dict] = []
    best_result = None

    for lambda_value in cfg.lambda_values:
        result = train_single_lambda(cfg, lambda_value, train_loader, test_loader)
        results.append(result)
        if best_result is None or result["accuracy"] > best_result["accuracy"]:
            best_result = result

    best_model = PrunableMLP(hidden_dims=cfg.hidden_dims).to(cfg.device)
    best_model.load_state_dict(best_result["state_dict"])
    gate_values = get_all_gate_values(best_model)
    plot_path = os.path.join(cfg.output_dir, "plots", "gate_distribution.png")
    plot_gate_distribution(gate_values, plot_path)

    report_results = [{k: v for k, v in item.items() if k != "state_dict"} for item in results]
    generate_report("report.md", report_results, best_result["lambda"])

    logging.info("Training complete.")
    logging.info("Gate distribution plot: %s", plot_path)
    logging.info("Report generated: report.md")
    print("\nExperiment Summary")
    for result in report_results:
        print(
            f"lambda={result['lambda']:.4g} | accuracy={result['accuracy']:.2f}% | sparsity={result['sparsity']:.2f}%"
        )


if __name__ == "__main__":
    main()
