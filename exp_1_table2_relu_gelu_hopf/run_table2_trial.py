"""
Single-trial script for Table 2 (ReLU vs GELU on Hopf link, width 3).

Refactored from ReLUlink/experiments/RELU_VS_GELU.py to run one
(depth, activation, seed) configuration and write a results JSON.

Network, optimizer, dataset, training loop, and early-stopping logic are
copied verbatim from the original `Width3Network` + `train_model` +
`ThickenedHopfLink` so the reproduction matches the canonical pipeline.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class ThickenedHopfLink:
    """Thickened Hopf link dataset (verbatim from RELU_VS_GELU.py)."""

    def __init__(self, n_points_per_curve=2000):
        self.n_points_per_curve = n_points_per_curve

    def generate_curve1(self, t_values):
        x = np.cos(t_values) + 0.3 * np.sin(100 * t_values)
        y = np.sin(t_values) + 0.3 * np.sin(100 * t_values)
        z = 0.5 * np.cos(100 * t_values)
        return np.column_stack([x, y, z])

    def generate_curve2(self, s_values):
        x = 1 + np.cos(s_values) + 0.3 * np.sin(100 * s_values)
        y = 0.5 * np.cos(100 * s_values)
        z = np.sin(s_values) + 0.3 * np.sin(100 * s_values)
        return np.column_stack([x, y, z])

    def generate_data(self, noise_std=0.01):
        t_values = np.linspace(0, 2 * np.pi, self.n_points_per_curve)
        s_values = np.linspace(0, 2 * np.pi, self.n_points_per_curve)
        curve1 = self.generate_curve1(t_values)
        curve2 = self.generate_curve2(s_values)
        curve1 += np.random.normal(0, noise_std, curve1.shape)
        curve2 += np.random.normal(0, noise_std, curve2.shape)
        labels1 = np.zeros(len(curve1))
        labels2 = np.ones(len(curve2))
        X = np.vstack([curve1, curve2])
        y = np.hstack([labels1, labels2])
        return X.astype(np.float32), y.astype(np.int64)


class Width3Network(nn.Module):
    """Verbatim port of Width3Network in RELU_VS_GELU.py."""

    def __init__(self, depth=12, activation='relu'):
        super().__init__()
        self.depth = depth
        self.activation_name = activation

        layers = []
        layers.append(nn.Linear(3, 3))  # input layer 3 -> 3

        for _ in range(depth - 2):
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            layers.append(nn.Linear(3, 3))

        # final activation + classifier head
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        layers.append(nn.Linear(3, 2))  # binary classification

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, val_loader, epochs=800, lr=1e-3, patience=200):
    """Verbatim port of train_model from RELU_VS_GELU.py (logging stripped)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        val_acc = 100.0 * correct / total
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_accuracies, best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--activation', type=str, required=True, choices=['relu', 'gelu'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_points_per_curve', type=int, default=2000)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / 'result.json'
    if out_file.exists():
        print(f"Already done: {out_file}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Seed (same order as outer loop in RELU_VS_GELU.py)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Data (verbatim split logic)
    gen = ThickenedHopfLink(n_points_per_curve=args.n_points_per_curve)
    X, y = gen.generate_data()
    n_train = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size, shuffle=False)

    model = Width3Network(depth=args.depth, activation=args.activation).to(device)
    t0 = time.time()
    train_losses, val_accuracies, best_val_acc = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
    )
    elapsed = time.time() - t0

    result = {
        'table': 2,
        'depth': args.depth,
        'activation': args.activation,
        'seed': args.seed,
        'best_val_acc': float(best_val_acc),
        'final_val_acc': float(val_accuracies[-1]) if val_accuracies else None,
        'total_epochs': len(train_losses),
        'epochs_max': args.epochs,
        'patience': args.patience,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'n_points_per_curve': args.n_points_per_curve,
        'train_size': int(n_train),
        'val_size': int(len(X) - n_train),
        'train_time_s': float(elapsed),
        'device': str(device),
        'optimizer': 'Adam',
        'loss': 'CrossEntropy',
    }
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out_file} (best_val_acc={best_val_acc:.2f}, epochs={len(train_losses)}, t={elapsed:.1f}s)")


if __name__ == '__main__':
    main()
