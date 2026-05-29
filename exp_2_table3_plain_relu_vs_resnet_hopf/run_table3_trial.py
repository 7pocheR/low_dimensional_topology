"""
Single-trial script for Table 3 (Plain ReLU vs ResNet on Hopf link, width 3).

Refactored from ReLUlink/experiments/relu_resnet_comparison.py to run one
(depth, arch, seed) configuration and write a results JSON.

Network, optimizer (AdamW + ReduceLROnPlateau), grad-clip, early stopping,
and dataset (6000 pt thickened Hopf link, 4000/2000 split) are copied
verbatim from the original `Width3ReLUNetwork`, `Width3ReLUResNet`,
`train_model`, and `ThickenedHopfLink` so the reproduction matches.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class ThickenedHopfLink:
    """Verbatim port from relu_resnet_comparison.py."""

    def __init__(self, n_points_per_curve=3000):
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


class Width3ReLUNetwork(nn.Module):
    """Verbatim port from relu_resnet_comparison.py."""

    def __init__(self, depth=3):
        super().__init__()
        self.depth = depth
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(3, 3))
            if i < depth - 1:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(3, 2)

    def forward(self, x):
        features = self.network(x)
        return self.classifier(features)


class Width3ReLUResNet(nn.Module):
    """Verbatim port from relu_resnet_comparison.py."""

    def __init__(self, depth=3):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = nn.Sequential(
                nn.Linear(3, 3),
                nn.ReLU(),
                nn.Linear(3, 3),
            )
            self.blocks.append(block)
        self.classifier = nn.Linear(3, 2)

    def forward(self, x):
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual
            x = torch.relu(x)
        return self.classifier(x)


def train_model(model, train_loader, val_loader, epochs=800, lr=1e-3, early_stop_patience=100):
    """Verbatim port of train_model from relu_resnet_comparison.py (logging stripped)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50,
    )

    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

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

        scheduler.step(avg_train_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_accuracies, best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--arch', type=str, required=True, choices=['relu', 'resnet'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_points_per_curve', type=int, default=3000)
    parser.add_argument('--n_train', type=int, default=4000)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / 'result.json'
    if out_file.exists():
        print(f"Already done: {out_file}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    gen = ThickenedHopfLink(n_points_per_curve=args.n_points_per_curve)
    X, y = gen.generate_data()
    n_train = args.n_train
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

    if args.arch == 'relu':
        model = Width3ReLUNetwork(depth=args.depth).to(device)
    else:
        model = Width3ReLUResNet(depth=args.depth).to(device)

    t0 = time.time()
    train_losses, val_accuracies, best_val_acc = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, early_stop_patience=args.patience,
    )
    elapsed = time.time() - t0

    result = {
        'table': 3,
        'depth': args.depth,
        'arch': args.arch,
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
        'optimizer': 'AdamW',
        'weight_decay': 1e-4,
        'scheduler': 'ReduceLROnPlateau(mode=min, factor=0.5, patience=50)',
        'grad_clip': 1.0,
        'loss': 'CrossEntropy',
    }
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out_file} (best_val_acc={best_val_acc:.2f}, epochs={len(train_losses)}, t={elapsed:.1f}s)")


if __name__ == '__main__':
    main()
