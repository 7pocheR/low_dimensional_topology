#!/usr/bin/env python3
"""
Retrain bird-deer CNN classifiers (GELU + ReLU, L8 no-skip) with checkpoint saving,
then evaluate on witness cycle points from linking detection.

Usage:
  python retrain_and_eval_witness.py --activation relu --seed 42
  python retrain_and_eval_witness.py --activation gelu --seed 42
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, accuracy_score
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'utils'))

DEFAULT_DATA_DIR = str(PROJECT_ROOT / 'data')
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / 'results' / 'cnn_binary_checkpoint')

BIRD_CLASS = 2
DEER_CLASS = 4


class FlattenedCIFAR10Dataset(Dataset):
    def __init__(self, images, labels, indices=None, transform=None):
        self.images = images
        self.labels = labels
        self.indices = indices  # original CIFAR-10 indices
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            from PIL import Image
            img = Image.fromarray(img)
            img = self.transform(img)
        orig_idx = self.indices[idx] if self.indices is not None else idx
        return img, label, orig_idx


def get_activation(name):
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'leaky_relu': nn.LeakyReLU(0.01),
        'swish': nn.SiLU(),
        'mish': nn.Mish(),
    }
    return activations[name]


class WidthBoundedCNN(nn.Module):
    def __init__(self, activation_name='relu', num_conv_blocks=2, use_skip=False):
        super().__init__()
        self.use_skip = use_skip
        act = lambda: get_activation(activation_name)

        channels = [3, 12, 48, 192]
        self.blocks = nn.ModuleList()

        for stage in range(3):
            in_c = channels[stage]
            out_c = channels[stage + 1]
            layers = []
            for b in range(num_conv_blocks):
                if b == num_conv_blocks - 1:
                    layers.append(nn.Conv2d(in_c if b == 0 else out_c, out_c, 3, stride=2, padding=1))
                else:
                    layers.append(nn.Conv2d(in_c if b == 0 else out_c, out_c, 3, stride=1, padding=1))
                layers.append(nn.BatchNorm2d(out_c))
                layers.append(act())
                in_c = out_c
            self.blocks.append(nn.Sequential(*layers))

        self.fc1 = nn.Linear(192 * 4 * 4, 3072)
        self.act_fc = act()
        self.fc2 = nn.Linear(3072, 2)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = self.act_fc(self.fc1(x))
        x = self.fc2(x)
        return x


def load_cifar10_binary_with_indices(seed=42, data_dir=DEFAULT_DATA_DIR):
    """Load CIFAR-10 bird vs deer, preserving original indices."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_full = datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_full = datasets.CIFAR10(root=data_dir, train=False, download=True)

    # Filter bird and deer, keep original indices
    train_mask = [i for i, (_, l) in enumerate(train_full) if l in [BIRD_CLASS, DEER_CLASS]]
    test_mask = [i for i, (_, l) in enumerate(test_full) if l in [BIRD_CLASS, DEER_CLASS]]

    images_train = [np.array(train_full.data[i]) for i in train_mask]
    labels_train = [0 if train_full.targets[i] == BIRD_CLASS else 1 for i in train_mask]
    train_orig_indices = train_mask

    images_test = [np.array(test_full.data[i]) for i in test_mask]
    labels_test = [0 if test_full.targets[i] == BIRD_CLASS else 1 for i in test_mask]
    test_orig_indices = test_mask

    # Split train into train/val (90/10)
    np.random.seed(seed)
    n = len(images_train)
    perm = np.random.permutation(n)
    split = int(0.9 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    train_ds = FlattenedCIFAR10Dataset(
        [images_train[i] for i in train_idx],
        [labels_train[i] for i in train_idx],
        [train_orig_indices[i] for i in train_idx],
        transform=train_transform
    )
    val_ds = FlattenedCIFAR10Dataset(
        [images_train[i] for i in val_idx],
        [labels_train[i] for i in val_idx],
        [train_orig_indices[i] for i in val_idx],
        transform=eval_transform
    )
    test_ds = FlattenedCIFAR10Dataset(
        images_test, labels_test, test_orig_indices,
        transform=eval_transform
    )

    return train_ds, val_ds, test_ds


def train_and_save(activation, seed=42, data_dir=DEFAULT_DATA_DIR, output_dir=DEFAULT_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {activation} L8 no-skip, seed={seed}, device={device}")

    train_ds, val_ds, test_ds = load_cifar10_binary_with_indices(seed, data_dir=data_dir)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = WidthBoundedCNN(activation_name=activation, num_conv_blocks=2, use_skip=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(100):
        model.train()
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        val_acc = correct / total
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: val_acc={val_acc:.4f}, best={best_val_acc:.4f}")

        if patience_counter >= 15:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best and evaluate
    model.load_state_dict(best_state)

    # Per-image evaluation on test set
    model.eval()
    all_preds = []
    all_labels = []
    all_indices = []
    with torch.no_grad():
        for imgs, labels, indices in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_indices.extend(indices.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_indices = np.array(all_indices)

    test_acc = (all_preds == all_labels).mean()
    print(f"  Test accuracy: {test_acc:.4f}")

    # Also evaluate on training set (for witness points)
    train_eval_ds = FlattenedCIFAR10Dataset(
        train_ds.images, train_ds.labels, train_ds.indices,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    train_eval_loader = DataLoader(train_eval_ds, batch_size=256, shuffle=False, num_workers=2)

    train_preds = []
    train_labels = []
    train_indices = []
    with torch.no_grad():
        for imgs, labels, indices in train_eval_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.numpy())
            train_indices.extend(indices.numpy())

    train_preds = np.array(train_preds)
    train_labels = np.array(train_labels)
    train_indices = np.array(train_indices)

    train_acc = (train_preds == train_labels).mean()
    print(f"  Train accuracy (eval mode): {train_acc:.4f}")

    # Save checkpoint and per-image predictions
    save_path = os.path.join(output_dir, f'{activation}_L8_noskip_s{seed}')
    os.makedirs(save_path, exist_ok=True)

    torch.save(best_state, os.path.join(save_path, 'best_model.pt'))

    np.savez(os.path.join(save_path, 'test_predictions.npz'),
             preds=all_preds, labels=all_labels, indices=all_indices)
    np.savez(os.path.join(save_path, 'train_predictions.npz'),
             preds=train_preds, labels=train_labels, indices=train_indices)

    results = {
        'activation': activation,
        'seed': seed,
        'test_accuracy': float(test_acc),
        'train_accuracy': float(train_acc),
        'best_val_accuracy': float(best_val_acc),
        'n_test': len(all_preds),
        'n_train': len(train_preds),
    }
    with open(os.path.join(save_path, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Saved to {save_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', required=True, choices=['relu', 'gelu'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    train_and_save(args.activation, args.seed, data_dir=args.data_dir, output_dir=args.output_dir)
