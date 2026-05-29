#!/usr/bin/env python3
"""
Train width-bounded CNN on CIFAR-10 binary classification for UNLINKED pair.
Control experiment: frog (6) vs ship (8) - only 18.2% linking consistency.

Hypothesis: Monotonic and non-monotonic activations should perform similarly
since there's no topological barrier for this unlinked pair.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision import datasets, transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_DIR = os.environ.get('CIFAR10_DATA_DIR', os.path.join(PROJECT_ROOT, 'data'))
DEFAULT_OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.join(PROJECT_ROOT, 'results', 'cnn_unlinked'))

# Unlinked pair: frog vs ship (18.2% linking consistency)
CLASS_A = 6  # frog
CLASS_B = 8  # ship
CLASS_A_NAME = 'frog'
CLASS_B_NAME = 'ship'


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class CIFAR10BinaryDataset(Dataset):
    """CIFAR-10 dataset filtered to two classes with transforms."""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class ConvBlock(nn.Module):
    """Convolutional block with optional skip connection."""
    def __init__(self, in_channels, out_channels, stride, activation, use_skip=False):
        super().__init__()
        self.use_skip = use_skip
        self.conv = nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, stride=stride, padding=1)
        self.activation = activation

        if use_skip:
            if stride != 1 or in_channels != out_channels:
                self.projection = nn.Conv2d(in_channels, out_channels,
                                           kernel_size=1, stride=stride, padding=0)
            else:
                self.projection = nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        if self.use_skip:
            identity = self.projection(x)
            out = out + identity
        return out


class WidthBoundedCNN(nn.Module):
    """Width-bounded CNN for binary classification."""
    def __init__(self, num_conv_blocks=1, activation='relu', use_skip=False):
        super().__init__()

        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'selu':
            act_fn = nn.SELU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'swish':
            act_fn = Swish()
        elif activation == 'mish':
            act_fn = Mish()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.activation = activation
        self.use_skip = use_skip

        # Block 1: 3×32×32 → 12×16×16
        layers = []
        in_ch = 3
        for i in range(num_conv_blocks - 1):
            layers.append(ConvBlock(in_ch, in_ch, stride=1,
                                   activation=act_fn, use_skip=use_skip))
        layers.append(ConvBlock(in_ch, 12, stride=2,
                               activation=act_fn, use_skip=use_skip))
        self.block1 = nn.Sequential(*layers)

        # Block 2: 12×16×16 → 48×8×8
        layers = []
        in_ch = 12
        for i in range(num_conv_blocks - 1):
            layers.append(ConvBlock(in_ch, in_ch, stride=1,
                                   activation=act_fn, use_skip=use_skip))
        layers.append(ConvBlock(in_ch, 48, stride=2,
                               activation=act_fn, use_skip=use_skip))
        self.block2 = nn.Sequential(*layers)

        # Block 3: 48×8×8 → 192×4×4
        layers = []
        in_ch = 48
        for i in range(num_conv_blocks - 1):
            layers.append(ConvBlock(in_ch, in_ch, stride=1,
                                   activation=act_fn, use_skip=use_skip))
        layers.append(ConvBlock(in_ch, 192, stride=2,
                               activation=act_fn, use_skip=use_skip))
        self.block3 = nn.Sequential(*layers)

        self.fc1 = nn.Linear(192 * 4 * 4, 3072)
        self.fc2 = nn.Linear(3072, 2)
        self.act_fn = act_fn

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.act_fn(self.fc1(x))
        x = self.fc2(x)
        return x


def load_cifar10_binary(class_a, class_b, seed=42, use_augmentation=True, data_dir=DEFAULT_DATA_DIR):
    """Load CIFAR-10, filter to two classes."""
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset_full = datasets.CIFAR10(root=data_dir,
                                          train=True, download=True, transform=None)
    test_dataset_full = datasets.CIFAR10(root=data_dir,
                                         train=False, download=True, transform=None)

    def filter_classes(dataset, ca, cb):
        images, labels = [], []
        for img, label in dataset:
            if label == ca:
                images.append(img)
                labels.append(0)
            elif label == cb:
                images.append(img)
                labels.append(1)
        return images, labels

    images_train_full, labels_train_full = filter_classes(train_dataset_full, class_a, class_b)
    images_test, labels_test = filter_classes(test_dataset_full, class_a, class_b)

    np.random.seed(seed)
    n = len(images_train_full)
    perm = np.random.permutation(n)
    n_val = int(n * 0.1)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    images_train = [images_train_full[i] for i in train_idx]
    labels_train = [labels_train_full[i] for i in train_idx]
    images_val = [images_train_full[i] for i in val_idx]
    labels_val = [labels_train_full[i] for i in val_idx]

    train_dataset = CIFAR10BinaryDataset(images_train, labels_train, transform=train_transform)
    val_dataset = CIFAR10BinaryDataset(images_val, labels_val, transform=eval_transform)
    test_dataset = CIFAR10BinaryDataset(images_test, labels_test, transform=eval_transform)

    return train_dataset, val_dataset, test_dataset


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(X)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            all_preds.append(model(X).argmax(1).cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'confusion_matrix': cm.tolist(),
        'class_a_accuracy': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'class_b_accuracy': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'class_a_as_b': int(fp),
        'class_b_as_a': int(fn),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', type=str, required=True,
                       choices=['relu', 'elu', 'selu', 'leaky_relu', 'gelu', 'swish', 'mish'])
    parser.add_argument('--num-conv-blocks', type=int, required=True, choices=[1, 2, 3])
    parser.add_argument('--use-skip', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_layers = 3 * args.num_conv_blocks + 2
    config_name = f"unlinked_{args.activation}{'_skip' if args.use_skip else ''}_L{total_layers}_B{args.num_conv_blocks}"

    print(f"="*60)
    print(f"UNLINKED PAIR CONTROL: {CLASS_A_NAME} vs {CLASS_B_NAME}")
    print(f"Linking consistency: 18.2% (essentially unlinked)")
    print(f"="*60)
    print(f"Config: {config_name}, Device: {device}")

    # Load data
    train_dataset, val_dataset, test_dataset = load_cifar10_binary(
        CLASS_A, CLASS_B, args.seed, use_augmentation=True, data_dir=args.data_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create model
    model = WidthBoundedCNN(
        num_conv_blocks=args.num_conv_blocks,
        activation=args.activation,
        use_skip=args.use_skip
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_results = evaluate(model, val_loader, device)
        scheduler.step(1 - val_results['accuracy'])

        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train_acc={train_acc:.4f}, val_acc={val_results['accuracy']:.4f}")

        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    test_results = evaluate(model, test_loader, device)
    train_results = evaluate(model, train_loader, device)

    result = {
        'config': {
            'activation': args.activation,
            'num_conv_blocks': args.num_conv_blocks,
            'total_layers': total_layers,
            'use_skip': args.use_skip,
            'task': 'cnn_unlinked_control',
            'class_a': CLASS_A_NAME,
            'class_b': CLASS_B_NAME,
            'linking_consistency': 0.182
        },
        'n_params': n_params,
        'test_accuracy': test_results['accuracy'],
        'train_accuracy': train_results['accuracy'],
        'best_val_accuracy': best_val_acc,
        'class_a_accuracy': test_results['class_a_accuracy'],
        'class_b_accuracy': test_results['class_b_accuracy'],
        'confusion_matrix': test_results['confusion_matrix']
    }

    print(f"\nResults:")
    print(f"  Test accuracy: {result['test_accuracy']:.4f}")
    print(f"  {CLASS_A_NAME} accuracy: {result['class_a_accuracy']:.4f}")
    print(f"  {CLASS_B_NAME} accuracy: {result['class_b_accuracy']:.4f}")

    result_file = os.path.join(args.output_dir, f'result_{config_name}.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"Saved to: {result_file}")


if __name__ == '__main__':
    main()
