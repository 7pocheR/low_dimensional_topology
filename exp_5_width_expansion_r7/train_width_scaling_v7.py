"""
Width-scaling experiment with MULTIPLE COPIES of linked spheres.

V7: Same as V5 but with L1-ordered grid placement instead of first-axis-only.

Key idea: Place k copies of the S^n × S^n link at L1-ordered grid points.
Each copy contributes a local entanglement region.
With k copies, the network must be wrong in k local patches.

Uses targeted thickening and fixed training dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import json
import time
from pathlib import Path
from itertools import islice
from typing import Generator, List, Tuple

SQRT2 = np.sqrt(2.0)
Vec = Tuple[int, ...]


# =============================================================================
# L1-ordered grid enumeration
# =============================================================================

def shell_vectors(d: int, k: int) -> Generator[Vec, None, None]:
    """
    Yield all integer vectors in Z^d with L1 norm exactly k, i.e. sum_i |x_i| = k.
    Deterministic order; each vector appears exactly once.
    """
    if d < 0 or k < 0:
        return
    if d == 0:
        if k == 0:
            yield ()
        return
    if d == 1:
        if k == 0:
            yield (0,)
        else:
            yield (-k,)
            yield (k,)
        return

    for x in range(-k, k + 1):
        rem = k - abs(x)
        for tail in shell_vectors(d - 1, rem):
            yield (x,) + tail


def l1_nondecreasing_Zn(d: int) -> Generator[Vec, None, None]:
    """
    Infinite generator enumerating Z^d in nondecreasing L1 norm.
    """
    if d <= 0:
        raise ValueError("Dimension d must be a positive integer.")
    k = 0
    while True:
        yield from shell_vectors(d, k)
        k += 1


def first_n_vectors(d: int, n: int) -> List[Vec]:
    """
    Return first n vectors from L1-ordered enumeration of Z^d.
    """
    if n < 0:
        raise ValueError("n must be nonnegative.")
    return list(islice(l1_nondecreasing_Zn(d), n))


def get_copy_centers(num_copies: int, spacing: float, dim: int) -> np.ndarray:
    """
    Generate copy centers using L1-ordered grid, scaled by spacing.
    """
    vecs = first_n_vectors(dim, num_copies)
    return np.array(vecs, dtype=float) * spacing


# =============================================================================
# Embeddings with MULTIPLE COPIES
# =============================================================================

def sample_on_sphere(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform sample from S^{dim-1} in R^{dim}."""
    x = rng.normal(size=dim)
    return x / np.linalg.norm(x)


def sample_uniform_ball(dim: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform in the dim-dimensional ball of radius rho."""
    direction = sample_on_sphere(dim, rng)
    r = rho * (rng.random() ** (1.0 / dim))
    return r * direction


def tildeA(u: np.ndarray) -> np.ndarray:
    """
    u: (n+1,) on S^n -> (2n+1,) in R^{2n+1}.
    Output structure: [X (n coords), Z (1 coord), Y (n coords)]
    tildeA is non-zero in X and Z, zero in Y.
    """
    n = u.size - 1
    u1, upr = u[0], u[1:]
    a = 1.0 - u1 / SQRT2
    X = upr / a
    Z = -u1 / (SQRT2 * a)
    Y = np.zeros(n, dtype=float)
    return np.concatenate([X, np.array([Z]), Y])


def tildeB(v: np.ndarray) -> np.ndarray:
    """
    v: (n+1,) on S^n -> (2n+1,) in R^{2n+1}.
    Output structure: [X (n coords), Z (1 coord), Y (n coords)]
    tildeB is non-zero in Y and Z, zero in X.
    """
    n = v.size - 1
    v1, vpr = v[0], v[1:]
    b = 1.0 - v1 / SQRT2
    X = np.zeros(n, dtype=float)
    Z = v1 / (SQRT2 * b)
    Y = vpr / b
    return np.concatenate([X, np.array([Z]), Y])


def generate_multi_copy_dataset(n: int, num_samples: int, num_copies: int,
                                 spacing: float, rho: float, seed: int,
                                 targeted_thickening: bool = True):
    """
    Generate dataset with multiple copies of linked spheres.

    Args:
        n: Dimension parameter (S^n × S^n in R^{2n+1})
        num_samples: Total number of samples
        num_copies: Number of copies of the link
        spacing: Distance between copy centers
        rho: Thickening radius
        seed: Random seed
        targeted_thickening: If True, thicken A in Y-subspace, B in X-subspace
    """
    rng = np.random.default_rng(seed)
    d = 2 * n + 1

    half = num_samples // 2
    samples_per_copy_A = half // num_copies
    samples_per_copy_B = half // num_copies

    A_pts = []
    B_pts = []

    # V7 CHANGE: Generate copy centers using L1-ordered grid (not first-axis-only)
    centers = get_copy_centers(num_copies, spacing, d)

    for copy_idx in range(num_copies):
        center = centers[copy_idx]

        # Generate A points for this copy
        for _ in range(samples_per_copy_A):
            u = sample_on_sphere(n + 1, rng)
            base = tildeA(u)

            if targeted_thickening:
                # Thicken in Y-subspace (last n coordinates)
                noise = np.zeros(d)
                noise_in_subspace = sample_uniform_ball(n, rho, rng)
                noise[n+1:] = noise_in_subspace
            else:
                # Ambient ball thickening
                noise = sample_uniform_ball(d, rho, rng)

            A_pts.append(base + noise + center)

        # Generate B points for this copy
        for _ in range(samples_per_copy_B):
            v = sample_on_sphere(n + 1, rng)
            base = tildeB(v)

            if targeted_thickening:
                # Thicken in X-subspace (first n coordinates)
                noise = np.zeros(d)
                noise_in_subspace = sample_uniform_ball(n, rho, rng)
                noise[:n] = noise_in_subspace
            else:
                # Ambient ball thickening
                noise = sample_uniform_ball(d, rho, rng)

            B_pts.append(base + noise + center)

    A_pts = np.array(A_pts, dtype=np.float32)
    B_pts = np.array(B_pts, dtype=np.float32)

    X = np.vstack([A_pts, B_pts])
    y = np.concatenate([
        np.zeros(len(A_pts), dtype=np.int64),
        np.ones(len(B_pts), dtype=np.int64)
    ])

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# =============================================================================
# Activation functions
# =============================================================================

class NegSlopeReLU(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x >= 0, x, self.alpha * x)


class Abs(nn.Module):
    """Absolute value activation - key non-monotonic function for topological transformations."""
    def forward(self, x):
        return torch.abs(x)


def get_activation(name: str):
    activations = {
        'relu': nn.ReLU,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'leaky_relu': lambda: nn.LeakyReLU(0.01),
        'gelu': nn.GELU,
        'swish': nn.SiLU,
        'mish': nn.Mish,
        'abs': Abs,
        'negslope_001': lambda: NegSlopeReLU(-0.01),
        'negslope_01': lambda: NegSlopeReLU(-0.1),
        'negslope_05': lambda: NegSlopeReLU(-0.5),
        'negslope_10': lambda: NegSlopeReLU(-1.0),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]()


# =============================================================================
# FFN Model
# =============================================================================

class FFNBlock(nn.Module):
    def __init__(self, dim: int, activation: nn.Module, skip: bool = False, norm: str = None):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = activation
        self.skip = skip
        if norm == 'layernorm':
            self.norm = nn.LayerNorm(dim)
        elif norm == 'rmsnorm':
            self.norm = nn.RMSNorm(dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(dim, elementwise_affine=False)
        else:
            self.norm = None

    def forward(self, x):
        out = self.linear(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.activation(out)
        if self.skip:
            out = out + x
        return out


class FFN(nn.Module):
    def __init__(self, input_dim: int, depth: int, activation_name: str, skip: bool = False, hidden_dim: int = None, norm: str = None):
        super().__init__()
        h = hidden_dim if hidden_dim is not None else input_dim
        self.norm_type = norm
        layers = []
        # First layer: input_dim -> h (if h != input_dim)
        if h != input_dim:
            layers.append(nn.Linear(input_dim, h))
            layers.append(get_activation(activation_name))
        for _ in range(depth):
            layers.append(FFNBlock(h, get_activation(activation_name), skip=skip, norm=norm))
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(h, 2)

    def forward(self, x):
        return self.output(self.layers(x))


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)

    return total_loss / total, correct / total


def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(X)
            total_loss += criterion(logits, y).item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)

    return total_loss / total, correct / total


# =============================================================================
# Main
# =============================================================================

def run_experiment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    n = args.n
    input_dim = 2 * n + 1

    print(f"\n{'='*60}")
    print(f"n={n}, depth={args.depth}, act={args.activation}, skip={args.skip}")
    print(f"MULTI-COPY V7: {args.num_copies} copies, spacing={args.spacing}")
    print(f"L1-ordered grid placement (not first-axis-only)")
    print(f"Targeted thickening: {args.targeted}, rho={args.rho}")
    print(f"FIXED dataset: train={args.train_samples:,}, val/test={args.val_test_samples:,}")
    print(f"{'='*60}\n")

    # Generate FIXED datasets
    print("Generating fixed training set...")
    X_train, y_train = generate_multi_copy_dataset(
        n, args.train_samples, args.num_copies, args.spacing, args.rho,
        seed=args.seed, targeted_thickening=args.targeted
    )

    print("Generating fixed validation set...")
    X_val, y_val = generate_multi_copy_dataset(
        n, args.val_test_samples, args.num_copies, args.spacing, args.rho,
        seed=args.seed + 100000, targeted_thickening=args.targeted
    )

    print("Generating fixed test set...")
    X_test, y_test = generate_multi_copy_dataset(
        n, args.val_test_samples, args.num_copies, args.spacing, args.rho,
        seed=args.seed + 200000, targeted_thickening=args.targeted
    )

    # Create DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Model
    hidden_dim = args.width if args.width is not None else input_dim
    norm_type = getattr(args, 'norm', None)
    model = FFN(input_dim, args.depth, args.activation, args.skip, hidden_dim=hidden_dim, norm=norm_type).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params * 4 / 1e6:.1f} MB)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc, best_epoch, patience_counter = 0.0, 0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\nTraining...")
    t0 = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc, best_epoch, patience_counter = val_acc, epoch, 0
            torch.save(model.state_dict(), args.output_dir / 'best_model.pt')
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}: train={train_acc:.4f}, val={val_acc:.4f}, best={best_val_acc:.4f}")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - t0

    # Test
    model.load_state_dict(torch.load(args.output_dir / 'best_model.pt'))
    test_loss, test_acc = eval_model(model, test_loader, criterion, device)

    print(f"\n{'='*60}")
    print(f"TEST ACC: {test_acc:.4f} (val={best_val_acc:.4f} @ epoch {best_epoch+1})")
    print(f"Time: {train_time:.1f}s")
    print(f"{'='*60}")

    # Save results
    results = {
        'n': n, 'input_dim': input_dim, 'hidden_dim': hidden_dim, 'depth': args.depth,
        'activation': args.activation, 'skip': args.skip, 'norm': norm_type,
        'num_copies': args.num_copies, 'spacing': args.spacing,
        'targeted_thickening': args.targeted, 'rho': args.rho,
        'train_samples': args.train_samples, 'val_test_samples': args.val_test_samples,
        'n_params': n_params,
        'version': 'v7_l1grid',
        'test_acc': test_acc, 'val_acc': best_val_acc, 'best_epoch': best_epoch + 1,
        'total_epochs': len(history['train_acc']), 'train_time': train_time,
        'lr': args.lr, 'batch_size': args.batch_size, 'seed': args.seed,
    }

    with open(args.output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    np.savez(args.output_dir / 'history.npz', **{k: np.array(v) for k, v in history.items()})

    print(f"Saved to {args.output_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--activation', type=str, required=True,
                        choices=['relu', 'elu', 'selu', 'leaky_relu',
                                 'gelu', 'swish', 'mish', 'abs',
                                 'negslope_001', 'negslope_01', 'negslope_05', 'negslope_10'])
    parser.add_argument('--skip', action='store_true')
    parser.add_argument('--width', type=int, default=None, help='Hidden width (default: input_dim = 2n+1)')
    parser.add_argument('--norm', type=str, default=None, choices=['layernorm', 'rmsnorm'], help='Normalization layer (default: none)')
    parser.add_argument('--num_copies', type=int, default=100, help='Number of link copies')
    parser.add_argument('--spacing', type=float, default=10.0, help='Spacing between copies')
    parser.add_argument('--targeted', action='store_true', default=True, help='Use targeted thickening')
    parser.add_argument('--no_targeted', action='store_false', dest='targeted')
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--train_samples', type=int, default=1_000_000)
    parser.add_argument('--val_test_samples', type=int, default=100_000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default=None)

    args = parser.parse_args()

    if args.output_dir is None:
        skip_str = 'skip' if args.skip else 'noskip'
        args.output_dir = Path(f'results/multi_copy/n{args.n}_d{args.depth}_{args.activation}_{skip_str}_c{args.num_copies}')
    else:
        args.output_dir = Path(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(args)
