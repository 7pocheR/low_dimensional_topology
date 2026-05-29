"""
Generate thickened linked sphere datasets for width-scaling experiments.

S^n and S^n linked in R^{2n+1} with tubular neighborhood thickening.
Uses verified tildeA/tildeB embeddings from verify_linking_correct.py.

Thickening strategies:
1. Uniform ball in normal space (bounded, guaranteed disjoint)
2. Gaussian in normal space (smoother, truncated for safety)

Disjointness: d_min = 2(sqrt(2)-1) ~ 0.828, so rho < 0.414 is safe.
We use rho=0.15 for moderate difficulty.
"""

import numpy as np
from math import pi
from scipy.special import gamma
import argparse
import os

SQRT2 = np.sqrt(2.0)


# =============================================================================
# Verified embeddings from linking number computation
# =============================================================================

def tildeA(u: np.ndarray) -> np.ndarray:
    """u: (n+1,) on S^n -> (2n+1,) in R^{2n+1}."""
    u = np.asarray(u, dtype=float)
    n = u.size - 1
    u1, upr = u[0], u[1:]
    a = 1.0 - u1 / SQRT2
    X = upr / a
    Z = -u1 / (SQRT2 * a)
    Y = np.zeros(n, dtype=float)
    return np.concatenate([X, np.array([Z]), Y])


def tildeB(v: np.ndarray) -> np.ndarray:
    """v: (n+1,) on S^n -> (2n+1,) in R^{2n+1}."""
    v = np.asarray(v, dtype=float)
    n = v.size - 1
    v1, vpr = v[0], v[1:]
    b = 1.0 - v1 / SQRT2
    X = np.zeros(n, dtype=float)
    Z = v1 / (SQRT2 * b)
    Y = vpr / b
    return np.concatenate([X, np.array([Z]), Y])


def d_tildeA(u: np.ndarray, du: np.ndarray) -> np.ndarray:
    """Directional derivative of tildeA at u along du (tangent to S^n)."""
    u = np.asarray(u, dtype=float)
    du = np.asarray(du, dtype=float)
    n = u.size - 1
    u1, upr = u[0], u[1:]
    du1, dupr = du[0], du[1:]
    a = 1.0 - u1 / SQRT2

    dX = dupr / a + upr * (du1 / (SQRT2 * a * a))
    dZ = -(du1) / (SQRT2 * a) - (u1 * du1) / (2.0 * a * a)
    dY = np.zeros(n, dtype=float)
    return np.concatenate([dX, np.array([dZ]), dY])


def d_tildeB(v: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """Directional derivative of tildeB at v along dv (tangent to S^n)."""
    v = np.asarray(v, dtype=float)
    dv = np.asarray(dv, dtype=float)
    n = v.size - 1
    v1, vpr = v[0], v[1:]
    dv1, dvpr = dv[0], dv[1:]
    b = 1.0 - v1 / SQRT2

    dY = dvpr / b + vpr * (dv1 / (SQRT2 * b * b))
    dZ = (dv1) / (SQRT2 * b) + (v1 * dv1) / (2.0 * b * b)
    dX = np.zeros(n, dtype=float)
    return np.concatenate([dX, np.array([dZ]), dY])


# =============================================================================
# Sampling utilities
# =============================================================================

def sample_on_sphere(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform sample from S^{dim-1} in R^{dim}."""
    x = rng.normal(size=dim)
    return x / np.linalg.norm(x)


def sample_uniform_ball(dim: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform sample from the dim-dimensional Euclidean ball of given radius."""
    direction = sample_on_sphere(dim, rng)
    r = radius * (rng.random() ** (1.0 / dim))  # correct radial law
    return r * direction


def sample_truncated_gaussian(dim: int, sigma: float, max_radius: float,
                               rng: np.random.Generator) -> np.ndarray:
    """Sample from truncated Gaussian in R^dim with given sigma, rejecting if |x| > max_radius."""
    while True:
        x = rng.normal(scale=sigma, size=dim)
        if np.linalg.norm(x) <= max_radius:
            return x


def orthonormal_columns(A: np.ndarray) -> np.ndarray:
    """Return Q with orthonormal columns spanning col(A)."""
    Q, _ = np.linalg.qr(A)
    return Q


def tangent_basis_on_sphere(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Return an orthonormal basis for T_u S^n (tangent space at u on unit sphere).
    Returns (n+1) x n matrix whose columns are orthonormal and perpendicular to u.
    """
    d = u.size  # n+1
    n = d - 1
    W = rng.normal(size=(d, n))
    W = W - np.outer(u, u @ W)  # project to tangent space
    Q, _ = np.linalg.qr(W)
    return Q[:, :n]


def normal_basis_from_tangent(T: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Given orthonormal tangent basis T (D x n) in ambient R^D,
    return orthonormal normal basis N (D x (D-n)).
    """
    D, n = T.shape
    k = D - n
    R = rng.normal(size=(D, k))
    R = R - T @ (T.T @ R)  # project to orthogonal complement
    N = orthonormal_columns(R)[:, :k]
    return N


# =============================================================================
# Tubular thickening samplers
# =============================================================================

def sample_tube_uniform(n: int, N: int, rho: float, component: str,
                        seed: int = 0) -> np.ndarray:
    """
    Sample N points from tubular neighborhood of tildeA(S^n) or tildeB(S^n)
    using uniform ball sampling in normal space.

    Args:
        n: dimension of sphere (S^n in R^{2n+1})
        N: number of points to sample
        rho: tube radius (must be < sqrt(2)-1 ~ 0.414 for disjointness)
        component: "A" or "B"
        seed: random seed

    Returns:
        (N, 2n+1) array of points
    """
    rng = np.random.default_rng(seed)
    D = 2 * n + 1
    pts = np.empty((N, D), dtype=float)

    embed_fn = tildeA if component.upper() == "A" else tildeB
    deriv_fn = d_tildeA if component.upper() == "A" else d_tildeB

    for i in range(N):
        # Sample basepoint on S^n
        u = sample_on_sphere(n + 1, rng)
        x = embed_fn(u)

        # Get tangent basis on S^n
        Eu = tangent_basis_on_sphere(u, rng)

        # Pushforward to ambient tangent space via derivative
        J = np.column_stack([deriv_fn(u, Eu[:, j]) for j in range(n)])
        T = orthonormal_columns(J)[:, :n]  # D x n orthonormal tangent

        # Normal basis (D x (n+1))
        Nbasis = normal_basis_from_tangent(T, rng)

        # Sample in normal ball
        z = sample_uniform_ball(n + 1, rho, rng)
        pts[i] = x + Nbasis @ z

    return pts


def sample_tube_gaussian(n: int, N: int, sigma: float, max_radius: float,
                         component: str, seed: int = 0) -> np.ndarray:
    """
    Sample N points from tubular neighborhood using truncated Gaussian in normal space.

    Args:
        n: dimension of sphere
        N: number of points
        sigma: Gaussian std dev in normal directions
        max_radius: truncation radius (for disjointness guarantee)
        component: "A" or "B"
        seed: random seed
    """
    rng = np.random.default_rng(seed)
    D = 2 * n + 1
    pts = np.empty((N, D), dtype=float)

    embed_fn = tildeA if component.upper() == "A" else tildeB
    deriv_fn = d_tildeA if component.upper() == "A" else d_tildeB

    for i in range(N):
        u = sample_on_sphere(n + 1, rng)
        x = embed_fn(u)

        Eu = tangent_basis_on_sphere(u, rng)
        J = np.column_stack([deriv_fn(u, Eu[:, j]) for j in range(n)])
        T = orthonormal_columns(J)[:, :n]

        Nbasis = normal_basis_from_tangent(T, rng)
        z = sample_truncated_gaussian(n + 1, sigma, max_radius, rng)
        pts[i] = x + Nbasis @ z

    return pts


# =============================================================================
# Dataset generation
# =============================================================================

def generate_linked_dataset(n: int, N_per_class: int, rho: float = 0.15,
                            thickening: str = "uniform", seed: int = 0):
    """
    Generate a binary classification dataset of linked spheres.

    Args:
        n: sphere dimension (S^n x S^n in R^{2n+1})
        N_per_class: points per class
        rho: tube radius or Gaussian truncation
        thickening: "uniform" or "gaussian"
        seed: random seed

    Returns:
        X: (2*N_per_class, 2n+1) features
        y: (2*N_per_class,) labels (0 for A, 1 for B)
    """
    if thickening == "uniform":
        A_pts = sample_tube_uniform(n, N_per_class, rho, "A", seed)
        B_pts = sample_tube_uniform(n, N_per_class, rho, "B", seed + 1000)
    elif thickening == "gaussian":
        sigma = rho / 3.0  # 3-sigma truncation
        A_pts = sample_tube_gaussian(n, N_per_class, sigma, rho, "A", seed)
        B_pts = sample_tube_gaussian(n, N_per_class, sigma, rho, "B", seed + 1000)
    else:
        raise ValueError(f"Unknown thickening: {thickening}")

    X = np.vstack([A_pts, B_pts])
    y = np.concatenate([np.zeros(N_per_class), np.ones(N_per_class)])

    return X.astype(np.float32), y.astype(np.int64)


def generate_unlinked_dataset(n: int, N_per_class: int, rho: float = 0.15,
                              separation: float = 5.0, thickening: str = "uniform",
                              seed: int = 0):
    """
    Generate unlinked spheres (control) by translating one sphere far away.
    Same intrinsic geometry, but no topological linking.
    """
    if thickening == "uniform":
        A_pts = sample_tube_uniform(n, N_per_class, rho, "A", seed)
        B_pts = sample_tube_uniform(n, N_per_class, rho, "B", seed + 1000)
    else:
        sigma = rho / 3.0
        A_pts = sample_tube_gaussian(n, N_per_class, sigma, rho, "A", seed)
        B_pts = sample_tube_gaussian(n, N_per_class, sigma, rho, "B", seed + 1000)

    # Translate B far away along first coordinate
    B_pts[:, 0] += separation

    X = np.vstack([A_pts, B_pts])
    y = np.concatenate([np.zeros(N_per_class), np.ones(N_per_class)])

    return X.astype(np.float32), y.astype(np.int64)


# =============================================================================
# Verification: check tubes are disjoint
# =============================================================================

def verify_disjointness(A_pts: np.ndarray, B_pts: np.ndarray) -> dict:
    """Check minimum distance between point clouds."""
    from scipy.spatial.distance import cdist

    # Subsample for speed if large
    n_sample = min(1000, len(A_pts), len(B_pts))
    idx_A = np.random.choice(len(A_pts), n_sample, replace=False)
    idx_B = np.random.choice(len(B_pts), n_sample, replace=False)

    dists = cdist(A_pts[idx_A], B_pts[idx_B])
    min_dist = dists.min()
    mean_dist = dists.mean()

    return {
        "min_distance": min_dist,
        "mean_distance": mean_dist,
        "disjoint": min_dist > 0
    }


# =============================================================================
# Main: generate and save datasets
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate linked sphere datasets")
    parser.add_argument("--n", type=int, default=2, help="Sphere dimension (default: 2)")
    parser.add_argument("--N", type=int, default=5000, help="Points per class")
    parser.add_argument("--rho", type=float, default=0.15, help="Tube radius")
    parser.add_argument("--thickening", choices=["uniform", "gaussian"], default="uniform")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"GENERATING LINKED SPHERE DATASET")
    print(f"  n = {args.n} (S^{args.n} x S^{args.n} in R^{2*args.n+1})")
    print(f"  N = {args.N} points per class")
    print(f"  rho = {args.rho}")
    print(f"  thickening = {args.thickening}")
    print("=" * 60)

    # Generate linked dataset
    X_linked, y_linked = generate_linked_dataset(
        n=args.n, N_per_class=args.N, rho=args.rho,
        thickening=args.thickening, seed=args.seed
    )

    # Generate unlinked control
    X_unlinked, y_unlinked = generate_unlinked_dataset(
        n=args.n, N_per_class=args.N, rho=args.rho,
        thickening=args.thickening, seed=args.seed
    )

    # Verify disjointness
    A_linked = X_linked[y_linked == 0]
    B_linked = X_linked[y_linked == 1]
    disjoint_info = verify_disjointness(A_linked, B_linked)

    print(f"\nLinked dataset verification:")
    print(f"  Min distance between classes: {disjoint_info['min_distance']:.4f}")
    print(f"  Mean distance: {disjoint_info['mean_distance']:.4f}")
    print(f"  Disjoint: {disjoint_info['disjoint']}")

    # Save datasets
    linked_path = os.path.join(args.output_dir, f"linked_S{args.n}_R{2*args.n+1}.npz")
    unlinked_path = os.path.join(args.output_dir, f"unlinked_S{args.n}_R{2*args.n+1}.npz")

    np.savez(linked_path, X=X_linked, y=y_linked)
    np.savez(unlinked_path, X=X_unlinked, y=y_unlinked)

    print(f"\nSaved:")
    print(f"  {linked_path}")
    print(f"  {unlinked_path}")

    # Summary statistics
    print(f"\nDataset shapes:")
    print(f"  Linked: X={X_linked.shape}, y={y_linked.shape}")
    print(f"  Unlinked: X={X_unlinked.shape}, y={y_unlinked.shape}")
