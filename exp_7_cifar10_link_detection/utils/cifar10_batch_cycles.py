"""
Batch computation of k-NN graphs and fundamental cycles for all CIFAR-10 classes.

This module precomputes the graph structures needed for linking detection,
allowing efficient reuse across all 45 class pairs.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import sys

sys.path.append('../src')
from link_detector import build_knn_graph, fundamental_cycle_basis, cycles_as_polylines


def build_all_graphs(
    pca_data: Dict[int, np.ndarray],
    k: int = 15,
    epsilon: float = 0.7,
    mutual: bool = True
) -> Dict[int, Dict[int, set]]:
    """
    Build k-NN graphs for all 10 classes.

    Args:
        pca_data: Dict mapping class_id to PCA-projected points (N, 3)
        k: Number of nearest neighbors
        epsilon: Maximum edge length
        mutual: Whether to use mutual k-NN

    Returns:
        Dict mapping class_id to adjacency dict
    """
    graphs = {}

    print(f"Building k-NN graphs for all classes (k={k}, epsilon={epsilon}, mutual={mutual})...")

    for class_id in range(10):
        start_time = time.time()
        X = pca_data[class_id]
        adj = build_knn_graph(X, k=k, epsilon=epsilon, mutual=mutual)

        n_edges = sum(len(neighbors) for neighbors in adj.values()) // 2
        isolated = sum(1 for node in adj if len(adj[node]) == 0)

        elapsed = time.time() - start_time
        print(f"  Class {class_id}: {len(X)} nodes, {n_edges} edges, "
              f"{isolated} isolated ({100*isolated/len(X):.1f}%), {elapsed:.2f}s")

        graphs[class_id] = adj

    return graphs


def extract_all_cycles(
    graphs: Dict[int, Dict[int, set]],
    min_cycle_length: int = 20
) -> Tuple[Dict[int, List[List[int]]], Dict[int, dict]]:
    """
    Extract fundamental cycles for all classes.

    Args:
        graphs: Dict mapping class_id to adjacency dict
        min_cycle_length: Minimum cycle length to keep

    Returns:
        cycles: Dict mapping class_id to list of cycles
        stats: Dict mapping class_id to cycle statistics
    """
    cycles = {}
    stats = {}

    print(f"\nExtracting fundamental cycles (min_length={min_cycle_length})...")

    for class_id in range(10):
        start_time = time.time()
        adj = graphs[class_id]

        # Extract all fundamental cycles
        all_cycles = fundamental_cycle_basis(adj)

        # Filter by length
        filtered_cycles = [c for c in all_cycles if len(c) >= min_cycle_length]

        elapsed = time.time() - start_time

        # Compute statistics
        if filtered_cycles:
            lengths = [len(c) - 1 for c in filtered_cycles]  # -1 because cycles repeat first vertex
            stats[class_id] = {
                'total_cycles': len(all_cycles),
                'filtered_cycles': len(filtered_cycles),
                'min_length': min(lengths),
                'max_length': max(lengths),
                'median_length': np.median(lengths),
                'mean_length': np.mean(lengths)
            }
        else:
            stats[class_id] = {
                'total_cycles': len(all_cycles),
                'filtered_cycles': 0,
                'min_length': 0,
                'max_length': 0,
                'median_length': 0,
                'mean_length': 0
            }

        cycles[class_id] = filtered_cycles

        print(f"  Class {class_id}: {len(all_cycles)} total -> {len(filtered_cycles)} filtered, "
              f"lengths [{stats[class_id]['min_length']:.0f}-{stats[class_id]['max_length']:.0f}], "
              f"{elapsed:.2f}s")

    return cycles, stats


def precompute_cycle_polylines(
    cycles: Dict[int, List[List[int]]],
    pca_data: Dict[int, np.ndarray]
) -> Dict[int, List[np.ndarray]]:
    """
    Convert cycle vertex indices to 3D polylines for all classes.

    Args:
        cycles: Dict mapping class_id to list of cycles (vertex indices)
        pca_data: Dict mapping class_id to PCA-projected points

    Returns:
        Dict mapping class_id to list of polylines (N, 3) arrays
    """
    polylines = {}

    for class_id in range(10):
        X = pca_data[class_id]
        class_cycles = cycles[class_id]
        polylines[class_id] = [cycles_as_polylines(c, X) for c in class_cycles]

    return polylines


class CIFAR10CycleData:
    """
    Container class for precomputed CIFAR-10 cycle data.

    Holds all the data needed for pairwise linking detection.
    """

    def __init__(
        self,
        pca_data: Dict[int, np.ndarray],
        k: int = 15,
        epsilon: float = 0.7,
        mutual: bool = True,
        min_cycle_length: int = 20
    ):
        """
        Initialize and precompute all cycle data.

        Args:
            pca_data: Dict mapping class_id to PCA-projected points
            k: Number of nearest neighbors
            epsilon: Maximum edge length
            mutual: Whether to use mutual k-NN
            min_cycle_length: Minimum cycle length to keep
        """
        self.pca_data = pca_data
        self.k = k
        self.epsilon = epsilon
        self.mutual = mutual
        self.min_cycle_length = min_cycle_length

        # Precompute graphs
        print("=" * 60)
        print("PRECOMPUTING CIFAR-10 CYCLE DATA")
        print("=" * 60)
        print(f"Parameters: k={k}, epsilon={epsilon}, mutual={mutual}, min_cycle_length={min_cycle_length}")

        self.graphs = build_all_graphs(pca_data, k, epsilon, mutual)
        self.cycles, self.cycle_stats = extract_all_cycles(self.graphs, min_cycle_length)
        self.polylines = precompute_cycle_polylines(self.cycles, pca_data)

        # Summary
        self._print_summary()

    def _print_summary(self):
        """Print summary of precomputed data."""
        print("\n" + "=" * 60)
        print("PRECOMPUTATION SUMMARY")
        print("=" * 60)

        total_cycles = sum(len(self.cycles[i]) for i in range(10))
        print(f"Total cycles across all classes: {total_cycles}")

        print(f"\n{'Class':<12} {'Nodes':<8} {'Cycles':<8} {'Lengths':<20}")
        print("-" * 60)
        for class_id in range(10):
            from cifar10_data_loader import CIFAR10_CLASSES
            n_nodes = len(self.pca_data[class_id])
            n_cycles = len(self.cycles[class_id])
            s = self.cycle_stats[class_id]
            if n_cycles > 0:
                length_str = f"[{s['min_length']:.0f}-{s['max_length']:.0f}], med={s['median_length']:.0f}"
            else:
                length_str = "N/A"
            print(f"{CIFAR10_CLASSES[class_id]:<12} {n_nodes:<8} {n_cycles:<8} {length_str:<20}")

    def get_class_cycles(self, class_id: int) -> List[List[int]]:
        """Get cycles (vertex indices) for a class."""
        return self.cycles[class_id]

    def get_class_polylines(self, class_id: int) -> List[np.ndarray]:
        """Get polylines (3D coordinates) for a class."""
        return self.polylines[class_id]

    def get_class_points(self, class_id: int) -> np.ndarray:
        """Get PCA-projected points for a class."""
        return self.pca_data[class_id]

    def get_pair_data(self, class_i: int, class_j: int) -> Tuple:
        """
        Get all data needed for linking detection between two classes.

        Returns:
            (polylines_i, polylines_j, cycles_i, cycles_j, points_i, points_j)
        """
        return (
            self.polylines[class_i],
            self.polylines[class_j],
            self.cycles[class_i],
            self.cycles[class_j],
            self.pca_data[class_i],
            self.pca_data[class_j]
        )

    def has_cycles(self, class_id: int) -> bool:
        """Check if a class has any cycles."""
        return len(self.cycles[class_id]) > 0

    def can_test_pair(self, class_i: int, class_j: int) -> bool:
        """Check if both classes have cycles for testing."""
        return self.has_cycles(class_i) and self.has_cycles(class_j)


def create_cifar10_cycle_data(
    k: int = 15,
    epsilon: float = 0.7,
    mutual: bool = True,
    min_cycle_length: int = 20,
    data_dir: str = '../data'
) -> CIFAR10CycleData:
    """
    Convenience function to create CIFAR10CycleData with default settings.
    """
    from cifar10_data_loader import load_or_create_cifar10_pca

    pca_data, metadata = load_or_create_cifar10_pca(data_dir=data_dir)

    return CIFAR10CycleData(
        pca_data=pca_data,
        k=k,
        epsilon=epsilon,
        mutual=mutual,
        min_cycle_length=min_cycle_length
    )


if __name__ == "__main__":
    # Test the module
    import argparse

    parser = argparse.ArgumentParser(description='Precompute CIFAR-10 cycle data')
    parser.add_argument('--k', type=int, default=15, help='k for k-NN (default: 15)')
    parser.add_argument('--epsilon', type=float, default=0.7, help='epsilon threshold (default: 0.7)')
    parser.add_argument('--min-cycle-length', type=int, default=20, help='Minimum cycle length (default: 20)')
    parser.add_argument('--no-mutual', action='store_true', help='Use standard k-NN instead of mutual')
    args = parser.parse_args()

    cycle_data = create_cifar10_cycle_data(
        k=args.k,
        epsilon=args.epsilon,
        mutual=not args.no_mutual,
        min_cycle_length=args.min_cycle_length
    )

    print("\n" + "=" * 60)
    print("TESTABLE PAIRS")
    print("=" * 60)

    testable_pairs = []
    for i in range(10):
        for j in range(i + 1, 10):
            if cycle_data.can_test_pair(i, j):
                testable_pairs.append((i, j))

    print(f"Total pairs with cycles in both classes: {len(testable_pairs)} / 45")
