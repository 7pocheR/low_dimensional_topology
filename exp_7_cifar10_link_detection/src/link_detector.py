"""
Topological linking detection for point clouds in R^3.

This module implements algorithms to detect topological linking between
curves extracted from spatial graphs built on point clouds.
"""

import math
from collections import deque
from typing import List, Tuple, Dict, Optional, Set

import numpy as np

Array = np.ndarray


# ---------------------------
# Graph construction utilities
# ---------------------------

def pairwise_distances_chunked(X: Array, Y: Array, chunk: int = 4096) -> Array:
    """Compute pairwise Euclidean distances between rows of X and Y in chunks.
    Returns an (len(X), len(Y)) array.
    """
    n, m = len(X), len(Y)
    D = np.empty((n, m), dtype=float)
    for i in range(0, n, chunk):
        Xi = X[i:i+chunk]
        # (a-b)^2 = a^2 + b^2 - 2ab
        a2 = np.sum(Xi*Xi, axis=1)[:, None]
        b2 = np.sum(Y*Y, axis=1)[None, :]
        D[i:i+chunk] = np.sqrt(np.maximum(a2 + b2 - 2.0 * Xi.dot(Y.T), 0.0))
    return D


def build_knn_graph(points: Array, k: int = 6, epsilon: float = float("inf"), mutual: bool = False) -> Dict[int, Set[int]]:
    """Build undirected k-NN graph with a maximum edge-length cap epsilon.

    Args:
        mutual: If True, use mutual k-NN (edge only if both nodes are k-NN of each other)

    Returns adjacency dict {u: set(neighbors)} with vertices 0..n-1.
    """
    X = np.asarray(points, dtype=float)
    n = len(X)
    if n == 0:
        return { }

    # Try to use scipy cKDTree if available for speed; else fall back.
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(X)
        dists, idxs = tree.query(X, k=min(k+1, n))  # include self at index 0
        # Ensure 2D shape
        if k+1 > 1 and np.ndim(idxs) == 1:
            idxs = idxs[:, None]
            dists = dists[:, None]
    except Exception:
        # Fallback to brute force in chunks
        D = pairwise_distances_chunked(X, X)
        idxs = np.argsort(D, axis=1)[:, :min(k+1, n)]
        dists = np.take_along_axis(D, idxs, axis=1)

    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}

    if mutual:
        # Mutual k-NN: edge only if both nodes are k-NN of each other
        knn_sets = {}
        for i in range(n):
            neighbors = set()
            for jpos in range(idxs.shape[1]):
                j = int(idxs[i, jpos])
                if j != i and float(dists[i, jpos]) <= epsilon:
                    neighbors.add(j)
            knn_sets[i] = neighbors

        # Add edges only if mutual
        for i in range(n):
            for j in knn_sets[i]:
                if i in knn_sets[j]:  # Mutual
                    u, v = (i, j) if i < j else (j, i)
                    adj[u].add(v)
                    adj[v].add(u)
    else:
        # Standard k-NN
        for i in range(n):
            for jpos in range(idxs.shape[1]):
                j = int(idxs[i, jpos])
                if j == i:
                    continue
                d = float(dists[i, jpos])
                if d <= epsilon:
                    u, v = (i, j) if i < j else (j, i)
                    adj[u].add(v)
                    adj[v].add(u)
    return adj


# ---------------------------
# Cycle basis via spanning forest
# ---------------------------

def spanning_forest(adj: Dict[int, Set[int]]) -> Tuple[Dict[int, int], Dict[int, int], List[Tuple[int,int]]]:
    """Build a spanning forest using BFS. Returns (parent, depth, tree_edges).

    parent[u] = parent of u (or -1 for roots)
    depth[u] = BFS depth from component root
    tree_edges = list of undirected edges (u,v) with u<v that are in the forest
    """
    n = len(adj)
    visited = [False]*n
    parent = {i: -1 for i in adj}
    depth = {i: 0 for i in adj}
    tree_edges: List[Tuple[int,int]] = []

    for start in adj:
        if visited[start]:
            continue
        visited[start] = True
        q = deque([start])
        parent[start] = -1
        depth[start] = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    depth[v] = depth[u] + 1
                    tree_edges.append((u, v) if u < v else (v, u))
                    q.append(v)
    return parent, depth, tree_edges


def path_in_tree(u: int, v: int, parent: Dict[int,int], depth: Dict[int,int]) -> List[int]:
    """Return the unique simple path from u to v in the spanning forest defined by parent/depth."""
    # Move up from deeper node until same depth
    a, b = u, v
    path_a = [a]
    path_b = [b]
    da, db = depth[a], depth[b]
    while da > db:
        a = parent[a]
        path_a.append(a)
        da -= 1
    while db > da:
        b = parent[b]
        path_b.append(b)
        db -= 1
    # Climb in tandem to LCA
    while a != b:
        a = parent[a]
        b = parent[b]
        path_a.append(a)
        path_b.append(b)
    lca = a
    # Full path u -> ... -> LCA -> ... -> v
    path_b.pop()  # remove LCA duplicate
    path = path_a + path_b[::-1]
    return path


def fundamental_cycle_basis(adj: Dict[int, Set[int]]) -> List[List[int]]:
    """Return a fundamental cycle basis (list of cycles, each as a list of vertices in order).

    For every non-tree edge (u,v), the cycle is the unique tree path between u and v, then the closing edge to u.
    """
    parent, depth, tree_edges = spanning_forest(adj)
    tree_edge_set = { (u,v) if u < v else (v,u) for (u,v) in tree_edges }

    cycles: List[List[int]] = []

    # Iterate all edges (u<v); if not a tree edge and u,v are connected in tree, make the fundamental cycle.
    seen_edges = set()
    for u in adj:
        for v in adj[u]:
            if u < v:
                e = (u, v)
            else:
                e = (v, u)
            if e in seen_edges:
                continue
            seen_edges.add(e)

            if e in tree_edge_set:
                continue  # skip tree edges

            # They are in the same component iff they share an ancestor path. If either parent[u]==-1 and parent[v]==-1 and u!=v, no path; skip.
            # Our BFS forest ensures path_in_tree works as long as u and v are in the same connected component; we can detect that by walking up to root.
            def root(x: int) -> int:
                while parent[x] != -1:
                    x = parent[x]
                return x
            if root(u) != root(v):
                # Non-tree edge connecting different components should not exist; skip defensively
                continue

            path = path_in_tree(u, v, parent, depth)
            # Close the cycle back to u by adding u again at the end for clarity (optional)
            cycle = path + [u]
            cycles.append(cycle)

    # Normalize cycles to be simple (no immediate repeats) and with last vertex equal to first for convenience
    norm_cycles: List[List[int]] = []
    for cyc in cycles:
        if cyc[0] != cyc[-1]:
            cyc = cyc + [cyc[0]]
        # Remove redundant immediate repeats
        clean = [cyc[0]]
        for w in cyc[1:]:
            if w != clean[-1]:
                clean.append(w)
        if len(clean) >= 4:  # at least 3 edges + closing
            norm_cycles.append(clean)
    return norm_cycles


# ---------------------------
# Linking number computation
# ---------------------------

def orthonormal_basis_perp(w: Array) -> Tuple[Array, Array]:
    """Given unit vector w (3,), return two unit vectors u,v forming an ONB of the plane orthogonal to w."""
    w = np.asarray(w, dtype=float)
    w = w / np.linalg.norm(w)
    # Pick a vector not parallel to w
    t = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(w, t)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return u, v


def project_points(points: Array, w: Array) -> Array:
    """Project 3D points to 2D coordinates on plane perpendicular to w."""
    u, v = orthonormal_basis_perp(w)
    U = np.stack([u, v], axis=1)  # 3x2
    return points @ U  # (n,3) * (3,2) -> (n,2)


def _segment_intersection_params_2d(p1, p2, q1, q2, tol=1e-12):
    """Return (t,u) with p(t)=p1+t(p2-p1), q(u)=q1+u(q2-q1) for interior 2D intersection, else None."""
    dp = p2 - p1
    dq = q2 - q1
    A = np.array([[dp[0], -dq[0]], [dp[1], -dq[1]]], dtype=float)
    b = q1 - p1
    det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    if abs(det) < tol:
        return None
    t = ( b[0]*A[1,1] - b[1]*A[0,1]) / det
    u = ( A[0,0]*b[1] - A[1,0]*b[0]) / det
    if tol < t < 1.0 - tol and tol < u < 1.0 - tol:
        return t, u
    return None


def linking_number_via_projection(P, Q, w=None, tries=12, tol=1e-12):
    """
    Proper projection-based lk using over/under ordering.
    P, Q: arrays of shape (L,3), (M,3), closed (last == first).
    """
    P = np.asarray(P, float); Q = np.asarray(Q, float)

    def once(dir3):
        dir3 = dir3 / np.linalg.norm(dir3)
        projP = project_points(P, dir3)  # (.,2)
        projQ = project_points(Q, dir3)
        s = 0
        for i in range(len(P)-1):
            p1, p2 = projP[i], projP[i+1]
            dp3 = P[i+1] - P[i]
            for j in range(len(Q)-1):
                q1, q2 = projQ[j], projQ[j+1]
                dq3 = Q[j+1] - Q[j]
                params = _segment_intersection_params_2d(p1, p2, q1, q2, tol=tol)
                if params is None:
                    continue
                t, u = params
                # 3D points at the crossing
                xP = P[i] + t*dp3
                xQ = Q[j] + u*dq3
                hP = float(np.dot(xP, dir3))
                hQ = float(np.dot(xQ, dir3))
                if abs(hP - hQ) < 1e-12:
                    # near-degenerate (same height) – ignore or re-pick direction
                    continue
                # order tangents as (over, under)
                t_over, t_under = (dp3, dq3) if hP > hQ else (dq3, dp3)
                orient = float(np.dot(np.cross(t_over, t_under), dir3))
                if abs(orient) < 1e-14:
                    continue
                s += 1 if orient > 0 else -1
        return int(round(s/2))

    if w is not None:
        return once(np.asarray(w, float))

    last = None
    for _ in range(tries):
        d = np.random.normal(size=3)
        cur = once(d)
        if last is None:
            last = cur
        elif cur == last:
            return cur
        else:
            last = cur
    return None


def gauss_linking_number_numeric(P: Array, Q: Array, nsub: int = 4, eps: float = 1e-12) -> float:
    """
    Midpoint quadrature over subdivided segments.
    Returns a float close to an integer (divide by 4π inside).
    """
    P = np.asarray(P, dtype=float); Q = np.asarray(Q, dtype=float)
    assert P.shape[1] == 3 and Q.shape[1] == 3

    minis_P = []
    for i in range(len(P)-1):
        a, b = P[i], P[i+1]
        d = (b - a) / nsub
        for k in range(nsub):
            minis_P.append((a + (k + 0.5)*d, d))

    minis_Q = []
    for j in range(len(Q)-1):
        c, d = Q[j], Q[j+1]
        e = (d - c) / nsub
        for k in range(nsub):
            minis_Q.append((c + (k + 0.5)*e, e))

    total, comp = 0.0, 0.0  # Kahan summation
    for x, dx in minis_P:
        for y, dy in minis_Q:
            r = x - y
            r2 = float(r.dot(r))
            if r2 < eps:
                continue
            num = float(np.dot(r, np.cross(dx, dy)))
            inc = num / (r2**1.5)
            yk = inc - comp
            tk = total + yk
            comp = (tk - total) - yk
            total = tk
    return total / (4.0*math.pi)


def cycles_as_polylines(cycle: List[int], points: Array) -> Array:
    """Convert a cycle given as a vertex index sequence (last == first) into an array of 3D points."""
    return points[np.array(cycle, dtype=int)]


# ---------------------------
# Main detection function
# ---------------------------

def detect_linking(
    X1: Array,
    X2: Array,
    k: int = 6,
    epsilon: float = float("inf"),
    max_projection_tries: int = 12,
    mutual: bool = False,
) -> Dict:
    """Build spatial graphs from point clouds and decide if there exist loops C1 ⊂ G1, C2 ⊂ G2 with lk(C1,C2) ≠ 0.

    Returns a dict with keys:
        decision: bool
        witness: Optional[dict] with keys {"cycle1", "cycle2", "lk"}
        stats: information about graph sizes and cycle counts
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    assert X1.ndim == 2 and X1.shape[1] == 3, "X1 must be (n1, 3)"
    assert X2.ndim == 2 and X2.shape[1] == 3, "X2 must be (n2, 3)"

    adj1 = build_knn_graph(X1, k=k, epsilon=epsilon, mutual=mutual)
    adj2 = build_knn_graph(X2, k=k, epsilon=epsilon, mutual=mutual)

    cycles1 = fundamental_cycle_basis(adj1)
    cycles2 = fundamental_cycle_basis(adj2)

    # Early exit if no cycles in either graph
    if len(cycles1) == 0 or len(cycles2) == 0:
        return {
            "decision": False,
            "witness": None,
            "stats": {
                "n1": len(X1), "n2": len(X2),
                "m1": sum(len(s) for s in adj1.values())//2,
                "m2": sum(len(s) for s in adj2.values())//2,
                "b1": len(cycles1), "b2": len(cycles2),
            }
        }

    # Try all pairs; stop at first nonzero lk
    for i, cyc1 in enumerate(cycles1):
        P = cycles_as_polylines(cyc1, X1)
        for j, cyc2 in enumerate(cycles2):
            Q = cycles_as_polylines(cyc2, X2)

            lk_val = None
            # Try Gauss integral first
            lk_float = gauss_linking_number_numeric(P, Q, nsub=4)
            if abs(lk_float - round(lk_float)) < 0.1:
                lk_val = int(round(lk_float))
            else:
                # Fallback to projection method
                lk_val = linking_number_via_projection(P, Q, w=None, tries=max_projection_tries)

            if lk_val is not None and lk_val != 0:
                return {
                    "decision": True,
                    "witness": {
                        "cycle1": cyc1,
                        "cycle2": cyc2,
                        "lk": int(lk_val),
                    },
                    "stats": {
                        "n1": len(X1), "n2": len(X2),
                        "m1": sum(len(s) for s in adj1.values())//2,
                        "m2": sum(len(s) for s in adj2.values())//2,
                        "b1": len(cycles1), "b2": len(cycles2),
                        "tested_pairs_up_to": (i, j),
                    }
                }

    return {
        "decision": False,
        "witness": None,
        "stats": {
            "n1": len(X1), "n2": len(X2),
            "m1": sum(len(s) for s in adj1.values())//2,
            "m2": sum(len(s) for s in adj2.values())//2,
            "b1": len(cycles1), "b2": len(cycles2),
        }
    }