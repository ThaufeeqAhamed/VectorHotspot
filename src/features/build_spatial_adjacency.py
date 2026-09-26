#!/usr/bin/env python3
"""
Phase 8: Precompute H3 Spatial Adjacency Matrices
VectorHotspot Project

Constructs and caches row-normalized sparse CSR adjacency matrices for:
- W1: k=1 ring (6 immediate contiguous neighbors)
- W2: k=2 ring (12 concentric surrounding neighbors)
over the 620,742 H3 Resolution 7 cells in India.
"""

import pandas as pd
import numpy as np
from scipy import sparse
import h3
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
H3_GRID_PATH = PROCESSED_DATA_DIR / "india_h3_grid_res7.csv"
OUTPUT_NPZ = PROCESSED_DATA_DIR / "h3_adjacency_res7.npz"


def build_and_save_adjacency():
    print("=" * 70)
    print(" BUILDING H3 SPATIAL ADJACENCY MATRICES (RES 7)")
    print("=" * 70)

    print("Loading H3 grid...")
    grid = pd.read_csv(H3_GRID_PATH, usecols=['h3_index'])
    hexes = grid['h3_index'].values
    n = len(hexes)
    h3_to_idx = {h: i for i, h in enumerate(hexes)}
    print(f"Total hexagons: {n:,}")

    t0 = time.time()
    rows_k1, cols_k1, vals_k1 = [], [], []
    rows_k2, cols_k2, vals_k2 = [], [], []

    print("Iterating hexagons to construct k=1 and k=2 neighbor rings...")
    for i, h in enumerate(hexes):
        # Ring 1 (k=1, 6 neighbors)
        r1 = [h3_to_idx[nbr] for nbr in h3.grid_ring(h, 1) if nbr in h3_to_idx]
        if r1:
            w1 = 1.0 / len(r1)
            rows_k1.extend([i] * len(r1))
            cols_k1.extend(r1)
            vals_k1.extend([w1] * len(r1))

        # Ring 2 (k=2, 12 neighbors)
        r2 = [h3_to_idx[nbr] for nbr in h3.grid_ring(h, 2) if nbr in h3_to_idx]
        if r2:
            w2 = 1.0 / len(r2)
            rows_k2.extend([i] * len(r2))
            cols_k2.extend(r2)
            vals_k2.extend([w2] * len(r2))

        if (i + 1) % 150000 == 0 or i == n - 1:
            print(f"  Processed {i+1:,}/{n:,} hexagons ({(i+1)/n*100:.1f}%)")

    W1 = sparse.csr_matrix((vals_k1, (rows_k1, cols_k1)), shape=(n, n), dtype=np.float32)
    W2 = sparse.csr_matrix((vals_k2, (rows_k2, cols_k2)), shape=(n, n), dtype=np.float32)

    t_build = time.time() - t0
    print(f"\nConstructed in {t_build:.1f}s:")
    print(f"  W1 (k=1 ring): {W1.nnz:,} edges ({W1.nnz/n:.2f} avg neighbors/cell)")
    print(f"  W2 (k=2 ring): {W2.nnz:,} edges ({W2.nnz/n:.2f} avg neighbors/cell)")

    print(f"Saving adjacency matrices to {OUTPUT_NPZ}...")
    np.savez_compressed(
        OUTPUT_NPZ,
        W1_data=W1.data, W1_indices=W1.indices, W1_indptr=W1.indptr, W1_shape=W1.shape,
        W2_data=W2.data, W2_indices=W2.indices, W2_indptr=W2.indptr, W2_shape=W2.shape
    )
    print(f"File size: {OUTPUT_NPZ.stat().st_size / (1024*1024):.2f} MB")
    print("Done.")


if __name__ == "__main__":
    build_and_save_adjacency()
